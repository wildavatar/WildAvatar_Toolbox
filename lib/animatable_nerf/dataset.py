import torch.utils.data as data
import numpy as np
import json
import os
import imageio
import cv2
from lib.config import cfg
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils
import json
from lib.smpl.smpl_numpy import SMPL

class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        super(Dataset, self).__init__()

        self.data_root = os.path.join(data_root, human)
        self.human = human
        self.split = split

        with open(os.path.join(self.data_root, ann_file), "r+") as f:
            annots = json.load(f)

        start = cfg.begin_ith_frame
        interval = cfg.frame_interval
        if cfg.eval:
            start = start + 1

        self.ims = np.array([
            np.array(id + ".png") for id in list(annots.keys())
        ])[start::interval]
        self.cam_inds = np.array([
            np.arange(len(self.ims))
        ]).ravel()
        self.cams = {}
        K_all, T_all, R_all, D_all = [], [], [], []
        cams_intri = np.array([
            np.array(annots[id]['cam_intrinsics'])[None] for id in list(annots.keys())
        ])[start::interval]
        cams_extri = np.array([
            np.array(annots[id]['cam_extrinsics'])[None] for id in list(annots.keys())
        ])[start::interval]
        for i in range(len(self.ims)):
            K_all.append(cams_intri[i].reshape(3,3).astype(np.float32))
            R_all.append(cams_extri[i,...,:3,:3].reshape(3,3).astype(np.float32))
            T_all.append(cams_extri[i,...,:3,3:4].reshape(3,1).astype(np.float32))
            D_all.append(np.zeros((5,1)).astype(np.float32))
        self.cams['K'] = K_all
        self.cams['R'] = R_all
        self.cams['T'] = T_all
        self.cams['D'] = D_all
        
        self.num_cams = 1
        self.lbs_root = 'data/zju_mocap/lbs'
        joints = np.load(os.path.join(self.lbs_root, 'joints.npy'))
        self.joints = joints.astype(np.float32)
        self.parents = np.load(os.path.join(self.lbs_root, 'parents.npy'))
        weights = np.load(os.path.join(self.lbs_root, 'weights.npy'))
        self.weights = weights.astype(np.float32)
        self.big_A = self.load_bigpose()

        if cfg.test_novel_pose or cfg.aninerf_animation:
            training_joints_path = os.path.join(self.lbs_root, 'training_joints.npy')
            if os.path.exists(training_joints_path):
                self.training_joints = np.load(training_joints_path)

        self.nrays = cfg.N_rand
        self.smpl_model = SMPL(sex='neutral', model_dir='data/smplx/smpl/SMPL_NEUTRAL.pkl')
        
    def load_bigpose(self):
        big_poses = np.zeros([len(self.joints), 3]).astype(np.float32).ravel()
        angle = 30
        big_poses[5] = np.deg2rad(angle)
        big_poses[8] = np.deg2rad(-angle)
        big_poses = big_poses.reshape(-1, 3)
        big_A = if_nerf_dutils.get_rigid_transformation(
            big_poses, self.joints, self.parents)
        big_A = big_A.astype(np.float32)
        return big_A

    def get_mask(self, index):
        msk_path = os.path.join(self.data_root, 'masks',
                                self.ims[index])
        msk_cihp = imageio.imread(msk_path)
        if len(msk_cihp.shape) == 3:
            msk_cihp = msk_cihp[..., 0]
        msk_cihp = (msk_cihp != 0).astype(np.uint8)
        msk = msk_cihp
        orig_msk = msk.copy()

        return msk, orig_msk
    
    def prepare_smpl_params(self, pose_index):
        with open(os.path.join(self.data_root, "metadata.json"), "r+") as f:
            params_ori = json.load(f)
        params = {}
        params['shapes'] = np.array(params_ori[pose_index]['betas']).astype(np.float32)
        params['poses'] = np.array(params_ori[pose_index]['poses']).astype(np.float32)[None]
        params['R'] = np.eye(3).astype(np.float32)
        params['Th'] = np.zeros(3).astype(np.float32)
        return params
    
    def prepare_input(self, i):
        # read xyz in the world coordinate system
        params = self.prepare_smpl_params(i)
        xyz, _ = self.smpl_model(params['poses'], params['shapes'].reshape(-1))
        wxyz = (np.matmul(xyz, params['R'].transpose()) + params['Th']).astype(np.float32)
        # transform smpl from the world coordinate to the smpl coordinate
        
        Th = params['Th'].astype(np.float32)
        R = params['R'].astype(np.float32)
        Rh = cv2.Rodrigues(R)[0].astype(np.float32)
        pxyz = np.dot(wxyz - Th, R).astype(np.float32)

        # calculate the skeleton transformation
        poses = params['poses'].reshape(-1, 3)
        joints = self.joints
        parents = self.parents
        A, canonical_joints = if_nerf_dutils.get_rigid_transformation(
            poses, joints, parents, return_joints=True)

        posed_joints = np.dot(canonical_joints, R.T) + Th

        # find the nearest training frame
        if (cfg.test_novel_pose or cfg.aninerf_animation) and hasattr(self, "training_joints"):
            nearest_frame_index = np.linalg.norm(self.training_joints -
                                                 posed_joints[None],
                                                 axis=2).mean(axis=1).argmin()
        else:
            nearest_frame_index = 0

        poses = poses.ravel().astype(np.float32)

        return wxyz, pxyz, A, Rh, Th, poses, nearest_frame_index

    def __getitem__(self, index):
        img_path = os.path.join(self.data_root, "images", self.ims[index])
        img = imageio.imread(img_path).astype(np.float32) / 255.
        msk, orig_msk = self.get_mask(index)

        H, W = img.shape[:2]
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        orig_msk = cv2.resize(orig_msk, (W, H),
                              interpolation=cv2.INTER_NEAREST)

        cam_ind = self.cam_inds[index]
        K = np.array(self.cams['K'][cam_ind])
        D = np.array(self.cams['D'][cam_ind])
        img = cv2.undistort(img, K, D)
        msk = cv2.undistort(msk, K, D)
        orig_msk = cv2.undistort(orig_msk, K, D)

        R = np.array(self.cams['R'][cam_ind])
        T = np.array(self.cams['T'][cam_ind])

        # reduce the image resolution by ratio
        H, W = int(img.shape[0] * cfg.ratio), int(img.shape[1] * cfg.ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        orig_msk = cv2.resize(orig_msk, (W, H),
                              interpolation=cv2.INTER_NEAREST)
        if cfg.mask_bkgd:
            img[msk == 0] = 0
        K[:2] = K[:2] * cfg.ratio

        i = os.path.basename(img_path)[:-4]
        # read v_shaped
        if cfg.get('use_bigpose', False):
            vertices_path = os.path.join(self.lbs_root, 'bigpose_vertices.npy')
        else:
            vertices_path = os.path.join(self.lbs_root, 'tvertices.npy')
        tvertices = np.load(vertices_path).astype(np.float32)
        tbounds = if_nerf_dutils.get_bounds(tvertices)

        wpts, pvertices, A, Rh, Th, poses, nearest_frame_index = self.prepare_input(i)

        pbounds = if_nerf_dutils.get_bounds(pvertices)
        wbounds = if_nerf_dutils.get_bounds(wpts)

        rgb, ray_o, ray_d, near, far, coord, mask_at_box = if_nerf_dutils.sample_ray_h36m(
            img, msk, K, R, T, wbounds, self.nrays, self.split)

        if cfg.erode_edge:
            orig_msk = if_nerf_dutils.crop_mask_edge(orig_msk)
        occupancy = orig_msk[coord[:, 0], coord[:, 1]]

        # nerf
        ret = {
            'rgb': rgb,
            'occupancy': occupancy,
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            'mask_at_box': mask_at_box
        }

        # blend weight
        meta = {
            'A': A,
            'big_A': self.big_A,
            'poses': poses,
            'weights': self.weights,
            'tvertices': tvertices,
            'pvertices': pvertices,
            'pbounds': pbounds,
            'wbounds': wbounds,
            'tbounds': tbounds
        }
        ret.update(meta)

        # transformation
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        meta = {'R': R, 'Th': Th, 'H': H, 'W': W}
        ret.update(meta)

        latent_index = index // self.num_cams
        bw_latent_index = index // self.num_cams
        if cfg.test_novel_pose or cfg.aninerf_animation:
            if 'h36m' in self.data_root:
                latent_index = 0
            else:
                latent_index = cfg.num_train_frame - 1
            latent_index = nearest_frame_index
        meta = {
            'latent_index': latent_index,
            'bw_latent_index': bw_latent_index,
            'frame_index': index,
            'cam_ind': cam_ind
        }
        ret.update(meta)

        return ret

    def __len__(self):
        return len(self.ims)
