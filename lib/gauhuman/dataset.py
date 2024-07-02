from torch.utils.data import Dataset
import numpy as np
import os
import imageio
import cv2
from smpl.smpl_numpy import SMPL
import json

from utils.graphics_utils import getWorld2View2, getProjectionMatrix_refine, focal2fov
from utils.general_utils import EasyDict
import torch

def get_rays(H, W, K, R, T):
    # calculate the camera origin
    rays_o = -np.dot(R.T, T).ravel()
    # calculate the world coodinates of pixels
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
    pixel_world = np.dot(pixel_camera - T.ravel(), R)
    # calculate the ray direction
    rays_d = pixel_world - rays_o[None, None]
    rays_o = np.broadcast_to(rays_o, rays_d.shape)
    return rays_o, rays_d

def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d

def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy

def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 4]]], 1) # 4,5,7,6,4
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask

def get_near_far(bounds, ray_o, ray_d):
    """calculate intersections with 3d bounding box"""
    bounds = bounds + np.array([-0.01, 0.01])[:, None]
    ray_d[ray_d==0.0] = 1e-8
    nominator = bounds[None] - ray_o[:, None]
    # calculate the step of intersections at six planes of the 3d bounding box
    d_intersect = (nominator / ray_d[:, None]).reshape(-1, 6)
    # calculate the six interections
    p_intersect = d_intersect[..., None] * ray_d[:, None] + ray_o[:, None]
    # calculate the intersections located at the 3d bounding box
    min_x, min_y, min_z, max_x, max_y, max_z = bounds.ravel()
    eps = 1e-6
    p_mask_at_box = (p_intersect[..., 0] >= (min_x - eps)) * \
                    (p_intersect[..., 0] <= (max_x + eps)) * \
                    (p_intersect[..., 1] >= (min_y - eps)) * \
                    (p_intersect[..., 1] <= (max_y + eps)) * \
                    (p_intersect[..., 2] >= (min_z - eps)) * \
                    (p_intersect[..., 2] <= (max_z + eps))
    # obtain the intersections of rays which intersect exactly twice
    mask_at_box = p_mask_at_box.sum(-1) == 2
    # TODO
    # mask_at_box = p_mask_at_box.sum(-1) >= 1

    p_intervals = p_intersect[mask_at_box][p_mask_at_box[mask_at_box]].reshape(
        -1, 2, 3)

    # calculate the step of intersections
    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]
    norm_ray = np.linalg.norm(ray_d, axis=1)
    d0 = np.linalg.norm(p_intervals[:, 0] - ray_o, axis=1) / norm_ray
    d1 = np.linalg.norm(p_intervals[:, 1] - ray_o, axis=1) / norm_ray
    near = np.minimum(d0, d1)
    far = np.maximum(d0, d1)

    return near, far, mask_at_box

def sample_ray_YouTube_batch(img, msk, K, R, T, bounds, image_scaling, white_back=False, use_matting=False):

    H, W = img.shape[:2]
    H, W = int(H * image_scaling), int(W * image_scaling)

    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    if use_matting:
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_LANCZOS4)
    else:
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)

    K_scale = np.copy(K)
    K_scale[:2, :3] = K_scale[:2, :3] * image_scaling
    ray_o, ray_d = get_rays(H, W, K_scale, R, T)
    pose = np.concatenate([R, T], axis=1)
    
    bound_mask = get_bound_2d_mask(bounds, K_scale, pose, H, W)

    mask_bkgd = True

    msk = msk * bound_mask
    bound_mask[msk == 100] = 0
    if mask_bkgd:
        # img[bound_mask != 1] = 1 if white_back else 0
        img[bound_mask != 1] = 0 

    # rgb = img.reshape(-1, 3).astype(np.float32)
    ray_o = ray_o.reshape(-1, 3).astype(np.float32)
    ray_d = ray_d.reshape(-1, 3).astype(np.float32)
    near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
    near = near.astype(np.float32)
    far = far.astype(np.float32)

    near_all = np.zeros_like(ray_o[:,0])
    far_all = np.ones_like(ray_o[:,0])
    near_all[mask_at_box] = near 
    far_all[mask_at_box] = far 
    near = near_all
    far = far_all

    coord = np.zeros([len(ray_o), 2]).astype(np.int64)
    bkgd_msk = msk

    return img, ray_o, ray_d, near, far, coord, mask_at_box, bkgd_msk, bound_mask


class YouTubeDatasetBatch(Dataset):
    def __init__(self, data_root=None, split='test', multi_person=False, num_instance=1, poses_start=0, poses_interval=1, poses_num=20, image_scaling=1, white_back=False, sample_obs_view=False, fix_obs_view=True, resolution=None, same_view=False, humanlist_txt=None, use_smplmask=False, use_sview=False, use_matting=False, metadata_json="metadata.json"):
        super(YouTubeDatasetBatch, self).__init__()
        self.data_root = data_root
        self.split = split
        self.image_scaling = image_scaling
        self.white_back = white_back
        self.sample_obs_view = sample_obs_view
        self.fix_obs_view = fix_obs_view
        self.camera_view_num = 1
        self.use_smplmask = use_smplmask
        self.use_sview = use_sview
        self.use_matting = use_matting
        self.metadata_json = metadata_json
        if humanlist_txt is None:
            self.humanlist_txt = "human_list.txt"
        else:
            self.humanlist_txt = humanlist_txt
        self.img_dir = "images"
        if self.use_matting:
            self.mask_dir = "mam_masks"
        else:
            self.mask_dir = "masks"
        self.smpl_mask_dir = "smpl_masks"
        self.output_view = [x for x in range(self.camera_view_num)]
        self.same_view = same_view

        self.poses_start = poses_start # start index 0
        self.poses_interval = poses_interval # interval 1
        self.poses_num = poses_num # number of used poses

        # observation pose and view
        self.obs_pose_index = None
        self.obs_view_index = None

        self.multi_person = multi_person

        if self.multi_person:
            humans_data_root = os.path.dirname(data_root)
            self.humans_list = os.path.join(humans_data_root, self.humanlist_txt)
            with open(self.humans_list) as f:
                humans_name = f.readlines()[0:num_instance]
        
        self.all_humans = [data_root] if not multi_person else [os.path.join(humans_data_root, x.strip()) for x in humans_name]
        print('num of subjects: ', len(self.all_humans))
        self.num_instance = len(self.all_humans)
        
        self.cams_all_intri = []
        self.cams_all_extri = []
        self.ims_all = []
        for subject_root in self.all_humans:
            ann_file = os.path.join(subject_root, self.metadata_json)
            with open(ann_file, "r+") as f:
                annots = json.load(f)
            cams_intri = np.array([
                np.array(annots[id]['cam_intrinsics'])[None] for id in list(annots.keys())[self.poses_start:self.poses_start + self.poses_num * self.poses_interval][::self.poses_interval]
            ])
            self.cams_all_intri.append(cams_intri)
            cams_extri = np.array([
                np.array(annots[id]['cam_extrinsics'])[None] for id in list(annots.keys())[self.poses_start:self.poses_start + self.poses_num * self.poses_interval][::self.poses_interval]
            ])
            self.cams_all_extri.append(cams_extri)
            # ims = np.array([
            #     np.array(annots[id]['imgname'])[None] for id in list(annots.keys())[self.poses_start:self.poses_start + self.poses_num * self.poses_interval][::self.poses_interval]
            # ])
            ims = np.array([
                np.array(id + ".png")[None] for id in list(annots.keys())[self.poses_start:self.poses_start + self.poses_num * self.poses_interval][::self.poses_interval]
            ])
            self.ims_all.append(ims)

        # prepare t pose and vertex
        self.smpl_model = SMPL(sex='neutral', model_dir='assets/SMPL_NEUTRAL.pkl')
        self.big_pose_params = self.big_pose_params()
        t_vertices, _ = self.smpl_model(self.big_pose_params['poses'], self.big_pose_params['shapes'].reshape(-1))
        self.t_vertices = t_vertices.astype(np.float32)
        min_xyz = np.min(self.t_vertices, axis=0)
        max_xyz = np.max(self.t_vertices, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        min_xyz[2] -= 0.1
        max_xyz[2] += 0.1
        self.t_world_bounds = np.stack([min_xyz, max_xyz], axis=0)
        # self.test_bbox_size()

    def test_bbox_size(self, min_length=16):
        import tqdm
        bad_list = []
        for index in tqdm.tqdm(range(self.__len__()*self.camera_view_num)):
            batch = self.__getitem__(index)
            bound_mask = batch['bound_mask']
            x, y, w, h = cv2.boundingRect(bound_mask[0].astype(np.uint8))
            original_image = batch['original_image']
            img = original_image[:, y:y+h, x:x+w]
            _, this_w, this_h = img.shape
            if this_w < min_length or this_h < min_length:
                print(batch['image_name'])
                bad_list.append(batch['image_name'] + "\n")
        with open("temp.txt","w+") as f:
            f.writelines(bad_list)
        
    def get_mask(self, pose_index, view_index):
        msk_path = os.path.join(self.data_root, self.mask_dir,
                                self.ims[pose_index][view_index].split('/')[-1])
        msk = imageio.imread(msk_path)
        if len(msk.shape) == 3:
            msk = msk[...,0]
        msk[msk!=0]=255
        return msk
    
    def get_smpl_mask(self, pose_index, view_index):
        msk_path = os.path.join(self.data_root, self.smpl_mask_dir,
                                self.ims[pose_index][view_index].split('/')[-1])
        msk = imageio.imread(msk_path)
        if len(msk.shape) == 3:
            msk = msk[...,0]
        msk[msk!=0]=255
        return msk
    
    def prepare_smpl_params(self, smpl_path, pose_index):
        with open(smpl_path, "r+") as f:
            params_ori = json.load(f)
        params = {}
        params['shapes'] = np.array(params_ori[pose_index]['betas']).astype(np.float32)
        params['poses'] = np.array(params_ori[pose_index]['poses']).astype(np.float32)[None]
        params['R'] = np.eye(3).astype(np.float32)
        params['Th'] = np.zeros(3).astype(np.float32)
        return params
    
    def prepare_input(self, smpl_path, pose_index):

        params = self.prepare_smpl_params(smpl_path, pose_index)
        xyz, _ = self.smpl_model(params['poses'], params['shapes'].reshape(-1))
        xyz = (np.matmul(xyz, params['R'].transpose()) + params['Th']).astype(np.float32)
        vertices = xyz

        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        world_bounds = np.stack([min_xyz, max_xyz], axis=0)

        return world_bounds, vertices, params

    def big_pose_params(self):

        big_pose_params = {}
        # big_pose_params = copy.deepcopy(params)
        big_pose_params['R'] = np.eye(3).astype(np.float32)
        big_pose_params['Th'] = np.zeros((1,3)).astype(np.float32)
        big_pose_params['shapes'] = np.zeros((1,10)).astype(np.float32)
        big_pose_params['poses'] = np.zeros((1,72)).astype(np.float32)
        big_pose_params['poses'][0, 5] = 45/180*np.array(np.pi)
        big_pose_params['poses'][0, 8] = -45/180*np.array(np.pi)
        big_pose_params['poses'][0, 23] = -30/180*np.array(np.pi)
        big_pose_params['poses'][0, 26] = 30/180*np.array(np.pi)

        return big_pose_params
    
    def get_meta_params(self):
        num_param = self.ims_all[0].shape[0]
        smpl_path = os.path.join(self.data_root, self.metadata_json)
        index_map = {}
        pose_params = []
        for i in range(num_param):
            frame_id = str(self.ims_all[0][i,0]).split("/")[-1][:-4]
            index_map[frame_id] = i
            world_bounds, vertices, params = self.prepare_input(smpl_path, frame_id)
            pose_params.append(params['poses'].astype(np.float32))
        pose_params = np.array(pose_params)
        shape_params = np.expand_dims(params['shapes'].astype(np.float32), axis=0)
        
        return index_map, torch.from_numpy(pose_params), torch.from_numpy(shape_params)
    
    def save_meta_params(self, index_map, pose_params, shape_params):
        smpl_path = os.path.join(self.data_root, self.metadata_json)
        new_smpl_path = os.path.join(self.data_root, "metadata_rec.json")
        with open(smpl_path, "r+") as f:
            metadata = json.load(f)
            
        for k,v in index_map.items():
            metadata[k]['poses'] = pose_params[v][0].detach().cpu().numpy().tolist()
        
        with open(new_smpl_path, "w+") as f:
            json.dump(metadata, f)
    
    def sview_select(self):
        angles = np.arange(-np.pi, np.pi, 2*np.pi/20)
        tgt_angle = np.random.choice(angles)
        src_angles = []
        for i in range(self.cams_extri.shape[0]):
            src_angles.append(cv2.Rodrigues(self.cams_extri[i, 0, :3, :3])[0][2])
        src_angles = np.array(src_angles)
        angle_diff = src_angles - tgt_angle
        circles = (angle_diff+np.pi/2) // (np.pi)
        angle_diff -= circles*(np.pi)
        return np.argmin(abs(angle_diff))
    
    def __getitem__(self, index):
        """
            pose_index: [0, number of used poses), the pose index of selected poses
            view_index: [0, number of all view)
            mask_at_box_all: mask of 2D bounding box which is the 3D bounding box projection 
                training: all one array, not fixed length, no use for training
                Test: zero and one array, fixed length which can be reshape to (H,W)
            bkgd_msk_all: mask of foreground and background
                trainning: for acc loss
                test: no use
        """

        instance_idx = index // (self.poses_num) if self.multi_person else 0
        pose_index = (index % self.poses_num)
        view_index = 0

        self.data_root = self.all_humans[instance_idx]
        self.ims = self.ims_all[instance_idx]
        self.cams_intri = self.cams_all_intri[instance_idx]
        self.cams_extri = self.cams_all_extri[instance_idx]

        if pose_index >= len(self.ims):
            pose_index = np.random.randint(len(self.ims))
        if self.obs_pose_index is not None and self.same_view:
            pose_index = self.obs_pose_index

        img_all, ray_o_all, ray_d_all, near_all, far_all = [], [], [], [], []
        mask_at_box_all, bkgd_msk_all, mask_at_box_large_all = [], [], []
        obs_img_all, obs_K_all, obs_R_all, obs_T_all = [], [], [], []
            
        # Load image, mask, K, D, R, T
        img_path = os.path.join(self.data_root, self.img_dir, self.ims[pose_index][view_index].split("/")[-1])
        img = np.array(imageio.imread(img_path).astype(np.float32) / 255.)
        H, W = img.shape[:2]
        # # TODO
        # if min(H, W)> 512:
        #     self.image_scaling = 512 / max(H, W)
        # else:
        #     self.image_scaling = 1
        msk = np.array(self.get_mask(pose_index, view_index)) / 255.
        img[msk == 0] = 1 if self.white_back else 0

        K = np.array(self.cams_intri[pose_index, view_index])
        R = np.array(self.cams_extri[pose_index, view_index][:3,:3])
        T = np.array(self.cams_extri[pose_index, view_index][:3,3:4])

        # rescaling
        if self.image_scaling != 1:
            H, W = int(H * self.image_scaling), int(W * self.image_scaling)
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            if self.use_matting:
                msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_LANCZOS4)
            else:
                msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
            K[:2] = K[:2]*self.image_scaling

        # Prepare the smpl input, including the current pose and canonical pose
        # i: the pose index of all poses this person has, not the pose index of getitem input
        i = os.path.basename(img_path)[:-4]

        smpl_path = os.path.join(self.data_root, self.metadata_json)
        world_bounds, vertices, params = self.prepare_input(smpl_path, i)
        params = {
            # 'poses': np.squeeze(np.expand_dims(params['poses'].astype(np.float32), axis=0), axis=-1),
            'poses': params['poses'].astype(np.float32),
            'R': params['R'].astype(np.float32),
            'Th': params['Th'].astype(np.float32).reshape(1,3),
            'shapes': np.expand_dims(params['shapes'].astype(np.float32), axis=0)}

        # Sample rays in target space world coordinate
        img, ray_o, ray_d, near, far, coord_, mask_at_box, bkgd_msk, bound_mask = sample_ray_YouTube_batch(
                img, msk, K, R, T, world_bounds, 1.0, use_matting=self.use_matting)

        mask_at_box_large = mask_at_box

        # vising joints for debuging
        # joints3d = self.smpl_model(pose=params['poses'], beta=params['shapes'].reshape(-1))[1]
        # joints3d_cam = np.dot(joints3d, np.linalg.inv(R)) + T.ravel()
        # joints2d = np.dot(joints3d_cam, K.T)
        # joints2d = joints2d / joints2d[:,2:]
        # this_img = img.copy() * 255
        # for joint in joints2d:
        #     cv2.circle(this_img, (int(joint[0]), int(joint[1])), 2, (255,0,0), 2)
        # cv2.imwrite("tmp.png", this_img)
        # print("ok")
        
        img = np.transpose(img, (2,0,1))

        # target view
        img_all.append(img)
        ray_o_all.append(ray_o)
        ray_d_all.append(ray_d)
        near_all.append(near)
        far_all.append(far)
        mask_at_box_all.append(mask_at_box)
        bkgd_msk_all.append(bkgd_msk)
        mask_at_box_large_all.append(mask_at_box_large)


        
        # training obs view data preparation
        if self.split == 'train':
            if self.sample_obs_view:
                self.obs_view_index = np.random.randint(self.camera_view_num)
            elif self.fix_obs_view:
                self.obs_view_index = 0

        if self.obs_pose_index is not None:
            obs_pose_index = int(self.obs_pose_index)
        elif self.same_view:
            obs_pose_index = pose_index
        elif self.use_sview:
            obs_pose_index = self.sview_select()
        else:
            obs_pose_index = np.random.randint(self.poses_num)

        # Load image, mask, K, D, R, T in observation space
        obs_img_path = os.path.join(self.data_root, self.img_dir, self.ims[obs_pose_index][self.obs_view_index].split("/")[-1])
        obs_img = np.array(imageio.imread(obs_img_path).astype(np.float32) / 255.)
        obs_msk = np.array(self.get_mask(obs_pose_index, self.obs_view_index)) / 255.
        obs_img[obs_msk == 0] = 1 if self.white_back else 0

        obs_K = np.array(self.cams_intri[obs_pose_index, self.obs_view_index])
        obs_R = np.array(self.cams_extri[obs_pose_index, self.obs_view_index][:3,:3])
        obs_T = np.array(self.cams_extri[obs_pose_index, self.obs_view_index][:3,3:4])
        # rescaling
        if self.image_scaling != 1:
            obs_img = cv2.resize(obs_img, (W, H), interpolation=cv2.INTER_AREA)
            if self.use_matting:
                obs_msk = cv2.resize(obs_msk, (W, H), interpolation=cv2.INTER_LANCZOS4)
            else:
                obs_msk = cv2.resize(obs_msk, (W, H), interpolation=cv2.INTER_NEAREST)
            obs_K[:2] = obs_K[:2]*self.image_scaling

        obs_img = np.transpose(obs_img, (2,0,1))

        # Prepare smpl in the observation space
        # i: the pose index of all poses this person has, not the pose index of getitem input
        obs_i = os.path.basename(obs_img_path)[:-4]
        _, obs_vertices, obs_params = self.prepare_input(smpl_path, obs_i)
        obs_params = {
            # 'poses': np.squeeze(np.expand_dims(obs_params['poses'].astype(np.float32), axis=0), axis=-1),
            'poses': obs_params['poses'].astype(np.float32),
            'R': obs_params['R'].astype(np.float32),
            'Th': obs_params['Th'].astype(np.float32).reshape(1,3),
            'shapes': np.expand_dims(obs_params['shapes'].astype(np.float32), axis=0)}

        # obs view
        obs_img_all.append(obs_img)
        obs_K_all.append(obs_K)
        obs_R_all.append(obs_R)
        obs_T_all.append(obs_T)

        # target view
        img_all = np.stack(img_all, axis=0)
        ray_o_all = np.stack(ray_o_all, axis=0)
        ray_d_all = np.stack(ray_d_all, axis=0)
        near_all = np.stack(near_all, axis=0)[...,None]
        far_all = np.stack(far_all, axis=0)[...,None]
        mask_at_box_all = np.stack(mask_at_box_all, axis=0)
        bkgd_msk_all = np.stack(bkgd_msk_all, axis=0)
        mask_at_box_large_all = np.stack(mask_at_box_large_all, axis=0)

        # obs view 
        obs_img_all = np.stack(obs_img_all, axis=0)
        obs_K_all = np.stack(obs_K_all, axis=0)
        obs_R_all = np.stack(obs_R_all, axis=0)
        obs_T_all = np.stack(obs_T_all, axis=0)

        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        w2c = np.eye(4)
        w2c[:3,:3] = R
        w2c[:3,3:4] = T
        # get the world-to-camera transform and set R, T
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3:4]
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        w2c = np.eye(4)
        w2c[:3,:3] = obs_R
        w2c[:3,3:4] = obs_T
        # get the world-to-camera transform and set R, T
        obs_R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        obs_T = w2c[:3, 3:4]
        
        focalX = K[0,0]
        focalY = K[1,1]
        trans = [0.0, 0.0, 0.0]
        scale = 1.0
        znear = 0.001
        zfar = 1000
        world_view_transform = torch.tensor(getWorld2View2(R, T[:,0], trans, scale)).transpose(0, 1)
        projection_matrix = getProjectionMatrix_refine(torch.Tensor(K), H, W, znear, zfar).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        obs_world_view_transform = torch.tensor(getWorld2View2(obs_R, obs_T[:,0], trans, scale)).transpose(0, 1)
        obs_projection_matrix = getProjectionMatrix_refine(torch.Tensor(obs_K), H, W, znear, zfar).transpose(0, 1)
        obs_full_proj_transform = (obs_world_view_transform.unsqueeze(0).bmm(obs_projection_matrix.unsqueeze(0))).squeeze(0)
        
        camera_center = world_view_transform.inverse()[3, :3]
        FovX = focal2fov(focalX, H)
        FovY = focal2fov(focalY, W)
        img.transpose(1,2,0)
        obs_img.transpose(1,2,0)
        ret = EasyDict({
            # shared settings
            'FoVx': FovX,
            'FoVy': FovY,
            'big_pose_smpl_param': self.big_pose_params,
            'big_pose_world_bound': self.t_world_bounds,
            'big_pose_world_vertex': self.t_vertices,
            'camera_center': camera_center,
            'image_height': H,
            'image_width': W,
            'scale': scale,
            'z_far': zfar,
            'z_near': znear,
            'trans': trans,
            # 'pose_id': index,

            # tgt settings
            'K': K,
            'R': R,
            'T': T,
            'smpl_param': params,
            'world_vertex': vertices,
            'original_image': img,
            'bkgd_mask': bkgd_msk[None],
            'world_view_transform': world_view_transform,
            'full_proj_transform': full_proj_transform,
            'projection_matrix': projection_matrix,
            'bound_mask': bound_mask[None],

            # obs settings
            'obs_K': obs_K,
            'obs_R': obs_R,
            'obs_T': obs_T,
            'obs_smpl_param': obs_params,
            'obs_world_vertex': obs_vertices,
            'obs_img': obs_img,
            'obs_msk': obs_msk,
            'obs_world_view_transform': obs_world_view_transform,
            'obs_full_proj_transform': obs_full_proj_transform,
            'obs_projection_matrix': obs_projection_matrix,
            # 'obs_bound_mask': obs_bound_mask[None],
            
            # test settings
            'pose_index': pose_index,
            'obs_pose_index': obs_pose_index,
            'mask_at_box_large_all': mask_at_box_large_all,
            'image_name': img_path,
            # 'world_bound': world_bounds,
        })
        if self.use_smplmask:
            smplmsk = np.array(self.get_smpl_mask(pose_index, view_index)) / 255.
            if self.image_scaling != 1:
                smplmsk = cv2.resize(smplmsk, (W, H), interpolation=cv2.INTER_NEAREST)
            smplmsk = smplmsk * bound_mask
            unsure_msk = np.logical_and(smplmsk, np.logical_not(msk))
            ret['unsure_mask'] = unsure_msk
        return ret

    def __len__(self):
        return self.num_instance * self.poses_num * self.camera_view_num
