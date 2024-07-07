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

def sample_ray_WildAvatar_batch(img, msk, K, R, T, bounds):
    H, W = img.shape[:2]

    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)

    K_scale = np.copy(K)
    ray_o, ray_d = get_rays(H, W, K_scale, R, T)
    pose = np.concatenate([R, T], axis=1)
    
    bound_mask = get_bound_2d_mask(bounds, K_scale, pose, H, W)

    mask_bkgd = True

    msk = msk * bound_mask
    bound_mask[msk == 100] = 0
    if mask_bkgd:
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


class WildAvatarDatasetBatch(Dataset):
    def __init__(self, data_root=None, poses_start=0, poses_interval=1, poses_num=20, white_back=False, metadata_json="metadata.json"):
        super(WildAvatarDatasetBatch, self).__init__()
        self.data_root = data_root
        self.white_back = white_back
        self.camera_view_num = 1
        self.metadata_json = metadata_json
        self.img_dir = "images"
        self.mask_dir = "masks"
        self.output_view = [x for x in range(self.camera_view_num)]

        self.poses_start = poses_start # start index 0
        self.poses_interval = poses_interval # interval 1
        self.poses_num = poses_num # number of used poses

        self.all_humans = [data_root]
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

    def get_mask(self, pose_index, view_index):
        msk_path = os.path.join(self.data_root, self.mask_dir,
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
    
    def __getitem__(self, index):
        instance_idx = 0
        pose_index = (index % self.poses_num)
        view_index = 0

        self.data_root = self.all_humans[instance_idx]
        self.ims = self.ims_all[instance_idx]
        self.cams_intri = self.cams_all_intri[instance_idx]
        self.cams_extri = self.cams_all_extri[instance_idx]

        if pose_index >= len(self.ims):
            pose_index = np.random.randint(len(self.ims))

        img_all, ray_o_all, ray_d_all, near_all, far_all = [], [], [], [], []
        mask_at_box_all, bkgd_msk_all, mask_at_box_large_all = [], [], []
            
        # Load image, mask, K, D, R, T
        img_path = os.path.join(self.data_root, self.img_dir, self.ims[pose_index][view_index].split("/")[-1])
        img = np.array(imageio.imread(img_path).astype(np.float32) / 255.)
        H, W = img.shape[:2]
        msk = np.array(self.get_mask(pose_index, view_index)) / 255.
        img[msk == 0] = 1 if self.white_back else 0

        K = np.array(self.cams_intri[pose_index, view_index])
        R = np.array(self.cams_extri[pose_index, view_index][:3,:3])
        T = np.array(self.cams_extri[pose_index, view_index][:3,3:4])

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
        img, ray_o, ray_d, near, far, coord_, mask_at_box, bkgd_msk, bound_mask = sample_ray_WildAvatar_batch(
                img, msk, K, R, T, world_bounds)

        mask_at_box_large = mask_at_box

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

        # target view
        img_all = np.stack(img_all, axis=0)
        ray_o_all = np.stack(ray_o_all, axis=0)
        ray_d_all = np.stack(ray_d_all, axis=0)
        near_all = np.stack(near_all, axis=0)[...,None]
        far_all = np.stack(far_all, axis=0)[...,None]
        mask_at_box_all = np.stack(mask_at_box_all, axis=0)
        bkgd_msk_all = np.stack(bkgd_msk_all, axis=0)
        mask_at_box_large_all = np.stack(mask_at_box_large_all, axis=0)

        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        w2c = np.eye(4)
        w2c[:3,:3] = R
        w2c[:3,3:4] = T
        # get the world-to-camera transform and set R, T
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3:4]
        
        focalX = K[0,0]
        focalY = K[1,1]
        trans = [0.0, 0.0, 0.0]
        scale = 1.0
        znear = 0.001
        zfar = 1000
        world_view_transform = torch.tensor(getWorld2View2(R, T[:,0], trans, scale)).transpose(0, 1)
        projection_matrix = getProjectionMatrix_refine(torch.Tensor(K), H, W, znear, zfar).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        
        camera_center = world_view_transform.inverse()[3, :3]
        FovX = focal2fov(focalX, H)
        FovY = focal2fov(focalY, W)
        img.transpose(1,2,0)
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

            # test settings
            'pose_index': pose_index,
            'mask_at_box_large_all': mask_at_box_large_all,
            'image_name': img_path,
        })
        return ret

    def __len__(self):
        return self.num_instance * self.poses_num * self.camera_view_num
