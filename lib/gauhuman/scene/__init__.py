#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import torch
from utils.system_utils import mkdir_p
from smpl.smpl_numpy import SMPL
import numpy as np
from scene.gaussian_model import BasicPointCloud
from utils.sh_utils import SH2RGB

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        if self.loaded_iter:
            self.cameras_extent = 0.0
            t_vertices = self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud-{}.ply".format(self.loaded_iter)))
            shs = np.random.random((t_vertices.shape[0], 3)) / 255.0
            pcd = BasicPointCloud(points=t_vertices, colors=SH2RGB(shs), normals=np.zeros((t_vertices.shape[0], 3)))
            self.gaussians.create_from_pcd(pcd, self.cameras_extent)
        else:
            def get_vertives1(x_range, y_range, z_range, d_grid = 0.01):
                vgrid = np.mgrid[x_range[0]:x_range[1]:d_grid, y_range[0]:y_range[1]:d_grid, z_range[0]:z_range[1]:d_grid]
                vgrid = vgrid.transpose(1,2,3,0)
                return vgrid.reshape(-1, 3)
                
            self.cameras_extent = args.cameras_extent
            smpl_model = SMPL(sex='neutral', model_dir='assets/SMPL_NEUTRAL.pkl')
            self.big_pose_params = self.prepare_big_pose_params()
            t_vertices, _ = smpl_model(self.big_pose_params['poses'], self.big_pose_params['shapes'].reshape(-1))
            dress_vertices = get_vertives1(x_range=[-0.5,0.5],y_range=[-0.8,-0.3],z_range=[-0.15,0.15],d_grid=0.02).astype(np.float32)
            # hair_vertices = get_vertives1(x_range=[-0.15,0.15],y_range=[0.2,0.7],z_range=[-0.3,0.1],d_grid=0.02)
            # t_vertices = np.concatenate((t_vertices, dress_vertices), axis=0)
            # t_vertices = self.gaussians.load_ply("/home/shimin/code/human/avatar/gauhuman-org/output/zju_mocap/CoreView_386/point_cloud/iteration_2000/point_cloud-2000.ply")
            t_vertices = t_vertices.astype(np.float32)
            self.gaussians.smpl_xyz = torch.from_numpy(t_vertices).cuda()
            # self.gaussians.smpl_xyz = torch.from_numpy(np.concatenate((t_vertices, dress_vertices), axis=0)).cuda()
            # t_vertices = np.concatenate((t_vertices, t_vertices*1.2), axis=0)
            shs = np.random.random((t_vertices.shape[0], 3)) / 255.0
            pcd = BasicPointCloud(points=t_vertices, colors=SH2RGB(shs), normals=np.zeros((t_vertices.shape[0], 3)))
            self.gaussians.create_from_pcd(pcd, self.cameras_extent)
            # self.gaussians.save_ply_temp(self.gaussians.get_xyz)
            # print("Saved at ./temp.ply")

        if self.gaussians.motion_offset_flag:
            model_path = os.path.join(self.model_path, "mlp_ckpt", "iteration_" + str(self.loaded_iter), "ckpt.pth")
            if os.path.exists(model_path):
                ckpt = torch.load(model_path, map_location='cuda:0')
                self.gaussians.pose_decoder.load_state_dict(ckpt['pose_decoder'])
                self.gaussians.lweight_offset_decoder.load_state_dict(ckpt['lweight_offset_decoder'])
    
    def prepare_big_pose_params(self):

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
    
    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud-{}.ply".format(iteration)))

        if self.gaussians.motion_offset_flag:
            model_path = os.path.join(self.model_path, "mlp_ckpt", "iteration_" + str(iteration), "ckpt.pth")
            mkdir_p(os.path.dirname(model_path))
            torch.save({
                'iter': iteration,
                'pose_decoder': self.gaussians.pose_decoder.state_dict(),
                'lweight_offset_decoder': self.gaussians.lweight_offset_decoder.state_dict(),
            }, model_path)