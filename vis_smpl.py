import os
import numpy as np
import cv2
import smplx
import torch
import argparse
from utils.smpl_utils import Renderer
from utils.meta_utils import read_meta


def vis_smpl(subject, ext="png", root_path="data/WildAvatar", save_smpl=True):
    metadata = read_meta(subject, root_path)
    intri = list(metadata.values())[0]['cam_intrinsics']
    focal_length = intri[0][0]
    img_w = intri[0][2] * 2
    img_h = intri[1][2] * 2
    
    render = Renderer(focal_length=focal_length, img_w=img_w, img_h=img_h,
                            faces=smpl_model.faces,
                            same_mesh_color=False)
    for frame_id, data in metadata.items():
        org_img_path = os.path.join(root_path, subject, "images", frame_id + "." + ext)
        assert os.path.exists(org_img_path)
        smpl_mask_save_path = org_img_path.replace("/images/", "/smpl_masks/").replace("/emc", "/emc-smpl_mask")
        smpl_save_path = org_img_path.replace("/images/", "/smpl/")
        cam_extrinsics = np.array(data['cam_extrinsics'])
        R = cam_extrinsics[:3, :3]
        
        pred_cam_full = torch.from_numpy(cam_extrinsics[:3,3])[None]
        poses = np.array(data['poses']).astype(np.float32)
        betas = np.array(data['betas']).astype(np.float32)
        poses = torch.from_numpy(poses[None])
        betas = torch.from_numpy(betas)
        chosen_vert_arr = smpl_model(betas=betas,
            body_pose=poses[:, 3:],
            global_orient=poses[:, :3],
            pose2rot=True,
            transl=pred_cam_full @ R)['vertices']
        chosen_vert_arr = chosen_vert_arr @ np.linalg.inv(R)
        smpl_color, smpl_mask = render.render_front_view(chosen_vert_arr)
        cv2.imwrite(smpl_mask_save_path, smpl_mask)
        if save_smpl:
            os.makedirs(os.path.dirname(smpl_save_path), exist_ok=True)
            cv2.imwrite(smpl_save_path, smpl_color)

def parse_args():
    parser = argparse.ArgumentParser(description='parse_args')
    
    parser.add_argument('--subject', type=str, default="__-ChmS-8m8")
    parser.add_argument('--ext', type=str, default="png")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    smpl_model = smplx.body_models.SMPL('assets/SMPL_NEUTRAL.pkl')
    vis_smpl(subject=args.subject, ext=args.ext)
