import os
import torch
from utils.loss_utils import ssim_metric
from scene import Scene, GaussianModel
from utils.loader_utils import collate_fn, data_to_device
import imageio
import numpy as np
from tqdm import tqdm
from utils.image_utils import psnr
from datasets.wildavatar_dataset import WildAvatarDatasetBatch

def test_single(tb_writer, scene : Scene, renderFunc, renderArgs, visualing=True, args=None):
    savedir_human = os.path.join(tb_writer.log_dir, 'novel_pose')
    os.makedirs(savedir_human, exist_ok=True)
    
    human_name = os.path.basename(args.source_path)
    viewpointset = WildAvatarDatasetBatch(data_root=os.path.join("data/WildAvatar", human_name), poses_start=0, poses_interval=1, poses_num=20, white_back=False)
    
    test_loader = iter(torch.utils.data.DataLoader(dataset=viewpointset, batch_size=1, collate_fn=collate_fn, num_workers=12))

    psnr_sub_view = []
    ssim_sub_view = []
    lpips_sub_view = []
    for pose_id, viewpoint in tqdm(enumerate(test_loader), desc="Views of {}".format(human_name), total=len(viewpointset)):
        viewpoint = data_to_device(viewpoint)
        render_output = renderFunc(viewpoint, scene.gaussians, *renderArgs, return_smpl_rot=True)
        image = torch.clamp(render_output["render"], 0.0, 1.0)
        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

        _, W, H = image.shape
        mask_at_box = viewpoint['mask_at_box_large_all'].reshape(viewpoint['mask_at_box_large_all'].shape[0], W, H)
        # metric on the whole img
        mask_at_box[:] = True
        
        psnr_ = psnr(image[:, mask_at_box[0]], gt_image[:, mask_at_box[0]]).mean().double().cpu().numpy()
        ssim, lpips = ssim_metric(image[:, mask_at_box[0]].cpu().numpy(), gt_image[:, mask_at_box[0]].cpu().numpy(), mask_at_box[0].cpu().numpy(), mask_at_box[0].shape[0], mask_at_box[0].shape[1])
        psnr_sub_view.append(psnr_)
        ssim_sub_view.append(ssim)
        lpips_sub_view.append(lpips)
        
        if visualing:
            rgb8 = 255 * (image.cpu().numpy()).transpose(1,2,0)
            gt_rgb8 = 255 * (gt_image.cpu().numpy()).transpose(1,2,0)
            rgb8 = np.concatenate([rgb8, gt_rgb8], axis=1)
            filename = os.path.join(savedir_human, '{:02d}_psnr{:04d}.png'.format(pose_id, int(psnr_*100)))
            img = rgb8
            imageio.imwrite(filename, img.astype(np.uint8))
            
    avg_psnr = np.array(psnr_sub_view).mean()
    np.save(savedir_human+'/psnr_{}.npy'.format(int(avg_psnr*100)), np.array(avg_psnr))
    avg_ssim = np.array(ssim_sub_view).mean()
    np.save(savedir_human+'/ssim_{}.npy'.format(int(avg_ssim*100)), np.array(avg_ssim))
    avg_lpips = np.array(lpips_sub_view).mean()
    np.save(savedir_human+'/lpips_{}.npy'.format(int(avg_lpips*100)), np.array(avg_lpips))
    torch.cuda.empty_cache()