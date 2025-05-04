import os
import json
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
from utils.datagen_utils import *


def depth_only_proj(path: str = '/project/Thesis/kubric-private/output/multiview_36_v3/train/1', num_t:int = 24, train_cams = 20, train_only = True, test_only = False, only_fg = True):
    
    
    num_views = len([view for view in os.listdir(path) if view.startswith('view')])
    views = []
    
    for idx in range(num_views):
        meta_i = read_cam_kubric(path, idx)
        views.append(meta_i)
    depths = []

    if train_only:
        start_cam = 0
        end_cam = train_cams
    elif test_only:
        start_cam = train_cams
        end_cam = num_views
    else:
        start_cam = 0
        end_cam = num_views

    # for all times
    for t in tqdm(range(num_t), desc='Processing frames'):
        # read all depths
        depths_t = []
        for idx in range(start_cam, end_cam):

            # depth = np.asarray(Image.open(os.path.join(path, depth_path, f'depth_{t:05d}.tiff')))
            depth, imgs, intrinsics, extrinsics, cam_ID = views[idx]
            depth = depth[t]

            seg = np.asarray(Image.open(os.path.join(path, f'view_{cam_ID}', f'segmentation_{t:05d}.png')))
            seg = seg.reshape(-1) >= only_fg
            # convert to 3d
            h, w = depth.shape
            v, u = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            

            #reshape to 1d
            x = u.reshape(-1)
            y = v.reshape(-1)
            depth = depth.reshape(-1)
            depth = np.stack((x, y, depth), axis=-1)
            depth = torch.tensor(
                depth.reshape(1, -1, 3), device='cuda', dtype=torch.float64
            )

            # Repeat intrinsics to match the first dimension of tracks and convert to torch tensor
            intrinsics = torch.tensor(
                np.repeat(np.linalg.inv(intrinsics)[None, ...], depth.shape[0], axis=0),
                device='cuda', dtype=torch.float64
            )

            # Repeat extrinsics to match the first dimension of tracks and convert to torch tensor
            extrinsics = torch.tensor(
                np.repeat(np.linalg.inv(extrinsics)[None, ...], depth.shape[0], axis=0),
                device='cuda', dtype=torch.float64
            )

            track = pixel_xy_and_camera_z_to_world_space(depth[..., :2], depth[..., 2:3],
                                                            intrinsics, extrinsics)[0, seg]

            depths_t.append(track.cpu().numpy())
        # stack the depths
        depths_t = np.concatenate(depths_t, axis=0)
        depths.append(depths_t)


    # save the depths

    np.save(os.path.join(path, f'depths_init.npy'), depths[0])
    np.savez_compressed(os.path.join(path, f'depths.npz'), *depths)

if __name__ == '__main__':
    path = '/project/Thesis/kubric-private/output/multiview_25/train/1'
    depth_only_proj(only_fg=True, train_cams=9, path=path)