import torch
import numpy as np
import warnings
import json
import os
from PIL import Image


def pixel_xy_and_camera_z_to_world_space(pixel_xy, camera_z, intrs_inv, extrs_inv):
    num_frames, num_points, _ = pixel_xy.shape
    assert pixel_xy.shape == (num_frames, num_points, 2)
    assert camera_z.shape == (num_frames, num_points, 1)
    assert intrs_inv.shape == (num_frames, 3, 3)
    assert extrs_inv.shape == (num_frames, 4, 4)

    pixel_xy_homo = torch.cat([pixel_xy, pixel_xy.new_ones(pixel_xy[..., :1].shape)], -1)
    camera_xyz = torch.einsum('Aij,ABj->ABi', intrs_inv, pixel_xy_homo) * camera_z
    camera_xyz_homo = torch.cat([camera_xyz, camera_xyz.new_ones(camera_xyz[..., :1].shape)], -1)
    world_xyz_homo = torch.einsum('Aij,ABj->ABi', extrs_inv, camera_xyz_homo)
    if not torch.allclose(
            world_xyz_homo[..., -1],
            world_xyz_homo.new_ones(world_xyz_homo[..., -1].shape),
            atol=0.1,
    ):
        warnings.warn(f"pixel_xy_and_camera_z_to_world_space found some homo coordinates not close to 1: "
                      f"the homo values are in {world_xyz_homo[..., -1].min()} – {world_xyz_homo[..., -1].max()}")
    world_xyz = world_xyz_homo[..., :-1]

    assert world_xyz.shape == (num_frames, num_points, 3)
    return world_xyz

def read_images_from_directory(dir :str):
    images = []
    for filename in sorted(os.listdir(dir)):
        filepath = os.path.join(dir, filename)
        if os.path.isfile(filepath) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = np.array(Image.open(filepath))
            images.append(img)
    return np.stack(images, axis=0)

def read_cam(sequence :str, idx :int):

    meta = json.load(open(os.path.join(sequence, 'train_meta.json')))

    cam_ID = meta['cam_id'][0][idx]
    intrinsics = meta['k'][0][idx]
    extrinsics = meta['w2c'][0][idx]
    depth = np.load(os.path.join(sequence, 'dynamic3dgs_depth', 'depths_' f"{cam_ID:02d}.npy"))
    imgs = read_images_from_directory(os.path.join(sequence, 'ims', f"{cam_ID}"))

    return depth, imgs, intrinsics, extrinsics, cam_ID

