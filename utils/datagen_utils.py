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

def read_images_from_directory(dir :str, prefix = None):
    images = []
    if prefix is None:
        for filename in sorted(os.listdir(dir)):
            filepath = os.path.join(dir, filename)
            if os.path.isfile(filepath) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
                img = np.array(Image.open(filepath))
                images.append(img)
    else:
        for filename in sorted(os.listdir(dir)):
            if filename.startswith(prefix):
                filepath = os.path.join(dir, filename)
                if os.path.isfile(filepath) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
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

def read_cam_dex(sequence :str, idx :int):

    meta = np.load(os.path.join(sequence, f'view_{idx:02d}', 'intrinsics_extrinsics.npz'))

    cam_ID = idx
    intrinsics = meta['intrinsics'][:3, :3]
    extrinsics = meta['extrinsics']
    depth = read_images_from_directory(os.path.join(sequence, f'view_{idx:02d}', 'depth'))/1000
    imgs = read_images_from_directory(os.path.join(sequence, f'view_{idx:02d}', 'rgb'))

    return depth, imgs, intrinsics, extrinsics, cam_ID

def quaternion_to_rotation_matrix(quaternions):
    """
    Converts a batch of quaternions to corresponding rotation matrices.

    Args:
        quaternions (torch.Tensor): Tensor of shape (..., 4) representing quaternions.

    Returns:
        torch.Tensor: Tensor of shape (..., 3, 3) representing rotation matrices.
    """
    assert quaternions.shape[-1] == 4, "Input quaternions must have shape (..., 4)"
    
    q = quaternions / quaternions.norm(dim=-1, keepdim=True)
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Compute rotation matrix elements
    R = torch.zeros((*q.shape[:-1], 3, 3), dtype=q.dtype, device=q.device)
    R[..., 0, 0] = 1 - 2 * (qy ** 2 + qz ** 2)
    R[..., 0, 1] = 2 * (qx * qy - qz * qw)
    R[..., 0, 2] = 2 * (qx * qz + qy * qw)
    R[..., 1, 0] = 2 * (qx * qy + qz * qw)
    R[..., 1, 1] = 1 - 2 * (qx ** 2 + qz ** 2)
    R[..., 1, 2] = 2 * (qy * qz - qx * qw)
    R[..., 2, 0] = 2 * (qx * qz - qy * qw)
    R[..., 2, 1] = 2 * (qy * qz + qx * qw)
    R[..., 2, 2] = 1 - 2 * (qx ** 2 + qy ** 2)

    return R

def read_cam_kubric(sequence :str, idx :int):

    meta = json.load(open(os.path.join(sequence, f'view_{idx}', 'metadata.json')))

    cam_ID = idx
    intrinsics = meta['camera']['K']
    extrinsics = meta['camera']['R']

    # Extracting the extrinsics Fixed Cams -> only first TS
    positions = torch.tensor(meta['camera']['positions'][0], dtype=torch.float64)
    quaternions = torch.tensor(meta['camera']['quaternions'][0], dtype=torch.float64)
    rotation_matrices = quaternion_to_rotation_matrix(quaternions)
    
    extrinsics_inv = torch.zeros((4, 4), dtype=torch.float64)
    extrinsics_inv[:3, :3] = rotation_matrices
    extrinsics_inv[:3, 3] = positions
    extrinsics_inv[3, 3] = 1
    extrinsics = extrinsics_inv.inverse().cpu().numpy()
    extrinsics[:3, :] = np.diag([1, -1, -1]) @ extrinsics[:3, :]

    # Change the intrinsics to the format
    w, h = meta['metadata']["resolution"]
    intrinsics = np.diag([w, h, 1]) @ intrinsics @ np.diag([1, -1, -1])

    depth = read_images_from_directory(os.path.join(sequence, f'view_{idx}'), prefix='depth_')

    imgs = read_images_from_directory(os.path.join(sequence, f'view_{idx}'), prefix='rgba_')[..., :3]

    return depth, imgs, intrinsics, extrinsics, cam_ID

