import torch
import numpy as np
import warnings
import json
import os
from PIL import Image
from scipy.ndimage import gaussian_filter
import imageio
import pathlib

NOISE_SCALE = 10


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

def depth_from_euclidean_to_z(depth, sensor_width, focal_length):
        n_frames, h, w = depth.shape
        sensor_height = sensor_width / w * h
        pixel_centers_x = (np.arange(-w / 2, w / 2, dtype=np.float32) + 0.5) / w * sensor_width
        pixel_centers_y = (np.arange(-h / 2, h / 2, dtype=np.float32) + 0.5) / h * sensor_height

        # Calculate squared distance from the center of the image
        pixel_centers_x, pixel_centers_y = np.meshgrid(pixel_centers_x, pixel_centers_y, indexing="xy")
        squared_distance_from_center = np.square(pixel_centers_x) + np.square(pixel_centers_y)

        # Calculate rescaling factor for each pixel
        z_to_eucl_rescaling = np.sqrt(1 + squared_distance_from_center / focal_length ** 2)

        # Apply the rescaling to each depth value
        depth_z = depth / z_to_eucl_rescaling
        return depth_z

def read_images_from_directory(dir :str, prefix = None):
    images = []
    if prefix is None:
        for filename in sorted(os.listdir(dir)):
            filepath = os.path.join(dir, filename)
            if os.path.isfile(filepath) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = np.array(Image.open(filepath))
                images.append(img)
            elif os.path.isfile(filepath) and filename.lower().endswith(('.tiff')):
                img = np.asarray(imageio.v2.imread(pathlib.Path(filepath).read_bytes(), format="tiff"), dtype=np.float64)[:,:, 0]
                images.append(img)
    else:
        for filename in sorted(os.listdir(dir)):
            if filename.startswith(prefix):
                filepath = os.path.join(dir, filename)
                if os.path.isfile(filepath) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = np.array(Image.open(filepath))
                    images.append(img)
                elif os.path.isfile(filepath) and filename.lower().endswith(('.tiff')):
                    img = np.asarray(imageio.v2.imread(pathlib.Path(filepath).read_bytes(), format="tiff"), dtype=np.float64)[:,:, 0]
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

def add_depth_noise(depth, noise_type='gaussian', sigma=0.01, outlier_ratio=0.05, max_noise=0.05, smooth_sigma=1.0):
    """
    Add noise to depth values with optional 2D smoothing.
    
    Args:
        depth (np.ndarray): Input depth values
        noise_type (str): Type of noise ('gaussian', 'saltpepper', or 'both')
        sigma (float): Standard deviation for Gaussian noise
        outlier_ratio (float): Ratio of outlier pixels for salt and pepper noise
        max_noise (float): Maximum magnitude of outlier noise
        smooth_sigma (float): Standard deviation for Gaussian smoothing (set to 0 to disable)
    
    Returns:
        np.ndarray: Depth with added noise
    """
    if not isinstance(depth, np.ndarray):
        depth = np.array(depth)
    
    noisy_depth = depth.copy()
    
    if noise_type in ['gaussian', 'both']:
        # Add Gaussian noise
        noise = np.random.normal(0, sigma, depth.shape)
        
        # Smooth the noise in 2D space
        if smooth_sigma > 0:
            noise = gaussian_filter(noise, sigma=smooth_sigma)
        
        noisy_depth += NOISE_SCALE*noise
    
    if noise_type in ['saltpepper', 'both']:
        # Add salt and pepper noise (outliers)
        mask_shape = depth.shape
        outlier_mask = np.random.random(mask_shape) < outlier_ratio
        
        # Generate outlier noise values
        outlier_noise = np.random.uniform(-max_noise, max_noise, mask_shape)
        
        # Smooth the outlier noise in 2D space
        if smooth_sigma > 0:
            outlier_noise = gaussian_filter(outlier_noise, sigma=smooth_sigma)
        
        noisy_depth[outlier_mask] += outlier_noise[outlier_mask]
    
    # Ensure we don't have negative depth values
    noisy_depth = np.maximum(noisy_depth, 0)
    
    return noisy_depth

def read_cam_kubric(sequence :str, idx :int, depth_euclid = False, query_points = None, noise_sigma: float = None):

    meta = json.load(open(os.path.join(sequence, f'view_{idx}', 'metadata.json')))

    cam_ID = idx
    intrinsics = meta['camera']['K']
    extrinsics = meta['camera']['R']
    sensor_width = meta['camera']['sensor_width']
    focal_length = meta['camera']['focal_length']

    # Extracting the extrinsics Fixed Cams -> only first TS
    positions = torch.tensor(meta['camera']['positions'][0], dtype=torch.float64)
    quaternions = torch.tensor(meta['camera']['quaternions'][0], dtype=torch.float64)
    rotation_matrices = quaternion_to_rotation_matrix(quaternions)
    
    extrinsics_inv = torch.zeros((4, 4), dtype=torch.float64)
    extrinsics_inv[:3, :3] = rotation_matrices
    extrinsics_inv[:3, 3] = positions
    extrinsics_inv[3, 3] = 1
    extrinsics = extrinsics_inv.inverse().cpu().numpy()
    extrinsics = np.diag([1, -1, -1, 1]) @ extrinsics

    # Change the intrinsics to the format
    w, h = meta['metadata']["resolution"]
    intrinsics = np.diag([w, h, 1]) @ intrinsics @ np.diag([1, -1, -1])

    depth = read_images_from_directory(os.path.join(sequence, f'view_{idx}'), prefix='depth_')
    if query_points is not None:
        q_depth = np.zeros((depth.shape[0], query_points.shape[1]), dtype=np.float32)
        for i in range(depth.shape[0]):
            x = query_points[:, 0]
            y = query_points[:, 1]
            x0 = x
            y0 = y
            x1 = x0 + 1
            y1 = y0 + 1

            x0 = torch.clamp(x0, 0, depth.shape[2] - 1)
            x1 = torch.clamp(x1, 0, depth.shape[2] - 1)
            y0 = torch.clamp(y0, 0, depth.shape[1] - 1)
            y1 = torch.clamp(y1, 0, depth.shape[1] - 1)

            Ia = depth[i, y0, x0]
            Ib = depth[i, y1, x0]
            Ic = depth[i, y0, x1]
            Id = depth[i, y1, x1]

            wa = (x1 - x) * (y1 - y)
            wb = (x1 - x) * (y - y0)
            wc = (x - x0) * (y1 - y)
            wd = (x - x0) * (y - y0)

            q_depth[i] = wa * Ia + wb * Ib + wc * Ic + wd * Id
        depth = q_depth
    if not depth_euclid:
        depth = depth_from_euclidean_to_z(depth, sensor_width, focal_length)

    if noise_sigma is not None:
        depth = add_depth_noise(depth, noise_type='gaussian', sigma=noise_sigma, smooth_sigma=1)


    # Add noise to the depth if requested
    if 'noise' in meta.get('depth_noise', {}):
        noise_params = meta.get('depth_noise', {})
        depth = add_depth_noise(
            depth,
            noise_type=noise_params.get('type', 'gaussian'),
            sigma=noise_params.get('sigma', 0.01),
            outlier_ratio=noise_params.get('outlier_ratio', 0.05),
            max_noise=noise_params.get('max_noise', 0.05)
        )

    imgs = read_images_from_directory(os.path.join(sequence, f'view_{idx}'), prefix='rgba_')[..., :3]

    return depth, imgs, intrinsics, extrinsics, cam_ID

