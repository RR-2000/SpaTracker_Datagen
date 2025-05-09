# %%

#-------- import the base packages -------------
import sys
import os
from easydict import EasyDict as edict

import torch
import torch.nn.functional as F
from base64 import b64encode
import numpy as np
from PIL import Image
import cv2
import argparse
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip
import torchvision.transforms as transforms

#-------- import spatialtracker -------------
from models.spatracker.predictor import SpaTrackerPredictor
from models.spatracker.utils.visualizer import Visualizer, read_video_from_path

#-------- import Depth Estimator -------------
from mde import MonoDEst

#-------- import utils -------------
from utils.datagen_utils import *

def predict_tracks(depths, images, downsample, query_frame, query_points=None):
    # read the video
    video = torch.from_numpy(images).permute(0, 3, 1, 2)[None].float()
    transform = transforms.Compose([
        transforms.CenterCrop((int(384*args.crop_factor),
                                int(512*args.crop_factor))),  
    ])
    _, T, _, H, W = video.shape
    segm_mask = np.ones((H, W), dtype=np.uint8)
    segm_mask = cv2.resize(segm_mask, (W, H), interpolation=cv2.INTER_NEAREST)
    if args.crop:
        video = transform(video)
        segm_mask = transform(torch.from_numpy(segm_mask[None, None]))[0,0].numpy()
    _, _, _, H, W = video.shape
    # adjust the downsample factor
    if H > W:
        downsample = max(downsample, 640//H)
    elif H < W:
        downsample = max(downsample, 960//W)
    else:
        downsample = max(downsample, 640//H)

    video = F.interpolate(video[0], scale_factor=downsample,
                        mode='bilinear', align_corners=True)[None]
    vidLen = video.shape[1]
    idx = torch.range(0, vidLen-1, args.fps).long()
    video = video[:, idx]
    # save the first image
    img0 = video[0,0].permute(1,2,0).detach().cpu().numpy()


    # cv2.imwrite(os.path.join(outdir, f'{args.vid_name}_ref.png'), img0[:,:,::-1])
    # cv2.imwrite(os.path.join(outdir, f'{args.vid_name}_seg.png'), segm_mask*255)

    S_lenth = 12       # [8, 12, 16] choose one you want
    model = SpaTrackerPredictor(
    checkpoint=os.path.join(
        './checkpoints/spaT_final.pth',
        ),
        interp_shape = (384, 512),
        seq_length = S_lenth
    )
    if torch.cuda.is_available():
        model = model.cuda()
        video = video.cuda()

    depths = torch.from_numpy(depths).float().cuda()[:,None]

    pred_tracks, pred_visibility, T_Firsts = (
                                    model(video, video_depth=depths,
                                    grid_size=grid_size, backward_tracking=args.backward,
                                    depth_predictor=None, grid_query_frame=query_frame, queries=query_points,
                                    segm_mask=torch.from_numpy(segm_mask)[None, None], wind_length=S_lenth)
                                        )
    
        
    return pred_tracks[0], pred_visibility[0], T_Firsts[0]


def getSplits(tracks, split: int = 0.5, seed: int = 42):
    np.random.seed(seed)
    num_points = tracks.shape[1]
    split_idx = np.random.permutation(num_points)
    return tracks[:, split_idx[int(num_points*split):int(num_points*split + 200)],:]

def query_cam_select(queried_points, save_dir: str, num_train: int = 20):
    queried_results = []
    combined_results = []
    for cam_ID in range(num_train):
        queried_results.append(np.expand_dims(np.load(os.path.join(save_dir, f'query_cam_{cam_ID}_traj.npy')), axis = 0))

    queried_results = np.concatenate(queried_results, axis = 0)

    for trackID in range(queried_points.shape[0]):
        init_dist = np.linalg.norm(queried_results[:,0,trackID] -  queried_points[trackID], axis = -1)
        best_cam = np.argmin(init_dist, axis = 0)
        combined_results.append(queried_results[best_cam,:, [trackID]])

    combined_results = np.concatenate(combined_results, axis = 0).transpose(1,0,2)

    np.save(os.path.join(save_dir, f'query_combined.npy'), combined_results)


    return


if __name__ == '__main__':
    # set the arguments
    parser = argparse.ArgumentParser()
    # add the video and segmentation
    parser.add_argument('--root', type=str, default='/cluster/scratch/rrajaraman/data/kubric_multiview_003/dense', help='path to the video')
    parser.add_argument('--vid_name', type=str, default='1', help='path to the video')
    # set the gpu
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    # set the model
    parser.add_argument('--model', type=str, default='spatracker', help='model name')
    # set the downsample factor
    parser.add_argument('--downsample', type=float, default=0.8, help='downsample factor')
    parser.add_argument('--grid_size', type=int, default=50, help='grid size')
    # set the results outdir
    parser.add_argument('--outdir', type=str, default='./kubric_traj', help='output directory')
    # set the fps
    parser.add_argument('--fps', type=float, default=1, help='fps')
    # draw the track length
    parser.add_argument('--len_track', type=int, default=10, help='len_track')
    parser.add_argument('--fps_vis', type=int, default=30, help='len_track')
    # crop the video
    parser.add_argument('--crop', action='store_true', help='whether to crop the video')
    parser.add_argument('--crop_factor', type=float, default=1, help='whether to crop the video')
    # backward tracking
    parser.add_argument('--backward', action='store_true', help='whether to backward the tracking')
    # if visualize the support points
    parser.add_argument('--vis_support', action='store_true', help='whether to visualize the support points')
    # set the visualized point size
    parser.add_argument('--point_size', type=int, default=3, help='point size')


    args = parser.parse_args()

    fps_vis = args.fps_vis

    # set input
    root_dir = args.root
    vid_dir = os.path.join(root_dir, args.vid_name + '.mp4')
    seg_dir = os.path.join(root_dir, args.vid_name + '.png')
    outdir = args.outdir
    os.path.exists(outdir) or os.makedirs(outdir)
    # set the paras
    grid_size = args.grid_size
    model_type = args.model
    downsample = args.downsample
    # set the gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    sequence_path = os.path.join(root_dir, args.vid_name)

    num_cams = 25

    query_frame = 0

    gt_tracks = getSplits(np.load(os.path.join(sequence_path, 'tracks_3d.npz'))['tracks_3d'])

    for idx in tqdm(range(num_cams)):
        # read the camera
        depth, imgs, intrinsics, extrinsics, cam_ID = read_cam_kubric(sequence_path, idx, False)

        query_points = gt_tracks[0]
        # Project query points onto the camera frame using intrinsics and extrinsics
        query_points_homogeneous = np.concatenate([query_points, np.ones((query_points.shape[0], 1))], axis=-1)

        query_points_cam_frame = (extrinsics @ query_points_homogeneous.T).T
        query_points_cam_frame = query_points_cam_frame[:, :3] / query_points_cam_frame[:, 3:4]
        query_points_cam_frame = (intrinsics @ query_points_cam_frame.T).T
        query_points_cam_frame = query_points_cam_frame[:, :2] / query_points_cam_frame[:, 2:3]
        query_points_cam_frame = np.concatenate(
            [np.zeros((query_points_cam_frame.shape[0], 1)), query_points_cam_frame], axis=-1
        )
        query_points_cam_frame = torch.tensor(query_points_cam_frame, device='cuda', dtype=torch.float32).unsqueeze(0)

        pred_tracks, pred_visibility, T_Firsts = predict_tracks(depth, imgs, downsample, query_frame)

        # Repeat intrinsics to match the first dimension of pred_tracks and convert to torch tensor
        intrinsics = torch.tensor(
            np.repeat(np.linalg.inv(intrinsics)[None, ...], pred_tracks.shape[0], axis=0),
            device=pred_tracks.device, dtype=pred_tracks.dtype
        )
        # Repeat extrinsics to match the first dimension of pred_tracks and convert to torch tensor
        extrinsics = torch.tensor(
            np.repeat(np.linalg.inv(extrinsics)[None, ...], pred_tracks.shape[0], axis=0),
            device=pred_tracks.device, dtype=pred_tracks.dtype
        )

        true_traj = pixel_xy_and_camera_z_to_world_space(pred_tracks[..., :2], pred_tracks[..., 2:3],
                                                        intrinsics, extrinsics)
        
        # save the trajectory
        os.path.exists(os.path.join(outdir, args.vid_name)) or os.makedirs(os.path.join(outdir, args.vid_name))
        np.save(os.path.join(outdir, args.vid_name, f'grid_cam_{cam_ID}_traj.npy'), true_traj.cpu().numpy())



        # Predict Query Points
        # depth, imgs, intrinsics, extrinsics, cam_ID = read_cam_kubric(sequence_path, idx, False)

        pred_tracks, pred_visibility, T_Firsts = predict_tracks(depth, imgs, downsample, query_frame, query_points=query_points_cam_frame)
        
        true_traj = pixel_xy_and_camera_z_to_world_space(pred_tracks[..., :2], pred_tracks[..., 2:3],
                                                        intrinsics, extrinsics)
        
        # save the trajectory
        os.path.exists(os.path.join(outdir, args.vid_name)) or os.makedirs(os.path.join(outdir, args.vid_name))
        np.save(os.path.join(outdir, args.vid_name, f'query_cam_{cam_ID}_traj.npy'), true_traj.cpu().numpy())
    
    query_cam_select(gt_tracks[0], save_dir = os.path.join(outdir, args.vid_name), num_train = 20)







