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

def predict_tracks(depths, images, downsample, query_frame):
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
                                    depth_predictor=None, grid_query_frame=query_frame,
                                    segm_mask=torch.from_numpy(segm_mask)[None, None], wind_length=S_lenth)
                                        )
    
        
    return pred_tracks[0], pred_visibility[0], T_Firsts[0]


if __name__ == '__main__':
    # set the arguments
    parser = argparse.ArgumentParser()
    # add the video and segmentation
    parser.add_argument('--root', type=str, default='/cluster/scratch/rrajaraman/data/panoptic_d3dgs', help='path to the video')
    parser.add_argument('--vid_name', type=str, default='basketball', help='path to the video')
    # set the gpu
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    # set the model
    parser.add_argument('--model', type=str, default='spatracker', help='model name')
    # set the downsample factor
    parser.add_argument('--downsample', type=float, default=0.8, help='downsample factor')
    parser.add_argument('--grid_size', type=int, default=50, help='grid size')
    # set the results outdir
    parser.add_argument('--outdir', type=str, default='./panoptic_traj', help='output directory')
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

    num_cams = len(os.listdir(os.path.join(sequence_path, 'seg')))

    query_frame = 0

    for idx in tqdm(range(num_cams)):
        # read the camera
        depth, imgs, intrinsics, extrinsics, cam_ID = read_cam(sequence_path, idx)

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
        np.save(os.path.join(outdir, args.vid_name, f'cam_{cam_ID}_traj.npy'), true_traj.cpu().numpy())

        vis = Visualizer(save_dir=outdir, grayscale=True, 
                        fps=fps_vis, pad_value=0, linewidth=args.point_size,
                        tracks_leave_trace=args.len_track)
        msk_query = (T_Firsts == query_frame)
        # visualize the all points
        if args.vis_support:
            video_vis = vis.visualize(video=video, tracks=pred_tracks[..., :2],
                                    visibility=pred_visibility,
                                    filename=args.vid_name+"_spatracker")







