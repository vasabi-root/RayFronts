# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import argparse
from collections import defaultdict
from functools import partial
import gc
import math
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random
from tqdm import tqdm
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.io import read_video, write_video
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from einops import rearrange

from datasets import load_dataset_builder, load_dataset
from datasets.distributed import split_dataset_by_node

from transformers import AutoModel, CLIPImageProcessor
import torch
import torchvision
from rayfronts.image_encoders.naradio import NARadioEncoder

def get_robust_pca(features: torch.Tensor, m: float = 2, remove_first_component=False, skip: int = 0):
    # features: (N, C)
    # m: a hyperparam controlling how many std dev outside for outliers
    assert len(features.shape) == 2, "features should be (N, C)"
    reduction_mat = torch.pca_lowrank(features, q=3 + skip, niter=20)[2]
    reduction_mat = reduction_mat[:, skip:]
    colors = features @ reduction_mat
    if remove_first_component:
        colors_min = colors.min(dim=0).values
        colors_max = colors.max(dim=0).values
        tmp_colors = (colors - colors_min) / (colors_max - colors_min)
        fg_mask = tmp_colors[..., 0] < 0.2
        reduction_mat = torch.pca_lowrank(features[fg_mask], q=3, niter=20)[2]
        colors = features @ reduction_mat
    else:
        fg_mask = torch.ones_like(colors[:, 0]).bool()
    d = torch.abs(colors[fg_mask] - torch.median(colors[fg_mask], dim=0).values)
    mdev = torch.median(d, dim=0).values
    s = d / mdev
    try:
        rins = colors[fg_mask][s[:, 0] < m, 0]
        gins = colors[fg_mask][s[:, 1] < m, 1]
        bins = colors[fg_mask][s[:, 2] < m, 2]
        rgb_min = torch.tensor([rins.min(), gins.min(), bins.min()])
        rgb_max = torch.tensor([rins.max(), gins.max(), bins.max()])
    except:
        rins = colors
        gins = colors
        bins = colors
        rgb_min = torch.tensor([rins.min(), gins.min(), bins.min()])
        rgb_max = torch.tensor([rins.max(), gins.max(), bins.max()])

    return reduction_mat, rgb_min.to(reduction_mat), rgb_max.to(reduction_mat)


def get_pca_map(
    feature_map: torch.Tensor,
    img_size,
    interpolation="bicubic",
    return_pca_stats=False,
    pca_stats=None,
    skip_components: int = 0,
):
    """
    feature_map: (1, h, w, C) is the feature map of a single image.
    """
    if feature_map.shape[0] != 1:
        # make it (1, h, w, C)
        feature_map = feature_map[None]
    if pca_stats is None:
        reduct_mat, color_min, color_max = get_robust_pca(
            feature_map.reshape(-1, feature_map.shape[-1]), skip=skip_components,
        )
    else:
        reduct_mat, color_min, color_max = pca_stats
    pca_color = feature_map @ reduct_mat
    pca_color = (pca_color - color_min) / (color_max - color_min)
    pca_color = pca_color.clamp(0, 1)
    pca_color = F.interpolate(
        pca_color.permute(0, 3, 1, 2),
        size=img_size,
        mode=interpolation,
    ).permute(0, 2, 3, 1)
    pca_color = pca_color.cpu().numpy().squeeze(0)
    if return_pca_stats:
        return pca_color, (reduct_mat, color_min, color_max)
    return pca_color

def read_frames_from_dir(dir):
    dir = Path(dir)
    frame_names = [name for name in sorted(os.listdir(dir)) if name[-3:].lower() in ['jpg', 'png', 'jpeg']]
    if len(frame_names) == 0:
        raise FileNotFoundError(f'There are no supported frames in a dir {dir!r}')
    
    return torch.stack([pil_to_tensor(Image.open(dir/name)) for name in frame_names])
    

def cat_frames(frames: torch.Tensor, direction='vertiacal'):
    '''
    Concatenates frames by direction.  
    `direction` can be 'vertical' or 'horizontal'.  
    Frames are concatenated starting from the frames[0] and up/left direction
    '''
    # assert a.shape == b.shape
    match direction:
        case 'horizontal':
            return rearrange(frames, 'n h w c -> h (n w) c').float().cpu()
            # return torch.cat(frames, dim=1)
        case 'vertical':
            return rearrange(frames, 'n h w c -> (n h) w c').float().cpu()
        case _:
            raise Exception(f'Unsupported direction: {direction}')
        
def cat_frames_into_grid(frames, rows, cols):
    assert rows*cols == len(frames)
    
    grid = torch.Tensor()
    rows_list = []
    for i in range(rows):
        row = frames[i*cols:(i+1)*cols]
        rows_list.append(cat_frames(row, 'horizontal'))
    
    return cat_frames(rows_list, 'vertical')

def calc_grid(frame_shape, frame_num) -> Tuple[int]:
    '''Returns (rows, cols)'''
    height = frame_shape[-2]
    width = frame_shape[-1]
    
    if height > width:
        return 1, frame_num
    if frame_num < 4:
        return frame_num, 1
    return frame_num // 2, 2

def add_title_to_frame(frame: torch.Tensor, title: str):
    img = to_pil_image(frame.permute(2, 0, 1)/255, 'RGB')
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 16)
    d = ImageDraw.Draw(img)
    height = int(frame.shape[0])
    width = int(frame.shape[1])
    d.text((width//2, 0), title, anchor='ma', font=font, fill='white')
    return pil_to_tensor(img).permute(1, 2, 0)

@torch.no_grad
def calc_similarities(text_features: torch.Tensor, features: torch.Tensor, cos_threshold=0.03) -> torch.Tensor:
    img_features_expanded = features.expand(len(text_features), *features.shape[1:]).cuda()
    text_features_expanded = text_features.unsqueeze(1).unsqueeze(1)

    print(img_features_expanded.shape, text_features_expanded.shape)
    similarities = torch.cosine_similarity(img_features_expanded, text_features_expanded, dim=-1).unsqueeze(-1)
    return similarities

@torch.no_grad
def normalize_cos(cosine_val):
    return (cosine_val+1)/2

@torch.no_grad
def get_query_map(text_features: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    assert text_features.shape[0] == 1 # currently supported only one query
    
    similarity = torch.cosine_similarity(features.cuda(), text_features, dim=-1).unsqueeze(-1)
    similarity = normalize_cos(similarity)
    # similarity = torch.where(similarity > (cos_threshold+1)/2, 1.0, 0.0)
    return similarity.expand(*similarity.shape[:3], 3)

@torch.no_grad
def upscale_res(tensor, desired_res):
    if tensor.shape[-3:-1] == desired_res:
        return tensor
    return torch.nn.functional.interpolate(
        # tensor.expand([1, tensor.shape[-3], tensor.shape[-2], 3]).permute(0, 3, 1, 2)
        (tensor[None] if len(tensor.shape) == 3 else tensor).permute(0, 3, 1, 2),
        # tensor.permute(2, 0, 1),
        size=desired_res,
        mode='bilinear',
    ).permute(0, 2, 3, 1).squeeze(0)

@torch.no_grad
def get_masked_frame(frame: torch.Tensor, similarity: torch.Tensor, cos_threshold=0.1):
    # features_color = torch.from_numpy(get_pca_map(lang_aligned_feat_map_permuted.float(), input_res)).permute(2, 0, 1).cuda()
    # h, w = similarity.shape[1:3]
    # similarity_united_hw = rearrange(similarity, 'b h w c -> b (h w) c').float()
    # normalized_sim = torch.softmax(similarity_united_hw, dim=1)
    # normalized_sim = rearrange(normalized_sim, 'b (h w) c -> b h w c', h=h, w=w).float()
    similarity_mask = torch.where(similarity > normalize_cos(cos_threshold), 1.0, 0.0)
    similarity_mask_scaled = upscale_res(similarity_mask, frame.shape[-2:])
    frame_masked = (frame.permute(0, 2, 3, 1) * similarity_mask_scaled)
    
    # frame_masked = add_title_to_frame(f'cosine thresh: {cos_threshold:.1}', frame * similarity_mask_scaled) / 255
    # frame_masked = (frame[0] * similarity_mask_scaled).cuda()
    # return to_pil_image(add_title_to_frame(query, album))
    return frame_masked

def rescale_res(resolution: Tuple[int], desired: Tuple[int]):
    rows, cols = resolution[-2], resolution[-1]
    if len(desired) == 1: # desired min side
        desired_min_side = desired[0]
        if rows <= cols:
            return desired_min_side, int(desired_min_side*cols/rows)
        return int(desired_min_side*rows/cols), desired_min_side
    return desired

@torch.no_grad
def transform_tensor(tensor: torch.Tensor, desired_res):
    return F.interpolate(tensor.float().cuda()/255, desired_res, mode='bilinear')

results = []


@torch.inference_mode()
def main(rank: int = 0, world_size: int = 1):
    '''
    Computes the PCA features for every frame in a supplied video and renders them into a new video.
    '''

    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    cv2.setNumThreads(1)

    device = torch.device('cuda', local_rank)
    parser = argparse.ArgumentParser(description='Visual Model Features in Video')
    parser.add_argument('-v', '--model-version', default='c-radio_v3-h',
                        help='Which radio model to load.'
    )
    parser.add_argument('--video', type=str, required=True,
                        help='Path to the video. Can be a directory containing frames (in this case, set --fps key also)')
    parser.add_argument('--fps', type=int, default=None, 
                        help='Desired FPS rate of a video. Requred if video is a directory with frames. ')
    parser.add_argument('--query', type=str, required=True,
                        help='Query to the image. Objects should be separated by \'; \' delimiter')
    parser.add_argument('--cos-thresh', type=float, default=0.1,
                        help='Cosine threshold to separate objects from background during cosine similarity analysis')
    parser.add_argument('--output', type=str, required=True,
                        help='Where to store the output video')
    parser.add_argument('-r', '--resolution', nargs='+', type=int, default=None,
                        help='The input image resolution.'
                             ' If one value is specified, the shortest dimension is resized to this.'
                             ' If two, the image is center cropped.'
                             ' If not specified, center cropped 378px is used.'
                             ' Default: The RADIO model\'s preferred resolution.'
    )
    parser.add_argument('--max-dim', default=False, action='store_true', help='Resize the max dimension to the specified resolution')
    parser.add_argument('--resize-multiple', type=int, default=16,
                        help='Resize images with dimensions a multiple of this value.'
                             ' This should be equal to the patch size of a ViT (e.g. RADIOv1)'
    )
    parser.add_argument('--vitdet-window-size', default=None, type=int, help='Enable ViTDet at the specific window size')
    parser.add_argument('--patch-size', default=16, type=int, help='The model patch size')
    parser.add_argument('--torchhub-repo',
                        help="Path to the Torchhub repo", default="NVlabs/RADIO"
    )
    parser.add_argument('--side-by-side', default=False, action='store_true',
                        help='Render the original frame and the PCA frame side-by-side')
    parser.add_argument('--audio', default=False, action='store_true',
                        help='Encode the audio in the output video')
    parser.add_argument('--video-codec', default='libx264', type=str, help='The video codec to use')
    parser.add_argument('--batch-size', type=int, default=8, help='The processing batch size')
    parser.add_argument('--force-reload', default=False, action='store_true', help='Reload the torch.hub codebase')

    args, _ = parser.parse_known_args()

    adaptor_names = ['siglip2-g']
    # model, preprocessor, info = load_model(args.model_version, vitdet_window_size=args.vitdet_window_size, adaptor_names=adaptor_names,
    #                                        torchhub_repo=args.torchhub_repo, force_reload=args.force_reload)
    input_res = [640, 352]

    enc = NARadioEncoder(model_version=args.model_version, lang_model="siglip2-g", # radio_v2.5-b  or  siglip
                         device=device)    
    
    enc.model.to(device=device).eval()
    
    query = [args.query] #.split('; ')

    if args.resolution is None:
        args.resolution = (enc.model.preferred_resolution.height, enc.model.preferred_resolution.width)

    patch_size = getattr(enc.model, 'patch_size', None) or args.patch_size

    if args.resize_multiple is None:
        args.resize_multiple = getattr(enc.model, 'min_resolution_step', patch_size)


    if os.path.isfile(args.video):
        input_video = read_video(args.video, output_format='TCHW')
        input_frames = input_video[0]
        fps = input_video[2]['video_fps']
    elif os.path.isdir(args.video):
        input_frames = read_frames_from_dir(args.video)
        fps = args.fps
        
    input_res = enc.get_nearest_size(*rescale_res(input_frames[0].shape, args.resolution))
    enc.input_resolution = input_res
    print(f'{enc.input_resolution[0]=}, {enc.input_resolution[1]=}')
        
    with torch.no_grad():
        text_features = enc.encode_labels(query)
    
    all_features = []
    tx_frames = []

    batch_size = args.batch_size
    
    num_frames = len(adaptor_names)+1
    if args.side_by_side: num_frames += 1
    
    features_map = {adaptor_name: [] for adaptor_name in adaptor_names}
    features_map[query[0]] = []
    
    cos_thresh_key = f'cosine threshold: {args.cos_thresh}'
    features_map[cos_thresh_key] = []
    
    with torch.no_grad():
        for b in tqdm(range(0, len(input_frames), batch_size)):
            curr_frames = input_frames[b:b+batch_size]
            curr_frames = transform_tensor(curr_frames, input_res)

            tx_frames.append(curr_frames)

            curr_frames = curr_frames.cuda()
            
            num_rows = curr_frames.shape[-2] // patch_size
            num_cols = curr_frames.shape[-1] // patch_size
            if b >= len(input_frames) - batch_size:
                pass
            
            # print(f'{curr_frames.shape=} {curr_frames.dtype=} {curr_frames.min()=} {curr_frames.max()=}')

            with torch.autocast(device.type, dtype=torch.bfloat16):
                feat_map = enc.encode_image_to_feat_map(curr_frames)
                
                # тот самый шаг маппинга пространства плотных признаков к пространству текстовых
                lang_aligned_feat_map = enc.align_spatial_features_with_language(feat_map)
                features = lang_aligned_feat_map.permute(0, 2, 3, 1)

                # features = rearrange(features, 'b (h w) c -> b h w c', h=num_rows, w=num_cols).float()
                # features = rearrange(features, 'b h w c -> b (h w) c').float()
                query_map = get_query_map(text_features, features)
                
                features_map[adaptor_names[0]].append(features.cpu())
                features_map[query[0]].append(query_map.cpu())
                features_map[cos_thresh_key].append(get_masked_frame(curr_frames, query_map, args.cos_thresh))
                
            torch.cuda.empty_cache()
            
            
    tx_frames = torch.cat(tx_frames)
    
    colored_frames = []
    if args.side_by_side:
        original_frames = []
        for frame in tx_frames.permute(0, 2, 3, 1) * 255:
            original_frames.append(add_title_to_frame(frame, 'original'))
        colored_frames.append(torch.stack(original_frames))
        
    for adaptor_name, all_features in features_map.items():
        all_features = torch.cat(all_features)
        
        num_keyframes = 30
        kf_stride = max(all_features.shape[0] // num_keyframes, 1)

        # We'll use this to compute the PCA
        sub_features = all_features[::kf_stride]
        pca_stats = get_robust_pca(sub_features.flatten(0, 2).float())

        output_frames = []
        for raw_frame, features in zip(tx_frames, all_features):
            img_size = raw_frame.shape[-2:]
            if all_features.shape[-1] != 3:
                pca_features = torch.from_numpy(get_pca_map(features.float(), img_size, pca_stats=pca_stats, interpolation='bilinear'))
            else:
                pca_features = upscale_res(features, input_res)
                # pca_features = torch.nn.functional.interpolate(
                #     features[None].permute(0, 3, 1, 2),
                #     size=img_size,
                #     mode='bilinear',
                # ).permute(0, 2, 3, 1).squeeze(0)
            
            pca_features = pca_features.mul_(255).byte()
            
            titled = add_title_to_frame(pca_features, adaptor_name)
            output_frames.append(titled)

        output_frames = torch.stack(output_frames)
        colored_frames.append(output_frames)
    
    del features_map
    gc.collect()
    torch.cuda.empty_cache()
    
    colored_frames = torch.stack(colored_frames, dim=1)
    rows, cols = calc_grid(input_frames[0].shape, int(colored_frames.shape[1]))
    grid_frames = []
    for frames in colored_frames:
        grid_frames.append(cat_frames_into_grid(frames, rows, cols))
    
    del colored_frames
    gc.collect()
    torch.cuda.empty_cache()
    
    grid_frames = torch.stack(grid_frames, dim=0)
    
    extra_args = dict()
    if args.audio:
        extra_args.update(dict(
            audio_array=input_video[1],
            audio_fps=input_video[2]['audio_fps'],
        ))

    dirname = os.path.dirname(args.output)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    options = {
        'crf': '18',  # Lower CRF for better quality
        'preset': 'slow',  # Use a slower preset for better compression efficiency
        'profile': 'high',  # Use high profile for advanced features
    }
    write_video(args.output, grid_frames, fps, video_codec=args.video_codec, options=options, **extra_args)


if __name__ == '__main__':
    main()
