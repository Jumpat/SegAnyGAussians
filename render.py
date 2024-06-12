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

import torch
from scene import Scene, GaussianModel, FeatureGaussianModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_contrastive_feature, render_mask
import torchvision

from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, target, precomputed_mask = None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    mask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "mask")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(mask_path, exist_ok=True)

    if target == 'feature':
        render_func = render_contrastive_feature
    elif target == 'contrastive_feature':
        render_func = render_contrastive_feature
    elif target == 'xyz':
        render_func = render
    else:
        render_func = render

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        res = render_func(view, gaussians, pipeline, background)

        if target == 'seg':
            assert precomputed_mask is not None and 'Rendering 2D segmentation mask requires a precomputed mask.'
            mask_res = render_mask(view, gaussians, pipeline, background, precomputed_mask=precomputed_mask)

        rendering = res["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        if target == 'seg':
            mask = mask_res["mask"]
            mask[mask < 0.5] = 0
            mask[mask != 0] = 1
            mask = mask[0, :, :]
            torchvision.utils.save_image(mask, os.path.join(mask_path, '{0:05d}'.format(idx) + ".png"))
        if target == 'seg' or target == 'scene':
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        elif 'feature' in target:
            torch.save(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".pt"))
        elif target == 'xyz':
            torch.save(rendering, os.path.join(render_path, 'xyz_{0:05d}'.format(idx) + ".pt"))
        
        
        

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, segment : bool = False, target = 'scene', idx = 0, precomputed_mask = None):
    dataset.need_features = dataset.need_masks = False
    if segment:
        assert target == 'seg' or target == 'coarse_seg_everything' or precomputed_mask is not None and "Segmentation only works with target seg!"
    gaussians, feature_gaussians = None, None
    with torch.no_grad():
        if precomputed_mask is not None:
            if '.pt' in precomputed_mask:
                precomputed_mask = torch.load(precomputed_mask)
            elif '.npy' in precomputed_mask:
                import numpy as np
                precomputed_mask = torch.from_numpy(np.load(precomputed_mask)).cuda()
                precomputed_mask[precomputed_mask > 0] = 1
                precomputed_mask[precomputed_mask != 1] = 0
                precomputed_mask = precomputed_mask.bool()

        if target == 'scene' or target == 'seg' or target == 'coarse_seg_everything' or target == 'xyz':
            gaussians = GaussianModel(dataset.sh_degree)
        if target == 'feature' or target == 'coarse_seg_everything' or target == 'contrastive_feature':
            feature_gaussians = FeatureGaussianModel(dataset.feature_dim)

        scene = Scene(dataset, gaussians, feature_gaussians, load_iteration=iteration, shuffle=False, mode='eval', target=target if target != 'xyz' and precomputed_mask is None else 'scene')

        if segment:
            gaussians.segment(precomputed_mask)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        if 'feature' in target:
            gaussians = feature_gaussians
            bg_color = [1 for i in range(dataset.feature_dim)] if dataset.white_background else [0 for i in range(dataset.feature_dim)]

        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, target, precomputed_mask=precomputed_mask)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, target, precomputed_mask=precomputed_mask)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--segment", action="store_true")
    parser.add_argument('--target', default='scene', const='scene', nargs='?', choices=['scene', 'seg', 'feature', 'coarse_seg_everything', 'contrastive_feature', 'xyz'])
    parser.add_argument('--idx', default=0, type=int)
    parser.add_argument('--precomputed_mask', default=None, type=str)

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    if not hasattr(args, 'precomputed_mask'):
        args.precomputed_mask = None
    if args.precomputed_mask is not None:
        print("Using precomputed mask " + args.precomputed_mask)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.segment, args.target, args.idx, args.precomputed_mask)