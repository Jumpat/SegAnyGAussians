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

import os
import torch
from random import randint
from gaussian_renderer import render_contrastive_feature
from scene import Scene, GaussianModel, FeatureGaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args

import numpy as np
from matplotlib import pyplot as plt
from segment_anything import (SamAutomaticMaskGenerator, SamPredictor,
                              sam_model_registry)

import torch

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def training(dataset, opt, pipe, iteration):

    dataset.need_features = True
    dataset.need_masks = True

    gaussians = GaussianModel(dataset.sh_degree)

    feature_gaussians = FeatureGaussianModel(dataset.feature_dim)
    sample_rate = 1.0
    scene = Scene(dataset, gaussians, feature_gaussians, load_iteration=iteration, shuffle=False, target='contrastive_feature', mode='train', sample_rate=sample_rate)

    feature_gaussians.change_to_segmentation_mode(opt, "contrastive_feature", fixed_feature=False)

    sam_proj = torch.nn.Sequential(
        torch.nn.Linear(256, 64, bias=True),
        torch.nn.LayerNorm(64),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(64, 64, bias=True),
        torch.nn.LayerNorm(64),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(64, dataset.feature_dim, bias=True)
    )

    sam_proj = sam_proj.cuda()
    sam_proj.train()
    param_group = {'params': sam_proj.parameters(), 'lr': opt.feature_lr, 'name': 'f'}
    feature_gaussians.optimizer.add_param_group(param_group)

    del gaussians
    torch.cuda.empty_cache()

    background = torch.ones([dataset.feature_dim], dtype=torch.float32, device="cuda") if dataset.white_background else torch.zeros([dataset.feature_dim], dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    first_iter = 0
    viewpoint_stack = None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam = viewpoint_stack.pop()

        feature_gaussians.update_learning_rate(iteration)

        sam_features = viewpoint_cam.original_features.cuda()
        H,W = sam_features.shape[-2:]

        # N_mask, H, W
        sam_masks = viewpoint_cam.original_masks
        sam_masks = torch.nn.functional.interpolate(sam_masks.unsqueeze(0), size=sam_features.shape[-2:] , mode='nearest').squeeze()
        nonzero_masks = sam_masks.sum(dim=(1,2)) > 0
        sam_masks = sam_masks[nonzero_masks,:,:]
        full_resolution_sam_masks = viewpoint_cam.original_masks
        full_resolution_sam_masks = full_resolution_sam_masks[nonzero_masks,:,:]

        low_dim_sam_features = sam_proj(
            sam_features.reshape(-1, H*W).permute([1,0])
            ).permute([1,0]).reshape(dataset.feature_dim, H, W)

        # NHW, NCHW
        prototypes = (sam_masks.unsqueeze(1) * low_dim_sam_features).sum(dim = (2,3))
        prototypes /= sam_masks.sum(dim=(1,2)).unsqueeze(-1)

        render_pkg_feat = render_contrastive_feature(viewpoint_cam, feature_gaussians, pipe, background, nonlinear=None, dropout=-1)
        rendered_features = render_pkg_feat["render"]
        pp = torch.einsum('NC, CHW -> NHW', prototypes, rendered_features)

        prob = torch.sigmoid(pp)
        
        full_resolution_sam_masks = torch.nn.functional.interpolate(full_resolution_sam_masks.unsqueeze(0), size=prob.shape[-2:] , mode='bilinear').squeeze()
        full_resolution_sam_masks[full_resolution_sam_masks <= 0.5] = 0

        bce_contrastive_loss = full_resolution_sam_masks * torch.log(prob + 1e-8) + ((1 - full_resolution_sam_masks) * torch.log(1 - prob + 1e-8))
        bce_contrastive_loss = -bce_contrastive_loss.mean()

        rands = torch.rand(feature_gaussians.get_point_features.shape[0], device=prob.device)
        reg_loss = torch.relu(torch.einsum('NC,KC->NK', feature_gaussians.get_point_features[rands > 0.9, :], prototypes)).mean()
        loss = bce_contrastive_loss + 0.1 * reg_loss

        NHW = sam_masks
        N,H,W = NHW.shape
        NL = NHW.view(N,-1)
        intersection = torch.einsum('NL,NC->LC', NL, NL)
        union = NL.sum(dim = 0, keepdim = True) + NL.sum(dim = 0, keepdim = True).T - intersection
        similarity = intersection / (union + 1e-5)
        HWHW = similarity.view(H,W,H,W)
        HWHW[HWHW == 0] = -1
        norm_rendered_feature = torch.nn.functional.normalize(torch.nn.functional.interpolate(rendered_features.unsqueeze(0), (H,W), mode = 'bilinear').squeeze(), dim=0, p=2)
        correspondence = torch.relu(torch.einsum('CHW,CJK->HWJK', norm_rendered_feature, norm_rendered_feature))
        corr_loss = -HWHW * correspondence

        loss += corr_loss.mean()

        loss.backward()

        feature_gaussians.optimizer.step()
        feature_gaussians.optimizer.zero_grad(set_to_none = True)

        iter_end.record()

        if iteration % 10 == 0:
            progress_bar.set_postfix({
                                    "3D Loss": f"{bce_contrastive_loss.item():.{3}f}",
                                      "Corr Loss": f"{corr_loss.mean().item():.{3}f}",
                                      })
            progress_bar.update(10)
    
    
    scene.save_feature(iteration, target = 'contrastive_feature')
    torch.save(sam_proj.state_dict(), os.path.join(scene.model_path, "point_cloud/iteration_{}/".format(iteration) + "sam_proj.pt"))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--target', default='contrastive_feature', const='contrastive_feature', nargs='?', choices=['scene', 'seg', 'feature', 'coarse_seg_everything', 'contrastive_feature'])
    parser.add_argument("--iteration", default=-1, type=int)
    
    args = get_combined_args(parser, target_cfg_file = 'cfg_args')
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.iteration)

    # All done
    print("\nTraining complete.")
