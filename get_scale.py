import torch


import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from argparse import ArgumentParser, Namespace
import cv2

from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel, FeatureGaussianModel

import gaussian_renderer
import importlib
importlib.reload(gaussian_renderer)

import os
FEATURE_DIM = 32

DATA_ROOT = './data/nerf_llff_data_for_3dgs/'
# MODEL_PATH = './output/figurines_lerf_poses/'
# MODEL_PATH = './output/figurines/'

ALLOW_PRINCIPLE_POINT_SHIFT = False


def get_combined_args(parser : ArgumentParser):
    # cmdlne_string = ['--model_path', model_path]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args()
    
    target_cfg_file = "cfg_args"

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, target_cfg_file)
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file found: {}".format(cfgfilepath))
        pass
    args_cfgfile = eval(cfgfile_string)

    # for k in args_cfgfile.__dict__.keys():
        # print(k, args_cfgfile.__dict__[k], "?")

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v

    # for k in merged_dict.keys():
        # print(k, merged_dict[k])
    return Namespace(**merged_dict)

def generate_grid_index(depth):
    h, w = depth.shape
    grid = torch.meshgrid([torch.arange(h), torch.arange(w)])
    grid = torch.stack(grid, dim=-1)
    return grid


if __name__ == '__main__':
    
    parser = ArgumentParser(description="Get scales for SAM masks")

    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--segment", action="store_true")
    parser.add_argument('--idx', default=0, type=int)
    parser.add_argument('--precomputed_mask', default=None, type=str)

    parser.add_argument("--image_root", default='/datasets/nerf_data/360_v2/garden/', type=str)

    args = get_combined_args(parser)

    dataset = model.extract(args)
    dataset.need_features = False
    dataset.need_masks = False

    # ALLOW_PRINCIPLE_POINT_SHIFT = 'lerf' in args.model_path
    dataset.allow_principle_point_shift = ALLOW_PRINCIPLE_POINT_SHIFT

    feature_gaussians = None
    scene_gaussians = GaussianModel(dataset.sh_degree)

    scene = Scene(dataset, scene_gaussians, feature_gaussians, load_iteration=-1, feature_load_iteration=-1, shuffle=False, mode='eval', target='scene')


    assert os.path.exists(os.path.join(dataset.source_path, 'images')) and "Please specify a valid image root."
    assert os.path.join(dataset.source_path, 'sam_masks') and "Please run extract_segment_everything_masks first."

    from tqdm import tqdm
    images_masks = {}
    for i, image_path in tqdm(enumerate(sorted(os.listdir(os.path.join(dataset.source_path, 'images'))))):
        # print(image_path)
        image = cv2.imread(os.path.join(os.path.join(dataset.source_path, 'images'), image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = torch.load(os.path.join(os.path.join(dataset.source_path, 'sam_masks'), image_path.replace('jpg', 'pt').replace('JPG', 'pt').replace('png', 'pt')))
        # N_mask, C

        images_masks[image_path.split('.')[0]] = masks.cpu().float()


    OUTPUT_DIR = os.path.join(args.image_root, 'mask_scales')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cameras = scene.getTrainCameras()

    background = torch.zeros(scene_gaussians.get_mask.shape[0], 3, device = 'cuda')

    for it, view in tqdm(enumerate(cameras)):

        rendered_pkg = gaussian_renderer.render_with_depth(view, scene_gaussians, pipeline.extract(args), background)

        depth = rendered_pkg['depth']

        # plt.imshow(depth.detach().cpu().squeeze().numpy())
        corresponding_masks = images_masks[view.image_name]

        # generate_grid_index(depth.squeeze())[50, 1]

        depth = depth.cpu().squeeze()

        grid_index = generate_grid_index(depth)

        points_in_3D = torch.zeros(depth.shape[0], depth.shape[1], 3).cpu()
        points_in_3D[:,:,-1] = depth

        # caluculate cx cy fx fy with FoVx FoVy
        cx = depth.shape[1] / 2
        cy = depth.shape[0] / 2
        fx = cx / np.tan(cameras[0].FoVx / 2)
        fy = cy / np.tan(cameras[0].FoVy / 2)


        points_in_3D[:,:,0] = (grid_index[:,:,0] - cx) * depth / fx
        points_in_3D[:,:,1] = (grid_index[:,:,1] - cy) * depth / fy

        upsampled_mask = torch.nn.functional.interpolate(corresponding_masks.unsqueeze(1), mode = 'bilinear', size = (depth.shape[0], depth.shape[1]), align_corners = False)

        eroded_masks = torch.conv2d(
            upsampled_mask.float(),
            torch.full((3, 3), 1.0).view(1, 1, 3, 3),
            padding=1,
        )
        eroded_masks = (eroded_masks >= 5).squeeze()  # (num_masks, H, W)

        scale = torch.zeros(len(corresponding_masks))
        for mask_id in range(len(corresponding_masks)):
            
            point_in_3D_in_mask = points_in_3D[eroded_masks[mask_id] == 1]

            scale[mask_id] = (point_in_3D_in_mask.std(dim=0) * 2).norm()

        torch.save(scale, os.path.join(OUTPUT_DIR, view.image_name + '.pt'))