import os
from PIL import Image
import cv2
import torch
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
from segment_anything import (SamAutomaticMaskGenerator, SamPredictor,
                              sam_model_registry)

if __name__ == '__main__':
    
    parser = ArgumentParser(description="SAM segment everything masks extracting params")
    
    parser.add_argument("--image_root", default='/datasets/nerf_data/360_v2/garden/', type=str)
    parser.add_argument("--sam_checkpoint_path", default='./third_party/segment-anything/sam_ckpt/sam_vit_h_4b8939.pth', type=str)
    parser.add_argument("--sam_arch", default="vit_h", type=str)
    parser.add_argument("--downsample", default=1, type=int)
    parser.add_argument("--downsample_type", default='image', type=str, choices=['image', 'mask'], help="Downsample then segment, or segment then downsample.")

    args = parser.parse_args()
    
    print("Initializing SAM...")
    model_type = args.sam_arch
    sam = sam_model_registry[model_type](checkpoint=args.sam_checkpoint_path).to('cuda')
    predictor = SamPredictor(sam)
    
    # custom
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.88,
        box_nms_thresh=0.7,
        stability_score_thresh=0.95,
        crop_n_layers=0,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )

    downsample_manually = False
    if args.downsample == "1" or args.downsample_type == 'mask':
        IMAGE_DIR = os.path.join(args.image_root, 'images')
    else:
        IMAGE_DIR = os.path.join(args.image_root, 'images_'+str(args.downsample))
        if not os.path.exists(IMAGE_DIR):
            IMAGE_DIR = os.path.join(args.image_root, 'images')
            downsample_manually = True
            print("No downsampled images, do it manually.")

    assert os.path.exists(IMAGE_DIR) and "Please specify a valid image root"
    OUTPUT_DIR = os.path.join(args.image_root, 'sam_masks')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Extracting SAM segment everything masks...")
    
    for path in tqdm(sorted(os.listdir(IMAGE_DIR))):
        name = path.split('.')[0]
        img = cv2.imread(os.path.join(IMAGE_DIR, path))
        if downsample_manually:
            img = cv2.resize(img,dsize=(img.shape[1] // args.downsample, img.shape[0] // args.downsample),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
        masks = mask_generator.generate(img)
        # print(len(masks))
        mask_list = []
        for m in masks:
            m_score = torch.from_numpy(m['segmentation']).float().to('cuda')

            if args.downsample_type == 'mask':
                m_score = torch.nn.functional.interpolate(m_score.unsqueeze(0).unsqueeze(0), size=(img.shape[0] // args.downsample, img.shape[1] // args.downsample) , mode='bilinear', align_corners=False).squeeze()
                m_score[m_score >= 0.5] = 1
                m_score[m_score != 1] = 0
                m_score = m_score.bool()

            if len(m_score.unique()) < 2:
                continue
            else:
                mask_list.append(m_score.bool())
        masks = torch.stack(mask_list, dim=0)

        torch.save(masks, os.path.join(OUTPUT_DIR, name+'.pt'))