import os
from PIL import Image
import cv2
import torch
from tqdm import tqdm
from argparse import ArgumentParser

from clip_utils.clip_utils import load_clip
from clip_utils import get_features_from_image_and_masks

if __name__ == '__main__':
    
    parser = ArgumentParser(description="Get CLIP features with SAM masks")
    
    parser.add_argument("--image_root", default='./data/360_v2/garden/', type=str)

    args = parser.parse_args()

    clip_model = load_clip()
    clip_model.eval()


    OUTPUT_DIR = os.path.join(args.image_root, 'clip_features')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with torch.no_grad():
        for i, image_path in tqdm(enumerate(sorted(os.listdir(os.path.join(args.image_root, 'images'))))):
            # print(image_path)
            image = cv2.imread(os.path.join(os.path.join(args.image_root, 'images'), image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            masks = torch.load(os.path.join(os.path.join(args.image_root, 'sam_masks'), image_path.replace('jpg', 'pt').replace('JPG', 'pt').replace('png', 'pt')))
            # N_mask, C
            features = get_features_from_image_and_masks(clip_model, image, masks, background = 0.)

            torch.save(features, os.path.join(OUTPUT_DIR, image_path.replace('jpg', 'pt').replace('JPG', 'pt').replace('png', 'pt')))

    torch.cuda.empty_cache()