python extract_segment_everything_masks.py --image_root $1 --downsample 4;
python get_scale.py --image_root $1 --model_path $2;
python get_clip_features.py --image_root $1;