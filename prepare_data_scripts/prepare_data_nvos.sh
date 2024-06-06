# CUDA_VISIBLE_DEVICES=0 python extract_segment_everything_masks.py --image_root ./data/nerf_llff_data_for_3dgs/fern --downsample 8;
# CUDA_VISIBLE_DEVICES=0 python get_scale.py --image_root ./data/nerf_llff_data_for_3dgs/fern --model_path ./output/fern;
# CUDA_VISIBLE_DEVICES=0 python get_clip_features.py --image_root ./data/nerf_llff_data_for_3dgs/fern;

# CUDA_VISIBLE_DEVICES=0 python extract_segment_everything_masks.py --image_root ./data/nerf_llff_data_for_3dgs/flower --downsample 8;
# CUDA_VISIBLE_DEVICES=0 python get_scale.py --image_root ./data/nerf_llff_data_for_3dgs/flower --model_path ./output/flower;
# CUDA_VISIBLE_DEVICES=0 python get_clip_features.py --image_root ./output/flower; #????

# CUDA_VISIBLE_DEVICES=0 python extract_segment_everything_masks.py --image_root ./data/nerf_llff_data_for_3dgs/fortress --downsample 8;
# CUDA_VISIBLE_DEVICES=0 python get_scale.py --image_root ./data/nerf_llff_data_for_3dgs/fortress --model_path ./output/fortress;
# CUDA_VISIBLE_DEVICES=0 python get_clip_features.py --image_root ./data/nerf_llff_data_for_3dgs/fortress;

# CUDA_VISIBLE_DEVICES=0 python extract_segment_everything_masks.py --image_root ./data/nerf_llff_data_for_3dgs/horns --downsample 8;
# CUDA_VISIBLE_DEVICES=0 python get_scale.py --image_root ./data/nerf_llff_data_for_3dgs/horns --model_path ./output/horns;
# CUDA_VISIBLE_DEVICES=0 python get_clip_features.py --image_root ./data/nerf_llff_data_for_3dgs/horns;

# CUDA_VISIBLE_DEVICES=0 python extract_segment_everything_masks.py --image_root ./data/nerf_llff_data/leaves --downsample 8;
# CUDA_VISIBLE_DEVICES=0 python get_scale.py --image_root ./data/nerf_llff_data/leaves --model_path ./output/leaves;
# CUDA_VISIBLE_DEVICES=0 python get_clip_features.py --image_root ./data/nerf_llff_data/leaves;

# CUDA_VISIBLE_DEVICES=0 python extract_segment_everything_masks.py --image_root ./data/nerf_llff_data_for_3dgs/orchids --downsample 8;
# CUDA_VISIBLE_DEVICES=0 python get_scale.py --image_root ./data/nerf_llff_data_for_3dgs/orchids --model_path ./output/orchids;
# CUDA_VISIBLE_DEVICES=0 python get_clip_features.py --image_root ./data/nerf_llff_data_for_3dgs/orchids;

CUDA_VISIBLE_DEVICES=1 python extract_segment_everything_masks.py --image_root ./data/nerf_llff_data_for_3dgs/trex --downsample 8;
CUDA_VISIBLE_DEVICES=1 python get_scale.py --image_root ./data/nerf_llff_data_for_3dgs/trex --model_path ./output/trex;
CUDA_VISIBLE_DEVICES=1 python get_clip_features.py --image_root ./data/nerf_llff_data_for_3dgs/trex;