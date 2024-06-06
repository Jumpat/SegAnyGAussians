# CUDA_VISIBLE_DEVICES=1 python extract_segment_everything_masks.py --image_root ./data/fork --downsample 4;
# CUDA_VISIBLE_DEVICES=1 python get_scale.py --image_root ./data/fork --model_path ./output/fork;
# CUDA_VISIBLE_DEVICES=1 python get_clip_features.py --image_root ./data/fork;

# CUDA_VISIBLE_DEVICES=1 python extract_segment_everything_masks.py --image_root ./data/lego_3dgs --downsample 4;
# CUDA_VISIBLE_DEVICES=1 python get_scale.py --image_root ./data/lego_3dgs --model_path ./output/lego;
# CUDA_VISIBLE_DEVICES=1 python get_clip_features.py --image_root ./data/lego_3dgs;

CUDA_VISIBLE_DEVICES=0 python extract_segment_everything_masks.py --image_root ./data/nerf_real_360/pinecone --downsample 8;
CUDA_VISIBLE_DEVICES=0 python get_scale.py --image_root ./data/nerf_real_360/pinecone --model_path ./output/pinecone;
CUDA_VISIBLE_DEVICES=0 python get_clip_features.py --image_root ./data/nerf_real_360/pinecone;

# CUDA_VISIBLE_DEVICES=0 python extract_segment_everything_masks.py --image_root ./data/nerf_llff_data_for_3dgs/room --downsample 8;
# CUDA_VISIBLE_DEVICES=0 python get_scale.py --image_root ./data/nerf_llff_data_for_3dgs/room --model_path ./output/room;
# CUDA_VISIBLE_DEVICES=0 python get_clip_features.py --image_root ./data/nerf_llff_data_for_3dgs/room;

# CUDA_VISIBLE_DEVICES=1 python extract_segment_everything_masks.py --image_root ./data/tandt/truck --downsample 4;
# CUDA_VISIBLE_DEVICES=1 python get_scale.py --image_root ./data/tandt/truck --model_path ./output/truck;
# CUDA_VISIBLE_DEVICES=1 python get_clip_features.py --image_root ./data/tandt/truck;