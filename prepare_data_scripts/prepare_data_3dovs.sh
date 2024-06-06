CUDA_VISIBLE_DEVICES=2 python extract_segment_everything_masks.py --image_root ./data/3dovs/bed --downsample 4;
CUDA_VISIBLE_DEVICES=2 python get_scale.py --image_root ./data/3dovs/bed --model_path ./output/3dovs-bed;
# CUDA_VISIBLE_DEVICES=2 python get_clip_features.py --image_root ./data/3dovs/bed;

# CUDA_VISIBLE_DEVICES=2 python extract_segment_everything_masks.py --image_root ./data/3dovs/bench --downsample 4;
# CUDA_VISIBLE_DEVICES=2 python get_scale.py --image_root ./data/3dovs/bench --model_path ./output/3dovs-bench;
# CUDA_VISIBLE_DEVICES=2 python get_clip_features.py --image_root ./data/3dovs/bench;

# CUDA_VISIBLE_DEVICES=2 python extract_segment_everything_masks.py --image_root ./data/3dovs/room --downsample 4;
# CUDA_VISIBLE_DEVICES=2 python get_scale.py --image_root ./data/3dovs/room --model_path ./output/3dovs-room;
# CUDA_VISIBLE_DEVICES=2 python get_clip_features.py --image_root ./data/3dovs/room;

# CUDA_VISIBLE_DEVICES=2 python extract_segment_everything_masks.py --image_root ./data/3dovs/sofa --downsample 4;
# CUDA_VISIBLE_DEVICES=2 python get_scale.py --image_root ./data/3dovs/sofa --model_path ./output/3dovs-sofa;
# CUDA_VISIBLE_DEVICES=2 python get_clip_features.py --image_root ./data/3dovs/sofa;

# CUDA_VISIBLE_DEVICES=2 python extract_segment_everything_masks.py --image_root ./data/3dovs/lawn --downsample 8;
# CUDA_VISIBLE_DEVICES=2 python get_scale.py --image_root ./data/3dovs/lawn --model_path ./output/3dovs-lawn;
# CUDA_VISIBLE_DEVICES=2 python get_clip_features.py --image_root ./data/3dovs/lawn;

# CUDA_VISIBLE_DEVICES=2 python extract_segment_everything_masks.py --image_root ./data/3dovs/table --downsample 4;
# CUDA_VISIBLE_DEVICES=2 python get_scale.py --image_root ./data/3dovs/table --model_path ./output/3dovs-table;
# CUDA_VISIBLE_DEVICES=2 python get_clip_features.py --image_root ./data/3dovs/table;