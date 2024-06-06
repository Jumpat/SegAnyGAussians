# CUDA_VISIBLE_DEVICES=3 python extract_segment_everything_masks.py --image_root ./data/Replica/office_0/Sequence_1_colmap --downsample 4;
CUDA_VISIBLE_DEVICES=3 python get_scale.py --image_root ./data/Replica/office_0/Sequence_1_colmap --model_path ./output/replica_office_0;
CUDA_VISIBLE_DEVICES=3 python get_clip_features.py --image_root ./data/Replica/office_0/Sequence_1_colmap;

# CUDA_VISIBLE_DEVICES=3 python extract_segment_everything_masks.py --image_root ./data/Replica/office_1/Sequence_1_colmap --downsample 4;
CUDA_VISIBLE_DEVICES=3 python get_scale.py --image_root ./data/Replica/office_1/Sequence_1_colmap --model_path ./output/replica_office_1;
CUDA_VISIBLE_DEVICES=3 python get_clip_features.py --image_root ./data/Replica/office_1/Sequence_1_colmap;

# CUDA_VISIBLE_DEVICES=3 python extract_segment_everything_masks.py --image_root ./data/Replica/office_2/Sequence_1_colmap --downsample 4;
CUDA_VISIBLE_DEVICES=3 python get_scale.py --image_root ./data/Replica/office_2/Sequence_1_colmap --model_path ./output/replica_office_2;
CUDA_VISIBLE_DEVICES=3 python get_clip_features.py --image_root ./data/Replica/office_2/Sequence_1_colmap;

# CUDA_VISIBLE_DEVICES=3 python extract_segment_everything_masks.py --image_root ./data/Replica/office_3/Sequence_1_colmap --downsample 4;
CUDA_VISIBLE_DEVICES=3 python get_scale.py --image_root ./data/Replica/office_3/Sequence_1_colmap --model_path ./output/replica_office_3;
CUDA_VISIBLE_DEVICES=3 python get_clip_features.py --image_root ./data/Replica/office_3/Sequence_1_colmap;

# CUDA_VISIBLE_DEVICES=3 python extract_segment_everything_masks.py --image_root ./data/Replica/office_4/Sequence_1_colmap --downsample 4;
CUDA_VISIBLE_DEVICES=3 python get_scale.py --image_root ./data/Replica/office_4/Sequence_1_colmap --model_path ./output/replica_office_4;
CUDA_VISIBLE_DEVICES=3 python get_clip_features.py --image_root ./data/Replica/office_4/Sequence_1_colmap;

# CUDA_VISIBLE_DEVICES=3 python extract_segment_everything_masks.py --image_root ./data/Replica/room_0/Sequence_1_colmap --downsample 4;
CUDA_VISIBLE_DEVICES=3 python get_scale.py --image_root ./data/Replica/room_0/Sequence_1_colmap --model_path ./output/replica_room_0;
CUDA_VISIBLE_DEVICES=3 python get_clip_features.py --image_root ./data/Replica/room_0/Sequence_1_colmap;

# CUDA_VISIBLE_DEVICES=3 python extract_segment_everything_masks.py --image_root ./data/Replica/room_1/Sequence_1_colmap --downsample 4;
CUDA_VISIBLE_DEVICES=3 python get_scale.py --image_root ./data/Replica/room_1/Sequence_1_colmap --model_path ./output/replica_room_1;
CUDA_VISIBLE_DEVICES=3 python get_clip_features.py --image_root ./data/Replica/room_1/Sequence_1_colmap;

# CUDA_VISIBLE_DEVICES=3 python extract_segment_everything_masks.py --image_root ./data/Replica/room_2/Sequence_1_colmap --downsample 4;
CUDA_VISIBLE_DEVICES=3 python get_scale.py --image_root ./data/Replica/room_2/Sequence_1_colmap --model_path ./output/replica_room_2;
CUDA_VISIBLE_DEVICES=3 python get_clip_features.py --image_root ./data/Replica/room_2/Sequence_1_colmap;