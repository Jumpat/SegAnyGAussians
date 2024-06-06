CUDA_VISIBLE_DEVICES=3 python train_contrastive_feature.py --model_path ./output/replica_office_0 --iterations 10000 --feature_lr 0.0025 --num_sampled_rays 1600 --smooth_K 8
CUDA_VISIBLE_DEVICES=3 python train_contrastive_feature.py --model_path ./output/replica_office_1 --iterations 10000 --feature_lr 0.0025 --num_sampled_rays 1600 --smooth_K 8
CUDA_VISIBLE_DEVICES=3 python train_contrastive_feature.py --model_path ./output/replica_office_2 --iterations 10000 --feature_lr 0.0025 --num_sampled_rays 1600 --smooth_K 8
CUDA_VISIBLE_DEVICES=3 python train_contrastive_feature.py --model_path ./output/replica_office_3 --iterations 10000 --feature_lr 0.0025 --num_sampled_rays 1600 --smooth_K 8
CUDA_VISIBLE_DEVICES=3 python train_contrastive_feature.py --model_path ./output/replica_office_4 --iterations 10000 --feature_lr 0.0025 --num_sampled_rays 1600 --smooth_K 8
CUDA_VISIBLE_DEVICES=3 python train_contrastive_feature.py --model_path ./output/replica_room_0 --iterations 10000 --feature_lr 0.0025 --num_sampled_rays 1600 --smooth_K 8
CUDA_VISIBLE_DEVICES=3 python train_contrastive_feature.py --model_path ./output/replica_room_1 --iterations 10000 --feature_lr 0.0025 --num_sampled_rays 1600 --smooth_K 8
CUDA_VISIBLE_DEVICES=3 python train_contrastive_feature.py --model_path ./output/replica_room_2 --iterations 10000 --feature_lr 0.0025 --num_sampled_rays 1600 --smooth_K 8
