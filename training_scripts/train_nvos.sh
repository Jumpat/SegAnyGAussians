CUDA_VISIBLE_DEVICES=2 python train_contrastive_feature.py --model_path ./output/fern --iterations 10000 --feature_lr 0.0025 --num_sampled_rays 1000
CUDA_VISIBLE_DEVICES=2 python train_contrastive_feature.py --model_path ./output/flower --iterations 10000 --feature_lr 0.0025 --num_sampled_rays 1000
CUDA_VISIBLE_DEVICES=2 python train_contrastive_feature.py --model_path ./output/fortress --iterations 10000 --feature_lr 0.0025 --num_sampled_rays 1000
CUDA_VISIBLE_DEVICES=2 python train_contrastive_feature.py --model_path ./output/horns --iterations 10000 --feature_lr 0.0025 --num_sampled_rays 1000
CUDA_VISIBLE_DEVICES=2 python train_contrastive_feature.py --model_path ./output/leaves --iterations 10000 --feature_lr 0.0025 --num_sampled_rays 1000
CUDA_VISIBLE_DEVICES=2 python train_contrastive_feature.py --model_path ./output/orchids --iterations 10000 --feature_lr 0.0025 --num_sampled_rays 1000
CUDA_VISIBLE_DEVICES=2 python train_contrastive_feature.py --model_path ./output/trex --iterations 10000 --feature_lr 0.0025 --num_sampled_rays 1000