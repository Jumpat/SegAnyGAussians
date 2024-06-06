CUDA_VISIBLE_DEVICES=1 python train_contrastive_feature.py --model_path ./output/3dovs-bed --iterations 10000 --feature_lr 0.0025 --num_sampled_ray 1000
CUDA_VISIBLE_DEVICES=1 python train_contrastive_feature.py --model_path ./output/3dovs-bench --iterations 10000 --feature_lr 0.0025 --num_sampled_ray 1000
CUDA_VISIBLE_DEVICES=1 python train_contrastive_feature.py --model_path ./output/3dovs-room --iterations 10000 --feature_lr 0.0025 --num_sampled_ray 1000
CUDA_VISIBLE_DEVICES=1 python train_contrastive_feature.py --model_path ./output/3dovs-sofa --iterations 10000 --feature_lr 0.0025 --num_sampled_ray 1000
CUDA_VISIBLE_DEVICES=1 python train_contrastive_feature.py --model_path ./output/3dovs-lawn --iterations 10000 --feature_lr 0.0025 --num_sampled_ray 1000