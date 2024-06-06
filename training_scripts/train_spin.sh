CUDA_VISIBLE_DEVICES=0 python train_contrastive_feature.py --model_path ./output/fork --iterations 10000 --feature_lr 0.0025 --num_sampled_rays 1000
CUDA_VISIBLE_DEVICES=0 python train_contrastive_feature.py --model_path ./output/lego --iterations 10000 --feature_lr 0.0025 --num_sampled_rays 1000
CUDA_VISIBLE_DEVICES=0 python train_contrastive_feature.py --model_path ./output/pinecone --iterations 10000 --feature_lr 0.0025 --num_sampled_rays 1000
CUDA_VISIBLE_DEVICES=0 python train_contrastive_feature.py --model_path ./output/room --iterations 10000 --feature_lr 0.0025 --num_sampled_rays 1000
CUDA_VISIBLE_DEVICES=1 python train_contrastive_feature.py --model_path ./output/truck --iterations 10000 --feature_lr 0.0025 --num_sampled_rays 1000