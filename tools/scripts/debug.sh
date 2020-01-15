# CUDA_VISIBLE_DEVICES=1,3 python train.py --dataroot ../datasets/font_cls --batch_size 4 --checkpoints_dir ./debug --dataset_mode classification --gpu_ids 0,1

CUDA_VISIBLE_DEVICES=1,3 python cls_train.py --dataroot ../datasets/font_cls --batch_size 64 --checkpoints_dir ./debug --dataset_mode classification --gpu_ids 0,1