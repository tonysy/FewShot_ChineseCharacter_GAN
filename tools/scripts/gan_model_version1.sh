# CUDA_VISIBLE_DEVICES=0 python font_train.py \
#     --dataroot ../datasets/font_cls \
#     --batch_size 4 \
#     --checkpoints_dir ./save/gan_model_version1 \
#     --dataset_mode font_aligned \
#     --model font_pix2pix \
#     --netG  font_256 \
#     --direction BtoA \
#     --gpu_ids 0

CUDA_VISIBLE_DEVICES=0,1,2,3 python font_train.py \
    --dataroot ../datasets/font_cls \
    --batch_size  64\
    --checkpoints_dir ./save/gan_model_version1 \
    --dataset_mode font_aligned \
    --model font_pix2pix \
    --netG  font_256 \
    --direction BtoA \
    --gpu_ids 0,1,2,3

# CUDA_VISIBLE_DEVICES=0,1,2,3 python font_train.py \
#     --dataroot ../datasets/font_cls \
#     --batch_size 16 \
#     --model
#     --checkpoints_dir ./save/gan_model_version1 \
#     --dataset_mode font_pix2pix \
#     --model font_pix2pix \
#     --netG  font_256 \
#     --gpu_ids 0,1,2,3
