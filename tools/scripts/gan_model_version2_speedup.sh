# CUDA_VISIBLE_DEVICES=0,1,2,3 python font_train.py \
#     --dataroot ../datasets/font_cls \
#     --batch_size  64 \
#     --checkpoints_dir ./save/gan_model_version2_speedup \
#     --dataset_mode font_aligned_pkl \
#     --model font_pix2pix_v2 \
#     --netG font_256 \
#     --netD fontbasic \
#     --direction BtoA \
#     --gpu_ids 0,1,2,3

CUDA_VISIBLE_DEVICES=2,3 python font_train.py \
    --dataroot ../datasets/font_cls \
    --batch_size  64 \
    --checkpoints_dir ./save/gan_model_version2_speedup \
    --dataset_mode font_aligned_pkl \
    --model font_pix2pix_v2 \
    --netG font_256 \
    --netD fontbasic \
    --direction BtoA \
    --gpu_ids 0,1