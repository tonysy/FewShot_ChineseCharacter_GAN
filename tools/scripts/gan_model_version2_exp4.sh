# CUDA_VISIBLE_DEVICES=0 python font_train.py \
CUDA_VISIBLE_DEVICES=2,3 python font_train.py \
    --dataroot ../datasets/font_cls \
    --batch_size  64 \
    --checkpoints_dir ./save/gan_model_version2_exp4 \
    --dataset_mode font_aligned \
    --model font_pix2pix_v2 \
    --netG font_256 \
    --netD fontbasic \
    --direction BtoA \
    --lambda_content_loss 20.0 \
    --lambda_cat_loss 2.0 \
    --gpu_ids 0,1
    #,1,2,3
