CUDA_VISIBLE_DEVICES=0 python font_train.py \
    --dataroot ../datasets/font_cls \
    --batch_size  4 \
    --checkpoints_dir ./save/gan_model_version2 \
    --dataset_mode font_aligned \
    --model font_pix2pix_v2 \
    --netG font_256 \
    --netD fontbasic \
    --direction BtoA \
    --gpu_ids 0
    #,1,2,3
