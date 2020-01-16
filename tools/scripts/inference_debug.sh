python inference_api.py --dataroot None \
    --model font_pix2pix_v2 \
    --netG font_256 \
    --netD fontbasic \
    --direction BtoA \
    --checkpoints_dir ./save/inference/ \
    --ckpt_path ./save/gan_model_version2_exp4/experiment_name/50_net_G.pth \
    --gpu_ids 0