python inference_api.py --dataroot None \
    --model font_pix2pix_v2 \
    --netG font_256 \
    --netD fontbasic \
    --direction BtoA \
    --checkpoints_dir ./save/inference/ \
    --cat_emb_path ./ckpt_and_files/mean_fc_feat.pt \
    --cls_ckpt ./ckpt_and_files/model_best.pth.tar \
    --ckpt_path ./ckpt_and_files/50_net_G.pth \
    --gpu_ids 0
