CUDA_VISIBLE_DEVICES=1 python train_s2me.py --model1 unet --model2 ynet_ffc --sup Scribble --exp s2me-ent-css_5.0_25k \
--mps True --mps_type entropy --cps True