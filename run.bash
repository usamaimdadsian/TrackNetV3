#!/usr/bin/bash
# python predict.py --video_file /mnt/linux/codes/harry/PadelBuddy/gen_data/padel_short.mp4 --tracknet_file ckpts/TrackNet_best.pt --inpaintnet_file ckpts/InpaintNet_best.pt --save_dir prediction --output_video
# python predict_new.py --video_file /mnt/linux/codes/harry/PadelBuddy/gen_data/padel_short.mp4 --tracknet_file ckpts/TrackNet_best.pt --inpaintnet_file ckpts/InpaintNet_best.pt --save_dir prediction --output_video
python predict_new.py --video_file /mnt/linux/codes/harry/PadelBuddy/gen_data/padel_short.mp4 --tracknet_file ckpts/TrackNet_best.pt --save_dir prediction --output_video
