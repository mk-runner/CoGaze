#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ../main_v0702_v0826.py \
--task "pretrain" \
--phase "finetune" \
--data_name "mimic_cxr" \
--version "v0702-v0826-pretrain" \
--ann_path "six_work_mimic_cxr_annotation_similar_case_v0702_gaze.json" \
--images_dir "/MIMIC-CXR/files" \
--view_position_path "/MIMIC-CXR/view-positions-dict.json" \
--eye_gaze_dir "/MIMIC-Eye-Gaze-Heatmap" \
--batch_size 80 \
--test_batch_size 80 \
--num_workers 8 \
--patience 10 \
--num_nodes 1 \
--encoder_max_length 300 \
--max_length 100 \
--learning_rate 5.0e-5 \
--save_best_model "yes" \
--save_last_model 'no' \
--ckpt_zoo_dir "/checkpoints" \
--max_epochs 50
