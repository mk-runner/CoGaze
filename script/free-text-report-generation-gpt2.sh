#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python ../main_v0702_v0826.py \
--task "report-generation" \
--phase "finetune" \
--data_name "mimic_cxr" \
--version "v0702-v0826-final-v1014-beam10" \
--ann_path "six_work_mimic_cxr_annotation_similar_case_v0702_v0826.json" \
--images_dir "files" \
--view_position_path "/MIMIC-CXR/view-positions-dict.json" \
--eye_gaze_dir "/MIMIC-Eye-Gaze-Heatmap/" \
--batch_size 64 \
--test_batch_size 64 \
--num_workers 12 \
--patience 2 \
--num_beams 10 \
--encoder_max_length 300 \
--max_length 100 \
--learning_rate 5.0e-5 \
--save_best_model "yes" \
--save_last_model 'no' \
--load "/results/mimic_cxr/pretrain/finetune_v0702-v0826-pretrain_2025_08_27_15/checkpoint/best_model.pt" \
--ckpt_zoo_dir "/checkpoints/" \
--num_beams 10 \
--max_epochs 30
