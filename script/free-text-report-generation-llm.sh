#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python ../main_v0702_v0904.py \
--task "report-generation-lora" \
--phase "finetune" \
--data_name "mimic_cxr" \
--version "stage2-v0702-v0904-llm" \
--ann_path "six_work_mimic_cxr_annotation_similar_case_v0702_v0826.json" \
--images_dir "MIMIC-CXR/files" \
--view_position_path "/MIMIC-CXR/view-positions-dict.json" \
--eye_gaze_dir "/MIMIC-Eye-Gaze-Heatmap" \
--batch_size 6 \
--test_batch_size 6 \
--num_workers 10 \
--patience 2 \
--num_beams 3 \
--encoder_max_length 300 \
--max_length 100 \
--accumulate_grad_batches 2 \
--learning_rate 5.0e-5 \
--load "finetune_stage2-v0702-v0905-not-precision_2025_09_05_18-best/checkpoint/best_model.pt" \
--save_best_model "yes" \
--save_last_model 'no' \
--llm_use_lora "yes" \
--ckpt_zoo_dir "/checkpoints" \
--max_epochs 30
