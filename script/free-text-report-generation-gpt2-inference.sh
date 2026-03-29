#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python ../main_v0702_v0826.py \
--task "report-generation" \
--phase "inference" \
--data_name "mimic_cxr" \
--version "v0702-v0826-final-v1014-beam10" \
--ann_path "cogaze_mimic_cxr_annotation_similar_case_v0702_v0826.json" \
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
--test_ckpt_path "https://huggingface.co/MK-runner/CoGaze/blob/main/distilgpt2_mimic_free_text_report_generation_best_model.pt" \
--ckpt_zoo_dir "/ckpt_zoo_dir/" \
--num_beams 10 \
--max_epochs 30
