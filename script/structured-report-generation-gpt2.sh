#!/bin/bash


CUDA_VISIBLE_DEVICES=3 python ../main_v0702_v0904_srrg.py \
--task "report-generation" \
--phase "finetune" \
--data_name "srrg" \
--version "srrg-beam10-3-devices" \
--ann_path "six_work_srrg_annotation_v0702_v0826.json" \
--mimic_dir "/MIMIC-CXR" \
--chexpert_dir "/dataset/CheXper-Plus/PNG" \
--view_position_path "/MIMIC-CXR/view-positions-dict.json" \
--batch_size 48 \
--test_batch_size 48 \
--num_workers 12 \
--patience 2 \
--num_beams 10 \
--encoder_max_length 300 \
--max_length 150 \
--accumulate_grad_batches 2 \
--learning_rate 5.0e-5 \
--load "/script/results/mimic_cxr/report-generation/finetune_v0702-v0826-final-v1014-beam10_2025_10_14_17/checkpoint/best_model.pt" \
--save_best_model "yes" \
--save_last_model 'no' \
--ckpt_zoo_dir "/checkpoints" \
--max_epochs 40
