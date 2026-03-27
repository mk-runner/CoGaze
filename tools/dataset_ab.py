import copy
import math
import random
import re
import os
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers.image_transforms import convert_to_rgb
from transformers.image_utils import infer_channel_dimension_format


def choose_images(view_positions):
    if len(view_positions) == 1:
        return 0
    if 'PA' in view_positions:
        candidates = [i for i, v in enumerate(view_positions) if v == 'PA']
        return random.choice(candidates)
    elif 'AP' in view_positions:
        candidates = [i for i, v in enumerate(view_positions) if v == 'AP']
        return random.choice(candidates)
    else:
        return random.randrange(len(view_positions))


def resize_heatmap(image, processor):
    # resize
    image = np.array(image)
    image = convert_to_rgb(image)
    input_data_format = infer_channel_dimension_format(image)
    crop_size = processor.crop_size
    size = {'shortest_edge': crop_size['height']}
    image = processor.resize(image=image, size=size, resample=processor.resample, input_data_format=input_data_format)
    image = processor.center_crop(image=image, size=crop_size, input_data_format=input_data_format)

    # convert RGB to gray
    image = torch.tensor(image, dtype=torch.float).permute(2, 0, 1)  # (c, h, w)

    # convert gray
    image = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
    image = image.unsqueeze(0)  # (1, H, W)
    return image


def get_gaze_regions(heatmap, edge_ratio=0.1, top_k=3):
    """
    the topk highest frequency colors, and find the top_color that appears most in the edge area (majority voting)
    """
    h, w = heatmap.shape

    # Count color frequencies
    flat = heatmap.clone().reshape(-1)
    colors, counts = torch.unique(flat, return_counts=True)
    top_colors = colors[np.argsort(-counts)[:top_k]]

    # Get edge regions (10% top/bottom/left/right)
    margin_h = int(h * edge_ratio)
    margin_w = int(w * edge_ratio)

    edge_mask = torch.zeros((h, w), dtype=torch.bool, device=heatmap.device)
    edge_mask[:margin_h, :] = True  # top
    edge_mask[-margin_h:, :] = True  # bottom
    edge_mask[:, :margin_w] = True  # left
    edge_mask[:, -margin_w:] = True  # right

    edge_pixels = heatmap[edge_mask]

    # Find the top_color that appears most in the edge area
    uni_edge_pixels, uni_counts = torch.unique(edge_pixels.reshape(-1), return_counts=True)
    unique_freq_map = dict(zip(uni_edge_pixels.tolist(), uni_counts.tolist()))

    top_colors_freqs = [unique_freq_map.get(val.item(), 0) for val in top_colors]
    final_bg_color = top_colors[np.argmax(top_colors_freqs)]

    # background regions are set to zero, otherwise not change.
    new_heatmap = torch.where(heatmap <= final_bg_color, torch.tensor(0), heatmap)
    return new_heatmap.reshape(-1)


def get_ground_truth_heatmap(image_path, processor, patch_size):
    image = Image.open(image_path).convert('RGB')
    # resize
    image = np.array(image)
    image = convert_to_rgb(image)
    input_data_format = infer_channel_dimension_format(image)
    crop_size = processor.crop_size
    size = {'shortest_edge': crop_size['height']}
    image = processor.resize(image=image, size=size, resample=processor.resample, input_data_format=input_data_format)
    image = processor.center_crop(image=image, size=crop_size, input_data_format=input_data_format)

    # convert RGB to gray
    image = torch.tensor(image, dtype=torch.float).permute(2, 0, 1)  # (c, h, w)
    _, h, w = image.shape
    # convert gray
    image = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]

    image = image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    saliency = F.adaptive_avg_pool2d(image, output_size=(h // patch_size, w // patch_size))

    # only record eye-gaze regions
    saliency = get_gaze_regions(heatmap=saliency.squeeze(0).squeeze(0), top_k=5)  # (num_patches,)
    saliency = torch.clamp(saliency, min=0.0)
    return saliency


def generate_saliency(gaze_image, gaze_idx, patch_size, image_size, batch_size, device):
    num_patches = (image_size // patch_size) ** 2 + 1  # has cls_token
    mix_saliency = torch.zeros(batch_size, num_patches).to(device)
    if len(gaze_image) != 0:
        if isinstance(gaze_image, list):
            gaze_image = torch.stack(gaze_image, dim=0).to(device)
        b, c, h, w = gaze_image.shape
        saliency = F.adaptive_avg_pool2d(gaze_image, output_size=(h // patch_size, w // patch_size))

        # add cls_token
        saliency = saliency.reshape(b, -1)
        cls_token = torch.zeros(b, 1, device=device)
        saliency = torch.cat([cls_token, saliency], dim=1)

        # normalize
        min_saliency = saliency.min(dim=1, keepdim=True)[0]
        max_saliency = saliency.max(dim=1, keepdim=True)[0]
        saliency = (saliency - min_saliency) / (max_saliency - min_saliency + 1e-8)
        mix_saliency[gaze_idx] = saliency
    return mix_saliency


class AlignDataset(Dataset):
    def __init__(self, args, split):
        ann = json.loads(open(args['ann_path'], 'r').read())[split]

        self.examples = []
        for item in tqdm(ann):
            # delete the sample that has no clinical findings
            if len(item['findings_factual_serialization']) == 0:
                continue
            report = item['findings'].strip()
            indication = item['indication_pure']
            history = item['history_pure']
            self.examples.append({
                'dicom_id': item["id"],
                'subject_id': item['subject_id'],
                'study_id': item["study_id"],
                'image_path': item['anchor_scan']['image_path'][0],
                'view_position': item['anchor_scan']['view_position'][0],
                'report': f'[CLS] [FINDINGS] {report}',
                'indication': f'[INDICATION] {indication}',
                'history': f'[HISTORY] {history}',
                'gaze_heatmap': item['sentence_gaze_heatmap'],
                'disease_labels': item['disease_labels']
            })
        # self.examples = self.examples[:10]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        return example


class GlobalCollateFn:
    def __init__(self, args, processor, pad_token, patch_size):
        self.args = args
        self.processor = processor
        self.pad_token = pad_token
        self.patch_size = patch_size

    def __call__(self, data):
        # convert dict
        data = pd.DataFrame(data).to_dict(orient='list')
        batch_size = len(data['dicom_id'])

        # obtain valid_information
        batch_knowledge, batch_image, batch_patient_id = [], [], []  # batch_patient_id for image-text pair alignment
        for i in range(batch_size):
            # obtain the clinical prompt (indication + ' ' + history)
            knowledge = f"[CLS] {data['indication'][i].strip()} [SEP] {data['history'][i].strip()}".strip()
            knowledge = knowledge if knowledge != '[CLS] [INDICATION] [SEP] [HISTORY]' else self.pad_token
            batch_knowledge.append(knowledge)

            # obtain and chest X-rays and its view position
            image_path = data['image_path'][i]
            image = Image.open(os.path.join(self.args['images_dir'], image_path)).convert('RGB')
            image = self.processor(image, return_tensors='pt').pixel_values[0]
            batch_image.append(image)

            # obtain batch_patient_id (ensure multiple images from the same patient to align their report)
            patient_id = f"{data['subject_id'][i]}_{data['study_id'][i]}"
            batch_patient_id.append(patient_id)

        batch_image = torch.stack(batch_image, dim=0)
        disease_labels = torch.from_numpy(np.array(data['disease_labels'])).long()
        batch = {
            'dicom_id': data['dicom_id'],
            'patient_id': np.array(batch_patient_id),
            'image': batch_image,
            'report': data['report'],
            'view_position': data['view_position'],
            'knowledge': batch_knowledge,
            'disease_labels': disease_labels,
        }
        return batch


class AlignCollateFn:
    def __init__(self, args, processor, pad_token, patch_size):
        self.args = args
        self.processor = processor
        self.patch_size = patch_size
        self.pad_token = pad_token

    def __call__(self, data):
        # convert dict
        data = pd.DataFrame(data).to_dict(orient='list')
        batch_size = len(data['dicom_id'])

        # obtain valid_information
        batch_knowledge, batch_image, batch_patient_id = [], [], []  # batch_patient_id for image-text pair alignment
        batch_transcript_sen_id, batch_transcript_sen, batch_heatmap_sen = [], [], []  # sentence-level
        batch_transcript_report_id, batch_transcript_report, batch_heatmap_report = [], [], []  # report-level
        for i in range(batch_size):

            # obtain the clinical prompt (indication + ' ' + history)
            knowledge = f"[CLS] {data['indication'][i].strip()} [SEP] {data['history'][i].strip()}".strip()
            knowledge = knowledge if knowledge != '[CLS] [INDICATION] [SEP] [HISTORY]' else self.pad_token
            batch_knowledge.append(knowledge)

            # obtain and chest X-rays and its view position
            image_path = data['image_path'][i]
            image = Image.open(os.path.join(self.args['images_dir'], image_path)).convert('RGB')
            image = self.processor(image, return_tensors='pt').pixel_values[0]
            batch_image.append(image)

            # obtain batch_patient_id (ensure multiple images from the same patient to align their report)
            patient_id = f"{data['subject_id'][i]}_{data['study_id'][i]}"
            batch_patient_id.append(patient_id)

            # obtain eye-gaze data
            if data['gaze_heatmap'][i] is not None:
                # report-level gaze heatmap
                report_heatmap_path = os.path.join(self.args['eye_gaze_dir'], data['gaze_heatmap'][i]['image_dir'],
                                                   'report_heatmap.jpg')
                report_transcript = data['gaze_heatmap'][i]['transcript_full_text']
                current_heatmap = get_ground_truth_heatmap(report_heatmap_path, self.processor, self.patch_size)
                batch_heatmap_report.append(current_heatmap)
                batch_transcript_report.append(f"[CLS] [TRANSCRIPT] {report_transcript}")
                batch_transcript_report_id.append(patient_id)
                # sentence-level gaze heatmap
                # obtain eye-gaze data
                gaze_dir = os.path.join(self.args['eye_gaze_dir'], data['gaze_heatmap'][i]['image_dir'])
                gaze_lookup = json.load(open(os.path.join(gaze_dir, 'sentence-heatmap-lookup.json')))

                # notably, only the sentence-level heatmap is considered
                for key, transcript in gaze_lookup.items():
                    if key == 'full_text':
                        # not consider report-level heatmap. Since report length is too big, waste computational head
                        continue
                    heatmap_path = os.path.join(gaze_dir, f"{key}_heatmap.jpg")
                    current_heatmap = get_ground_truth_heatmap(heatmap_path, self.processor, self.patch_size)
                    batch_heatmap_sen.append(current_heatmap)
                    batch_transcript_sen.append(f"[CLS] [TRANSCRIPT] {transcript}")
                    batch_transcript_sen_id.append(patient_id)

        batch_image = torch.stack(batch_image, dim=0)
        batch_heatmap_sen = torch.stack(batch_heatmap_sen, dim=0) if len(batch_heatmap_sen) != 0 else None
        batch_heatmap_report = torch.stack(batch_heatmap_report, dim=0) if len(batch_heatmap_report) != 0 else None
        disease_labels = torch.from_numpy(np.array(data['disease_labels'])).long()
        batch = {
            'dicom_id': data['dicom_id'],
            'patient_id': np.array(batch_patient_id),
            'image': batch_image,
            'report': data['report'],
            'view_position': data['view_position'],
            'knowledge': batch_knowledge,
            'heatmap_image_sen': batch_heatmap_sen,
            'transcript_sen': batch_transcript_sen,
            'transcript_sen_id': batch_transcript_sen_id,
            'heatmap_image_report': batch_heatmap_report,
            'transcript_report': batch_transcript_report,
            'transcript_report_id': batch_transcript_report_id,
            'disease_labels': disease_labels,
        }
        return batch


class End2EndDataset(Dataset):
    """
    current train, # gaze is 1711
    current val, # gaze is 10
    current test, # gaze is 434
    """
    def __init__(self, args, split):
        ann = json.loads(open(args['ann_path'], 'r').read())[split]

        self.examples = []
        # statistic gaze-heatmap
        gaze_ids = []
        for item in ann:
            if len(item['findings_factual_serialization']) == 0 or item['sentence_gaze_heatmap'] is None:
                continue
            gaze_ids.append(item['sentence_gaze_heatmap']['image_dir'])
        min_num_gaze = math.ceil(len(gaze_ids) * args['gaze_ratio'])
        valid_gaze_ids = random.sample(gaze_ids, min_num_gaze)
        print(f"all number of gaze is {len(gaze_ids)}, randomly sample {len(valid_gaze_ids)}")
        for item in tqdm(ann):
            # delete the sample that has no clinical findings
            if len(item['findings_factual_serialization']) == 0:
                continue
            report = item['findings'].strip()
            indication = item['indication_pure']
            history = item['history_pure']
            # convert disease labels into negative/positive status
            # 3.uncertain into 1.positive; 2.negative into 0.blank
            # 1. positive; 0: negative
            disease_labels = []
            for x in item['disease_labels']:
                if x == 3:
                    disease_labels.append(1)
                elif x == 2:
                    disease_labels.append(0)
                else:
                    disease_labels.append(x)
            # disease_labels = [1 if x == 3 else 0 if x == 2 else x for x in item['disease_labels']]
            gaze_heatmap = item['sentence_gaze_heatmap']
            if gaze_heatmap is not None and gaze_heatmap['image_dir'] in valid_gaze_ids:
                pass
            else:
                gaze_heatmap = None
            self.examples.append({
                'dicom_id': item["id"],
                'subject_id': item['subject_id'],
                'study_id': item["study_id"],
                'image_path': item['anchor_scan']['image_path'][0],
                'view_position': item['anchor_scan']['view_position'][0],
                'report': f'[CLS] [FINDINGS] {report}',
                'indication': f'[INDICATION] {indication}',
                'history': f'[HISTORY] {history}',
                'gaze_heatmap': gaze_heatmap,
                'disease_labels': disease_labels
            })
        # self.examples = self.examples[:10]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        return example


class End2EndCollateFn:
    def __init__(self, args, processor, pad_token, patch_size, bos_token, eos_token):
        self.args = args
        self.processor = processor
        self.patch_size = patch_size
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

    def __call__(self, data):
        # convert dict
        data = pd.DataFrame(data).to_dict(orient='list')
        batch_size = len(data['dicom_id'])
        # report: patient-level (a report corresponding to multiple images)
        # gaze heatmap: image-level (a gaze heatmap corresponding to the same dicom_id)

        # obtain valid_information
        batch_knowledge, batch_image, batch_patient_id = [], [], []  # batch_patient_id for image-text pair alignment
        batch_transcript_sen_id, batch_transcript_sen, batch_heatmap_sen = [], [], []  # sentence-level
        batch_transcript_report_id, batch_transcript_report, batch_heatmap_report = [], [], []  # report-level
        for i in range(batch_size):

            # obtain the clinical prompt (indication + ' ' + history)
            knowledge = f"[CLS] {data['indication'][i].strip()} [SEP] {data['history'][i].strip()}".strip()
            knowledge = knowledge if knowledge != '[CLS] [INDICATION] [SEP] [HISTORY]' else self.pad_token
            batch_knowledge.append(knowledge)

            # obtain and chest X-rays and its view position
            image_path = data['image_path'][i]
            image = Image.open(os.path.join(self.args['images_dir'], image_path)).convert('RGB')
            image = self.processor(image, return_tensors='pt').pixel_values[0]
            batch_image.append(image)

            # obtain batch_patient_id (ensure multiple images from the same patient to align their report)
            patient_id = f"{data['subject_id'][i]}_{data['study_id'][i]}"  # for instance-level alignment
            batch_patient_id.append(patient_id)
            cur_dicom_id = data['dicom_id'][i]                 # for gaze-soft-supervised

            # obtain eye-gaze data
            if data['gaze_heatmap'][i] is not None:
                # report-level gaze heatmap
                report_heatmap_path = os.path.join(self.args['eye_gaze_dir'], data['gaze_heatmap'][i]['image_dir'],
                                                   'report_heatmap.jpg')
                report_transcript = data['gaze_heatmap'][i]['transcript_full_text']
                current_heatmap = get_ground_truth_heatmap(report_heatmap_path, self.processor, self.patch_size)
                batch_heatmap_report.append(current_heatmap)
                batch_transcript_report.append(f"[CLS] [TRANSCRIPT] {report_transcript}")
                batch_transcript_report_id.append(cur_dicom_id)
                # sentence-level gaze heatmap
                gaze_dir = os.path.join(self.args['eye_gaze_dir'], data['gaze_heatmap'][i]['image_dir'])
                gaze_lookup = json.load(open(os.path.join(gaze_dir, 'sentence-heatmap-lookup.json')))

                # notably, only the sentence-level heatmap is considered
                for key, transcript in gaze_lookup.items():
                    if key == 'full_text':
                        # not consider report-level heatmap. Since report length is too big, waste computational head
                        continue
                    heatmap_path = os.path.join(gaze_dir, f"{key}_heatmap.jpg")
                    current_heatmap = get_ground_truth_heatmap(heatmap_path, self.processor, self.patch_size)
                    batch_heatmap_sen.append(current_heatmap)
                    batch_transcript_sen.append(f"[CLS] [TRANSCRIPT] {transcript}")
                    batch_transcript_sen_id.append(cur_dicom_id)

        batch_image = torch.stack(batch_image, dim=0)
        batch_heatmap_sen = torch.stack(batch_heatmap_sen, dim=0) if len(batch_heatmap_sen) != 0 else None
        batch_heatmap_report = torch.stack(batch_heatmap_report, dim=0) if len(batch_heatmap_report) != 0 else None
        disease_labels = torch.from_numpy(np.array(data['disease_labels'])).long()
        batch = {
            'dicom_id': data['dicom_id'],
            'patient_id': np.array(batch_patient_id),
            'image': batch_image,
            'view_position': data['view_position'],
            'align_report': data['report'],
            'report': [f'{self.bos_token} {item.split("[CLS] [FINDINGS] ")[-1]} {self.eos_token}' for item in
                       data['report']],
            'knowledge': batch_knowledge,
            'heatmap_image_sen': batch_heatmap_sen,
            'transcript_sen': batch_transcript_sen,
            'transcript_sen_id': batch_transcript_sen_id,
            'heatmap_image_report': batch_heatmap_report,
            'transcript_report': batch_transcript_report,
            'transcript_report_id': batch_transcript_report_id,
            'disease_labels': disease_labels,
        }
        return batch


class DiseasePredicationDataset(Dataset):
    def __init__(self, args, split):
        ann = json.loads(open(args['ann_path'], 'r').read())[split]

        self.examples = []
        for item in tqdm(ann):
            # delete the sample that has no clinical findings
            if len(item['findings_factual_serialization']) == 0:
                continue
            report = item['findings'].strip()
            indication = item['indication_pure']
            history = item['history_pure']
            # convert disease labels into negative/positive status
            # 3.uncertain into 1.positive; 2.negative into 0.blank
            # 1. positive; 0: negative
            disease_labels = []
            for x in item['disease_labels']:
                if x == 3:
                    disease_labels.append(1)
                elif x == 2:
                    disease_labels.append(0)
                else:
                    disease_labels.append(x)
            # disease_labels = [1 if x == 3 else 0 if x == 2 else x for x in item['disease_labels']]
            self.examples.append({
                'dicom_id': item["id"],
                'subject_id': item['subject_id'],
                'study_id': item["study_id"],
                'image_path': item['anchor_scan']['image_path'][0],
                'view_position': item['anchor_scan']['view_position'][0],
                'report': f'[CLS] [FINDINGS] {report}',
                'indication': f'[INDICATION] {indication}',
                'history': f'[HISTORY] {history}',
                'disease_labels': disease_labels
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        return example


class DiseasePredicationCollateFn:
    def __init__(self, args, processor, pad_token, patch_size, bos_token, eos_token):
        self.args = args
        self.processor = processor
        self.patch_size = patch_size
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

    def __call__(self, data):
        # convert dict
        data = pd.DataFrame(data).to_dict(orient='list')
        batch_size = len(data['dicom_id'])
        # report: patient-level (a report corresponding to multiple images)
        # gaze heatmap: image-level (a gaze heatmap corresponding to the same dicom_id)

        # obtain valid_information
        batch_knowledge, batch_image, batch_patient_id = [], [], []  # batch_patient_id for image-text pair alignment
        for i in range(batch_size):

            # obtain the clinical prompt (indication + ' ' + history)
            knowledge = f"[CLS] {data['indication'][i].strip()} [SEP] {data['history'][i].strip()}".strip()
            knowledge = knowledge if knowledge != '[CLS] [INDICATION] [SEP] [HISTORY]' else self.pad_token
            batch_knowledge.append(knowledge)

            # obtain and chest X-rays and its view position
            image_path = data['image_path'][i]
            image = Image.open(os.path.join(self.args['images_dir'], image_path)).convert('RGB')
            image = self.processor(image, return_tensors='pt').pixel_values[0]
            batch_image.append(image)

            # obtain batch_patient_id (ensure multiple images from the same patient to align their report)
            patient_id = f"{data['subject_id'][i]}_{data['study_id'][i]}"  # for instance-level alignment
            batch_patient_id.append(patient_id)

        batch_image = torch.stack(batch_image, dim=0)
        disease_labels = torch.from_numpy(np.array(data['disease_labels'])).long()
        batch = {
            'dicom_id': data['dicom_id'],
            'patient_id': np.array(batch_patient_id),
            'image': batch_image,
            'view_position': data['view_position'],
            'align_report': data['report'],
            'report': [f'{self.bos_token} {item.split("[CLS] [FINDINGS] ")[-1]} {self.eos_token}' for item in
                       data['report']],
            'knowledge': batch_knowledge,
            'disease_labels': disease_labels,
        }
        return batch


class GenerationDataset(Dataset):
    def __init__(self, args, split):
        ann = json.loads(open(args['ann_path'], 'r').read())[split]

        self.examples = []
        for item in tqdm(ann):
            # delete the sample that has no clinical findings
            if len(item['findings_factual_serialization']) == 0:
                continue
            report = item['findings'].strip()
            indication = item['indication_pure']
            history = item['history_pure']
            self.examples.append({
                'dicom_id': item["id"],
                'subject_id': item['subject_id'],
                'study_id': item["study_id"],
                'image_path': item['anchor_scan']['image_path'][0],
                'view_position': item['anchor_scan']['view_position'][0],
                'report': f'{report}',
                'indication': f'[INDICATION] {indication}',
                'history': f'[HISTORY] {history}',
                'disease_labels': item['disease_labels'],
                'disease_prediction': item['disease_prediction'],
                'similar_case': item['similar_case']['findings'].split('[CLS] [FINDINGS] ')[-1],
            })
        # self.examples = self.examples[:2]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        return example


class GenerationCollateFn:
    def __init__(self, args, processor, pad_token, bos_token, eos_token):
        self.args = args
        self.processor = processor
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

    def __call__(self, data):
        # convert dict
        data = pd.DataFrame(data).to_dict(orient='list')
        batch_size = len(data['dicom_id'])

        # obtain valid_information
        batch_knowledge, batch_image, batch_patient_id = [], [], []  # batch_patient_id for image-text pair alignment
        for i in range(batch_size):
            # obtain the clinical prompt (indication + ' ' + history)
            knowledge = f"[CLS] {data['indication'][i].strip()} [SEP] {data['history'][i].strip()}".strip()
            knowledge = knowledge if knowledge != '[CLS] [INDICATION] [SEP] [HISTORY]' else self.pad_token
            batch_knowledge.append(knowledge)

            # obtain and chest X-rays and its view position
            image_path = data['image_path'][i]
            image = Image.open(os.path.join(self.args['images_dir'], image_path)).convert('RGB')
            image = self.processor(image, return_tensors='pt').pixel_values[0]
            batch_image.append(image)

            # obtain batch_patient_id (ensure multiple images from the same patient to align their report)
            patient_id = f"{data['subject_id'][i]}_{data['study_id'][i]}"
            batch_patient_id.append(patient_id)

        batch_image = torch.stack(batch_image, dim=0)
        disease_labels = torch.from_numpy(np.array(data['disease_labels'])).long()
        batch = {
            'dicom_id': data['dicom_id'],
            'patient_id': np.array(batch_patient_id),
            'image': batch_image,
            'report': [f'{self.bos_token} {item} {self.eos_token}' for item in data['report']],
            'view_position': data['view_position'],
            'knowledge': batch_knowledge,
            'disease_labels': disease_labels,
            'disease_prediction': data['disease_prediction'],
            'similar_case': data['similar_case'],
        }
        return batch


class GenerationLLMCollateFn:
    """
    llm_tokenizer add begin-of-string (default)
    """
    def __init__(self, args, processor, pad_token):
        self.args = args
        self.processor = processor
        self.pad_token = pad_token

    def __call__(self, data):
        # convert dict
        data = pd.DataFrame(data).to_dict(orient='list')
        batch_size = len(data['dicom_id'])

        # obtain valid_information
        batch_knowledge, batch_image, batch_patient_id = [], [], []  # batch_patient_id for image-text pair alignment
        for i in range(batch_size):
            # obtain the clinical prompt (indication + ' ' + history)
            knowledge = f"[CLS] {data['indication'][i].strip()} [SEP] {data['history'][i].strip()}".strip()
            knowledge = knowledge if knowledge != '[CLS] [INDICATION] [SEP] [HISTORY]' else self.pad_token
            batch_knowledge.append(knowledge)

            # obtain and chest X-rays and its view position
            image_path = data['image_path'][i]
            image = Image.open(os.path.join(self.args['images_dir'], image_path)).convert('RGB')
            image = self.processor(image, return_tensors='pt').pixel_values[0]
            batch_image.append(image)

            # obtain batch_patient_id (ensure multiple images from the same patient to align their report)
            patient_id = f"{data['subject_id'][i]}_{data['study_id'][i]}"
            batch_patient_id.append(patient_id)

        batch_image = torch.stack(batch_image, dim=0)
        disease_labels = torch.from_numpy(np.array(data['disease_labels'])).long()
        batch = {
            'dicom_id': data['dicom_id'],
            'patient_id': np.array(batch_patient_id),
            'image': batch_image,
            'report': data['report'],
            'view_position': data['view_position'],
            'knowledge': batch_knowledge,
            'disease_labels': disease_labels,
            'disease_prediction': data['disease_prediction'],
            'similar_case': data['similar_case'],
        }
        return batch


class SRRGGenerationDataset(Dataset):
    def __init__(self, args, split):
        ann = json.loads(open(args['ann_path'], 'r').read())[split]

        self.examples = []
        for item in tqdm(ann):
            image_paths, view_positions = item['image_paths'], item['view_positions']
            valid_idx = choose_images(view_positions)
            history = item['history_section']
            if history is None:
                history = ''
            else:
                history = history.strip()
            self.examples.append({
                'dicom_id': item["id"],
                'image_path': image_paths[valid_idx],
                'view_position': view_positions[valid_idx],
                'report': item['findings_section'].strip(),
                'history': history,
            })
        # self.examples = self.examples[:2]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        return example


class SRRGGenerationLLMCollateFn:
    """
    llm_tokenizer add begin-of-string (default)
    """
    def __init__(self, args, processor, pad_token):
        self.args = args
        self.processor = processor
        self.pad_token = pad_token

    def __call__(self, data):
        # convert dict
        data = pd.DataFrame(data).to_dict(orient='list')
        batch_size = len(data['dicom_id'])

        # obtain valid_information
        batch_knowledge, batch_image = [], []
        for i in range(batch_size):
            # obtain the clinical prompt (indication + ' ' + history)
            knowledge = f"[CLS] {data['history'][i]}".strip()
            knowledge = knowledge if knowledge != '[CLS]' else self.pad_token
            batch_knowledge.append(knowledge)

            # obtain and chest X-rays and its view position
            if 'mimic' in data['dicom_id'][i]:
                image_path = os.path.join(self.args['mimic_dir'], data['image_path'][i])
            else:
                image_path = os.path.join(self.args['chexpert_dir'], data['image_path'][i])
            image = Image.open(image_path).convert('RGB')
            image = self.processor(image, return_tensors='pt').pixel_values[0]
            batch_image.append(image)

        batch_image = torch.stack(batch_image, dim=0)
        batch = {
            'dicom_id': data['dicom_id'],
            'image': batch_image,
            'report': data['report'],
            'view_position': data['view_position'],
            'knowledge': batch_knowledge,
        }
        return batch


class SRRGGenerationGPT2CollateFn:
    """
    llm_tokenizer add begin-of-string (default)
    """
    def __init__(self, args, processor, pad_token, bos_token, eos_token):
        self.args = args
        self.processor = processor
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

    def __call__(self, data):
        # convert dict
        data = pd.DataFrame(data).to_dict(orient='list')
        batch_size = len(data['dicom_id'])

        # obtain valid_information
        batch_knowledge, batch_image = [], []
        for i in range(batch_size):
            # obtain the clinical prompt (indication + ' ' + history)
            knowledge = f"[CLS] {data['history'][i]}".strip()
            knowledge = knowledge if knowledge != '[CLS]' else self.pad_token
            batch_knowledge.append(knowledge)

            # obtain and chest X-rays and its view position
            if 'mimic' in data['dicom_id'][i]:
                image_path = os.path.join(self.args['mimic_dir'], data['image_path'][i])
            else:
                image_path = os.path.join(self.args['chexpert_dir'], data['image_path'][i])
            image = Image.open(image_path).convert('RGB')
            image = self.processor(image, return_tensors='pt').pixel_values[0]
            batch_image.append(image)

        batch_image = torch.stack(batch_image, dim=0)
        batch = {
            'dicom_id': data['dicom_id'],
            'image': batch_image,
            'report': [f'{self.bos_token} {item} {self.eos_token}' for item in data['report']],
            'view_position': data['view_position'],
            'knowledge': batch_knowledge,
        }
        return batch


class OnlyGazeDataset(Dataset):
    def __init__(self, args, split):
        ann = json.loads(open(args['ann_path'], 'r').read())[split]
        self.examples = []
        for item in ann:
            if item['sentence_gaze_heatmap'] is None:
                continue
            report = item['findings'].strip()
            indication = item['indication_pure'].strip()
            history = item['history_pure'].strip()
            # convert disease labels into negative/positive status
            # 3.uncertain into 1.positive; 2.negative into 0.blank
            # 1. positive; 0: negative
            disease_labels = []
            for x in item['disease_labels']:
                if x == 3:
                    disease_labels.append(1)
                elif x == 2:
                    disease_labels.append(0)
                else:
                    disease_labels.append(x)
            self.examples.append({
                'dicom_id': item["id"],
                'subject_id': item['subject_id'],
                'study_id': item["study_id"],
                'image_path': item['anchor_scan']['image_path'][0],
                'view_position': item['anchor_scan']['view_position'][0],
                'report': f'[CLS] [FINDINGS] {report}',
                'indication': f'[INDICATION] {indication}',
                'history': f'[HISTORY] {history}',
                'gaze_heatmap': item['sentence_gaze_heatmap'],
                'disease_labels': disease_labels
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        return example


class OnlyGazeCollateFn:
    def __init__(self, args, processor, pad_token, patch_size):
        self.args = args
        self.processor = processor
        self.pad_token = pad_token
        self.patch_size = patch_size

    def __call__(self, data):
        # convert dict
        data = pd.DataFrame(data).to_dict(orient='list')
        batch_size = len(data['dicom_id'])

        # obtain valid_information
        batch_knowledge, batch_image, batch_patient_id = [], [], []  # batch_patient_id for image-text pair alignment
        batch_transcript_sen_id, batch_transcript_sen, batch_heatmap_sen = [], [], []  # sentence-level
        batch_transcript_report_id, batch_transcript_report, batch_heatmap_report = [], [], []  # report-level
        for i in range(batch_size):

            # obtain the clinical prompt (indication + ' ' + history)
            knowledge = f"[CLS] {data['indication'][i].strip()} [SEP] {data['history'][i].strip()}".strip()
            knowledge = knowledge if knowledge != '[CLS] [INDICATION] [SEP] [HISTORY]' else self.pad_token
            batch_knowledge.append(knowledge)

            # obtain and chest X-rays and its view position
            image_path = data['image_path'][i]
            image = Image.open(os.path.join(self.args['images_dir'], image_path)).convert('RGB')
            image = self.processor(image, return_tensors='pt').pixel_values[0]
            batch_image.append(image)

            # obtain batch_patient_id (ensure multiple images from the same patient to align their report)
            patient_id = f"{data['subject_id'][i]}_{data['study_id'][i]}"
            batch_patient_id.append(patient_id)
            cur_dicom_id = data['dicom_id'][i]

            # obtain eye-gaze data

            # report-level gaze heatmap
            report_heatmap_path = os.path.join(self.args['eye_gaze_dir'], data['gaze_heatmap'][i]['image_dir'],
                                               'report_heatmap.jpg')
            report_transcript = data['gaze_heatmap'][i]['transcript_full_text']
            current_heatmap = get_ground_truth_heatmap(report_heatmap_path, self.processor, self.patch_size)
            batch_heatmap_report.append(current_heatmap)
            batch_transcript_report.append(f"[CLS] [TRANSCRIPT] {report_transcript}")
            batch_transcript_report_id.append(cur_dicom_id)
            # sentence-level gaze heatmap
            # obtain eye-gaze data
            gaze_dir = os.path.join(self.args['eye_gaze_dir'], data['gaze_heatmap'][i]['image_dir'])
            gaze_lookup = json.load(open(os.path.join(gaze_dir, 'sentence-heatmap-lookup.json')))

            for key, transcript in gaze_lookup.items():
                if key == 'full_text':
                    # not consider report-level heatmap. Since report length is too big, waste computational head
                    continue
                heatmap_path = os.path.join(gaze_dir, f"{key}_heatmap.jpg")
                current_heatmap = get_ground_truth_heatmap(heatmap_path, self.processor, self.patch_size)
                batch_heatmap_sen.append(current_heatmap)
                batch_transcript_sen.append(f"[CLS] [TRANSCRIPT] {transcript}")
                batch_transcript_sen_id.append(cur_dicom_id)

        batch_image = torch.stack(batch_image, dim=0)
        batch_heatmap_sen = torch.stack(batch_heatmap_sen, dim=0) if len(batch_heatmap_sen) != 0 else None
        batch_heatmap_report = torch.stack(batch_heatmap_report, dim=0) if len(batch_heatmap_report) != 0 else None
        disease_labels = torch.from_numpy(np.array(data['disease_labels'])).long()
        batch = {
            'dicom_id': data['dicom_id'],
            'patient_id': np.array(batch_patient_id),
            'image': batch_image,
            'report': data['report'],
            'view_position': data['view_position'],
            'knowledge': batch_knowledge,
            'heatmap_image_sen': batch_heatmap_sen,
            'transcript_sen': batch_transcript_sen,
            'transcript_sen_id': batch_transcript_sen_id,
            'heatmap_image_report': batch_heatmap_report,
            'transcript_report': batch_transcript_report,
            'transcript_report_id': batch_transcript_report_id,
            'disease_labels': disease_labels,
        }
        return batch

def temp():
    from transformers import AutoImageProcessor
    from torch.utils.data import DataLoader
    from tools.utils import setup_arguments

    args, logger = setup_arguments()
    image_processor = AutoImageProcessor.from_pretrained(args['rad_dino_path'], trust_remote_code=True)
    train_set = OnlyGazeDataset(args, 'train')
    collate_fn = OnlyGazeCollateFn(args, image_processor, '[PAD]', 14)
    loader = DataLoader(
        train_set,
        batch_size=4,
        num_workers=0,
        shuffle=True,
        prefetch_factor=None,
        collate_fn=collate_fn,
        drop_last=True,
    )
    for batch in loader:
        gaze_image, gaze_idx, transcript = [], [], []
        for i in range(batch['image'].shape[0]):
            if len(batch['heatmap_image'][i]) != 0:
                gaze_image.append(batch['heatmap_image'][i])
                gaze_idx.append(i)
                transcript.append(batch['transcript'][i])
        if len(gaze_image) != 0:
            print()

# temp()
