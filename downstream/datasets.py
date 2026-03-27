import os
import re
import cv2
import ast
import numpy as np
import pandas as pd
from PIL import Image
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor
from transformers.image_transforms import convert_to_rgb
from transformers.image_utils import infer_channel_dimension_format

from downstream.siim_mask_functions import rle2mask
from downstream.constants import *

import matplotlib.pyplot as plt


def visualize_image_and_mask(image, mask, processed_image, processed_mask):

    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('/my-figures/mask.png', dpi=600)
    fig, axes = plt.subplots(2, 3, figsize=(12, 12))

    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(mask, cmap='gray')
    axes[0, 1].set_title("Original Mask")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(image, cmap='gray')
    axes[0, 2].imshow(mask, cmap='coolwarm', alpha=0.5)
    axes[0, 2].axis('off')

    axes[1, 0].imshow(processed_image)
    axes[1, 0].set_title("Processed Image")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(processed_mask, cmap='gray')
    axes[1, 1].set_title("Processed Mask")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(processed_image, cmap='gray')
    axes[1, 2].imshow(processed_mask, cmap='coolwarm', alpha=0.5)
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()



def read_from_dicom(img_path):
    dcm = pydicom.dcmread(img_path)
    x = dcm.pixel_array.astype(np.float32)
    norm_x = x - x.min()
    norm_x = norm_x / norm_x.max()
    # scale to (0~255)
    norm_x = (norm_x * 255.0).astype(np.uint8)
    pil_image = Image.fromarray(norm_x).convert("RGB")
    view_position = dcm.ViewPosition if 'ViewPosition' in dcm else 'PA'
    sex = dcm.PatientSex if 'PatientSex' in dcm else ""
    age = dcm.PatientAge if 'PatientAge' in dcm else ""
    if sex == 'M':
        sex = 'Man' if age == '' else 'man'
    elif sex == 'F':
        sex = 'Woman' if age == '' else 'woman'

    if age != '':
        indication = f'A {age}-year-old {sex}'
    else:
        indication = f'{sex}'
    return pil_image, view_position, indication


def read_from_text(info_path):
    lines = []
    with open(info_path) as f:
        for line in f.readlines():
            current_line = line.strip()
            if len(current_line) != 0:
                lines.append(current_line)
    parts = re.split(r'(\d+)', lines[0])
    sex, age = parts[0].strip(), parts[1].strip()
    if lines[-1] == 'normal':
        target = 0
    else:
        target = 1
    if sex == 'male':
        indication = f'A {age}-year-old man'
    else:
        indication = f'A {age}-year-old woman'
    view_position = 'PA'

    # if len(lines) != 2:
    #     print(info_path, lines, indication, target)

    return target, indication, view_position


def get_resize_mask(image, processor):
    # resize
    image = np.array(image)
    image = np.repeat(image[:, :, np.newaxis], 3, axis=-1)
    image = convert_to_rgb(image)
    input_data_format = infer_channel_dimension_format(image)
    crop_size = processor.crop_size
    size = {'shortest_edge': crop_size['height']}
    image = processor.resize(image=image, size=size, resample=processor.resample, input_data_format=input_data_format)
    image = processor.center_crop(image=image, size=crop_size, input_data_format=input_data_format)

    return image[:, :, 0]


class RSNADataset(Dataset):
    def __init__(self, args, split="train"):
        super().__init__()
        if split == 'train':
            if args['data_ratio'] == 1.0:
                self.example = pd.read_csv(rsna_train.split('.txt')[0] + '.csv')
            elif args['data_ratio'] == 0.1:
                self.example = pd.read_csv(rsna_train_10.split('.txt')[0] + '.csv')
            elif args['data_ratio'] == 0.01:
                self.example = pd.read_csv(rsna_train_1.split('.txt')[0] + '.csv')
            else:
                raise NotImplementedError
        elif split == 'val':
            self.example = pd.read_csv(rsna_val.split('.txt')[0] + '.csv')
        else:
            self.example = pd.read_csv(rsna_test.split('.txt')[0] + '.csv')
        # print()
        # self.example = self.example[:100]

    def __getitem__(self, index):
        row = self.example.iloc[index]
        # get image
        img_path = os.path.join(rsna_images, row['patientId']+'.dcm')
        return {
            'id': row['patientId'],
            'image_path': img_path,
            'target': row['Target'],
            'bbox': row['bbox'] if row['bbox'] is not None else [[0, 0, 0, 0]]
        }

    def __len__(self):
        return len(self.example)


class RSNACollateFn:
    def __init__(self, processor, pad_token):
        self.processor = processor
        self.pad_token = pad_token

    def __call__(self, data):
        # convert dict
        data = pd.DataFrame(data).to_dict(orient='list')
        batch_size = len(data['id'])

        # obtain valid_information
        batch_knowledge, batch_image, batch_view_position = [], [], []
        batch_mask, batch_targets = [], []
        for i in range(batch_size):
            image_path = data['image_path'][i]
            image, view_position, indication = read_from_dicom(image_path)

            # obtain binary classification labels and corresponding bounding boxes
            target = data['target'][i]
            batch_targets.append(target)
            if target == 0:
                crop_size = self.processor.crop_size
                width, height = crop_size['width'], crop_size['height']
                mask = np.zeros((width, height))
            else:
                # In certain samples, it has multiple bounding boxes.
                mask = np.zeros(image.size)
                for bbox in eval(data['bbox'][i]):
                    x_min, y_min, w, h = bbox
                    x_min, y_min, w, h = int(x_min), int(y_min), int(w), int(h)
                    x_max, y_max = x_min + w, y_min + h
                    mask[y_min: y_max, x_min: x_max] = 1
                mask = get_resize_mask(mask, self.processor)
                # croped_mask = get_resize_mask(mask, self.processor)
                # croped_image = self.processor(image, return_tensors='pt').pixel_values[0].permute(1, 2, 0).numpy()
                # visualize_image_and_mask(np.array(image), mask, mask, mask)
            batch_mask.append(torch.from_numpy(mask))

            # obtain the clinical prompt (indication + ' ' + history)
            if len(indication) == 0:
                knowledge = self.pad_token
            else:
                knowledge = f"[CLS] [INDICATION] {indication.strip()} [SEP] [HISTORY]".strip()
            batch_knowledge.append(knowledge)

            # obtain and chest X-rays and its view position
            image = self.processor(image, return_tensors='pt').pixel_values[0]
            batch_image.append(image)
            batch_view_position.append(view_position)

        batch_image = torch.stack(batch_image, dim=0)
        batch_mask = torch.stack(batch_mask, dim=0)
        disease_labels = torch.from_numpy(np.array(batch_targets)).long()
        batch = {
            'id': data['id'],
            'image': batch_image,
            'view_position': batch_view_position,
            'mask': batch_mask,
            'knowledge': batch_knowledge,
            'disease_labels': disease_labels,
            'image_path': data['image_path']
        }
        return batch


class ShenZhenCXRDataset(Dataset):
    def __init__(self, args, split="train"):
        super().__init__()

        if split == 'train':
            self.example = pd.read_csv(shenzhen_train, header=None)
        elif split == 'val':
            self.example = pd.read_csv(shenzhen_val, header=None)
        else:
            self.example = pd.read_csv(shenzhen_test, header=None)
        self.example.columns = ['image_id', 'target']

    def __getitem__(self, index):
        row = self.example.iloc[index]
        # get image
        img_path = os.path.join(shenzhen_images, row['image_id'])
        img_info = os.path.join(shenzhen_info, row['image_id'].split('.png')[0] + '.txt')
        target, indication, view_position = read_from_text(img_info)
        # if target != row['target'].item():
        #     print(target, row['target'], row['image_id'])
        return {
            'id': row['image_id'],
            'image_path': img_path,
            'target': row['target'].item(),
            'indication': indication,
            'view_position': view_position,
        }

    def __len__(self):
        return len(self.example)


class ShenZhenCXRCollateFn:
    def __init__(self, processor, pad_token):
        self.processor = processor
        self.pad_token = pad_token

    def __call__(self, data):
        # convert dict
        data = pd.DataFrame(data).to_dict(orient='list')
        batch_size = len(data['id'])

        # obtain valid_information
        batch_knowledge, batch_image = [], []
        for i in range(batch_size):
            image_path = data['image_path'][i]
            image = Image.open(image_path).convert('RGB')
            image = self.processor(image, return_tensors='pt').pixel_values[0]
            batch_image.append(image)

            indication = data['indication'][i]
            # obtain the clinical prompt (indication + ' ' + history)
            if len(indication) == 0:
                knowledge = self.pad_token
            else:
                knowledge = f"[CLS] [INDICATION] {indication.strip()} [SEP] [HISTORY]".strip()
            batch_knowledge.append(knowledge)

        batch_image = torch.stack(batch_image, dim=0)
        disease_labels = torch.from_numpy(np.array(data['target'])).long()
        batch = {
            'id': data['id'],
            'image': batch_image,
            'view_position': data['view_position'],
            'knowledge': batch_knowledge,
            'disease_labels': disease_labels,
            'image_path': data['image_path']
        }
        return batch


class SIIMDataset(Dataset):
    """
    https://academictorrents.com/details/6ef7c6d039e85152c4d0f31d83fa70edc4aba088
    https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation

    "The data is comprised of images in DICOM format and annotations in the
    form of image IDs and run-length-encoded (RLE) masks. Some of the images
    contain instances of pneumothorax (collapsed lung), which are indicated
    by encoded binary masks in the annotations. Some training images have
    multiple annotations. Images without pneumothorax have a mask value of -1."
    """
    def __init__(self, args, split="train"):
        super().__init__()

        if split == 'train':
            if args['data_ratio'] == 1.0:
                self.example = pd.read_csv(siim_train, header=None)
            elif args['data_ratio'] == 0.1:
                self.example = pd.read_csv(siim_train_10, header=None)
            elif args['data_ratio'] == 0.01:
                self.example = pd.read_csv(siim_train_1, header=None)
            else:
                raise NotImplementedError
        elif split == 'val':
            self.example = pd.read_csv(siim_val, header=None)
        else:
            self.example = pd.read_csv(siim_test, header=None)

        # obtain all information of these examples
        self.data_info = pd.read_csv(siim_labels)
        self.data_info.columns = ['id', 'image_path', 'target']

    def __getitem__(self, index):
        item_list = self.example.iloc[index][0].split('_')
        item_id = eval(item_list[0])
        row = self.data_info[self.data_info['id'] == item_id]

        # get image
        file_name = row['image_path'].item()
        file_name_split = file_name.split('.')
        file_name_prefix = '.'.join(file_name_split[:7])
        first_change = eval(file_name_split[7])
        file_name_suffix = '.'.join(file_name_split[8:-1])
        second_change = eval(file_name.split('.')[-1])
        image_path = os.path.join(siim_images,
                                  f"{file_name_prefix}.{first_change - 2}.{file_name_suffix}.{second_change - 1}",
                                  f"{file_name_prefix}.{first_change - 1}.{file_name_suffix}.{second_change - 2}",
                                  f"{file_name}.dcm")
        bbox = row['target'].item()
        if bbox != '-1':
            target = 1
        else:
            target = 0
        return {
            'id': item_id,
            'image_path': image_path,
            'target': target,
            'bbox': bbox
        }

    def __len__(self):
        return len(self.example)


class SIIMCollateFn:
    def __init__(self, processor, pad_token):
        self.processor = processor
        self.pad_token = pad_token

    def __call__(self, data):
        # convert dict
        data = pd.DataFrame(data).to_dict(orient='list')
        batch_size = len(data['id'])

        # obtain valid_information
        batch_knowledge, batch_image, batch_view_position = [], [], []
        batch_mask, batch_targets = [], []
        for i in range(batch_size):
            image_path = data['image_path'][i]
            image, view_position, indication = read_from_dicom(image_path)

            # obtain binary classification labels and corresponding bounding boxes
            target = data['target'][i]
            batch_targets.append(target)

            if target == 0:
                crop_size = self.processor.crop_size
                width, height = crop_size['width'], crop_size['height']
                mask = np.zeros((width, height))
            else:
                # In certain samples, it has multiple bounding boxes.
                width, height = image.size
                mask = rle2mask(data['bbox'][i], width, height)
                mask = torch.from_numpy(mask)
                mask = get_resize_mask(mask, self.processor)
                # visualize_image_and_mask(mask, mask, mask, mask)
            batch_mask.append(torch.from_numpy(mask))

            # obtain the clinical prompt (indication + ' ' + history)
            if len(indication) == 0:
                knowledge = self.pad_token
            else:
                knowledge = f"[CLS] [INDICATION] {indication.strip()} [SEP] [HISTORY]".strip()
            batch_knowledge.append(knowledge)

            # obtain and chest X-rays and its view position
            image = self.processor(image, return_tensors='pt').pixel_values[0]
            batch_image.append(image)
            batch_view_position.append(view_position)

        batch_image = torch.stack(batch_image, dim=0)
        batch_mask = torch.stack(batch_mask, dim=0).float()
        disease_labels = torch.from_numpy(np.array(batch_targets)).long()
        batch = {
            'id': data['id'],
            'image': batch_image,
            'view_position': batch_view_position,
            'mask': batch_mask,
            'knowledge': batch_knowledge,
            'disease_labels': disease_labels,
            'image_path': data['image_path']
        }
        return batch


class TBX11KDataset(Dataset):
    """
    tuberculosis diagnosis
    https://www.kaggle.com/datasets/vbookshelf/tbx11k-simplified
    """
    def __init__(self, args, split="train"):
        super().__init__()

        if split == 'train':
            if args['data_ratio'] == 1.0:
                self.example = pd.read_csv(tbx11k_train, header=None)
            elif args['data_ratio'] == 0.1:
                self.example = pd.read_csv(tbx11k_train_10, header=None)
            elif args['data_ratio'] == 0.01:
                self.example = pd.read_csv(tbx11k_train_1, header=None)
            else:
                raise NotImplementedError
        elif split == 'val':
            self.example = pd.read_csv(tbx11k_val, header=None)
        else:
            self.example = pd.read_csv(tbx11k_test, header=None)

        # obtain all information of these examples
        self.example = self.example.values.reshape(-1).tolist()

    def __getitem__(self, index):
        file_name = self.example[index] + '.png'
        return file_name

    def __len__(self):
        return len(self.example)


class TBX11KCollateFn:
    def __init__(self, processor, pad_token):
        self.processor = processor
        self.pad_token = pad_token
        self.raw_data_info = pd.read_csv(tbx11k_labels)
        self.data_info = self.raw_data_info.groupby("fname").first()
        self.map = {
            'tb': 1,
            'no_tb': 0
        }

    def __call__(self, data):
        images, targets, masks = [], [], []
        for file_name in data:
            # obtain mask
            rows = self.raw_data_info[self.raw_data_info["fname"] == file_name]
            mask = np.zeros([512, 512]).astype(np.uint8)
            for index, row in rows.iterrows():
                if row.target == "tb":
                    bbox = ast.literal_eval(row.bbox)
                    xywh = np.asarray([bbox["xmin"], bbox["ymin"], bbox["width"], bbox["height"]])
                    xywh = xywh.astype(int)
                    mask[xywh[1]: xywh[1] + xywh[3], xywh[0]: xywh[0] + xywh[2]] = 1

            mask = get_resize_mask(mask, self.processor)
            masks.append(torch.from_numpy(mask))

            # obtain targets
            target = self.data_info.loc[file_name, 'target']
            targets.append(self.map[target])

            # obtain image
            image_path = os.path.join(tbx11k_images, file_name)
            image = Image.open(image_path).convert('RGB')
            image = self.processor(image, return_tensors='pt').pixel_values[0]
            images.append(image)

        knowledge = [self.pad_token] * len(data)
        view_positions = ['PA'] * len(data)

        batch_image = torch.stack(images, dim=0)
        batch_mask = torch.stack(masks, dim=0).float()
        disease_labels = torch.from_numpy(np.array(targets)).long()
        batch = {
            'id': data,
            'image': batch_image,
            'view_position': view_positions,
            'mask': batch_mask,
            'knowledge': knowledge,
            'disease_labels': disease_labels,
        }
        return batch



class NIHDataset(Dataset):
    def __init__(self, args, split="train"):
        super().__init__()
        if split == 'train':
            if args['data_ratio'] == 1.0:
                self.example = pd.read_csv(nih_train.split('.txt')[0] + '.csv')
            elif args['data_ratio'] == 0.1:
                self.example = pd.read_csv(nih_train_10.split('.txt')[0] + '.csv')
            elif args['data_ratio'] == 0.01:
                self.example = pd.read_csv(nih_train_1.split('.txt')[0] + '.csv')
            else:
                raise NotImplementedError
        elif split == 'val':
            self.example = pd.read_csv(nih_val.split('.txt')[0] + '.csv')
        else:
            self.example = pd.read_csv(nih_test.split('.txt')[0] + '.csv')
        del self.example['No Finding']
        self.diseases_labels = list(self.example.columns)[-14:]

    def __getitem__(self, index):
        row = self.example.iloc[index]
        # get image
        img_path = os.path.join(nih_images, row['Image Index'])

        if row['Patient Gender'] == 'M':
            sex = 'man'
        elif row['Patient Gender'] == 'F':
            sex = 'woman'
        else:
            sex = ''
        context = f"Follow-up visit {row['Follow-up #']} of a {row['Patient Age']}-year-old {sex}."

        return {
            'id': row['Image Index'].split('.png')[0],
            'image_path': img_path,
            'target': row[self.diseases_labels].values.reshape(-1).tolist(),
            'view_position': row['View Position'],
            'context': context
        }

    def __len__(self):
        return len(self.example)


class NIHCollateFn:
    def __init__(self, processor, pad_token):
        self.processor = processor
        self.pad_token = pad_token

    def __call__(self, data):
        # convert dict
        data = pd.DataFrame(data).to_dict(orient='list')
        batch_size = len(data['id'])

        # obtain valid_information
        batch_image, batch_knowledge = [], []
        for i in range(batch_size):
            image_path = data['image_path'][i]

            image = Image.open(image_path).convert('RGB')
            image = self.processor(image, return_tensors='pt').pixel_values[0]
            batch_image.append(image)
            # obtain binary classification labels and corresponding bounding boxes
            context = data['context'][i]
            if len(context) == 0:
                knowledge = self.pad_token
            else:
                knowledge = f"[CLS] [INDICATION] {context} [SEP] [HISTORY]".strip()
            batch_knowledge.append(knowledge)

        batch_image = torch.stack(batch_image, dim=0)
        disease_labels = torch.from_numpy(np.array(data['target'])).long()
        batch = {
            'id': data['id'],
            'image': batch_image,
            'view_position': data['view_position'],
            'knowledge': batch_knowledge,
            'disease_labels': disease_labels,
        }
        return batch


def sta_tbx11k_labels_distributions(data_ratio):
    if data_ratio == 1.0:
        df = pd.read_csv(tbx11k_train, header=None)
    elif data_ratio == 0.1:
        df = pd.read_csv(tbx11k_train_10, header=None)
    elif data_ratio == 0.01:
        df = pd.read_csv(tbx11k_train_1, header=None)
    else:
        raise NotImplementedError
    raw_data_info = pd.read_csv(tbx11k_labels)
    data_info = raw_data_info.groupby("fname").first()
    df['Target'] = df[0].apply(lambda x: data_info.loc[x+'.png', 'target'])
    distribute = df['Target'].value_counts(normalize=False).to_dict()
    return [distribute['no_tb'], distribute['tb']]


def sta_rsna_labels_distributions(data_ratio):
    if data_ratio == 1.0:
        df = pd.read_csv(rsna_train.split('.txt')[0] + '.csv')
    elif data_ratio == 0.1:
        df = pd.read_csv(rsna_train_10.split('.txt')[0] + '.csv')
    elif data_ratio == 0.01:
        df = pd.read_csv(rsna_train_1.split('.txt')[0] + '.csv')
    else:
        raise NotImplementedError

    distribute = df['Target'].value_counts(normalize=False).to_dict()
    return [distribute[0], distribute[1]]


def sta_siim_labels_distributions(data_ratio):
    if data_ratio == 1.0:
        df = pd.read_csv(siim_train, header=None)
    elif data_ratio == 0.1:
        df = pd.read_csv(siim_train_10, header=None)
    elif data_ratio == 0.01:
        df = pd.read_csv(siim_train_1, header=None)
    else:
        raise NotImplementedError
    df['Target'] = df[0].apply(lambda x: eval(x.split('_')[2]))
    distribute = df['Target'].value_counts(normalize=False).to_dict()
    return [distribute[0], distribute[1]]


def sta_shenzhen_labels_distributions(data_ratio):
    df = pd.read_csv(shenzhen_train, header=None)
    distribute = df[1].value_counts(normalize=False).to_dict()
    return [distribute[0], distribute[1]]


def sta_nih_labels_distributions(data_ratio):
    if data_ratio == 1.0:
        df = pd.read_csv(nih_train.split('.txt')[0] + '.csv')
    elif data_ratio == 0.1:
        df = pd.read_csv(nih_train_10.split('.txt')[0] + '.csv')
    elif data_ratio == 0.01:
        df = pd.read_csv(nih_train_1.split('.txt')[0] + '.csv')
    else:
        raise NotImplementedError
    del df['No Finding']
    cols = list(df.columns)[-14:]
    distribute = df[cols].sum().tolist()
    # distribute = df['Target'].value_counts(normalize=False).to_dict()
    return distribute


def build_dataloader(dataset, args):
    pad_token = "[PAD]"
    image_processor = AutoImageProcessor.from_pretrained(args['rad_dino_path'], trust_remote_code=True)
    if args['data_name'] == 'rsna':
        collate_fn = RSNACollateFn(image_processor, pad_token)
    elif args['data_name'] == 'nih':  # 'nih'
        collate_fn = NIHCollateFn(image_processor, pad_token)
    elif args['data_name'] == 'siim':  # 'siim'
        collate_fn = SIIMCollateFn(image_processor, pad_token)
    elif args['data_name'] == 'shenzhen':  # 'shenzhen-CXR'
        collate_fn = ShenZhenCXRCollateFn(image_processor, pad_token)
    else:
        raise NotImplementedError

    return DataLoader(
        dataset,
        batch_size=args['batch_size'],
        num_workers=args['num_workers'],
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )


def temp_iteration():
    from downstream.utils import setup_arguments, setup_seed
    args, logger = setup_arguments()
    train_set = SIIMDataset(args, 'train')
    test_set = SIIMDataset(args, 'test')
    val_set = SIIMDataset(args, 'val')
    train_loader = build_dataloader(train_set, args)
    val_loader = build_dataloader(val_set, args)
    test_loader = build_dataloader(test_set, args)
    print("train_loader")
    for batch in train_loader:
        for t, mask in zip(batch['disease_labels'], batch['mask']):
            if t == 1:
                assert len(mask.unique()) == 2
    print("val_loader")
    for batch in val_loader:
        for t, mask in zip(batch['disease_labels'], batch['mask']):
            if t == 1:
                assert len(mask.unique()) == 2
    print("test_loader")
    for batch in test_loader:
        for t, mask in zip(batch['disease_labels'], batch['mask']):
            if t == 1:
                assert len(mask.unique()) == 2


# temp_iteration()

# sta_tbx11k_labels_distributions(1)