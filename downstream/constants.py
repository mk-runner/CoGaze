import os

#*******************constants_setup************
# =====================RSNA dataset ==================================
# phenumonia, only frontal image, binary classification, bounding box
rsna_dir = 'dataset/RSNA'
# labels
rsna_train_path = os.path.join(rsna_dir, 'stage_2_train_labels.csv')
# images
rsna_images = os.path.join(rsna_dir, 'stage_2_train_images')
# split
# rsna_train = 'benchx_split/RSNA/train.txt'
# rsna_train_1 = 'benchx_split/RSNA/train_1.txt'
# rsna_train_10 = 'benchx_split/RSNA/train_10.txt'
# rsna_val = 'benchx_split/RSNA/val.txt'
# rsna_test = 'benchx_split/RSNA/test.txt'
rsna_train = 'downstream/benchx_split/RSNA/train.txt'
rsna_train_1 = 'downstream/benchx_split/RSNA/train_1.txt'
rsna_train_10 = 'downstream/benchx_split/RSNA/train_10.txt'
rsna_val = 'downstream/benchx_split/RSNA/val.txt'
rsna_test = 'downstream/benchx_split/RSNA/test.txt'


# ======================NIH dataset ==================================
# (multi-label classification)
nih_dir = '/dataset/NIH-Chest-X-ray-dataset/data/'
# labels
nih_labels = os.path.join(nih_dir, 'Data_Entry_2017_v2020.csv')
# images
nih_images = os.path.join(nih_dir, 'images')
# split
nih_train = 'benchx_split/NIH_Chest_Xray/train.txt'
nih_train_1 = 'benchx_split/NIH_Chest_Xray/train_1.txt'
nih_train_10 = 'benchx_split/NIH_Chest_Xray/train_10.txt'
nih_val = 'benchx_split/NIH_Chest_Xray/val.txt'
nih_test = 'benchx_split/NIH_Chest_Xray/test.txt'
# nih_train = 'downstream/benchx_split/NIH_Chest_Xray/train.txt'
# nih_train_1 = 'downstream/benchx_split/NIH_Chest_Xray/train_1.txt'
# nih_train_10 = 'downstream/benchx_split/NIH_Chest_Xray/train_10.txt'
# nih_val = 'downstream/benchx_split/NIH_Chest_Xray/val.txt'
# nih_test = 'downstream/benchx_split/NIH_Chest_Xray/test.txt'


# ======================ShenZhen-CXR dataset ==================================
# (multi-label classification)
shenzhen_dir = '/dataset/shenzhen-CXR/ChinaSet_AllFiles'
# images
shenzhen_images = os.path.join(shenzhen_dir, 'CXR_png')
shenzhen_info = os.path.join(shenzhen_dir, 'ClinicalReadings')
# split
# shenzhen_train = 'chexworld_split/shenzhen/ShenzhenCXR_train_data.txt'
# shenzhen_val = 'chexworld_split/shenzhen/ShenzhenCXR_val_data.txt'
# shenzhen_test = 'chexworld_split/shenzhen/ShenzhenCXR_test_data.txt'
shenzhen_train = 'downstream/chexworld_split/shenzhen/ShenzhenCXR_train_data.txt'
shenzhen_val = 'downstream/chexworld_split/shenzhen/ShenzhenCXR_val_data.txt'
shenzhen_test = 'downstream/chexworld_split/shenzhen/ShenzhenCXR_test_data.txt'


# ======================SIIM dataset ==================================
# Download link: https://www.kaggle.com/datasets/vbookshelf/pneumothorax-chest-xray-images-and-masks

# (multi-label classification)
siim_dir = 'data/dataset/SIIM-ACR'
# labels
siim_labels = os.path.join(siim_dir, 'stage_2_train.csv')
# images
siim_images = os.path.join(siim_dir, 'dicom-images-train')
# split
# siim_train = 'benchx_split/SIIM/train.txt'
# siim_train_1 = 'benchx_split/SIIM/train_1.txt'
# siim_train_10 = 'benchx_split/SIIM/train_10.txt'
# siim_val = 'benchx_split/SIIM/val.txt'
# siim_test = 'benchx_split/SIIM/test.txt'

siim_train = 'downstream/benchx_split/SIIM/train.txt'
siim_train_1 = 'downstream/benchx_split/SIIM/train_1.txt'
siim_train_10 = 'downstream/benchx_split/SIIM/train_10.txt'
siim_val = 'downstream/benchx_split/SIIM/val.txt'
siim_test = 'downstream/benchx_split/SIIM/test.txt'


# ======================TBX11K dataset ==================================
# Download link: https://www.kaggle.com/datasets/vbookshelf/pneumothorax-chest-xray-images-and-masks

# (multi-label classification)
tbx11k_dir = '/dataset/tbx11k-simplified'
# labels
tbx11k_labels = os.path.join(tbx11k_dir, 'data.csv')
# images
tbx11k_images = os.path.join(tbx11k_dir, 'images')
# split
tbx11k_train = 'benchx_split/TBX11K/train.txt'
tbx11k_train_1 = 'benchx_split/TBX11K/train_1.txt'
tbx11k_train_10 = 'benchx_split/TBX11K/train_10.txt'
tbx11k_val = 'benchx_split/TBX11K/val.txt'
tbx11k_test = 'benchx_split/TBX11K/test.txt'

# tbx11k_train = 'downstream/benchx_split/TBX11K/train.txt'
# tbx11k_train_1 = 'downstream/benchx_split/TBX11K/train_1.txt'
# tbx11k_train_10 = 'downstream/benchx_split/TBX11K/train_10.txt'
# tbx11k_val = 'downstream/benchx_split/TBX11K/val.txt'
# tbx11k_test = 'downstream/benchx_split/TBX11K/test.txt'


pneumonia_prompts = [
    "A chest X-ray showing pneumonia.",
    "A chest X-ray showing lung consolidation and infiltrates.",
    "A chest radiograph with patchy opacities consistent with pneumonia.",
    "Chest X-ray demonstrating alveolar consolidation typical of bacterial pneumonia.",
    "A chest X-ray with signs of infection in both lungs.",
    "A chest X-ray showing inflammation and airspace disease.",
    "Radiograph showing bilateral consolidation due to pneumonia.",
    "A chest X-ray indicating acute lower respiratory infection.",
    "An X-ray image showing pneumonia with dense opacities.",
    "Chest X-ray with pulmonary infiltrates characteristic of pneumonia."
]
pneumothorax_prompts = [
    "A chest X-ray showing pneumothorax.",
    "Chest radiograph demonstrating collapsed right lung.",
    "A chest X-ray with visible pleural line and absent lung markings.",
    "A chest X-ray showing air in the pleural space.",
    "Radiograph showing left-sided pneumothorax.",
    "A chest X-ray showing partial lung collapse.",
    "Chest X-ray showing hyperlucent hemithorax due to pneumothorax.",
    "A chest X-ray demonstrating spontaneous pneumothorax.",
    "Chest radiograph showing absence of vascular markings in one lung field.",
    "X-ray image with signs of pneumothorax and pleural separation."
]
tuberculosis_prompts = [
    "A chest X-ray showing tuberculosis.",
    "A chest radiograph with cavitary lesions in the upper lobes.",
    "Chest X-ray showing fibrotic changes and nodular opacities consistent with tuberculosis.",
    "A chest X-ray showing patchy infiltrates in both lungs due to tuberculosis.",
    "Radiograph showing consolidation and cavity formation typical of pulmonary tuberculosis.",
    "Chest X-ray demonstrating apical opacities caused by tuberculosis infection.",
    "A chest X-ray showing scarring and nodules consistent with old tuberculosis.",
    "Chest radiograph showing upper lobe lesions with cavitation.",
    "A chest X-ray indicating pulmonary tuberculosis.",
    "X-ray showing lung cavitation and consolidation consistent with tuberculosis."
]
normal_prompts = [
    "A normal chest X-ray.",
    "A chest X-ray showing clear lungs and no abnormal findings.",
    "Chest radiograph without any signs of disease.",
    "A chest X-ray with normal appearance and no consolidation.",
    "Radiograph demonstrating no acute cardiopulmonary abnormality.",
    "Chest X-ray showing no evidence of infection or collapse.",
    "A chest X-ray showing healthy lungs.",
    "X-ray image without opacity or lesion.",
    "A normal chest radiograph with clear lung fields.",
    "A chest X-ray showing no pneumothorax, no pneumonia, no tuberculosis."
]

class_prompts = {
    "pneumonia": pneumonia_prompts,
    "pneumothorax": pneumothorax_prompts,
    "tuberculosis": tuberculosis_prompts,
    "normal": normal_prompts
}

