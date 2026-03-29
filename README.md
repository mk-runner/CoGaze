<div align="center">

# 🩺 Seeing Like Radiologists: Context- and Gaze-Guided Vision-Language Pretraining for Chest X-rays

[![arXiv](https://img.shields.io/badge/arXiv-2508.05353-b31b1b.svg)](https://arxiv.org/abs/2508.05353)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/MK-runner/PriorRG)
[![BibTeX](https://img.shields.io/badge/%F0%9F%93%96-BibTeX-yellow)](#-citation)

<img src="generated_reports/fig2.png" alt="Framework Overview" width="75%">

</div>

---

## 📰 News

- **[2026-03-28]** Official code and [model weights](https://huggingface.co/MK-runner/CoGaze) are now public.

---

## ⚙️ Installation

```bash
# Create environment
conda create -n priorrg python=3.10.16
conda activate CoGaze
````

**Core dependencies:**

* `transformers==4.43.3`
* `radgraph==0.09`
* `pytorch-lighting==2.5.1.post0`
* `torch==2.4.1`
* `torchvision==0.19.1`

---

## 🧩 Model Checkpoints

| Dataset       | Pretrained Checkpoints                                                                 | CoGaze (DistilGPT2) | Generated Free-text Reports                                                                                                           |
|---------------|----------------------------------------------------------------------------------------|----------------------|-----------------------------------------------------------------------------------------------------------------------------|
| **MIMIC-CXR** | https://huggingface.co/MK-runner/CoGaze/tree/main/checkpoints/mimic-cxr              | CoGaze (DistilGPT2)  | https://github.com/mk-runner/PriorRG/blob/main/generated_reports/mimic-cxr-generated-reports-24-03-2025_18-07-41.csv       |
---

## 📁 Dataset Structure for MIMIC-CXR Dataset

### 1. Medical Images

CoGaze is trained on **MIMIC-CXR** dataset from [PhysioNet](https://physionet.org/content/mimic-cxr/2.0.0/).

```
data/
├── p10/
│   └── p10000032/
│       └── s50414267/
│           ├── 02aa804e-....jpg
│           └── 174413ec-....jpg
├── p11/
└── ...
```

### 2. Radiology Reports

Organized by `study_id` to obtain longitudinal data.

| Dataset            | Processed File                                                                                                                                                      | Description                       |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| **MIMIC-CXR**      | [`priorrg_mimic_cxr_annotation.json`](https://huggingface.co/MK-runner/PriorRG/blob/main/checkpoints/mimic-cxr/radiology-reports/priorrg_mimic_cxr_annotation.json) | Report annotations for MIMIC-CXR  |
| **View Positions** | [`view_position_dict.json`](https://huggingface.co/MK-runner/PriorRG/blob/main/checkpoints/mimic-cxr/radiology-reports/view-positions-dict-mimic.json)              | Metadata for X-ray view positions |

### 3. Checkpoint Directory Layout

```
ckpt_zoo_dir/
├── chexbert.pth
├── radgraph/
├── google-bert/bert-base-uncased/
├── microsoft/BiomedVLP-CXR-BERT-specialized/
├── microsoft/rad-dino/
└── distilbert/distilgpt2/
```

> `chexbert.pth` and `radgraph` must be downloaded manually (see [MLRG](https://github.com/mk-runner/MLRG) for instructions).
> Other checkpoints will be automatically fetched during training.

### 4. Download Datasets for Classification and Segmentation

- NIH Chest X-rays: We used the [NIH Chest X-rays](https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset) dataset from Huggingface. 

- RSNA: We used the stage 2 data of the [RSNA Pneumonia](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge) dataset from Kaggle.

- SIIM: We used the stage 1 data of the [SIIM-ACR Pneumothorax Segmentation](https://www.kaggle.com/datasets/vbookshelf/pneumothorax-chest-xray-images-and-masks) dataset from Kaggle.

- TBX11K: We used the [TBX11K Simplified](https://www.kaggle.com/datasets/vbookshelf/tbx11k-simplified) dataset from Kaggle.
  
- Shenzhen: We used the [Shenzhen chest X-ray set](https://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip) dataset from NIH.

---

## 🚀 Inference

The script `main_single_sample_github.py` supports **four input configurations** for single-study inference:

| Input Type                | Description                                                                                                                                                                     |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 🩻 **Image only**         | Single X-ray without view position (`view_position='unk'`)                                                                                                                      |
| 🧭 **+ View position**    | Specify position (e.g., PA, AP, Lateral). See [`view_position_dict.json`](https://huggingface.co/MK-runner/PriorRG/blob/main/radiology-report/priorrg_view_position_v1.0.json). |
| 💬 **+ Clinical context** | Add optional clinical notes or findings                                                                                                                                         |
| 📜 **+ Prior study**      | Provide a previous X-ray for longitudinal reasoning                                                                                                                             |

> Example configurations are available in `main_single_sample_github.py`.

---

## 🧠 Training & Evaluation Pipeline (MIMIC-CXR)

```bash
# Pretraining (finetune mode)
bash script_github/mimic-cxr-pretraining-finetune.sh

# Pretraining (inference mode)
bash script_github/mimic-cxr-pretraining-inference.sh

# Report generation (finetune mode)
bash script_github/mimic-cxr-report-generation-finetune.sh

# Report generation (inference mode)
bash script_github/mimic-cxr-report-generation-inference.sh
```

---

## 📊 Evaluation

```python
def compute_performance_using_generated_reports():
    from tools.metrics.metrics import compute_all_scores, compute_chexbert_details_scores
    import pandas as pd

    mimic_cxr_generated_path = 'generated_reports/mimic-cxr-generated-reports-24-03-2025_18-07-41.csv'
    args = {
        'chexbert_path': "/home/miao/data/dataset/checkpoints/chexbert.pth",
        'bert_path': "/home/miao/data/dataset/checkpoints/bert-base-uncased",
        'radgraph_path': "/home/miao/data/dataset/checkpoints/radgraph",
    }

    data = pd.read_csv(mimic_cxr_generated_path)
    gts, gens = data['reference_report'].tolist(), data['generated_report'].tolist()
    scores = compute_all_scores(gts, gens, args)
    print(scores)
```

---

## 📊 More metrics
```python
{
    'BertScore': 0.589690089225769,
    'SemScore': 0.44889214634895325,
    '1/RadCliQ-V1': 1.0499188828999766,
    'RATEScore': 0.5711956463232671,
    'green': 0.3607354281809111,
    'chexbert_5_micro_f1': 0.5621201554249278,
    'chexbert_5_macro_f1': 0.49565410982343805,
    'chexbert_all_micro_p': 0.5410030133448127,
    'chexbert_all_micro_r': 0.4849508006945784,
    'chexbert_all_micro_f1': 0.5114457218435242,
    'chexbert_all_macro_p': 0.42861347185421145,
    'chexbert_all_macro_r': 0.36832540441255107,
    'chexbert_all_macro_f1': 0.37640516594538165,
    'BLEU_1': 0.4118609564738112, 'BLEU_2': 0.2895466207962516,
    'BLEU_3': 0.21973018011383075, 'BLEU_4': 0.17475057720959183,
    'METEOR': 0.1894554556994692, 'ROUGE_L': 0.3238645529898187, 'CIDer': 0.4069847807856516
}
```
---

## 📚 Citation

If you find this work helpful, please cite:

```bibtex

```

---

## 🙏 Acknowledgements

* [MLRG](https://github.com/mk-runner/MLRG): Dataset organization and evaluation tools
* [cvt2distilgpt2](https://github.com/aehrc/cvt2distilgpt2): Text generation initialization framework

---

<div align="center">

⭐️ **If you find this repository useful, please consider starring it!**
📬 For questions, open an issue or contact the authors.

</div>
