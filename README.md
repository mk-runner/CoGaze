<div align="center">

# 🩺 Seeing Like Radiologists: Context- and Gaze-Guided Vision-Language Pretraining for Chest X-rays

[![arXiv](https://img.shields.io/badge/arXiv-2508.05353-b31b1b.svg)](https://arxiv.org/abs/2508.05353)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/MK-runner/CoGaze)
[![BibTeX](https://img.shields.io/badge/%F0%9F%93%96-BibTeX-yellow)](#-citation)

<img src="generated_reports/fig2.png" alt="Framework Overview" width="100%">

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
| **MIMIC-CXR** | [Hugging Face](https://huggingface.co/MK-runner/CoGaze/blob/main/mimic_pretrain_best_model.pt)              | [Hugging Face](https://huggingface.co/MK-runner/CoGaze/blob/main/mimic_report_generation_best_model.pt)  | [Generated Reports](https://github.com/mk-runner/CoGaze/tree/main/generated_reports)       |
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

All restructured reports, including the annotation with eye-tracking data (`cogaze_mimic_cxr_annotation_similar_case_v0702_gaze.json`), the image–text pair annotation (`cogaze_mimic_cxr_annotation_similar_case_v0702_v0826.json`), and the SRRG image–text pair annotation (`cogaze_srrg_annotation_v0702_v0826.json`), are available on [Hugging Face](https://huggingface.co/MK-runner/CoGaze/tree/main/mimic-annotation).

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

> During training, additional checkpoints will be automatically downloaded or retrieved. You only need to specify the parameter `--online_ckpt "Yes"`.

### 4. Download Datasets for Classification and Segmentation

- NIH Chest X-rays: We used the [NIH Chest X-rays](https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset) dataset from Huggingface. 

- RSNA: We used the stage 2 data of the [RSNA Pneumonia](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge) dataset from Kaggle.

- SIIM: We used the stage 1 data of the [SIIM-ACR Pneumothorax Segmentation](https://www.kaggle.com/datasets/vbookshelf/pneumothorax-chest-xray-images-and-masks) dataset from Kaggle.

- TBX11K: We used the [TBX11K Simplified](https://www.kaggle.com/datasets/vbookshelf/tbx11k-simplified) dataset from Kaggle.
  
- Shenzhen: We used the [Shenzhen chest X-ray set](https://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip) dataset from NIH.

---

## 🧠 Training & Evaluation Pipeline (MIMIC-CXR)

```bash
# Pretraining (finetune mode)
bash script/pretrain.sh

# Free-text Report generation (finetune mode)
bash script/free-text-report-generation-gpt2.sh
bash script/free-text-report-generation-llm.sh

# Free-text Report generation (inference mode)
bash script/free-text-report-generation-gpt2.sh  # Please set the phase to inference by using "--phase inference", and provide the "test_ckpt_path" parameter.

# Structure Report generation (finetune mode)
bash script/structured-report-generation-gpt2.sh
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

## 📊 More metrics for free-text report generation generated by CoGaze (DistilGPT2)
```python
{
    'BertScore': 0.5956377387046814,
    'Radgraph-simple': 0.30690433233898795,
    'Radgraph-partial': 0.28076371917819565,
    'Radgraph-complete': 0.22603009157065043,
    'SemScore': 0.45877182483673096,
    '1/RadCliQ-V1': 1.082196619824061,
    'RATEScore': 0.5787309255637078,
    'chexbert_5_micro_f1': 0.5708835341365461,
    'chexbert_5_macro_f1': 0.49498245207765257,
    'chexbert_all_micro_p': 0.5544458762886598,
    'chexbert_all_micro_r': 0.4980706154736639,
    'chexbert_all_micro_f1': 0.5247484500457363,
    'chexbert_all_macro_p': 0.44258976034375364,
    'chexbert_all_macro_r': 0.37672752858687886,
    'chexbert_all_macro_f1': 0.3883859770668801,
    'BLEU_1': 0.4103171077382396,
    'BLEU_2': 0.28970066408787387,
    'BLEU_3': 0.22010546378006685,
    'BLEU_4': 0.17481171574606008,
    'METEOR': 0.19054219748683743,
    'ROUGE_L': 0.3257898419599922,
    'CIDer': 0.3962696560568994
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
