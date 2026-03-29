<div align="center">

# 🩺 CoGaze: Seeing Like Radiologists: Context- and Gaze-Guided Vision-Language Pretraining for Chest X-rays

[![arXiv](https://img.shields.io/badge/arXiv-2508.05353-b31b1b.svg)](https://arxiv.org/abs/**.**)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/MK-runner/CoGaze)
[![BibTeX](https://img.shields.io/badge/📖-BibTeX-yellow)](#-citation)

<img src="generated_reports/fig2.png" alt="Framework Overview" width="100%">

</div>

---

## ✨ Overview

**CoGaze** is a vision-language pretraining framework designed for **chest X-ray understanding**, inspired by how radiologists interpret medical images.  

It integrates:

- 👁️ Gaze information is used during pretraining, while downstream tasks (report generation, classification, and segmentation) do not require gaze data.
- 🧠 Context-aware reasoning  
- 📝 Free-text & structured report generation, supervised & zero-shot classification, segmentation, image-text retrieval

---

## 📰 News

- **[2026-03-28]** 🚀 Official code and pretrained models are released on [Hugging Face](https://huggingface.co/MK-runner/CoGaze)

---

## ⚙️ Installation

```bash
# Create conda environment
conda create -n cogaze python=3.10.16
conda activate cogaze
````

### 📦 Core Dependencies

```txt
transformers==4.43.3
radgraph==0.09
pytorch-lighting==2.5.1.post0
torch==2.4.1
torchvision==0.19.1
```

---

## 🧩 Model Zoo

| Dataset       | Pretrained Model                                                                                        | Report Generation Model                                                                                     | Outputs                                                                              |
| ------------- | ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| **MIMIC-CXR** | [CoGaze Pretrained Checkpoint](https://huggingface.co/MK-runner/CoGaze/blob/main/mimic_pretrain_best_model.pt) | [CoGaze (DistilGPT2)](https://huggingface.co/MK-runner/CoGaze/blob/main/distilgpt2_mimic_free_text_report_generation_best_model.pt) | [Generated Reports](https://github.com/mk-runner/CoGaze/tree/main/generated_reports) |

---

## 📁 Dataset Preparation

### 1️⃣ MIMIC-CXR Images

Dataset source: [PhysioNet](https://physionet.org/content/mimic-cxr/2.0.0/)

```
data/
├── p10/
│   └── p10000032/
│       └── s50414267/
│           ├── image1.jpg
│           └── image2.jpg
├── p11/
└── ...
```

---

### 2️⃣ Annotations & Reports

Available on 🤗 Hugging Face:

* Gaze heatmap
* Image-text pairs
* SRRG annotations

👉 [https://huggingface.co/MK-runner/CoGaze/tree/main/mimic-annotation](https://huggingface.co/MK-runner/CoGaze/tree/main/mimic-annotation)

---

### 3️⃣ Checkpoint Structure

```
ckpt_zoo_dir/
├── chexbert.pth
├── radgraph/
├── google-bert/
├── microsoft/
└── distilgpt2/
```

⚠️ **Manual download required:**

* `chexbert.pth`
* `radgraph`

See: [https://github.com/mk-runner/MLRG](https://github.com/mk-runner/MLRG)

💡 Tip: Enable automatic download during training:

```bash
--online_ckpt "Yes"
```

---

### 4️⃣ Additional Datasets

| Task           | Dataset                                                                                         |
| -------------- | ----------------------------------------------------------------------------------------------- |
| Classification | [NIH Chest X-rays](https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset)            |
| Detection      | [RSNA Pneumonia](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge)        |
| Segmentation   | [SIIM-ACR](https://www.kaggle.com/datasets/vbookshelf/pneumothorax-chest-xray-images-and-masks) |
| Tuberculosis   | [TBX11K](https://www.kaggle.com/datasets/vbookshelf/tbx11k-simplified)                          |
| External       | [Shenzhen Dataset](https://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip)            |

---

## 🧠 Training & Inference

### 🔹 Pretraining

```bash
bash script/pretrain.sh
```

---

### 🔹 Report Generation

#### Free-text (Training)

```bash
bash script/free-text-report-generation-gpt2.sh
bash script/free-text-report-generation-llm.sh
```

#### Free-text (Inference)

```bash
bash script/free-text-report-generation-gpt2-inference.sh
```

#### Structured Reports

```bash
bash script/structured-report-generation-gpt2.sh
```

---

## 📊 Evaluation

### 🔹 Compute Metrics

```python
from tools.metrics.metrics import compute_all_scores
import pandas as pd

data = pd.read_csv("generated_reports/xxx.csv")
gts = data['reference_report'].tolist()
gens = data['generated_report'].tolist()

scores = compute_all_scores(gts, gens, args)
print(scores)
```

---

### 📈 Performance (DistilGPT2)

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

```bibtex
@article{cogaze2026,
  title={Seeing Like Radiologists: Context- and Gaze-Guided Vision-Language Pretraining for Chest X-rays},
  author={...},
  year={2026}
}
```

---

## 🙏 Acknowledgements

* [MLRG](https://github.com/mk-runner/MLRG) — dataset & evaluation tools
* [cvt2distilgpt2](https://github.com/aehrc/cvt2distilgpt2) — text generation initialization

---

## ⭐ Support

If you find this project useful:

* ⭐ Star this repository
* 🐛 Open issues for questions or bugs
* 📬 Contact Kang Liu (kangliu422@gmail.com) for collaboration

---
