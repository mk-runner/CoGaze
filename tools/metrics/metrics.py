from sklearn.metrics import f1_score, recall_score, precision_score
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from .Radgraph import F1RadGraph
from .f1chexbert import F1CheXbert
import re
import numpy as np
import torch


def compute_nlg_scores(gts, res, args=None):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """

    # Set up scorers

    gts = {i: [re.sub(' +', ' ', gt.replace(".", " ."))] for i, gt in enumerate(gts)}
    res = {i: [re.sub(' +', ' ', hpy.replace(".", " ."))] for i, hpy in enumerate(res)}
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), 'CIDer'),
    ]
    eval_res = {}
    # Compute score for each metric
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res, verbose=0)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    return eval_res


def compute_ce_scores(gts, res, args):
    # gts and res is list, e.g., [str1, str2]
    # roberta-large
    # model_type = 'distilbert-base-uncased',
    # P, R, F1 = score(res, gts, model_type=args['bertscore_checkpoint'],
    #                  num_layers=5, batch_size=64, nthreads=4, all_layers=False, idf=False, baseline_path=None,
    #                  device='cuda' if torch.cuda.is_available() else 'cpu', lang='en', rescale_with_baseline=True)
    # bertscore = F1.mean().cpu().item()

    f1chexbert = F1CheXbert(chexbert_checkpoint=args['chexbert_path'], model_checkpoint=args['bert_path'],
                            tokenizer_checkpoint=args['bert_path'])
    accuracy, accuracy_per_sample, chexbert_all, chexbert_5 = f1chexbert(hyps=res, refs=gts)
    # default is chexbert_5_micro_f1
    # micro: each sample has the same weight; macro: each class has the same weight
    chexbert_5_micro_f1 = chexbert_5["micro avg"]
    chexbert_all_micro_f1 = chexbert_all["micro avg"]
    chexbert_5_macro_f1 = chexbert_5["macro avg"]
    chexbert_all_macro_f1 = chexbert_all["macro avg"]
    # chexbertscore = class_report_5["micro avg"]["f1-score"]

    # f1radgraph_partial = F1RadGraph(reward_level='partial', model_path=args['radgraph_path'])
    # partial_mean_reward, reward_list, hypothesis_ann_lists, reference_ann_lists = f1radgraph_partial(hyps=res, refs=gts)

    f1radgraph_all = F1RadGraph(reward_level='all', model_path=args['radgraph_path'])
    all_mean_reward, reward_list, hypothesis_ann_lists, reference_ann_lists = f1radgraph_all(hyps=res, refs=gts)

    metrics = {
        # "BERTScore": bertscore,
        "Radgraph-partial": all_mean_reward[1],
        "Radgraph-simple": all_mean_reward[0],
        "Radgraph-complete": all_mean_reward[2],
        "chexbert_5_micro_f1": chexbert_5_micro_f1["f1-score"],
        "chexbert_5_macro_f1": chexbert_5_macro_f1["f1-score"],
        "chexbert_all_micro_p": chexbert_all_micro_f1['precision'],
        "chexbert_all_micro_r": chexbert_all_micro_f1['recall'],
        "chexbert_all_micro_f1": chexbert_all_micro_f1["f1-score"],
        "chexbert_all_macro_p": chexbert_all_macro_f1['precision'],
        "chexbert_all_macro_r": chexbert_all_macro_f1['recall'],
        "chexbert_all_macro_f1": chexbert_all_macro_f1["f1-score"],
    }
    # all_mean_reward, reward_list, hypothesis_ann_lists, reference_ann_lists = f1radgraph_all(hyps=res, refs=gts)
    return metrics


def delete_organ(report):
    valid_organ = [
        "Lungs and Airways",
        "Pleura",
        "Cardiovascular",
        "Hila and Mediastinum",
        "Tubes, Catheters, and Support Devices",
        "Musculoskeletal and Chest Wall",
        "Abdominal",
        "Other",
    ]
    pattern = r'(' + '|'.join(map(re.escape, valid_organ)) + r'):-'
    # 查找每个 section 的位置
    matches = list(re.finditer(pattern, report))
    results = []
    for i, m in enumerate(matches):
        section = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(report)
        content = report[start:end].strip()

        # 按 “-” 分割内容，清理空白，并去掉空字符串
        items = [x.strip() for x in content.split('-') if x.strip()]
        # 在前面加回 “-” 符号（因为原文中每项以 - 开头）
        items = [x.strip('-').strip() for x in items]

        results.append(' '.join(items))
    return ' '.join(results)


def compute_all_scores(gts, gens, args):
    # compute clinical efficacy metrics
    ce_metrics = compute_ce_scores(gts, gens, args)

    # compute natural language generation (NLG) metrics
    nlg_metrics = compute_nlg_scores(gts, gens)
    ce_metrics.update(nlg_metrics)
    return ce_metrics


def compute_all_scores_delete_organ(gts, gens, args):
    # compute clinical efficacy metrics
    gts = [delete_organ(item) for item in gts]
    gens = [delete_organ(item) for item in gens]
    ce_metrics = compute_ce_scores(gts, gens, args)

    # compute natural language generation (NLG) metrics
    nlg_metrics = compute_nlg_scores(gts, gens)
    ce_metrics.update(nlg_metrics)
    return ce_metrics


def compute_chexbert_scores(gts, gens, args):
    # compute clinical efficacy metrics
    ce_metrics = compute_ce_scores(gts, gens, args)
    return ce_metrics


def compute_chexbert_details_scores(gts, res, args):
    f1chexbert = F1CheXbert(chexbert_checkpoint=args['chexbert_path'], model_checkpoint=args['bert_path'],
                            tokenizer_checkpoint=args['bert_path'])
    accuracy, accuracy_per_sample, chexbert_all, chexbert_5 = f1chexbert(hyps=res, refs=gts)
    # default is chexbert_5_micro_f1
    # micro: each sample has the same weight; macro: each class has the same weight
    del chexbert_all['weighted avg']
    del chexbert_all['samples avg']
    sample_num = chexbert_all['micro avg']['support']
    new_results = {}
    for key, value in chexbert_all.items():
        if 'avg' in key:
            new_results[key] = ['-', round(value['precision'], 3), round(value['recall'], 3),
                                round(value['f1-score'], 3)]
        else:
            new_results[key] = [f"{round(value['support'] * 100 / sample_num, 1)} ({int(value['support'])})",
                                round(value['precision'], 3), round(value['recall'], 3), round(value['f1-score'], 3)]
    return new_results


class DiseaseClassificationMetric:
    def __init__(self):
        self.label_set = [
            'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
            'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
            'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
            'Support Devices', 'No Finding'
        ]

    def _convert_to_binary(self, preds, targets):
        """
        self.id2status = {
            1: 'positive',
            2: 'negative',
            3: 'uncertain',
            0: 'blank'
        }
        """
        preds = preds.numpy()
        # convert uncertain into positive; convert blank into negative
        binary_preds = np.isin(preds, [1, 3]).astype(int)
        binary_targets = np.isin(targets, [1, 3]).astype(int)
        return binary_preds, binary_targets

    def compute_metrics(self, preds, targets):
        binary_preds, binary_targets = self._convert_to_binary(preds, targets)

        metrics = {
            'Precision': round(precision_score(binary_targets, binary_preds, average="micro"), 3),
            'Recall': round(recall_score(binary_targets, binary_preds, average="micro"), 3),
            'F1': round(f1_score(binary_targets, binary_preds, average="micro"), 3)
        }

        return metrics


