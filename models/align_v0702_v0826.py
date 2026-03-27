from typing import Dict

import torch
import transformers
import torchmetrics
from torch.cuda.amp import autocast
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.utilities import rank_zero_only
from torchmetrics import Accuracy, AUROC, AveragePrecision, Recall, Specificity, F1Score
from transformers import (AutoModel, AutoConfig, AutoImageProcessor, GPT2TokenizerFast,
                          PretrainedConfig, AutoTokenizer, LlamaForCausalLM)

from models.conversation_v0702_llm import conv_gazerg_llama32, SeparatorStyle, conv_gazerg_v0904
from models.perceiver_pytorch import Perceiver
from tools.metrics.chexbert import RadGraphMetrics, F1CheXbertMetrics
from tools.metrics.coco import COCOCaptionMetrics
from tools.metrics.report_logger import ReportLogger
from models.utils import *
from models.class_balanced_loss import MultiLabelDiseasesBalancedClassLoss
from tools.dataset_ab import *


class Pretrain(pl.LightningModule):
    def __init__(
            self,
            args: Dict,
            logger,
    ):
        super().__init__()
        self.args = args
        self.mylog = logger
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.prefetch_factor = args['prefetch_factor']
        self.is_stop_kl_loss = False
        self.is_stop_text_disease_loss = False
        self.is_stop_image_disease_loss = False
        self.val_best_scores = {
            "best_epoch": -1,
            "best_monitor_metric": 1e9,
        }

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_loss_metric = {
            'loss': torchmetrics.MeanMetric().to(device),
            'align_loss': torchmetrics.MeanMetric().to(device),
            'kl_loss': torchmetrics.MeanMetric().to(device),
            'img_disease_loss': torchmetrics.MeanMetric().to(device),
            'text_disease_loss': torchmetrics.MeanMetric().to(device),
        }
        self.val_loss_metric = {
            'loss': torchmetrics.MeanMetric().to(device),
            'align_loss': torchmetrics.MeanMetric().to(device),
            'kl_loss': torchmetrics.MeanMetric().to(device),
            'img_disease_loss': torchmetrics.MeanMetric().to(device),
            'text_disease_loss': torchmetrics.MeanMetric().to(device),
        }
        self.test_loss_metric = {
            'loss': torchmetrics.MeanMetric().to(device),
            'align_loss': torchmetrics.MeanMetric().to(device),
            'kl_loss': torchmetrics.MeanMetric().to(device),
            'img_disease_loss': torchmetrics.MeanMetric().to(device),
            'text_disease_loss': torchmetrics.MeanMetric().to(device),
        }

        # accuracy
        task = 'multilabel'
        num_labels = 14
        cur_num_classes = None
        self.report_acc = Accuracy(task=task, num_labels=num_labels, num_classes=cur_num_classes)
        self.vision_acc = Accuracy(task=task, num_labels=num_labels, num_classes=cur_num_classes)
        # auroc
        self.report_auroc = AUROC(task=task, num_labels=num_labels, num_classes=cur_num_classes)
        self.vision_auroc = AUROC(task=task, num_labels=num_labels, num_classes=cur_num_classes)
        # PR-AUC equals AP
        self.report_ap = AveragePrecision(task=task, num_labels=num_labels, num_classes=cur_num_classes)
        self.vision_ap = AveragePrecision(task=task, num_labels=num_labels, num_classes=cur_num_classes)
        # recall == sensitivity
        self.report_recall = Recall(task=task, num_labels=num_labels, num_classes=cur_num_classes)
        self.vision_recall = Recall(task=task, num_labels=num_labels, num_classes=cur_num_classes)
        # specificity
        self.report_specificity = Specificity(task=task, num_labels=num_labels, num_classes=cur_num_classes)
        self.vision_specificity = Specificity(task=task, num_labels=num_labels, num_classes=cur_num_classes)
        # f1-score
        self.report_f1 = F1Score(task=task, num_labels=num_labels, num_classes=cur_num_classes)
        self.vision_f1 = F1Score(task=task, num_labels=num_labels, num_classes=cur_num_classes)
        print("load model ...")

        # ==================define image encoder and text encoder =================
        # image encoder
        self.image_encoder = AutoModel.from_pretrained(args['rad_dino_path'], trust_remote_code=True)
        self.image_processor = AutoImageProcessor.from_pretrained(args['rad_dino_path'], trust_remote_code=True)
        self.image_encoder.eval()
        self.freeze_parameters(self.image_encoder)

        # text encoder
        # self.tokenizer = AutoTokenizer.from_pretrained(self.args['distilgpt2_path'], trust_remote_code=True)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(self.args['distilgpt2_path'], trust_remote_code=True)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]', 'sep_token': '[SEP]', 'cls_token': '[CLS]'})
        self.tokenizer.add_tokens(['[INDICATION]', '[HISTORY]', '[Similar Cases]', '[FINDINGS]', '[TRANSCRIPT]'])
        self.text_encoder = self.build_text_encoder()
        self.text_encoder.train()

        # ========================define pos_embed, projector, and classifier=============================
        decoder_config = AutoConfig.from_pretrained(self.args['distilgpt2_path'], trust_remote_code=True)
        hidden_size = decoder_config.hidden_size
        image_dim = self.image_encoder.config.hidden_size
        text_dim = self.text_encoder.config.hidden_size

        # define projector
        self.image_projector = VisionProjectorMLP(image_dim, hidden_size * 2, hidden_size, args['view_position_path'])
        self.text_projector = TextProjectorMLP(text_dim, hidden_size * 2, hidden_size)

        # ==============define prior-guided uni-modal features ============================
        self.perceiver = Perceiver(
            byte_dim=hidden_size,  # byte array dimension
            depth=args['perceiver_num_blocks'],
            # depth of net. depth * (cross attention -> self_per_cross_attn * self attention)
            num_latents=args['perceiver_num_latents'],  # number of latents
            latent_dim=hidden_size,  # latent dimension
            cross_heads=8,  # number of heads for cross attention. paper said 1
            latent_heads=8,  # number of heads for latent self attention, 8
            cross_dim_head=64,  # number of dimensions per cross attention head
            latent_dim_head=64,  # number of dimensions per latent self attention head
            attn_dropout=0.,
            ff_dropout=0.,
            weight_tie_layers=False,  # whether to weight tie layers (optional, as indicated in the diagram)
            self_per_cross_attn=1  # number of self attention blocks per cross attention
        )

        self.shared_latent = nn.Parameter(torch.randn(args['perceiver_num_latents'], hidden_size), requires_grad=True)
        self.latent_type_embed = nn.ParameterDict({
            'context': nn.Parameter(torch.randn(1, hidden_size), requires_grad=True),
            'image': nn.Parameter(torch.randn(1, hidden_size), requires_grad=True),
        })

        # ===========================define loss functions =================================
        # define classifier for visual and textual features
        self.vision_classifier = MultiLabelDiseaseClassifier(input_dim=hidden_size, pool_type='avg')
        self.report_classifier = MultiLabelDiseaseClassifier(input_dim=hidden_size, pool_type='avg')
        # classification_loss
        class_frequency = compute_binary_multilabel_class_frequency(args['ann_path'])  # statistic class frequency
        print("current class frequency is", class_frequency)
        self.multi_label_loss = MultiLabelDiseasesBalancedClassLoss(class_frequency, gamma=0.2, beta=0.9999,
                                                                    loss_type='sigmoid')

        # define temperature hyper-parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=True)
        self.logit_scale_gaze = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=True)
        print("finish loading model ...")

    def finetune_parameters(self, model):
        for para in model.parameters():
            para.requires_grad = True

    def freeze_parameters(self, model):
        for para in model.parameters():
            para.requires_grad = False

    def build_text_encoder(self):
        enc_config = AutoConfig.from_pretrained(self.args['cxr_bert_path'], trust_remote_code=True)
        enc_config.vocab_size = len(self.tokenizer)
        enc_config.eos_token_id = self.tokenizer.eos_token_id
        enc_config.bos_token_id = self.tokenizer.bos_token_id
        enc_config.pad_token_id = self.tokenizer.pad_token_id
        enc_config.num_hidden_layers = self.args['text_encoder_num_blocks']
        enc_config.max_length = 200
        return AutoModel.from_pretrained(
            self.args['cxr_bert_path'],
            config=enc_config,
            ignore_mismatched_sizes=True,
            trust_remote_code=True)

    @rank_zero_only
    def log_once(self, message):
        self.mylog.info(message)

    def setup(self, stage=None):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#setup
        """
        if stage == 'fit' or stage is None:
            # construct train_set and val_set
            self.train_set = End2EndDataset(self.args, 'train')
            self.val_set = End2EndDataset(self.args, 'val')

            print("No. of training & validation examples: {} & {}.".format(
                self.train_set.__len__(), self.val_set.__len__()))
            self.log_once("No. of training & validation examples: {} & {}.".format(
                self.train_set.__len__(), self.val_set.__len__()))
        if stage == "test" or stage is None:
            self.test_set = End2EndDataset(self.args, 'test')
            print("No. of test examples: {}.".format(self.test_set.__len__()))
            self.log_once("No. of test examples: {}.".format(self.test_set.__len__()))

    def train_dataloader(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#train-dataloader
        """
        pad_token = self.tokenizer.pad_token
        patch_size = self.image_encoder.config.patch_size
        bos_token, eos_token = self.tokenizer.bos_token, self.tokenizer.eos_token
        collate_fn = End2EndCollateFn(self.args, self.image_processor, pad_token, patch_size, bos_token, eos_token)
        return DataLoader(
            self.train_set,
            batch_size=self.args['batch_size'],
            num_workers=self.args['num_workers'],
            shuffle=True,
            prefetch_factor=self.prefetch_factor,
            collate_fn=collate_fn,
            drop_last=True,
        )

    def val_dataloader(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#val-dataloader
        """
        pad_token = self.tokenizer.pad_token
        patch_size = self.image_encoder.config.patch_size
        bos_token, eos_token = self.tokenizer.bos_token, self.tokenizer.eos_token
        collate_fn = End2EndCollateFn(self.args, self.image_processor, pad_token, patch_size, bos_token, eos_token)
        return DataLoader(
            self.val_set,
            batch_size=self.args['test_batch_size'],
            num_workers=self.args['num_workers'],
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
            collate_fn=collate_fn,
            drop_last=False
        )

    def test_dataloader(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#test-dataloader
        """
        pad_token = self.tokenizer.pad_token
        patch_size = self.image_encoder.config.patch_size
        bos_token, eos_token = self.tokenizer.bos_token, self.tokenizer.eos_token
        collate_fn = End2EndCollateFn(self.args, self.image_processor, pad_token, patch_size, bos_token, eos_token)
        return DataLoader(
            self.test_set,
            batch_size=self.args['test_batch_size'],
            num_workers=self.args['num_workers'],
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
            collate_fn=collate_fn,
            drop_last=False,
        )

    def configure_optimizers(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        """
        all_parameters = []
        for param in self.parameters():
            if not param.requires_grad:
                continue
            all_parameters.append(param)
        optimiser = torch.optim.AdamW(all_parameters, lr=self.args['learning_rate'])
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimiser, T_max=self.args['max_epochs'],
        #                                                        eta_min=1e-6)
        scheduler = ReduceLROnPlateau(optimiser, mode=self.args['monitor_mode'],
                                      factor=0.1, patience=self.args['patience'])
        return {
            "optimizer": optimiser,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self.args['monitor_metric'],
                'frequency': 1,  # the frequency of check
                'interval': 'epoch'
            }
        }

    def get_text_features(self, text, device):
        tokens = self.tokenizer(text, padding=True, return_tensors='pt',
                                max_length=self.args['encoder_max_length'], truncation=True)
        tokens['input_ids'] = tokens['input_ids'].to(device)
        tokens['attention_mask'] = tokens['attention_mask'].to(device)
        text_embed = self.text_encoder(input_ids=tokens['input_ids'],
                                       attention_mask=tokens['attention_mask'])['last_hidden_state']
        text_embed = self.text_projector(text_embed)
        return text_embed

    def get_vision_features(self, images, view_positions):
        with torch.no_grad():
            image_embed = self.image_encoder(images)['last_hidden_state']
        # add view positional embedding, fusion gaze_heatmap projector
        image_embed = self.image_projector(image_embed, view_positions)
        return image_embed

    def global_alignment_loss(self, global_image_embed, global_text_embed, patient_ids):
        # multi-positive contrastive learning
        patient_ids = np.array(patient_ids)
        labels = (patient_ids.reshape(-1, 1) == patient_ids.reshape(1, -1)).astype(int)
        labels = torch.from_numpy(labels).float().to(global_image_embed.device)
        labels = labels / labels.sum(1, keepdim=True)
        del patient_ids

        # normalize
        global_image_embed = F.normalize(global_image_embed, dim=-1, p=2)
        global_text_embed = F.normalize(global_text_embed, dim=-1, p=2)

        # calculate the InfoNCE loss
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * global_image_embed @ global_text_embed.t()
        logits_per_text = logits_per_image.t()

        loss_image = F.cross_entropy(logits_per_image, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        global_instance_loss = (loss_image + loss_text) / 2.0
        return global_instance_loss

    def soft_gaze_guidance(self, image_feat, transcript_feat, gaze_heatmap, dicom_id, transcript_id, topk_ratio=0.25):
        """

        Args:
            image_feat: (b, num_patches + 1, d) patch features for each sample (include [CLS] token)
            transcript_feat: (sum_i^{i=b} num_sentences, d) sentence cls features for transcripts
            gaze_heatmap: (sum_i^{i=b} num_sentences, num_patches) supervision signals
            dicom_id: numpy.array, (b,) each sample's patient-id
            transcript_id: numpy.array, (sum_i^{i=b},) each transcript's patient-id
        Returns:
            a loss
        """
        # normalization (not include cls token)
        image_feat = F.normalize(image_feat[:, 1:, :], dim=-1, p=2)
        transcript_feat = F.normalize(transcript_feat, dim=-1, p=2)
        transcript_id = np.array(transcript_id)

        # compute KL-divergence
        kl_individual = []
        logit_scale = self.logit_scale_gaze.exp()
        for i, p_id in enumerate(dicom_id):
            # obtain valid text features
            valid_idx = transcript_id == p_id
            if np.sum(valid_idx) == 0:
                continue
            valid_transcript_feat = transcript_feat[valid_idx]  # (cur_num_sentences, dim)

            # obtain valid image features
            valid_image_feat = image_feat[i]  # (num_patches, dim)

            # obtain valid gaze_heatmap
            t2i_heatmap = gaze_heatmap[valid_idx]  # (cur_num_sentences, num_patches)
            # normalize heatmap

            # compute logis and kl_divergence loss (transcript -> image)
            t2i_logis = logit_scale * valid_transcript_feat @ valid_image_feat.T
            # we can explore different topk_ratios (default is 0.25)
            kl_loss = bidirectional_js_loss_dynamic(t2i_logis, t2i_heatmap, topk_ratio=topk_ratio, i2t_weight=0.2)
            kl_individual.append(kl_loss)

        kl_loss = sum(kl_individual) / len(kl_individual)
        return kl_loss

    def get_shared_latent(self, mode: str, num: int):
        type_embed = self.latent_type_embed[mode]  # (1, d)
        latent = self.shared_latent + type_embed  # (l, d)
        return latent.unsqueeze(0).repeat(num, 1, 1)  # (b, l, d)

    def forward(self, batch, mode='train'):
        device = batch['image'].device
        batch_size = len(batch['image'])

        # =================== extract uni-modal features =======================
        # obtain vision features
        vision_feat = self.get_vision_features(batch['image'], batch['view_position'])  # (b, 1370, 768)

        # =================== fuse uni-modal features using attention =======================
        # obtain report and knowledge features
        knowledge_idx = [i for i, k in enumerate(batch['knowledge']) if k != self.tokenizer.pad_token]
        non_knowledge_idx = [i for i in range(batch_size) if i not in knowledge_idx]

        visual_latent = torch.zeros(batch_size, self.args['perceiver_num_latents'], vision_feat.shape[-1]).to(device)
        indication_latent = torch.zeros(batch_size, self.args['perceiver_num_latents'], vision_feat.shape[-1]).to(
            device)
        if len(knowledge_idx) != 0:
            knowledge_idx = np.array(knowledge_idx)
            context_context = [batch['knowledge'][i] for i in knowledge_idx]
            # 1: encode clinical context
            context_latent = self.get_shared_latent('context', len(knowledge_idx))
            context_embed = self.get_text_features(context_context, device)
            context_latent = self.perceiver(context_embed, latent=context_latent)
            # 2: using context_compact to guide image perceiver
            know_visual_latent = self.perceiver(vision_feat[knowledge_idx], latent=context_latent)
            visual_latent[knowledge_idx] = know_visual_latent
            indication_latent[knowledge_idx] = context_latent
        if len(non_knowledge_idx) != 0:
            non_knowledge_idx = np.array(non_knowledge_idx)
            # 1: using shared latent for image-only path
            image_latent = self.get_shared_latent('image', len(non_knowledge_idx))
            non_visual_latent = self.perceiver(vision_feat[non_knowledge_idx], latent=image_latent)
            visual_latent[non_knowledge_idx] = non_visual_latent
            indication_latent[non_knowledge_idx] = image_latent

        # =================== compute disease classification logits =======================
        vision_logits = self.vision_classifier(visual_latent)
        targets = torch.tensor(batch['disease_labels'], dtype=torch.long, device=device)
        # obtain report features
        report_feat = self.get_text_features(batch['align_report'], device)
        report_logits = self.report_classifier(report_feat)

        # ===== compute loss functions (instance-loss, kl-loss, and classification loss) ======
        # 1. compute instance-level cross-modal alignment loss
        instance_loss = self.global_alignment_loss(visual_latent.mean(dim=1), report_feat.mean(dim=1),
                                                   patient_ids=batch['patient_id'])
        loss_dict = {
            'loss': instance_loss,
            'align_loss': instance_loss,
        }

        # =========2. transcript-to-patch similarity=============
        # obtain transcript and heatmap
        transcript_feat, gaze_heatmap = [], []
        transcript_id = [*batch['transcript_sen_id'], *batch['transcript_report_id']]
        if batch['heatmap_image_sen'] is not None:
            # obtain transcript feat (obtain sentence and report transcript_feat in twice avoid waste memory)
            sen_transcript_feat = self.get_text_features(batch['transcript_sen'], device)
            report_transcript_feat = self.get_text_features(batch['transcript_report'], device)
            transcript_feat.append(sen_transcript_feat[:, 0, :])
            transcript_feat.append(report_transcript_feat[:, 0, :])
            transcript_feat = torch.cat(transcript_feat, dim=0)
            # obtain heatmaps
            gaze_heatmap = torch.cat([batch['heatmap_image_sen'], batch['heatmap_image_report']], dim=0)

        if len(transcript_feat) != 0:
            kl_loss = self.soft_gaze_guidance(vision_feat, transcript_feat, gaze_heatmap,
                                              batch['dicom_id'], transcript_id,
                                              self.args['topk_ratio'])

            loss_dict['kl_loss'] = kl_loss
            loss_dict['loss'] = loss_dict['loss'] + kl_loss

        # 2. compute image-based and text-based classifier loss
        vision_loss = self.multi_label_loss(vision_logits, targets)
        loss_dict['img_disease_loss'] = vision_loss
        loss_dict['loss'] = loss_dict['loss'] + 0.5 * vision_loss

        # 2. compute image-based and text-based classifier loss
        report_loss = self.multi_label_loss(report_logits, targets)
        loss_dict['text_disease_loss'] = report_loss
        loss_dict['loss'] = loss_dict['loss'] + 0.5 * report_loss

        if mode != 'train':
            loss_dict.update({
                'report_logits': report_logits,
                'vision_logits': vision_logits,
                'targets': targets
            })
        return loss_dict

    def training_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training-step
        """
        batch_size = len(batch['image'])
        loss_dict = self(batch, mode='train')

        self.log_dict({f'tra_step_{k}': v for k, v in loss_dict.items()}, on_step=True, on_epoch=False,
                      batch_size=batch_size, prog_bar=True, sync_dist=True)
        if batch_idx % self.args['log_every_n_steps'] == 0 or batch_idx + 1 == self.trainer.num_training_batches:
            cur_loss_item = ''
            with torch.no_grad():
                cur_loss_item += ', '.join([f"{k} = {round(v.detach().cpu().item(), 2)}" for k, v in loss_dict.items()])
            self.log_once(
                f"Epoch {self.current_epoch}, training step {batch_idx}/{self.trainer.num_training_batches}, "
                f"{cur_loss_item}, lr: {self.optimizers().param_groups[0]['lr']}")

        # update loss through mean_metric
        for key, loss in loss_dict.items():
            # if f"{key}" in self.train_loss_metric:
            self.train_loss_metric[f"{key}"].update(loss.detach())
        # Update and log scores for each validation metric:
        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation-step
        """
        # Inference:
        batch_size = len(batch['image'])
        loss_dict = self(batch, mode='val')
        vision_logits = loss_dict['vision_logits']
        report_logits = loss_dict['report_logits']
        disease_labels = loss_dict['targets']
        del loss_dict['vision_logits'], loss_dict['report_logits'], loss_dict['targets']
        # Logging:
        self.log_dict({f'val_step_{k}': v for k, v in loss_dict.items()}, on_epoch=False, on_step=True,
                      batch_size=batch_size, prog_bar=False, sync_dist=True)

        if batch_idx % self.args['log_every_n_steps'] == 0 or batch_idx + 1 == self.trainer.num_val_batches[0]:
            cur_loss_item = ''
            with torch.no_grad():
                cur_loss_item += ', '.join([f"{k} = {round(v.detach().item(), 2)}" for k, v in loss_dict.items()])
            self.log_once(
                f"Epoch {self.current_epoch}, validation step {batch_idx}/{self.trainer.num_val_batches[0]}, "
                f"{cur_loss_item}, lr: {self.optimizers().param_groups[0]['lr']}")

        for key, loss in loss_dict.items():
            self.val_loss_metric[f"{key}"].update(loss)

        # probability-based metrics
        self.report_auroc.update(report_logits, disease_labels)
        self.report_ap.update(report_logits, disease_labels)

        # label-based metrics
        self.report_recall.update(report_logits, disease_labels)
        self.report_f1.update(report_logits, disease_labels)
        self.report_specificity.update(report_logits, disease_labels)
        self.report_acc.update(report_logits, disease_labels)

        # probability-based metrics
        self.vision_auroc.update(vision_logits, disease_labels)
        self.vision_ap.update(vision_logits, disease_labels)

        # label-based metrics
        self.vision_recall.update(vision_logits, disease_labels)
        self.vision_f1.update(vision_logits, disease_labels)
        self.vision_specificity.update(vision_logits, disease_labels)
        self.vision_acc.update(vision_logits, disease_labels)

    def test_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#test-step
        """
        # Inference:
        batch_size = len(batch['image'])
        loss_dict = self(batch, mode='test')
        vision_logits = loss_dict['vision_logits']
        report_logits = loss_dict['report_logits']
        disease_labels = loss_dict['targets']
        del loss_dict['vision_logits'], loss_dict['report_logits'], loss_dict['targets']

        # Logging:
        self.log_dict({f'test_step_{k}': v for k, v in loss_dict.items()}, on_epoch=False, on_step=True,
                      batch_size=batch_size, prog_bar=True, sync_dist=True)
        if batch_idx % self.args['log_every_n_steps'] == 0 or batch_idx + 1 == self.trainer.num_test_batches[0]:
            cur_loss_item = ''
            with torch.no_grad():
                cur_loss_item += ', '.join([f"{k} = {round(v.detach().item(), 2)}" for k, v in loss_dict.items()])
            self.log_once(
                f"Epoch {self.current_epoch}, testing step {batch_idx}/{self.trainer.num_test_batches[0]}, {cur_loss_item}")
        for key, loss in loss_dict.items():
            if f"{key}" in self.test_loss_metric:
                self.test_loss_metric[f"{key}"].update(loss)

        # probability-based metrics
        self.report_auroc.update(report_logits, disease_labels)
        self.report_ap.update(report_logits, disease_labels)

        # label-based metrics
        self.report_recall.update(report_logits, disease_labels)
        self.report_f1.update(report_logits, disease_labels)
        self.report_specificity.update(report_logits, disease_labels)
        self.report_acc.update(report_logits, disease_labels)

        # probability-based metrics
        self.vision_auroc.update(vision_logits, disease_labels)
        self.vision_ap.update(vision_logits, disease_labels)

        # label-based metrics
        self.vision_recall.update(vision_logits, disease_labels)
        self.vision_f1.update(vision_logits, disease_labels)
        self.vision_specificity.update(vision_logits, disease_labels)
        self.vision_acc.update(vision_logits, disease_labels)

    def on_train_epoch_end(self):
        cur_all_loss = {}
        for key, metric in self.train_loss_metric.items():
            avg_metric = metric.compute()
            metric.reset()
            cur_all_loss[key] = avg_metric
        self.log_dict({f'train_epoch_{k}': v for k, v in cur_all_loss.items()}, on_epoch=True,
                      on_step=False, prog_bar=False)

        cur_loss_item = ', '.join([f"{k} = {round(v.item(), 2)}" for k, v in cur_all_loss.items()])
        self.log_once(
            f"Epoch {self.current_epoch}, Training is over, "
            f"{cur_loss_item}, lr: {self.optimizers().param_groups[0]['lr']}"
            "\n###############################################################"
        )

    def on_validation_epoch_end(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#on-validation-epoch-end
        """
        cur_all_loss = {}
        for key, metric in self.val_loss_metric.items():
            avg_metric = metric.compute()
            metric.reset()
            cur_all_loss[key] = avg_metric

        metrics = {
            'report_ap': self.report_ap,
            'vision_ap': self.vision_ap,
            'report_auroc': self.report_auroc,
            'vision_auroc': self.vision_auroc,
            'report_acc': self.report_acc,
            'vision_acc': self.vision_acc,
            'report_f1': self.report_f1,
            'vision_f1': self.vision_f1,
            'report_recall': self.report_recall,
            'vision_recall': self.vision_recall,
            'report_specificity': self.report_specificity,
            'vision_specificity': self.vision_specificity,
        }
        for name, metric in metrics.items():
            avg_metric = metric.compute()
            metric.reset()
            cur_all_loss[name] = torch.round(avg_metric, decimals=3)

        self.log_dict({f'val_epoch_{k}': v for k, v in cur_all_loss.items()}, on_epoch=True, on_step=False,
                      prog_bar=False)

        if cur_all_loss['loss'] < self.val_best_scores['best_monitor_metric']:
            self.val_best_scores = {
                "best_epoch": self.current_epoch,
                'best_monitor_metric': cur_all_loss['loss']
            }
            if self.args['save_best_model']:
                self.save_finetune_checkpoint('best')
        if self.args['save_last_model']:
            self.save_finetune_checkpoint('last')

        cur_loss_item = ', '.join([f"{k} = {round(v.item(), 3)}" for k, v in cur_all_loss.items()])
        self.log_once(
            "###############################################################\n"
            f"Epoch {self.current_epoch}, Validation is over, current val loss:"
            f"{cur_loss_item}, lr: {self.optimizers().param_groups[0]['lr']}\n"
            f"best epoch {self.val_best_scores['best_epoch']}, and optimal loss: {self.val_best_scores['best_monitor_metric']}\n"
        )

    def on_test_epoch_end(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#on-test-epoch-end
        """
        cur_all_loss = {}
        for key, metric in self.test_loss_metric.items():
            avg_metric = metric.compute()
            metric.reset()
            cur_all_loss[key] = avg_metric

        metrics = {
            'report_ap': self.report_ap,
            'vision_ap': self.vision_ap,
            'report_auroc': self.report_auroc,
            'vision_auroc': self.vision_auroc,
            'report_acc': self.report_acc,
            'vision_acc': self.vision_acc,
            'report_f1': self.report_f1,
            'vision_f1': self.vision_f1,
            'report_recall': self.report_recall,
            'vision_recall': self.vision_recall,
            'report_specificity': self.report_specificity,
            'vision_specificity': self.vision_specificity,
        }
        for name, metric in metrics.items():
            avg_metric = metric.compute()
            metric.reset()
            cur_all_loss[name] = torch.round(avg_metric, decimals=3)

        self.log_dict({f'test_epoch_{k}': v for k, v in cur_all_loss.items()}, on_epoch=True, on_step=False,
                      prog_bar=False)

        cur_loss_item = ', '.join([f"{k} = {round(v.item(), 3)}" for k, v in cur_all_loss.items()])
        self.log_once(
            "###############################################################\n"
            f"Epoch {self.current_epoch}, test is over, current loss: {cur_loss_item}\n"
        )

    def save_finetune_checkpoint(self, status):
        state_dict = {}
        for name, para in self.named_parameters():
            if para.requires_grad:
                state_dict[name] = para
        checkpoint = {
            'state_dict': state_dict,
            'optimizer_state': self.trainer.optimizers[0].state_dict(),
            'epoch': self.current_epoch
        }
        torch.save(checkpoint, f'{self.args["ckpt_dir"]}/best_model.pt')
        print(f"The {status} model is saved on epoch {self.current_epoch}!")


class ReportGeneration(pl.LightningModule):
    def __init__(
            self,
            args: Dict,
            logger,
    ):
        super().__init__()
        self.args = args
        self.mylog = logger
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.prefetch_factor = args['prefetch_factor']

        self.val_best_scores = {
            "best_epoch": -1,
            "best_monitor_metric": -1.0,
        }

        self.train_loss_metric = torchmetrics.MeanMetric()
        self.val_coco_metrics = COCOCaptionMetrics(metrics=["bleu", "cider", "rouge", "meteor"])
        self.test_coco_metrics = COCOCaptionMetrics(metrics=["bleu", "cider", "rouge", "meteor"], save=False)

        self.val_f1chexbert_metrics = F1CheXbertMetrics(
            chexbert_path=args['chexbert_path'],
            model_path=args['bert_path'],
            mbatch_size=args['test_batch_size'],
            exp_dir=args['exp_dir_trial'],
        )
        self.test_f1chexbert_metrics = F1CheXbertMetrics(
            chexbert_path=args['chexbert_path'],
            model_path=args['bert_path'],
            mbatch_size=args['test_batch_size'],
            exp_dir=args['exp_dir_trial'],
        )
        # Radgraph metrics:
        self.val_radgraph_metrics = RadGraphMetrics(
            radgraph_path=self.args['radgraph_path'],
            mbatch_size=args['test_batch_size'],
            exp_dir=args['exp_dir_trial'],
        )
        self.test_radgraph_metrics = RadGraphMetrics(
            radgraph_path=self.args['radgraph_path'],
            mbatch_size=args['test_batch_size'],
            exp_dir=args['exp_dir_trial'],
        )
        # Report logging:
        self.val_report_logger = ReportLogger(exp_dir=args['exp_dir_trial'], split='val_reports')
        self.test_report_logger = ReportLogger(exp_dir=args['exp_dir_trial'], split='test_reports')

        print("load model ...")

        # ==================define image encoder and text encoder =================
        # image encoder
        self.image_encoder = AutoModel.from_pretrained(args['rad_dino_path'], trust_remote_code=True)
        self.image_processor = AutoImageProcessor.from_pretrained(args['rad_dino_path'], trust_remote_code=True)
        self.image_encoder.eval()
        self.freeze_parameters(self.image_encoder)

        # text encoder
        self.tokenizer = GPT2TokenizerFast.from_pretrained(self.args['distilgpt2_path'], trust_remote_code=True)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]', 'sep_token': '[SEP]', 'cls_token': '[CLS]'})
        self.tokenizer.add_tokens(['[INDICATION]', '[HISTORY]', '[Similar Cases]', '[FINDINGS]', '[TRANSCRIPT]'])
        self.text_encoder = self.build_text_encoder()
        self.text_encoder.train()

        # ========================define pos_embed, projector, and classifier=============================
        decoder_config = AutoConfig.from_pretrained(self.args['distilgpt2_path'], trust_remote_code=True)
        hidden_size = decoder_config.hidden_size
        image_dim = self.image_encoder.config.hidden_size
        text_dim = self.text_encoder.config.hidden_size

        # define projector
        self.image_projector = VisionProjectorMLP(image_dim, hidden_size * 2, hidden_size, args['view_position_path'])
        self.text_projector = TextProjectorMLP(text_dim, hidden_size * 2, hidden_size)

        # ==============define prior-guided uni-modal features ============================
        self.perceiver = Perceiver(
            byte_dim=hidden_size,  # byte array dimension
            depth=args['perceiver_num_blocks'],
            # depth of net. depth * (cross attention -> self_per_cross_attn * self attention)
            num_latents=args['perceiver_num_latents'],  # number of latents
            latent_dim=hidden_size,  # latent dimension
            cross_heads=8,  # number of heads for cross attention. paper said 1
            latent_heads=8,  # number of heads for latent self attention, 8
            cross_dim_head=64,  # number of dimensions per cross attention head
            latent_dim_head=64,  # number of dimensions per latent self attention head
            attn_dropout=0.,
            ff_dropout=0.,
            weight_tie_layers=False,  # whether to weight tie layers (optional, as indicated in the diagram)
            self_per_cross_attn=1  # number of self attention blocks per cross attention
        )

        self.shared_latent = nn.Parameter(torch.randn(args['perceiver_num_latents'], hidden_size), requires_grad=True)
        self.latent_type_embed = nn.ParameterDict({
            'context': nn.Parameter(torch.randn(1, hidden_size), requires_grad=True),
            'image': nn.Parameter(torch.randn(1, hidden_size), requires_grad=True),
        })
        # ===========================define text decoder =================================
        self.language_model = self.build_text_decoder()

        print("finish loading model ...")

    def finetune_parameters(self, model):
        for para in model.parameters():
            para.requires_grad = True

    def freeze_parameters(self, model):
        for para in model.parameters():
            para.requires_grad = False

    def build_text_encoder(self):
        enc_config = AutoConfig.from_pretrained(self.args['cxr_bert_path'], trust_remote_code=True)
        enc_config.vocab_size = len(self.tokenizer)
        enc_config.eos_token_id = self.tokenizer.eos_token_id
        enc_config.bos_token_id = self.tokenizer.bos_token_id
        enc_config.pad_token_id = self.tokenizer.pad_token_id
        enc_config.num_hidden_layers = self.args['text_encoder_num_blocks']
        enc_config.max_length = 200
        return AutoModel.from_pretrained(
            self.args['cxr_bert_path'],
            config=enc_config,
            ignore_mismatched_sizes=True,
            trust_remote_code=True)

    def build_text_decoder(self):
        config = transformers.GPT2Config.from_pretrained(self.args['distilgpt2_path'])
        config.add_cross_attention = True
        config.is_decoder = True
        config.vocab_size = len(self.tokenizer)
        decoder = transformers.GPT2LMHeadModel(config=config)
        # Resize GPT2 embedding to include padding and beginning of sentence token:
        decoder.resize_token_embeddings(len(self.tokenizer))

        class DummyEncoder:
            main_input_name = 'dummy'

            class DummyConfig(PretrainedConfig):
                model_type = 'bert'

            config = DummyConfig()

            def __init__(self, hidden_size):
                self.config.hidden_size = hidden_size

            def forward(self, *args, **kwargs):
                pass

            def get_output_embeddings(cls):
                return None

        # Use Hugging Face Transformers EncoderDecoderModel to generate conditionally:
        dummy_encoder = DummyEncoder(hidden_size=decoder.config.hidden_size)

        # To be compatible with previous the framework (and hence, the available checkpoint):
        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder_decoder = transformers.EncoderDecoderModel(encoder=dummy_encoder, decoder=decoder)

        return Decoder()

    @rank_zero_only
    def log_once(self, message):
        self.mylog.info(message)

    def setup(self, stage=None):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#setup
        """
        eos_token = self.tokenizer.eos_token
        if stage == 'fit' or stage is None:
            # construct train_set and val_set
            self.train_set = GenerationDataset(self.args, 'train')
            self.val_set = GenerationDataset(self.args, 'val')

            print("No. of training & validation examples: {} & {}.".format(
                self.train_set.__len__(), self.val_set.__len__()))
            self.log_once("No. of training & validation examples: {} & {}.".format(
                self.train_set.__len__(), self.val_set.__len__()))
        if stage == "test" or stage is None:
            self.test_set = GenerationDataset(self.args, 'test')
            print("No. of test examples: {}.".format(self.test_set.__len__()))
            self.log_once("No. of test examples: {}.".format(self.test_set.__len__()))

    def train_dataloader(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#train-dataloader
        """
        pad_token = self.tokenizer.pad_token
        bos_token, eos_token = self.tokenizer.bos_token, self.tokenizer.eos_token
        collate_fn = GenerationCollateFn(self.args, self.image_processor, pad_token, bos_token, eos_token)
        return DataLoader(
            self.train_set,
            batch_size=self.args['batch_size'],
            num_workers=self.args['num_workers'],
            shuffle=True,
            prefetch_factor=self.prefetch_factor,
            collate_fn=collate_fn,
            drop_last=True,
        )

    def val_dataloader(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#val-dataloader
        """
        pad_token = self.tokenizer.pad_token
        bos_token, eos_token = self.tokenizer.bos_token, self.tokenizer.eos_token
        collate_fn = GenerationCollateFn(self.args, self.image_processor, pad_token, bos_token, eos_token)
        return DataLoader(
            self.val_set,
            batch_size=self.args['test_batch_size'],
            num_workers=self.args['num_workers'],
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
            collate_fn=collate_fn,
            drop_last=False
        )

    def test_dataloader(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#test-dataloader
        """
        pad_token = self.tokenizer.pad_token
        bos_token, eos_token = self.tokenizer.bos_token, self.tokenizer.eos_token
        collate_fn = GenerationCollateFn(self.args, self.image_processor, pad_token, bos_token, eos_token)
        return DataLoader(
            self.test_set,
            batch_size=self.args['test_batch_size'],
            num_workers=self.args['num_workers'],
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
            collate_fn=collate_fn,
            drop_last=False,
        )

    def configure_optimizers(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        """
        lang_parameters, other_parameters = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'language_model' in name:
                lang_parameters.append(param)
            else:
                other_parameters.append(param)
        optimiser = torch.optim.AdamW(
            [{'params': lang_parameters, 'lr': self.args['learning_rate'] * 10},
             {'params': other_parameters, 'lr': self.args['learning_rate']}])

        # update optimizer
        if self.optimizer_state_dict:
            optimiser.load_state_dict(self.optimizer_state_dict)
        del self.optimizer_state_dict
        scheduler = ReduceLROnPlateau(optimiser, mode=self.args['monitor_mode'],
                                      factor=0.1, patience=self.args['patience'])
        return {
            "optimizer": optimiser,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self.args['monitor_metric'],
                'frequency': 1,  # the frequency of check
                'interval': 'epoch'
            }
        }

    def get_text_features(self, text, device):
        tokens = self.tokenizer(text, padding=True, return_tensors='pt',
                                max_length=self.args['encoder_max_length'], truncation=True)
        tokens['input_ids'] = tokens['input_ids'].to(device)
        tokens['attention_mask'] = tokens['attention_mask'].to(device)
        text_embed = self.text_encoder(input_ids=tokens['input_ids'],
                                       attention_mask=tokens['attention_mask'])['last_hidden_state']
        text_embed = self.text_projector(text_embed)
        return text_embed

    def get_vision_features(self, images, view_positions):
        with torch.no_grad():
            image_embed = self.image_encoder(images)['last_hidden_state']
        # add view positional embedding projector
        image_embed = self.image_projector(image_embed, view_positions)
        return image_embed

    def get_shared_latent(self, mode: str, num: int):
        type_embed = self.latent_type_embed[mode]  # (1, d)
        latent = self.shared_latent + type_embed  # (l, d)
        return latent.unsqueeze(0).repeat(num, 1, 1)  # (b, l, d)

    def build_batch_prompt(self, device, responses, is_targets=True):
        tokens = self.tokenizer(responses, padding=True, return_tensors='pt',
                                max_length=200, truncation=True)
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)

        if is_targets:
            attention_mask = attention_mask[:, :-1]  # string + [eos]
            # initialize targets
            targets = input_ids[:, 1:].detach().clone().to(device)
            targets = targets.masked_fill(
                targets == self.tokenizer.pad_token_id, -100
            )  # mask padding

            input_ids = input_ids.detach()[:, :-1]
            input_ids[input_ids == self.tokenizer.sep_token_id] = self.tokenizer.pad_token_id
        else:
            targets = None

        return input_ids, attention_mask, targets

    def generate(self, encoder_outputs):
        """
        Autoregressive generate a prediction.

        Argument/s:
            num_beams - number of considered beams for the search (one beam is a greedy search).
            images - images for the encoder.

        Returns:
            Indices of the tokens for the predicted sequence.
        """
        with torch.no_grad():
            outputs = self.language_model.encoder_decoder.generate(
                max_length=self.args['max_length'],
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                num_beams=self.args['num_beams'],
                return_dict_in_generate=True,
                use_cache=True,
                encoder_outputs=encoder_outputs,
            )

        return outputs['sequences']

    def obtain_reference_reports(self, text):
        inputs = self.tokenizer(text, padding=True, max_length=self.args['max_length'],
                                truncation=True, return_tensors='pt')
        ref_reports = self.tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
        # delete illegal characters
        ref_reports = [re.sub(r'[^\x20-\x7E]', '', report.strip()) for report in ref_reports]
        return ref_reports

    def forward(self, batch, mode='train'):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#forward
        """
        device = batch['image'].device
        batch_size = len(batch['image'])

        # =================== extract uni-modal features =======================
        # obtain vision features
        vision_feat = self.get_vision_features(batch['image'], batch['view_position'])  # (b, 1370, 768)

        # obtain report and knowledge features
        knowledge_idx = [i for i, k in enumerate(batch['knowledge']) if k != self.tokenizer.pad_token]
        non_knowledge_idx = [i for i in range(batch_size) if i not in knowledge_idx]

        visual_latent = torch.zeros(batch_size, self.args['perceiver_num_latents'], vision_feat.shape[-1]).to(device)
        indication_latent = torch.zeros(batch_size, self.args['perceiver_num_latents'], vision_feat.shape[-1]).to(
            device)
        if len(knowledge_idx) != 0:
            knowledge_idx = np.array(knowledge_idx)
            context_context = [batch['knowledge'][i] for i in knowledge_idx]
            # 1: encode clinical context
            context_latent = self.get_shared_latent('context', len(knowledge_idx))
            context_embed = self.get_text_features(context_context, device)
            context_latent = self.perceiver(context_embed, latent=context_latent)
            # 2: using context_compact to guide image perceiver
            know_visual_latent = self.perceiver(vision_feat[knowledge_idx], latent=context_latent)
            visual_latent[knowledge_idx] = know_visual_latent
            indication_latent[knowledge_idx] = context_latent
        if len(non_knowledge_idx) != 0:
            non_knowledge_idx = np.array(non_knowledge_idx)
            # 1: using shared latent for image-only path
            image_latent = self.get_shared_latent('image', len(non_knowledge_idx))
            non_visual_latent = self.perceiver(vision_feat[non_knowledge_idx], latent=image_latent)
            visual_latent[non_knowledge_idx] = non_visual_latent
            indication_latent[non_knowledge_idx] = image_latent

        # ===== compute loss function (language modeling loss) ======
        # obtain input_embed
        encoder_outputs = torch.cat([visual_latent, indication_latent], dim=1)
        encoder_outputs = transformers.modeling_outputs.BaseModelOutput(last_hidden_state=encoder_outputs)

        if mode == 'train':
            # concatenation embeddings, attention_mask, and targets
            input_ids, attention_mask, targets = self.build_batch_prompt(device, batch['report'], is_targets=True)
            outputs = self.language_model.encoder_decoder(
                decoder_input_ids=input_ids,
                decoder_attention_mask=attention_mask,
                encoder_outputs=encoder_outputs,
                return_dict=True,
                labels=targets
            )
            return {"loss": outputs['loss']}
        else:
            outputs = self.generate(encoder_outputs)
            generated_reports = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # delete illegal characters
            generated_reports = [re.sub(r'[^\x20-\x7E]', '', report.strip()) for report in generated_reports]
            return generated_reports

    def training_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training-step
        """
        # Inference:
        batch_size = len(batch['image'])
        loss_dict = self(batch)

        self.log_dict({f'tra_step_{k}': v for k, v in loss_dict.items()}, on_step=True, on_epoch=False,
                      batch_size=batch_size, prog_bar=True, sync_dist=True)
        if batch_idx % self.args['log_every_n_steps'] == 0 or batch_idx + 1 == self.trainer.num_training_batches:
            cur_loss_item = ''
            with torch.no_grad():
                cur_loss_item += ', '.join([f"{k} = {round(v.detach().cpu().item(), 2)}" for k, v in loss_dict.items()])
            self.log_once(
                f"Epoch {self.current_epoch}, training step {batch_idx}/{self.trainer.num_training_batches}, "
                f"{cur_loss_item}, lr: {self.optimizers().param_groups[0]['lr']}")

        # update loss through mean_metric
        for key, loss in loss_dict.items():
            self.train_loss_metric[key].update(loss.detach())
        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation-step
        """
        # Inference:
        generated_reports = self(batch, mode='val')
        if self.current_epoch == 0:
            generated_reports = [text + " findings." for text in generated_reports]
        else:
            generated_reports = [text if len(text) > 0 else "There is no findings." for text in generated_reports]
        reference_reports = self.obtain_reference_reports(batch['report'])  # remove special tokens

        if batch_idx % self.args['log_every_n_steps'] == 0 or batch_idx + 1 == self.trainer.num_val_batches[0]:
            self.log_once(
                f"Epoch {self.current_epoch}, validation step {batch_idx}/{self.trainer.num_val_batches[0]}")

        # # Log reports:
        dicom_ids = batch['dicom_id']
        self.val_report_logger.update(generated_reports, dicom_ids=dicom_ids, reference_reports=reference_reports)

        # # Evaluate:
        self.val_f1chexbert_metrics.update(generated_reports, reference_reports, ids=dicom_ids)
        self.val_coco_metrics.update(generated_reports, reference_reports, ids=dicom_ids)
        self.val_radgraph_metrics.update(generated_reports, reference_reports, ids=dicom_ids)

    def test_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#test-step
        """
        generated_reports = self(batch, mode='test')

        # generated_reports = [text if len(text) > 0 else "..." for text in generated_reports]
        # reference_reports = self.obtain_reference_reports(batch['report'])  # remove special tokens
        #
        # if batch_idx % self.args['log_every_n_steps'] == 0 or batch_idx + 1 == self.trainer.num_test_batches[0]:
        #     self.log_once(
        #         f"Epoch {self.current_epoch}, test step {batch_idx}/{self.trainer.num_test_batches[0]}")
        #
        # # # Log reports:
        # dicom_ids = batch['dicom_id']
        # self.test_report_logger.update(generated_reports, dicom_ids=dicom_ids, reference_reports=reference_reports)
        #
        # # # Evaluate:
        # self.test_f1chexbert_metrics.update(generated_reports, reference_reports, ids=dicom_ids)
        # self.test_coco_metrics.update(generated_reports, reference_reports, ids=dicom_ids)
        # self.test_radgraph_metrics.update(generated_reports, reference_reports, ids=dicom_ids)

    def on_train_epoch_end(self):
        epoch_loss = self.train_loss_metric.compute()
        self.train_loss_metric.reset()
        self.log_once(
            f"Epoch {self.current_epoch}, Training is over, "
            f"training epoch loss = {epoch_loss}, lr: {self.optimizers().param_groups[0]['lr']}"
            "\n###############################################################"
        )

    def on_validation_epoch_end(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#on-validation-epoch-end
        """
        # Save reports:
        self.val_report_logger.compute(self.current_epoch)
        self.val_report_logger.reset()

        scores = {}
        # F1-radgraph
        output = self.val_radgraph_metrics.compute()
        scores.update(output)
        self.val_radgraph_metrics.reset()

        # chexbert
        output = self.val_f1chexbert_metrics.compute()
        scores.update(output)
        self.val_f1chexbert_metrics.reset()

        output = self.val_coco_metrics.compute()
        scores.update(output)
        self.val_coco_metrics.reset()

        # scores['RB'] = scores['F1-Radgraph-partial'] + scores['chen_bleu_4']
        # scores['RC'] = scores['F1-Radgraph-partial'] + scores['chexbert_all_micro_f1']
        scores['RCB'] = scores['F1-Radgraph-partial'] + scores['chen_bleu_4'] + scores['chexbert_all_micro_f1']

        self.log_dict({f'{k}': v for k, v in scores.items()}, on_step=False, on_epoch=True)

        if scores[self.args['monitor_metric']] > self.val_best_scores['best_monitor_metric']:
            self.val_best_scores = {
                "best_epoch": self.current_epoch,
                'best_monitor_metric': scores[self.args['monitor_metric']]
            }
            if self.args['save_best_model']:
                self.save_finetune_checkpoint('best')
        if self.args['save_last_model']:
            self.save_finetune_checkpoint('last')

        metrics_item = '\n'.join([f'{k}: {v}' for k, v in scores.items()])
        self.log_once(
            "###############################################################\n"
            f"Epoch {self.current_epoch}, Validation is over, current metrics:\n"
            f"best validation epoch: {self.val_best_scores['best_epoch']}, "
            f"best val_metrics: {self.args['monitor_metric']} = {self.val_best_scores['best_monitor_metric']}\n"
            f"{metrics_item} \n"
        )

    def on_test_epoch_end(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#on-test-epoch-end
        """

        # Save reports:
        self.test_report_logger.log(1)
        self.test_report_logger.compute(self.current_epoch)
        self.test_report_logger.reset()

        scores = {}
        output = self.test_radgraph_metrics.compute()
        scores.update(output)
        self.test_radgraph_metrics.reset()

        output = self.test_f1chexbert_metrics.compute()
        scores.update(output)
        self.test_f1chexbert_metrics.reset()

        output = self.test_coco_metrics.compute()
        scores.update(output)
        self.test_coco_metrics.reset()

        scores['RB'] = scores['F1-Radgraph-partial'] + scores['chen_bleu_4']
        scores['RC'] = scores['F1-Radgraph-partial'] + scores['chexbert_all_micro_f1']
        scores['RCB'] = scores['F1-Radgraph-partial'] + scores['chen_bleu_4'] + scores['chexbert_all_micro_f1']

        print('\n')
        print(scores)

        self.log_dict({f'{k}': v for k, v in scores.items()}, on_step=False, on_epoch=True)
        metrics_item = '\n'.join([f'{k}: {v}' for k, v in scores.items()])
        self.log_once(
            "###############################################################\n"
            f"test is over, current metrics:"
            f"{metrics_item} \n"
        )

    def save_finetune_checkpoint(self, status):
        state_dict = {}
        for name, para in self.named_parameters():
            if para.requires_grad:
                state_dict[name] = para
        checkpoint = {
            'state_dict': state_dict,
            'optimizer_state': self.trainer.optimizers[0].state_dict(),
            'epoch': self.current_epoch
        }
        torch.save(checkpoint, f'{self.args["ckpt_dir"]}/best_model.pt')
        print(f"The {status} model is saved on epoch {self.current_epoch}!")
        self.log_once(f"The {status} model is saved on epoch {self.current_epoch}!")



class ReportGenerationLLM(pl.LightningModule):
    def __init__(
            self,
            args: Dict,
            logger,
    ):
        super().__init__()
        self.args = args
        self.mylog = logger
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.prefetch_factor = args['prefetch_factor']

        self.val_best_scores = {
            "best_epoch": -1,
            "best_monitor_metric": -1.0,
        }

        self.train_loss_metric = torchmetrics.MeanMetric()
        self.val_coco_metrics = COCOCaptionMetrics(metrics=["bleu", "cider", "rouge", "meteor"])
        self.test_coco_metrics = COCOCaptionMetrics(metrics=["bleu", "cider", "rouge", "meteor"], save=False)

        self.val_f1chexbert_metrics = F1CheXbertMetrics(
            chexbert_path=args['chexbert_path'],
            model_path=args['bert_path'],
            mbatch_size=args['test_batch_size'],
            exp_dir=args['exp_dir_trial'],
        )
        self.test_f1chexbert_metrics = F1CheXbertMetrics(
            chexbert_path=args['chexbert_path'],
            model_path=args['bert_path'],
            mbatch_size=args['test_batch_size'],
            exp_dir=args['exp_dir_trial'],
        )
        # Radgraph metrics:
        self.val_radgraph_metrics = RadGraphMetrics(
            radgraph_path=self.args['radgraph_path'],
            mbatch_size=args['test_batch_size'],
            exp_dir=args['exp_dir_trial'],
        )
        self.test_radgraph_metrics = RadGraphMetrics(
            radgraph_path=self.args['radgraph_path'],
            mbatch_size=args['test_batch_size'],
            exp_dir=args['exp_dir_trial'],
        )
        # Report logging:
        self.val_report_logger = ReportLogger(exp_dir=args['exp_dir_trial'], split='val_reports')
        self.test_report_logger = ReportLogger(exp_dir=args['exp_dir_trial'], split='test_reports')

        print("load model ...")

        # ==================define image encoder and text encoder =================
        # image encoder
        self.image_encoder = AutoModel.from_pretrained(args['rad_dino_path'], trust_remote_code=True)
        self.image_processor = AutoImageProcessor.from_pretrained(args['rad_dino_path'], trust_remote_code=True)
        self.image_encoder.eval()
        self.freeze_parameters(self.image_encoder)

        # text encoder
        self.tokenizer = GPT2TokenizerFast.from_pretrained(self.args['distilgpt2_path'], trust_remote_code=True)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]', 'sep_token': '[SEP]', 'cls_token': '[CLS]'})
        self.tokenizer.add_tokens(['[INDICATION]', '[HISTORY]', '[Similar Cases]', '[FINDINGS]', '[TRANSCRIPT]'])
        self.text_encoder = self.build_text_encoder()
        self.text_encoder.train()
        self.finetune_parameters(self.text_encoder)

        # ========================define pos_embed, projector, and classifier=============================
        decoder_config = AutoConfig.from_pretrained(self.args['distilgpt2_path'], trust_remote_code=True)
        hidden_size = decoder_config.hidden_size
        image_dim = self.image_encoder.config.hidden_size
        text_dim = self.text_encoder.config.hidden_size

        # define projector
        self.image_projector = VisionProjectorMLP(image_dim, hidden_size * 2, hidden_size, args['view_position_path'])
        self.text_projector = TextProjectorMLP(text_dim, hidden_size * 2, hidden_size)

        # ==============define prior-guided uni-modal features ============================
        self.perceiver = Perceiver(
            byte_dim=hidden_size,  # byte array dimension
            depth=args['perceiver_num_blocks'],
            # depth of net. depth * (cross attention -> self_per_cross_attn * self attention)
            num_latents=args['perceiver_num_latents'],  # number of latents
            latent_dim=hidden_size,  # latent dimension
            cross_heads=8,  # number of heads for cross attention. paper said 1
            latent_heads=8,  # number of heads for latent self attention, 8
            cross_dim_head=64,  # number of dimensions per cross attention head
            latent_dim_head=64,  # number of dimensions per latent self attention head
            attn_dropout=0.,
            ff_dropout=0.,
            weight_tie_layers=False,  # whether to weight tie layers (optional, as indicated in the diagram)
            self_per_cross_attn=1  # number of self attention blocks per cross attention
        )

        self.shared_latent = nn.Parameter(torch.randn(args['perceiver_num_latents'], hidden_size), requires_grad=True)
        self.latent_type_embed = nn.ParameterDict({
            'context': nn.Parameter(torch.randn(1, hidden_size), requires_grad=True),
            'image': nn.Parameter(torch.randn(1, hidden_size), requires_grad=True),
        })
        # ===========================define large language models =================================
        # llama 3.2-3B
        self.llm_tokenizer = AutoTokenizer.from_pretrained(args['llama_path'], trust_remote_code=True,
                                                           padding_side="right",
                                                           use_fast=False,
                                                           )
        special_token = "<|finetune_right_pad_id|>"
        pad_token_id = self.llm_tokenizer.convert_tokens_to_ids(special_token)
        self.llm_tokenizer.pad_token_id = pad_token_id
        self.llm = LlamaForCausalLM.from_pretrained(
            args['llama_path'],
            device_map='auto'
        )
        self.llm.eval()
        self.freeze_parameters(self.llm)
        self.conv = conv_gazerg_v0904
        self.embed_tokens = self.llm.get_input_embeddings()

        # define lora-config
        if args['llm_use_lora']:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=args['llm_r'],
                lora_alpha=args['llm_alpha'],
                target_modules=['q_proj', 'v_proj'],
                lora_dropout=args['lora_dropout'],
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llm = get_peft_model(self.llm, lora_config)

        self.adapter = nn.Linear(hidden_size, self.llm.config.hidden_size)

        print("finish loading model ...")

    def finetune_parameters(self, model):
        for para in model.parameters():
            para.requires_grad = True

    def freeze_parameters(self, model):
        for para in model.parameters():
            para.requires_grad = False

    def build_text_encoder(self):
        enc_config = AutoConfig.from_pretrained(self.args['cxr_bert_path'], trust_remote_code=True)
        enc_config.vocab_size = len(self.tokenizer)
        enc_config.eos_token_id = self.tokenizer.eos_token_id
        enc_config.bos_token_id = self.tokenizer.bos_token_id
        enc_config.pad_token_id = self.tokenizer.pad_token_id
        enc_config.num_hidden_layers = self.args['text_encoder_num_blocks']
        enc_config.max_length = 200
        return AutoModel.from_pretrained(
            self.args['cxr_bert_path'],
            config=enc_config,
            ignore_mismatched_sizes=True,
            trust_remote_code=True)

    @rank_zero_only
    def log_once(self, message):
        self.mylog.info(message)

    def setup(self, stage=None):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#setup
        """
        if stage == 'fit' or stage is None:
            # construct train_set and val_set
            self.train_set = GenerationDataset(self.args, 'train')
            self.val_set = GenerationDataset(self.args, 'val')

            print("No. of training & validation examples: {} & {}.".format(
                self.train_set.__len__(), self.val_set.__len__()))
            self.log_once("No. of training & validation examples: {} & {}.".format(
                self.train_set.__len__(), self.val_set.__len__()))
        if stage == "test" or stage is None:
            self.test_set = GenerationDataset(self.args, 'test')
            print("No. of test examples: {}.".format(self.test_set.__len__()))
            self.log_once("No. of test examples: {}.".format(self.test_set.__len__()))

    def train_dataloader(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#train-dataloader
        """
        pad_token = self.tokenizer.pad_token
        collate_fn = GenerationLLMCollateFn(self.args, self.image_processor, pad_token)
        return DataLoader(
            self.train_set,
            batch_size=self.args['batch_size'],
            num_workers=self.args['num_workers'],
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
        )

    def val_dataloader(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#val-dataloader
        """
        pad_token = self.tokenizer.pad_token
        collate_fn = GenerationLLMCollateFn(self.args, self.image_processor, pad_token)
        return DataLoader(
            self.val_set,
            batch_size=self.args['test_batch_size'],
            num_workers=self.args['num_workers'],
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False
        )

    def test_dataloader(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#test-dataloader
        """
        pad_token = self.tokenizer.pad_token
        collate_fn = GenerationLLMCollateFn(self.args, self.image_processor, pad_token)
        return DataLoader(
            self.test_set,
            batch_size=self.args['test_batch_size'],
            num_workers=self.args['num_workers'],
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False,
        )

    def configure_optimizers(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        """
        lang_parameters, other_parameters = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'adapter' in name:
                lang_parameters.append(param)
            else:
                other_parameters.append(param)
        optimiser = torch.optim.AdamW(
            [{'params': lang_parameters, 'lr': self.args['learning_rate'] * 5},
             {'params': other_parameters, 'lr': self.args['learning_rate']}])
        scheduler = ReduceLROnPlateau(optimiser, mode=self.args['monitor_mode'],
                                      factor=0.1, patience=self.args['patience'])
        return {
            "optimizer": optimiser,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self.args['monitor_metric'],
                'frequency': 1,  # the frequency of check
                'interval': 'epoch'
            }
        }

    def get_text_features(self, text, device):
        tokens = self.tokenizer(text, padding=True, return_tensors='pt',
                                max_length=self.args['encoder_max_length'], truncation=True)
        tokens['input_ids'] = tokens['input_ids'].to(device)
        tokens['attention_mask'] = tokens['attention_mask'].to(device)
        text_embed = self.text_encoder(input_ids=tokens['input_ids'],
                                       attention_mask=tokens['attention_mask'])['last_hidden_state']
        text_embed = self.text_projector(text_embed)
        return text_embed

    def get_vision_features(self, images, view_positions):
        with torch.no_grad():
            image_embed = self.image_encoder(images)['last_hidden_state']
        # add view positional embedding projector
        image_embed = self.image_projector(image_embed, view_positions)
        return image_embed

    def get_shared_latent(self, mode: str, num: int):
        type_embed = self.latent_type_embed[mode]  # (1, d)
        latent = self.shared_latent + type_embed  # (l, d)
        return latent.unsqueeze(0).repeat(num, 1, 1)  # (b, l, d)

    def prompt_image_wrap(self, img_embeds):
        """
        merge embeddings from image embeds and its before prompt
        Args:
            img_embeds: (b, seq_len, dim)
        Returns:
            prompt
        """
        batch_size = img_embeds.shape[0]
        conv = self.conv.copy()
        conv.append_message(conv.roles[0], 'Input:\n1. Image: <image>')
        prompt = conv.get_prompt()
        p_before = prompt.split('<image>')[0]
        tokens = self.llm_tokenizer(p_before, return_tensors='pt')
        input_ids = tokens['input_ids'].to(img_embeds.device)
        p_before_embeds = self.embed_tokens(input_ids).expand(batch_size, -1, -1)

        wrapped_img_embeds = torch.cat([
            p_before_embeds,
            img_embeds,
        ], dim=1)
        # expand只能广播维度为1的，所以atts_img[:, :1]
        wrapped_atts_img = torch.ones(wrapped_img_embeds.size()[:-1], dtype=torch.long).to(img_embeds.device)

        return wrapped_img_embeds, wrapped_atts_img

    def build_batch_prompt_simple(self, device, diseases, similar_cases, responses=None, is_targets=True):
        # diseases: [dict1, dict2]
        prompts, prompts_prefix = [], []
        if responses is None:
            responses = [None] * len(similar_cases)

        if similar_cases is not None:
            # ensure similar cases max_length is self.args['max_length']
            self.llm_tokenizer.truncation_side = 'left'
            similar_cases_token = self.llm_tokenizer(similar_cases, padding=True, return_tensors='pt',
                                                     max_length=self.args['max_length'], truncation=True)
            similar_cases = self.llm_tokenizer.batch_decode(similar_cases_token['input_ids'], skip_special_tokens=True)
            self.llm_tokenizer.truncation_side = 'right'
        for disease, similar_case, response in zip(diseases, similar_cases, responses):
            disease_str = ', '.join(map(str, disease))
            similar_case = similar_case.strip()
            query = f'<image>\n2. Similar report: {similar_case}\n'
            query += f"3. Predicted diseases: {disease_str}\n"
            query += f"Instructions:\n- Use the Similar report as a draft.\n"
            query += f'- Revise based on the Image to ensure accuracy.\n'
            query += f'- Refer to Predicted diseases only if supported by the Image.\n'
            query += f'Output only the Findings section in formal radiology style.'

            # the string before response
            conv = self.conv.copy()
            conv.append_message(conv.roles[0], query)
            conv.append_message(conv.roles[1], response)
            prompt = conv.get_prompt()
            valid_prompt = prompt.split('<image>')[-1]
            valid_prompt_prefix = valid_prompt.split('<|start_header_id|>assistant<|end_header_id|>')[
                                      0] + '<|start_header_id|>assistant<|end_header_id|>\n\n'
            prompts.append(valid_prompt)
            prompts_prefix.append(valid_prompt_prefix)
        # the fist token is <begin_of_text>
        tokens = self.llm_tokenizer(prompts, padding=True, return_tensors='pt',
                                    max_length=400, truncation=True)
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)
        p_after_embeds = self.embed_tokens(input_ids)

        if is_targets:
            # initialize targets
            targets = input_ids.detach().to(device)
            targets = targets.masked_fill(
                targets == self.llm_tokenizer.pad_token_id, -100
            )  # mask padding
            prefix_encoding = self.llm_tokenizer(prompts_prefix, padding=True, return_tensors='pt',
                                                 max_length=400, truncation=True)
            prefix_lengths = prefix_encoding['attention_mask'].sum(dim=1)
            for i, l in enumerate(prefix_lengths):
                targets[i, :l] = -100  # right-shift
            targets = targets[:, 1:]   # the first token is bos_token
        else:
            targets = None
        return p_after_embeds[:, 1:], attention_mask[:, 1:], targets  # the first token is bos_token

    def obtain_reference_reports(self, text):
        inputs = self.llm_tokenizer(text, padding=True, max_length=self.args['max_length'],
                                    truncation=True, return_tensors='pt')
        ref_reports = self.llm_tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
        # delete illegal characters
        ref_reports = [re.sub(r'[^\x20-\x7E]', '', report.strip()) for report in ref_reports]
        return ref_reports

    def forward(self, batch, mode='train'):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#forward
        """
        device = batch['image'].device
        batch_size = len(batch['image'])
        disease_prediction = batch['disease_prediction']
        similar_case = batch['similar_case']

        # =================== extract uni-modal features =======================
        # obtain vision features
        vision_feat = self.get_vision_features(batch['image'], batch['view_position'])  # (b, 1370, 768)

        # obtain report and knowledge features
        knowledge_idx = [i for i, k in enumerate(batch['knowledge']) if k != self.tokenizer.pad_token]
        non_knowledge_idx = [i for i in range(batch_size) if i not in knowledge_idx]

        visual_latent = torch.zeros(batch_size, self.args['perceiver_num_latents'], vision_feat.shape[-1]).to(device)
        indication_latent = torch.zeros(batch_size, self.args['perceiver_num_latents'], vision_feat.shape[-1]).to(
            device)
        if len(knowledge_idx) != 0:
            knowledge_idx = np.array(knowledge_idx)
            context_context = [batch['knowledge'][i] for i in knowledge_idx]
            # 1: encode clinical context
            context_latent = self.get_shared_latent('context', len(knowledge_idx))
            context_embed = self.get_text_features(context_context, device)
            context_latent = self.perceiver(context_embed, latent=context_latent)
            # 2: using context_compact to guide image perceiver
            know_visual_latent = self.perceiver(vision_feat[knowledge_idx], latent=context_latent)
            visual_latent[knowledge_idx] = know_visual_latent
            indication_latent[knowledge_idx] = context_latent
        if len(non_knowledge_idx) != 0:
            non_knowledge_idx = np.array(non_knowledge_idx)
            # 1: using shared latent for image-only path
            image_latent = self.get_shared_latent('image', len(non_knowledge_idx))
            non_visual_latent = self.perceiver(vision_feat[non_knowledge_idx], latent=image_latent)
            visual_latent[non_knowledge_idx] = non_visual_latent
            indication_latent[non_knowledge_idx] = image_latent

        # ===== compute loss function (language modeling loss) ======
        # obtain input_embed
        encoder_outputs = torch.cat([visual_latent, indication_latent], dim=1)
        encoder_outputs = self.adapter(encoder_outputs)
        p_before_embeds, before_atts = self.prompt_image_wrap(encoder_outputs)

        if mode == 'train':
            # concatenation embeddings, attention_mask, and targets
            p_after_embeds, after_atts, after_targets = self.build_batch_prompt_simple(device, disease_prediction,
                                                                                       similar_case, batch['report'],
                                                                                       is_targets=True)
            before_targets = torch.ones_like(before_atts, dtype=torch.long).to(device).fill_(-100)

            targets = torch.cat([before_targets, after_targets], dim=1)
            inputs_embeds = torch.cat([p_before_embeds, p_after_embeds], dim=1)
            attention_mask = torch.cat([before_atts, after_atts], dim=1)

            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets
            )
            return {
                'loss': outputs.loss,
            }
        else:
            p_after_embeds, after_atts, _ = self.build_batch_prompt_simple(device, disease_prediction,
                                                                           similar_case, None,
                                                                           is_targets=False)
            inputs_embeds = torch.cat([p_before_embeds, p_after_embeds], dim=1)
            attention_mask = torch.cat([before_atts, after_atts], dim=1)
            stop_str = self.conv.sep if self.conv.sep_style not in {SeparatorStyle.TWO, SeparatorStyle.LLAMA_3,
                                                                    SeparatorStyle.MISTRAL} else self.conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.llm_tokenizer, inputs_embeds.shape[1])
            with torch.no_grad(), autocast(dtype=torch.float16):  # more suitable for inference phase
                output_ids = self.llm.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    do_sample=True if self.args['temperature'] > 0 else False,
                    temperature=self.args['temperature'],
                    num_beams=self.args['num_beams'],
                    min_new_tokens=self.args['min_new_tokens'],
                    max_new_tokens=self.args['max_new_tokens'],
                    use_cache=True,
                    pad_token_id=self.llm_tokenizer.pad_token_id,
                    stopping_criteria=[stopping_criteria],
                )

            generated_reports = self.llm_tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            # delete illegal characters
            generated_reports = [re.sub(r'[^\x20-\x7E]', '', report.strip()) for report in generated_reports]
            return generated_reports

    def training_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training-step
        """
        # Inference:
        batch_size = len(batch['image'])
        loss_dict = self(batch)

        self.log_dict({f'tra_step_{k}': v for k, v in loss_dict.items()}, on_step=True, on_epoch=False,
                      batch_size=batch_size, prog_bar=True, sync_dist=True)
        if batch_idx % self.args['log_every_n_steps'] == 0 or batch_idx + 1 == self.trainer.num_training_batches:
            cur_loss_item = ''
            with torch.no_grad():
                cur_loss_item += ', '.join([f"{k} = {round(v.detach().cpu().item(), 2)}" for k, v in loss_dict.items()])
            self.log_once(
                f"Epoch {self.current_epoch}, training step {batch_idx}/{self.trainer.num_training_batches}, "
                f"{cur_loss_item}, lr: {self.optimizers().param_groups[0]['lr']}")

        # update loss through mean_metric
        for key, loss in loss_dict.items():
            # if f"{key}" in self.train_loss_metric:
            self.train_loss_metric[f"{key}"].update(loss.detach())
        # Update and log scores for each validation metric:
        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation-step
        """
        # Inference:
        generated_reports = self(batch, mode='val')
        if self.current_epoch == 0:
            generated_reports = [text + " There is no findings." for text in generated_reports]
        else:
            generated_reports = [text if len(text) > 0 else "There is no findings." for text in generated_reports]
        reference_reports = self.obtain_reference_reports(batch['report'])  # remove special tokens

        if batch_idx % self.args['log_every_n_steps'] == 0 or batch_idx + 1 == self.trainer.num_val_batches[0]:
            self.log_once(
                f"Epoch {self.current_epoch}, validation step {batch_idx}/{self.trainer.num_val_batches[0]}")

        # # Log reports:
        dicom_ids = batch['dicom_id']
        self.val_report_logger.update(generated_reports, dicom_ids=dicom_ids, reference_reports=reference_reports)

        # # Evaluate:
        self.val_f1chexbert_metrics.update(generated_reports, reference_reports, ids=dicom_ids)
        self.val_coco_metrics.update(generated_reports, reference_reports, ids=dicom_ids)
        self.val_radgraph_metrics.update(generated_reports, reference_reports, ids=dicom_ids)

    def test_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#test-step
        """
        generated_reports = self(batch, mode='test')

        generated_reports = [text if len(text) > 0 else "..." for text in generated_reports]
        reference_reports = self.obtain_reference_reports(batch['report'])  # remove special tokens

        if batch_idx % self.args['log_every_n_steps'] == 0 or batch_idx + 1 == self.trainer.num_test_batches[0]:
            self.log_once(
                f"Epoch {self.current_epoch}, test step {batch_idx}/{self.trainer.num_test_batches[0]}")

        # # Log reports:
        dicom_ids = batch['dicom_id']
        self.test_report_logger.update(generated_reports, dicom_ids=dicom_ids, reference_reports=reference_reports)

        # # Evaluate:
        self.test_f1chexbert_metrics.update(generated_reports, reference_reports, ids=dicom_ids)
        self.test_coco_metrics.update(generated_reports, reference_reports, ids=dicom_ids)
        self.test_radgraph_metrics.update(generated_reports, reference_reports, ids=dicom_ids)

    def on_train_epoch_end(self):
        epoch_loss = self.train_loss_metric.compute()
        self.train_loss_metric.reset()
        self.log_once(
            f"Epoch {self.current_epoch}, Training is over, "
            f"training epoch loss = {epoch_loss}, lr: {self.optimizers().param_groups[0]['lr']}"
            "\n###############################################################"
        )

    def on_validation_epoch_end(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#on-validation-epoch-end
        """
        # Save reports:
        self.val_report_logger.compute(self.current_epoch)
        self.val_report_logger.reset()
        #
        scores = {}
        # F1-radgraph
        output = self.val_radgraph_metrics.compute()
        scores.update(output)
        self.val_radgraph_metrics.reset()

        # chexbert
        output = self.val_f1chexbert_metrics.compute()
        scores.update(output)
        self.val_f1chexbert_metrics.reset()

        output = self.val_coco_metrics.compute()
        scores.update(output)
        self.val_coco_metrics.reset()

        # scores['RB'] = scores['F1-Radgraph-partial'] + scores['chen_bleu_4']
        scores['RCB4'] = scores['F1-Radgraph-partial'] + scores['chen_bleu_4'] + scores['chexbert_all_micro_f1']
        scores['RCB'] = scores['F1-Radgraph-partial'] + scores['chen_bleu_2'] + scores['chexbert_all_micro_f1']
        # scores['RCB'] = scores['chen_bleu_4'] + scores['chexbert_all_micro_f1']

        self.log_dict({f'{k}': v for k, v in scores.items()}, on_step=False, on_epoch=True)

        if scores[self.args['monitor_metric']] > self.val_best_scores['best_monitor_metric']:
            self.val_best_scores = {
                "best_epoch": self.current_epoch,
                'best_monitor_metric': scores[self.args['monitor_metric']]
            }
            if self.args['save_best_model']:
                self.save_finetune_checkpoint('best')
        if self.args['save_last_model']:
            self.save_finetune_checkpoint('last')

        metrics_item = '\n'.join([f'{k}: {v}' for k, v in scores.items()])
        self.log_once(
            "###############################################################\n"
            f"Epoch {self.current_epoch}, Validation is over, current metrics:\n"
            f"best validation epoch: {self.val_best_scores['best_epoch']}, "
            f"best val_metrics: {self.args['monitor_metric']} = {self.val_best_scores['best_monitor_metric']}\n"
            f"{metrics_item} \n"
        )

    def on_test_epoch_end(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#on-test-epoch-end
        """

        # Save reports:
        self.test_report_logger.log(1)
        self.test_report_logger.compute(self.current_epoch)
        self.test_report_logger.reset()

        scores = {}
        output = self.test_radgraph_metrics.compute()
        scores.update(output)
        self.test_radgraph_metrics.reset()

        output = self.test_f1chexbert_metrics.compute()
        scores.update(output)
        self.test_f1chexbert_metrics.reset()

        output = self.test_coco_metrics.compute()
        scores.update(output)
        self.test_coco_metrics.reset()

        # scores['RB'] = scores['F1-Radgraph-partial'] + scores['chen_bleu_4']
        # scores['RC'] = scores['F1-Radgraph-partial'] + scores['chexbert_all_micro_f1']
        scores['RCB'] = scores['F1-Radgraph-partial'] + scores['chen_bleu_4'] + scores['chexbert_all_micro_f1']
        # scores['RCB'] = scores['chen_bleu_4'] + scores['chexbert_all_micro_f1']

        print('\n')
        print(scores)

        self.log_dict({f'{k}': v for k, v in scores.items()}, on_step=False, on_epoch=True)
        metrics_item = '\n'.join([f'{k}: {v}' for k, v in scores.items()])
        self.log_once(
            "###############################################################\n"
            f"test is over, current metrics:"
            f"{metrics_item} \n"
        )

    def save_finetune_checkpoint(self, status):
        state_dict = {}
        for name, para in self.named_parameters():
            if 'image_encoder' in name or 'llm' in name or 'embed_tokens' in name:
                continue
            state_dict[name] = para
        lora_state_dict, lora_config = None, None
        if hasattr(self, 'llm') and self.args['llm_use_lora']:
            from peft import get_peft_model_state_dict
            lora_state_dict = get_peft_model_state_dict(self.llm)
            lora_config = self.llm.peft_config['default'].to_dict()
        checkpoint = {
            'state_dict': state_dict,
            'lora_state_dict': lora_state_dict,
            'lora_config': lora_config,
            'optimizer_state': self.trainer.optimizers[0].state_dict(),
            'epoch': self.current_epoch
        }
        torch.save(checkpoint, f'{self.args["ckpt_dir"]}/{status}_model.pt')
        print(f"The {status} model is saved on epoch {self.current_epoch}!")


class ReportGenerationLoRA(pl.LightningModule):
    def __init__(
            self,
            args: Dict,
            logger,
    ):
        super().__init__()
        self.args = args
        self.mylog = logger
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.prefetch_factor = args['prefetch_factor']

        self.val_best_scores = {
            "best_epoch": -1,
            "best_monitor_metric": -1.0,
        }

        self.train_loss_metric = torchmetrics.MeanMetric()
        self.val_coco_metrics = COCOCaptionMetrics(metrics=["bleu", "cider", "rouge", "meteor"])
        self.test_coco_metrics = COCOCaptionMetrics(metrics=["bleu", "cider", "rouge", "meteor"], save=False)

        self.val_f1chexbert_metrics = F1CheXbertMetrics(
            chexbert_path=args['chexbert_path'],
            model_path=args['bert_path'],
            mbatch_size=args['test_batch_size'],
            exp_dir=args['exp_dir_trial'],
        )
        self.test_f1chexbert_metrics = F1CheXbertMetrics(
            chexbert_path=args['chexbert_path'],
            model_path=args['bert_path'],
            mbatch_size=args['test_batch_size'],
            exp_dir=args['exp_dir_trial'],
        )
        # Radgraph metrics:
        self.val_radgraph_metrics = RadGraphMetrics(
            radgraph_path=self.args['radgraph_path'],
            mbatch_size=args['test_batch_size'],
            exp_dir=args['exp_dir_trial'],
        )
        self.test_radgraph_metrics = RadGraphMetrics(
            radgraph_path=self.args['radgraph_path'],
            mbatch_size=args['test_batch_size'],
            exp_dir=args['exp_dir_trial'],
        )
        # Report logging:
        self.val_report_logger = ReportLogger(exp_dir=args['exp_dir_trial'], split='val_reports')
        self.test_report_logger = ReportLogger(exp_dir=args['exp_dir_trial'], split='test_reports')

        print("load model ...")

        # ==================define image encoder and text encoder =================
        # image encoder
        self.image_encoder = AutoModel.from_pretrained(args['rad_dino_path'], trust_remote_code=True)
        self.image_processor = AutoImageProcessor.from_pretrained(args['rad_dino_path'], trust_remote_code=True)
        self.image_encoder.eval()
        self.freeze_parameters(self.image_encoder)

        # text encoder
        self.tokenizer = GPT2TokenizerFast.from_pretrained(self.args['distilgpt2_path'], trust_remote_code=True)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]', 'sep_token': '[SEP]', 'cls_token': '[CLS]'})
        self.tokenizer.add_tokens(['[INDICATION]', '[HISTORY]', '[Similar Cases]', '[FINDINGS]', '[TRANSCRIPT]'])
        self.text_encoder = self.build_text_encoder()
        self.text_encoder.eval()
        self.freeze_parameters(self.text_encoder)

        # ========================define pos_embed, projector, and classifier=============================
        decoder_config = AutoConfig.from_pretrained(self.args['distilgpt2_path'], trust_remote_code=True)
        hidden_size = decoder_config.hidden_size
        image_dim = self.image_encoder.config.hidden_size
        text_dim = self.text_encoder.config.hidden_size

        # define projector
        self.image_projector = VisionProjectorMLP(image_dim, hidden_size * 2, hidden_size, args['view_position_path'])
        self.text_projector = TextProjectorMLP(text_dim, hidden_size * 2, hidden_size)
        self.image_projector.eval()
        self.freeze_parameters(self.image_projector)
        self.text_projector.eval()
        self.freeze_parameters(self.text_projector)

        # ==============define prior-guided uni-modal features ============================
        self.perceiver = Perceiver(
            byte_dim=hidden_size,  # byte array dimension
            depth=args['perceiver_num_blocks'],
            # depth of net. depth * (cross attention -> self_per_cross_attn * self attention)
            num_latents=args['perceiver_num_latents'],  # number of latents
            latent_dim=hidden_size,  # latent dimension
            cross_heads=8,  # number of heads for cross attention. paper said 1
            latent_heads=8,  # number of heads for latent self attention, 8
            cross_dim_head=64,  # number of dimensions per cross attention head
            latent_dim_head=64,  # number of dimensions per latent self attention head
            attn_dropout=0.,
            ff_dropout=0.,
            weight_tie_layers=False,  # whether to weight tie layers (optional, as indicated in the diagram)
            self_per_cross_attn=1  # number of self attention blocks per cross attention
        )
        self.perceiver.eval()
        self.freeze_parameters(self.perceiver)

        self.shared_latent = nn.Parameter(torch.randn(args['perceiver_num_latents'], hidden_size), requires_grad=False)
        self.latent_type_embed = nn.ParameterDict({
            'context': nn.Parameter(torch.randn(1, hidden_size), requires_grad=False),
            'image': nn.Parameter(torch.randn(1, hidden_size), requires_grad=False),
        })
        # ===========================define large language models =================================
        # llama 3.2-3B
        self.llm_tokenizer = AutoTokenizer.from_pretrained(args['llama_path'], trust_remote_code=True,
                                                           padding_side="right",
                                                           use_fast=False,
                                                           )
        special_token = "<|finetune_right_pad_id|>"
        pad_token_id = self.llm_tokenizer.convert_tokens_to_ids(special_token)
        self.llm_tokenizer.pad_token_id = pad_token_id
        self.llm = LlamaForCausalLM.from_pretrained(
            args['llama_path'],
            device_map='auto'
        )
        # define lora-config
        if args['llm_use_lora']:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=args['llm_r'],
                lora_alpha=args['llm_alpha'],
                target_modules=['q_proj', 'v_proj'],  # 'k_proj', 'o_proj'
                lora_dropout=args['lora_dropout'],
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llm = get_peft_model(self.llm, lora_config)
            self.llm.train()
        else:
            self.llm.eval()
            self.freeze_parameters(self.llm)
        self.conv = conv_gazerg_v0904
        self.embed_tokens = self.llm.get_input_embeddings()
        self.adapter = nn.Linear(hidden_size, self.llm.config.hidden_size)

        print("finish loading model ...")

    def finetune_parameters(self, model):
        for para in model.parameters():
            para.requires_grad = True

    def freeze_parameters(self, model):
        for para in model.parameters():
            para.requires_grad = False

    def build_text_encoder(self):
        enc_config = AutoConfig.from_pretrained(self.args['cxr_bert_path'], trust_remote_code=True)
        enc_config.vocab_size = len(self.tokenizer)
        enc_config.eos_token_id = self.tokenizer.eos_token_id
        enc_config.bos_token_id = self.tokenizer.bos_token_id
        enc_config.pad_token_id = self.tokenizer.pad_token_id
        enc_config.num_hidden_layers = self.args['text_encoder_num_blocks']
        enc_config.max_length = 200
        return AutoModel.from_pretrained(
            self.args['cxr_bert_path'],
            config=enc_config,
            ignore_mismatched_sizes=True,
            trust_remote_code=True)

    @rank_zero_only
    def log_once(self, message):
        self.mylog.info(message)

    def setup(self, stage=None):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#setup
        """
        if stage == 'fit' or stage is None:
            # construct train_set and val_set
            self.train_set = GenerationDataset(self.args, 'train')
            self.val_set = GenerationDataset(self.args, 'val')

            print("No. of training & validation examples: {} & {}.".format(
                self.train_set.__len__(), self.val_set.__len__()))
            self.log_once("No. of training & validation examples: {} & {}.".format(
                self.train_set.__len__(), self.val_set.__len__()))
        if stage == "test" or stage is None:
            self.test_set = GenerationDataset(self.args, 'test')
            print("No. of test examples: {}.".format(self.test_set.__len__()))
            self.log_once("No. of test examples: {}.".format(self.test_set.__len__()))

    def train_dataloader(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#train-dataloader
        """
        pad_token = self.tokenizer.pad_token
        collate_fn = GenerationLLMCollateFn(self.args, self.image_processor, pad_token)
        return DataLoader(
            self.train_set,
            batch_size=self.args['batch_size'],
            num_workers=self.args['num_workers'],
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
        )

    def val_dataloader(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#val-dataloader
        """
        pad_token = self.tokenizer.pad_token
        collate_fn = GenerationLLMCollateFn(self.args, self.image_processor, pad_token)
        return DataLoader(
            self.val_set,
            batch_size=self.args['test_batch_size'],
            num_workers=self.args['num_workers'],
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False
        )

    def test_dataloader(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#test-dataloader
        """
        pad_token = self.tokenizer.pad_token
        collate_fn = GenerationLLMCollateFn(self.args, self.image_processor, pad_token)
        return DataLoader(
            self.test_set,
            batch_size=self.args['test_batch_size'],
            num_workers=self.args['num_workers'],
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False,
        )

    def configure_optimizers(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        """
        adapter_parameters = list(self.adapter.parameters())
        lora_parameters = [p for p in self.llm.parameters() if p.requires_grad]

        optimiser = torch.optim.AdamW(
            [{'params': adapter_parameters, 'lr': self.args['learning_rate'] / 5},
             {'params': lora_parameters, 'lr': self.args['learning_rate']}])
        scheduler = ReduceLROnPlateau(optimiser, mode=self.args['monitor_mode'],
                                      factor=0.1, patience=self.args['patience'])
        return {
            "optimizer": optimiser,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self.args['monitor_metric'],
                'frequency': 1,  # the frequency of check
                'interval': 'epoch'
            }
        }

    def get_text_features(self, text, device):
        tokens = self.tokenizer(text, padding=True, return_tensors='pt',
                                max_length=self.args['encoder_max_length'], truncation=True)
        tokens['input_ids'] = tokens['input_ids'].to(device)
        tokens['attention_mask'] = tokens['attention_mask'].to(device)
        text_embed = self.text_encoder(input_ids=tokens['input_ids'],
                                       attention_mask=tokens['attention_mask'])['last_hidden_state']
        text_embed = self.text_projector(text_embed)
        return text_embed

    def get_vision_features(self, images, view_positions):
        with torch.no_grad():
            image_embed = self.image_encoder(images)['last_hidden_state']
        # add view positional embedding projector
        image_embed = self.image_projector(image_embed, view_positions)
        return image_embed

    def get_shared_latent(self, mode: str, num: int):
        type_embed = self.latent_type_embed[mode]  # (1, d)
        latent = self.shared_latent + type_embed  # (l, d)
        return latent.unsqueeze(0).repeat(num, 1, 1)  # (b, l, d)

    def prompt_image_wrap(self, img_embeds):
        """
        merge embeddings from image embeds and its before prompt
        Args:
            img_embeds: (b, seq_len, dim)
        Returns:
            prompt
        """
        batch_size = img_embeds.shape[0]
        conv = self.conv.copy()
        conv.append_message(conv.roles[0], 'Input:\n1. Image: <image>')
        prompt = conv.get_prompt()
        p_before = prompt.split('<image>')[0]
        tokens = self.llm_tokenizer(p_before, return_tensors='pt')
        input_ids = tokens['input_ids'].to(img_embeds.device)
        p_before_embeds = self.embed_tokens(input_ids).expand(batch_size, -1, -1)

        wrapped_img_embeds = torch.cat([
            p_before_embeds,
            img_embeds,
        ], dim=1)
        # expand只能广播维度为1的，所以atts_img[:, :1]
        wrapped_atts_img = torch.ones(wrapped_img_embeds.size()[:-1], dtype=torch.long).to(img_embeds.device)

        return wrapped_img_embeds, wrapped_atts_img

    def build_batch_prompt_simple(self, device, diseases, similar_cases, responses=None, is_targets=True):
        # diseases: [dict1, dict2]
        prompts, prompts_prefix = [], []
        if responses is None:
            responses = [None] * len(similar_cases)

        if similar_cases is not None:
            # ensure similar cases max_length is self.args['max_length']
            self.llm_tokenizer.truncation_side = 'left'
            similar_cases_token = self.llm_tokenizer(similar_cases, padding=True, return_tensors='pt',
                                                     max_length=self.args['max_length'], truncation=True)
            similar_cases = self.llm_tokenizer.batch_decode(similar_cases_token['input_ids'], skip_special_tokens=True)
            self.llm_tokenizer.truncation_side = 'right'
        for disease, similar_case, response in zip(diseases, similar_cases, responses):
            disease_str = ', '.join(map(str, disease))
            similar_case = similar_case.strip()
            query = f'<image>\n2. Similar report: {similar_case}\n'
            query += f"3. Predicted diseases: {disease_str}\n"
            query += f"Instructions:\n- Use the Similar report as a draft.\n"
            query += f'- Revise based on the Image to ensure accuracy.\n'
            query += f'- Refer to Predicted diseases only if supported by the Image.\n'
            query += f'Output only the Findings section in formal radiology style.'

            # the string before response
            conv = self.conv.copy()
            conv.append_message(conv.roles[0], query)
            conv.append_message(conv.roles[1], response)
            prompt = conv.get_prompt()
            valid_prompt = prompt.split('<image>')[-1]
            valid_prompt_prefix = valid_prompt.split('<|start_header_id|>assistant<|end_header_id|>')[
                                      0] + '<|start_header_id|>assistant<|end_header_id|>\n\n'
            prompts.append(valid_prompt)
            prompts_prefix.append(valid_prompt_prefix)
        # the fist token is <begin_of_text>
        tokens = self.llm_tokenizer(prompts, padding=True, return_tensors='pt',
                                    max_length=400, truncation=True)
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)
        p_after_embeds = self.embed_tokens(input_ids)

        if is_targets:
            # initialize targets
            targets = input_ids.detach().to(device)
            targets = targets.masked_fill(
                targets == self.llm_tokenizer.pad_token_id, -100
            )  # mask padding
            prefix_encoding = self.llm_tokenizer(prompts_prefix, padding=True, return_tensors='pt',
                                                 max_length=400, truncation=True)
            prefix_lengths = prefix_encoding['attention_mask'].sum(dim=1)
            for i, l in enumerate(prefix_lengths):
                targets[i, :l] = -100  # right-shift
            targets = targets[:, 1:]   # the first token is bos_token
        else:
            targets = None
        return p_after_embeds[:, 1:], attention_mask[:, 1:], targets  # the first token is bos_token

    def obtain_reference_reports(self, text):
        inputs = self.llm_tokenizer(text, padding=True, max_length=self.args['max_length'],
                                    truncation=True, return_tensors='pt')
        ref_reports = self.llm_tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
        # delete illegal characters
        ref_reports = [re.sub(r'[^\x20-\x7E]', '', report.strip()) for report in ref_reports]
        return ref_reports

    def forward(self, batch, mode='train'):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#forward
        """
        device = batch['image'].device
        batch_size = len(batch['image'])
        disease_prediction = batch['disease_prediction']
        similar_case = batch['similar_case']

        # =================== extract uni-modal features =======================
        # obtain vision features
        vision_feat = self.get_vision_features(batch['image'], batch['view_position'])  # (b, 1370, 768)

        # obtain report and knowledge features
        knowledge_idx = [i for i, k in enumerate(batch['knowledge']) if k != self.tokenizer.pad_token]
        non_knowledge_idx = [i for i in range(batch_size) if i not in knowledge_idx]

        visual_latent = torch.zeros(batch_size, self.args['perceiver_num_latents'], vision_feat.shape[-1]).to(device)
        indication_latent = torch.zeros(batch_size, self.args['perceiver_num_latents'], vision_feat.shape[-1]).to(
            device)
        if len(knowledge_idx) != 0:
            knowledge_idx = np.array(knowledge_idx)
            context_context = [batch['knowledge'][i] for i in knowledge_idx]
            # 1: encode clinical context
            context_latent = self.get_shared_latent('context', len(knowledge_idx))
            context_embed = self.get_text_features(context_context, device)
            context_latent = self.perceiver(context_embed, latent=context_latent)
            # 2: using context_compact to guide image perceiver
            know_visual_latent = self.perceiver(vision_feat[knowledge_idx], latent=context_latent)
            visual_latent[knowledge_idx] = know_visual_latent
            indication_latent[knowledge_idx] = context_latent
        if len(non_knowledge_idx) != 0:
            non_knowledge_idx = np.array(non_knowledge_idx)
            # 1: using shared latent for image-only path
            image_latent = self.get_shared_latent('image', len(non_knowledge_idx))
            non_visual_latent = self.perceiver(vision_feat[non_knowledge_idx], latent=image_latent)
            visual_latent[non_knowledge_idx] = non_visual_latent
            indication_latent[non_knowledge_idx] = image_latent

        # ===== compute loss function (language modeling loss) ======
        # obtain input_embed
        encoder_outputs = torch.cat([visual_latent, indication_latent], dim=1)
        encoder_outputs = self.adapter(encoder_outputs)
        p_before_embeds, before_atts = self.prompt_image_wrap(encoder_outputs)

        if mode == 'train':
            # concatenation embeddings, attention_mask, and targets
            p_after_embeds, after_atts, after_targets = self.build_batch_prompt_simple(device, disease_prediction,
                                                                                       similar_case, batch['report'],
                                                                                       is_targets=True)
            before_targets = torch.ones_like(before_atts, dtype=torch.long).to(device).fill_(-100)

            targets = torch.cat([before_targets, after_targets], dim=1)
            inputs_embeds = torch.cat([p_before_embeds, p_after_embeds], dim=1)
            attention_mask = torch.cat([before_atts, after_atts], dim=1)

            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets
            )
            return {
                'loss': outputs.loss,
            }
        else:
            p_after_embeds, after_atts, _ = self.build_batch_prompt_simple(device, disease_prediction,
                                                                           similar_case, None,
                                                                           is_targets=False)
            inputs_embeds = torch.cat([p_before_embeds, p_after_embeds], dim=1)
            attention_mask = torch.cat([before_atts, after_atts], dim=1)
            stop_str = self.conv.sep if self.conv.sep_style not in {SeparatorStyle.TWO, SeparatorStyle.LLAMA_3,
                                                                    SeparatorStyle.MISTRAL} else self.conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.llm_tokenizer, inputs_embeds.shape[1])
            with torch.no_grad(), autocast(dtype=torch.float16):  # more suitable for inference phase
                output_ids = self.llm.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    do_sample=True if self.args['temperature'] > 0 else False,
                    temperature=self.args['temperature'],
                    num_beams=self.args['num_beams'],
                    min_new_tokens=self.args['min_new_tokens'],
                    max_new_tokens=self.args['max_new_tokens'],
                    use_cache=True,
                    pad_token_id=self.llm_tokenizer.pad_token_id,
                    stopping_criteria=[stopping_criteria],
                )

            generated_reports = self.llm_tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            # delete illegal characters
            generated_reports = [re.sub(r'[^\x20-\x7E]', '', report.strip()) for report in generated_reports]
            return generated_reports

    def training_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training-step
        """
        # Inference:
        batch_size = len(batch['image'])
        loss_dict = self(batch)

        self.log_dict({f'tra_step_{k}': v for k, v in loss_dict.items()}, on_step=True, on_epoch=False,
                      batch_size=batch_size, prog_bar=True, sync_dist=True)
        if batch_idx % self.args['log_every_n_steps'] == 0 or batch_idx + 1 == self.trainer.num_training_batches:
            cur_loss_item = ''
            with torch.no_grad():
                cur_loss_item += ', '.join([f"{k} = {round(v.detach().cpu().item(), 2)}" for k, v in loss_dict.items()])
            self.log_once(
                f"Epoch {self.current_epoch}, training step {batch_idx}/{self.trainer.num_training_batches}, "
                f"{cur_loss_item}, lr: {self.optimizers().param_groups[0]['lr']}")

        # update loss through mean_metric
        for key, loss in loss_dict.items():
            # if f"{key}" in self.train_loss_metric:
            self.train_loss_metric[f"{key}"].update(loss.detach())
        # Update and log scores for each validation metric:
        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation-step
        """
        # Inference:
        generated_reports = self(batch, mode='val')
        if self.current_epoch == 0:
            generated_reports = [text + " There is no findings." for text in generated_reports]
        else:
            generated_reports = [text if len(text) > 0 else "There is no findings." for text in generated_reports]
        reference_reports = self.obtain_reference_reports(batch['report'])  # remove special tokens

        if batch_idx % self.args['log_every_n_steps'] == 0 or batch_idx + 1 == self.trainer.num_val_batches[0]:
            self.log_once(
                f"Epoch {self.current_epoch}, validation step {batch_idx}/{self.trainer.num_val_batches[0]}")

        # # Log reports:
        dicom_ids = batch['dicom_id']
        self.val_report_logger.update(generated_reports, dicom_ids=dicom_ids, reference_reports=reference_reports)

        # # Evaluate:
        self.val_f1chexbert_metrics.update(generated_reports, reference_reports, ids=dicom_ids)
        self.val_coco_metrics.update(generated_reports, reference_reports, ids=dicom_ids)
        self.val_radgraph_metrics.update(generated_reports, reference_reports, ids=dicom_ids)

    def test_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#test-step
        """
        generated_reports = self(batch, mode='test')

        generated_reports = [text if len(text) > 0 else "..." for text in generated_reports]
        reference_reports = self.obtain_reference_reports(batch['report'])  # remove special tokens

        if batch_idx % self.args['log_every_n_steps'] == 0 or batch_idx + 1 == self.trainer.num_test_batches[0]:
            self.log_once(
                f"Epoch {self.current_epoch}, test step {batch_idx}/{self.trainer.num_test_batches[0]}")

        # # Log reports:
        dicom_ids = batch['dicom_id']
        self.test_report_logger.update(generated_reports, dicom_ids=dicom_ids, reference_reports=reference_reports)

        # # Evaluate:
        self.test_f1chexbert_metrics.update(generated_reports, reference_reports, ids=dicom_ids)
        self.test_coco_metrics.update(generated_reports, reference_reports, ids=dicom_ids)
        self.test_radgraph_metrics.update(generated_reports, reference_reports, ids=dicom_ids)

    def on_train_epoch_end(self):
        epoch_loss = self.train_loss_metric.compute()
        self.train_loss_metric.reset()
        self.log_once(
            f"Epoch {self.current_epoch}, Training is over, "
            f"training epoch loss = {epoch_loss}, lr: {self.optimizers().param_groups[0]['lr']}"
            "\n###############################################################"
        )

    def on_validation_epoch_end(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#on-validation-epoch-end
        """
        # Save reports:
        self.val_report_logger.compute(self.current_epoch)
        self.val_report_logger.reset()
        #
        scores = {}
        # F1-radgraph
        output = self.val_radgraph_metrics.compute()
        scores.update(output)
        self.val_radgraph_metrics.reset()

        # chexbert
        output = self.val_f1chexbert_metrics.compute()
        scores.update(output)
        self.val_f1chexbert_metrics.reset()

        output = self.val_coco_metrics.compute()
        scores.update(output)
        self.val_coco_metrics.reset()

        # scores['RB'] = scores['F1-Radgraph-partial'] + scores['chen_bleu_4']
        scores['RCB4'] = scores['F1-Radgraph-partial'] + scores['chen_bleu_4'] + scores['chexbert_all_micro_f1']
        scores['RCB'] = scores['F1-Radgraph-partial'] + scores['chen_bleu_2'] + scores['chexbert_all_micro_f1']
        # scores['RCB'] = scores['chen_bleu_4'] + scores['chexbert_all_micro_f1']

        self.log_dict({f'{k}': v for k, v in scores.items()}, on_step=False, on_epoch=True)

        if scores[self.args['monitor_metric']] > self.val_best_scores['best_monitor_metric']:
            self.val_best_scores = {
                "best_epoch": self.current_epoch,
                'best_monitor_metric': scores[self.args['monitor_metric']]
            }
            if self.args['save_best_model']:
                self.save_finetune_checkpoint('best')
        if self.args['save_last_model']:
            self.save_finetune_checkpoint('last')

        metrics_item = '\n'.join([f'{k}: {v}' for k, v in scores.items()])
        self.log_once(
            "###############################################################\n"
            f"Epoch {self.current_epoch}, Validation is over, current metrics:\n"
            f"best validation epoch: {self.val_best_scores['best_epoch']}, "
            f"best val_metrics: {self.args['monitor_metric']} = {self.val_best_scores['best_monitor_metric']}\n"
            f"{metrics_item} \n"
        )

    def on_test_epoch_end(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#on-test-epoch-end
        """

        # Save reports:
        self.test_report_logger.log(1)
        self.test_report_logger.compute(self.current_epoch)
        self.test_report_logger.reset()

        scores = {}
        output = self.test_radgraph_metrics.compute()
        scores.update(output)
        self.test_radgraph_metrics.reset()

        output = self.test_f1chexbert_metrics.compute()
        scores.update(output)
        self.test_f1chexbert_metrics.reset()

        output = self.test_coco_metrics.compute()
        scores.update(output)
        self.test_coco_metrics.reset()

        # scores['RB'] = scores['F1-Radgraph-partial'] + scores['chen_bleu_4']
        # scores['RC'] = scores['F1-Radgraph-partial'] + scores['chexbert_all_micro_f1']
        scores['RCB'] = scores['F1-Radgraph-partial'] + scores['chen_bleu_4'] + scores['chexbert_all_micro_f1']
        # scores['RCB'] = scores['chen_bleu_4'] + scores['chexbert_all_micro_f1']

        print('\n')
        print(scores)

        self.log_dict({f'{k}': v for k, v in scores.items()}, on_step=False, on_epoch=True)
        metrics_item = '\n'.join([f'{k}: {v}' for k, v in scores.items()])
        self.log_once(
            "###############################################################\n"
            f"test is over, current metrics:"
            f"{metrics_item} \n"
        )

    def save_finetune_checkpoint(self, status):
        state_dict = {}
        for name, para in self.named_parameters():
            if 'image_encoder' in name or 'llm' in name or 'embed_tokens' in name:
                continue
            state_dict[name] = para
        lora_state_dict, lora_config = None, None
        if hasattr(self, 'llm') and self.args['llm_use_lora']:
            from peft import get_peft_model_state_dict
            lora_state_dict = get_peft_model_state_dict(self.llm)
            lora_config = self.llm.peft_config['default'].to_dict()
        checkpoint = {
            'state_dict': state_dict,
            'lora_state_dict': lora_state_dict,
            'lora_config': lora_config,
            'optimizer_state': self.trainer.optimizers[0].state_dict(),
            'epoch': self.current_epoch
        }
        torch.save(checkpoint, f'{self.args["ckpt_dir"]}/{status}_model.pt')
        print(f"The {status} model is saved on epoch {self.current_epoch}!")

