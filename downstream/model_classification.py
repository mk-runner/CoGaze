import math
from typing import Optional, Dict
from einops import rearrange

import torch
import torchmetrics
from torchmetrics.classification import AUROC, Accuracy, AveragePrecision, Recall, Specificity, F1Score, Dice
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.utilities import rank_zero_only
from transformers import AutoModel, AutoConfig, AutoImageProcessor, GPT2TokenizerFast
from models.perceiver_pytorch import Perceiver

from models.utils import *
from models.class_balanced_loss import BalancedClassLoss, MultiLabelBalancedClassLoss
from downstream.datasets import *
from downstream.segmentation_loss import MixedLoss, MixedSIIMLoss


class Classifier(pl.LightningModule):
    def __init__(
            self,
            args: Dict,
            logger,
            num_classes: int = 2,
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
            "best_monitor": -1.0,
        }

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_loss_metric = {
            'loss': torchmetrics.MeanMetric().to(device),
        }
        if 'nih' not in args['data_name']:
            if args['data_name'] != 'cx-rsna':
                task = 'multiclass'
                num_labels = None
                cur_num_classes = 2
            else:
                task = 'multiclass'
                num_labels = None
                cur_num_classes = 3
        else:  # nih or cx-nih
            task = 'multilabel'
            num_labels = num_classes
            cur_num_classes = None
        # accuracy
        self.val_acc = Accuracy(task=task, num_labels=num_labels, num_classes=cur_num_classes)
        self.test_acc = Accuracy(task=task, num_labels=num_labels, num_classes=cur_num_classes)
        # auroc
        self.val_auroc = AUROC(task=task, num_labels=num_labels, num_classes=cur_num_classes)
        self.test_auroc = AUROC(task=task, num_labels=num_labels, num_classes=cur_num_classes)
        # PR-AUC equals AP
        self.val_ap = AveragePrecision(task=task, num_labels=num_labels, num_classes=cur_num_classes)
        self.test_ap = AveragePrecision(task=task, num_labels=num_labels, num_classes=cur_num_classes)
        # recall == sensitivity
        self.val_recall = Recall(task=task, num_labels=num_labels, num_classes=cur_num_classes)
        self.test_recall = Recall(task=task, num_labels=num_labels, num_classes=cur_num_classes)
        # specificity
        self.val_specificity = Specificity(task=task, num_labels=num_labels, num_classes=cur_num_classes)
        self.test_specificity = Specificity(task=task, num_labels=num_labels, num_classes=cur_num_classes)
        # f1-score
        self.val_f1 = F1Score(task=task, num_labels=num_labels, num_classes=cur_num_classes)
        self.test_f1 = F1Score(task=task, num_labels=num_labels, num_classes=cur_num_classes)

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
        # define classifier head
        self.vision_classifier = DiseaseHead(input_dim=hidden_size, n_classes=num_classes,
                                             global_token=args['global_token'])
        # classification_loss
        if args['data_name'] == 'rsna':
            class_frequency = sta_rsna_labels_distributions(args['data_ratio'])
        elif args['data_name'] == 'covidx':
            class_frequency = sta_covidx_labels_distributions(args['data_ratio'])
        elif args['data_name'] == 'nih':
            class_frequency = sta_nih_labels_distributions(args['data_ratio'])
        elif args['data_name'] == 'cx-nih':
            class_frequency = sta_cx_nih_labels_distributions(args['data_ratio'])
        elif args['data_name'] == 'cx-rsna':
            class_frequency = sta_cx_rsna_labels_distributions(args['data_ratio'])
        elif args['data_name'] == 'siim':
            class_frequency = sta_siim_labels_distributions(args['data_ratio'])
        elif args['data_name'] == 'shenzhen':
            class_frequency = sta_shenzhen_labels_distributions(args['data_ratio'])
        else:
            raise NotImplementedError
        if 'nih' not in args['data_name']:  # other datasets except for nih and cx-nih
            if args['loss_type'] != 'cross-entropy':
                self.multi_label_loss = BalancedClassLoss(class_frequency, no_of_classes=num_classes,
                                                          gamma=0.2, beta=0.9999, loss_type=args['loss_type'])
            else:
                self.multi_label_loss = nn.CrossEntropyLoss()
        else:  # nih and cx-nih
            self.multi_label_loss = MultiLabelBalancedClassLoss(class_frequency, no_of_classes=num_classes,
                                                                gamma=0.2, beta=0.9999, loss_type=args['loss_type'])

        # # define temperature hyper-parameter
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
            if self.args['data_name'] == 'rsna':
                self.train_set = RSNADataset(self.args, 'train')
                self.val_set = RSNADataset(self.args, 'test')
            elif self.args['data_name'] == 'cx-rsna':
                self.train_set = CXRSNADataset(self.args, 'train')
                self.val_set = CXRSNADataset(self.args, 'test')
            elif self.args['data_name'] == 'covidx':
                self.train_set = COVIDxDataset(self.args, 'train')
                self.val_set = COVIDxDataset(self.args, 'test')
            elif self.args['data_name'] == 'nih':
                self.train_set = NIHDataset(self.args, 'train')
                self.val_set = NIHDataset(self.args, 'test')
            elif self.args['data_name'] == 'cx-nih':
                self.train_set = CXNIHDataset(self.args, 'train')
                self.val_set = CXNIHDataset(self.args, 'test')
            elif self.args['data_name'] == 'siim':
                self.train_set = SIIMDataset(self.args, 'train')
                self.val_set = SIIMDataset(self.args, 'test')
            elif self.args['data_name'] == 'shenzhen':
                self.train_set = ShenZhenCXRDataset(self.args, 'train')
                self.val_set = ShenZhenCXRDataset(self.args, 'test')
            else:
                raise NotImplementedError

            print("No. of training & validation examples: {} & {}.".format(
                self.train_set.__len__(), self.val_set.__len__()))
            self.log_once("No. of training & validation examples: {} & {}.".format(
                self.train_set.__len__(), self.val_set.__len__()))
        if stage == "test" or stage is None:
            if self.args['data_name'] == 'rsna':
                self.test_set = RSNADataset(self.args, 'test')
            elif self.args['data_name'] == 'cx-rsna':
                self.test_set = CXRSNADataset(self.args, 'test')
            elif self.args['data_name'] == 'covidx':
                self.test_set = COVIDxDataset(self.args, 'test')
            elif self.args['data_name'] == 'nih':
                self.test_set = NIHDataset(self.args, 'test')
            elif self.args['data_name'] == 'cx-nih':
                self.test_set = CXNIHDataset(self.args, 'test')
            elif self.args['data_name'] == 'siim':
                self.test_set = SIIMDataset(self.args, 'test')
            elif self.args['data_name'] == 'shenzhen':
                self.test_set = ShenZhenCXRDataset(self.args, 'test')
            else:
                raise NotImplementedError
            print("No. of test examples: {}.".format(self.test_set.__len__()))
            self.log_once("No. of test examples: {}.".format(self.test_set.__len__()))

    def train_dataloader(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#train-dataloader
        """
        pad_token = self.tokenizer.pad_token
        if self.args['data_name'] == 'rsna':
            collate_fn = RSNACollateFn(self.image_processor, pad_token)
        elif self.args['data_name'] == 'cx-rsna':
            collate_fn = CXRSNACollateFn(self.image_processor, pad_token)
        elif self.args['data_name'] == 'covidx':  # 'covidx'
            collate_fn = COVIDxCollateFn(self.image_processor, pad_token)
        elif self.args['data_name'] == 'nih':  # 'nih'
            collate_fn = NIHCollateFn(self.image_processor, pad_token)
        elif self.args['data_name'] == 'cx-nih':  # 'nih'
            collate_fn = CXNIHCollateFn(self.image_processor, pad_token)
        elif self.args['data_name'] == 'siim':  # 'siim'
            collate_fn = SIIMCollateFn(self.image_processor, pad_token)
        elif self.args['data_name'] == 'shenzhen':  # 'shenzhen-CXR'
            collate_fn = ShenZhenCXRCollateFn(self.image_processor, pad_token)
        else:
            raise NotImplementedError

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
        if self.args['data_name'] == 'rsna':
            collate_fn = RSNACollateFn(self.image_processor, pad_token)
        elif self.args['data_name'] == 'cx-rsna':
            collate_fn = CXRSNACollateFn(self.image_processor, pad_token)
        elif self.args['data_name'] == 'covidx':  # 'covidx'
            collate_fn = COVIDxCollateFn(self.image_processor, pad_token)
        elif self.args['data_name'] == 'nih':  # 'nih'
            collate_fn = NIHCollateFn(self.image_processor, pad_token)
        elif self.args['data_name'] == 'cx-nih':  # 'nih'
            collate_fn = CXNIHCollateFn(self.image_processor, pad_token)
        elif self.args['data_name'] == 'siim':  # 'siim'
            collate_fn = SIIMCollateFn(self.image_processor, pad_token)
        elif self.args['data_name'] == 'shenzhen':  # 'shenzhen-CXR'
            collate_fn = ShenZhenCXRCollateFn(self.image_processor, pad_token)
        else:
            raise NotImplementedError

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
        if self.args['data_name'] == 'rsna':
            collate_fn = RSNACollateFn(self.image_processor, pad_token)
        elif self.args['data_name'] == 'cx-rsna':
            collate_fn = CXRSNACollateFn(self.image_processor, pad_token)
        elif self.args['data_name'] == 'covidx':  # 'covidx'
            collate_fn = COVIDxCollateFn(self.image_processor, pad_token)
        elif self.args['data_name'] == 'nih':  # 'nih'
            collate_fn = NIHCollateFn(self.image_processor, pad_token)
        elif self.args['data_name'] == 'cx-nih':  # 'nih'
            collate_fn = CXNIHCollateFn(self.image_processor, pad_token)
        elif self.args['data_name'] == 'siim':  # 'siim'
            collate_fn = SIIMCollateFn(self.image_processor, pad_token)
        elif self.args['data_name'] == 'shenzhen':  # 'shenzhen-CXR'
            collate_fn = ShenZhenCXRCollateFn(self.image_processor, pad_token)
        else:
            raise NotImplementedError

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
        finetune_para, pretrain_para = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'vision_classifier' not in name:
                pretrain_para.append(param)
            else:
                finetune_para.append(param)
        optimiser = torch.optim.AdamW(
            [{'params': pretrain_para, 'lr': self.args['learning_rate']},
             {'params': finetune_para, 'lr': self.args['learning_rate'] * 10}])

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

    def get_shared_latent(self, mode: str, num: int):
        type_embed = self.latent_type_embed[mode]  # (1, d)
        latent = self.shared_latent + type_embed  # (l, d)
        return latent.unsqueeze(0).repeat(num, 1, 1)  # (b, l, d)

    def forward(self, batch):
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

        # ===== compute loss function (language modeling loss) ======
        # obtain input_embed
        if self.args['knowledge_feat']:
            encoder_outputs = torch.cat([visual_latent, indication_latent], dim=1)
        else:
            encoder_outputs = visual_latent
        # =================== compute disease classification logits =======================
        vision_logits = self.vision_classifier(encoder_outputs)

        # 2. compute image-based and text-based classifier loss
        targets = batch['disease_labels'].to(device)
        vision_loss = self.multi_label_loss(vision_logits, targets)

        return {
            'preds': vision_logits,
            'targets': targets,
            'loss': vision_loss
        }

    def training_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training-step
        """
        # Inference:
        batch_size = len(batch['image'])
        loss_dict = self(batch)

        self.log_dict({f'tra_step_loss': loss_dict['loss'].detach().cpu().item()}, on_step=True, on_epoch=False,
                      batch_size=batch_size, prog_bar=True, sync_dist=True)
        if batch_idx % self.args['log_every_n_steps'] == 0 or batch_idx + 1 == self.trainer.num_training_batches:
            self.log_once(
                f"Epoch {self.current_epoch}, training step {batch_idx}/{self.trainer.num_training_batches}, "
                f"loss: {loss_dict['loss'].detach().cpu().item()}, lr: {self.optimizers().param_groups[0]['lr']}")

        self.train_loss_metric['loss'].update(loss_dict['loss'].detach())
        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation-step
        """
        # Inference:
        batch_size = len(batch['image'])
        result = self(batch)

        # Logging:
        self.log_dict({f'val_step_loss': result['loss'].detach().cpu().item()}, on_epoch=False, on_step=True,
                      batch_size=batch_size, prog_bar=False, sync_dist=True)

        if batch_idx % self.args['log_every_n_steps'] == 0 or batch_idx + 1 == self.trainer.num_val_batches[0]:
            self.log_once(
                f"Epoch {self.current_epoch}, validation step {batch_idx}/{self.trainer.num_val_batches[0]}, "
                f"loss: {result['loss'].detach().cpu().item()}, lr: {self.optimizers().param_groups[0]['lr']}")

        # update metrics
        # probability-based metrics
        self.val_auroc.update(result['preds'], result['targets'])
        self.val_ap.update(result['preds'], result['targets'])

        # label-based metrics
        self.val_recall.update(result['preds'], result['targets'])
        self.val_f1.update(result['preds'], result['targets'])
        self.val_specificity.update(result['preds'], result['targets'])
        self.val_acc.update(result['preds'], result['targets'])

    def test_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#test-step
        """
        # Inference:
        batch_size = len(batch['image'])
        result = self(batch)

        # Logging:
        self.log_dict({f'test_step_loss': result['loss'].detach().cpu().item()}, on_epoch=False, on_step=True,
                      batch_size=batch_size, prog_bar=True, sync_dist=True)
        if batch_idx % self.args['log_every_n_steps'] == 0 or batch_idx + 1 == self.trainer.num_test_batches[0]:
            self.log_once(f"Epoch {self.current_epoch}, testing step {batch_idx}/{self.trainer.num_test_batches[0]}, "
                          f"loss: {result['loss'].detach().cpu().item()}")

        # update metrics
        # probability-based metrics
        self.test_auroc.update(result['preds'], result['targets'])
        self.test_ap.update(result['preds'], result['targets'])

        # label-based metrics
        self.test_recall.update(result['preds'], result['targets'])
        self.test_f1.update(result['preds'], result['targets'])
        self.test_specificity.update(result['preds'], result['targets'])
        self.test_acc.update(result['preds'], result['targets'])

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
        cur_all_metrics = {}
        metrics = {
            'ap': self.val_ap,
            'auroc': self.val_auroc,
            'acc': self.val_acc,
            'f1': self.val_f1,
            'recall': self.val_recall,
            'specificity': self.val_specificity,
        }
        for name, metric in metrics.items():
            avg_metric = metric.compute()
            metric.reset()
            cur_all_metrics[name] = torch.round(avg_metric, decimals=3)
        cur_all_metrics['monitor'] = cur_all_metrics['auroc'] + cur_all_metrics['f1']
        self.log_dict({f'val_{k}': v for k, v in cur_all_metrics.items()}, prog_bar=True)

        if cur_all_metrics['monitor'] > self.val_best_scores["best_monitor"]:
            self.val_best_scores = {
                "best_epoch": self.current_epoch,
                'best_monitor': cur_all_metrics['monitor']
            }
            if self.args['save_best_model']:
                self.save_finetune_checkpoint('best')

        cur_metrics_item = ', '.join([f"{k} = {round(v.item(), 3)}" for k, v in cur_all_metrics.items()])
        self.log_once(
            "###############################################################\n"
            f"Epoch {self.current_epoch}, Best epoch {self.val_best_scores['best_epoch']}. Validation is over, metrics:"
            f"{cur_metrics_item}, lr: {self.optimizers().param_groups[0]['lr']}\n"
        )

    def on_test_epoch_end(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#on-test-epoch-end
        """
        cur_all_metrics = {}
        metrics = {
            'ap': self.test_ap,
            'auroc': self.test_auroc,
            'acc': self.test_acc,
            'f1': self.test_f1,
            'recall': self.test_recall,
            'specificity': self.test_specificity,
        }
        for name, metric in metrics.items():
            avg_metric = metric.compute()
            metric.reset()
            cur_all_metrics[name] = torch.round(avg_metric, decimals=3)
        self.log_dict({f'test_{k}': v for k, v in cur_all_metrics.items()}, prog_bar=True)

        cur_metrics_item = ', '.join([f"{k} = {round(v.item(), 3)}" for k, v in cur_all_metrics.items()])
        self.log_once(
            "###############################################################\n"
            f"Epoch {self.current_epoch}, test is over, metrics:"
            f"{cur_metrics_item}\n"
        )

    def save_finetune_checkpoint(self, status):
        state_dict = {}
        for name, para in self.named_parameters():
            if "image_encoder" not in name:
                state_dict[name] = para
        checkpoint = {
            'state_dict': state_dict,
            'optimizer_state': self.trainer.optimizers[0].state_dict(),
            'epoch': self.current_epoch
        }
        torch.save(checkpoint, f'{self.args["ckpt_dir"]}/best_model.pt')
        print(f"The {status} model is saved on epoch {self.current_epoch}!")


class Segmenter(pl.LightningModule):
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
            "best_monitor": -1.0,
        }
        self.old_dice_list = []
        self.x_hyps = []
        self.x_refs = []

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_loss_metric = {
            'loss': torchmetrics.MeanMetric().to(device),
        }
        assert args['data_name'] in ['rsna', 'siim']

        # Dice (default is macro-Dice, compute a dice for a sample, and report average performance)
        self.val_dice = Dice(num_classes=2, average='macro', ignore_index=0)
        self.test_dice = Dice(num_classes=2, average='macro', ignore_index=0)

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
        # define segmentation head
        if args['data_name'] == 'rsna':
            self.loss_fn = MixedLoss(alpha=10)
        else:
            self.loss_fn = MixedSIIMLoss(alpha=10)
        self.segment = SegmentationHead()

        # # define temperature hyper-parameter
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
            trust_remote_code=True
        )

    @rank_zero_only
    def log_once(self, message):
        self.mylog.info(message)

    def setup(self, stage=None):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#setup
        """
        if stage == 'fit' or stage is None:
            # construct train_set and val_set
            if self.args['data_name'] == 'rsna':
                self.train_set = RSNADataset(self.args, 'train')
                self.val_set = RSNADataset(self.args, 'test')
            elif self.args['data_name'] == 'siim':
                self.train_set = SIIMDataset(self.args, 'train')
                self.val_set = SIIMDataset(self.args, 'test')
            else:
                raise NotImplementedError

            print("No. of training & validation examples: {} & {}.".format(
                self.train_set.__len__(), self.val_set.__len__()))
            self.log_once("No. of training & validation examples: {} & {}.".format(
                self.train_set.__len__(), self.val_set.__len__()))
        if stage == "test" or stage is None:
            if self.args['data_name'] == 'rsna':
                self.test_set = RSNADataset(self.args, 'test')
            elif self.args['data_name'] == 'siim':
                self.test_set = SIIMDataset(self.args, 'test')
            else:
                raise NotImplementedError
            print("No. of test examples: {}.".format(self.test_set.__len__()))
            self.log_once("No. of test examples: {}.".format(self.test_set.__len__()))

    def train_dataloader(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#train-dataloader
        """
        pad_token = self.tokenizer.pad_token
        if self.args['data_name'] == 'rsna':
            collate_fn = RSNACollateFn(self.image_processor, pad_token)
        elif self.args['data_name'] == 'siim':  # 'siim'
            collate_fn = SIIMCollateFn(self.image_processor, pad_token)
        else:
            raise NotImplementedError

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
        if self.args['data_name'] == 'rsna':
            collate_fn = RSNACollateFn(self.image_processor, pad_token)
        elif self.args['data_name'] == 'siim':  # 'siim'
            collate_fn = SIIMCollateFn(self.image_processor, pad_token)
        else:
            raise NotImplementedError

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
        if self.args['data_name'] == 'rsna':
            collate_fn = RSNACollateFn(self.image_processor, pad_token)
        elif self.args['data_name'] == 'siim':  # 'siim'
            collate_fn = SIIMCollateFn(self.image_processor, pad_token)
        else:
            raise NotImplementedError

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
        finetune_para, pretrain_para = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'segment' not in name:
                pretrain_para.append(param)
            else:
                finetune_para.append(param)
        optimiser = torch.optim.AdamW(
            [{'params': pretrain_para, 'lr': self.args['learning_rate']},
             {'params': finetune_para, 'lr': self.args['learning_rate'] * 5}])

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

    def get_shared_latent(self, mode: str, num: int):
        type_embed = self.latent_type_embed[mode]  # (1, d)
        latent = self.shared_latent + type_embed  # (l, d)
        return latent.unsqueeze(0).repeat(num, 1, 1)  # (b, l, d)

    def get_dice(self, probability, truth, threshold=0.5):
        # from MGCA
        batch_size = len(truth)
        with torch.no_grad():
            probability = probability.view(batch_size, -1)
            truth = truth.view(batch_size, -1)
            assert probability.shape == truth.shape

            p = (probability > threshold).float()
            t = (truth > 0.5).float()

            t_sum = t.sum(-1)  # (b, 1)
            p_sum = p.sum(-1)
            neg_index = torch.nonzero(t_sum == 0)  # (b,) only background samples
            pos_index = torch.nonzero(t_sum >= 1)

            dice_neg = (p_sum == 0).float()
            dice_pos = 2 * (p * t).sum(-1) / ((p + t).sum(-1))

            dice_neg = dice_neg[neg_index]
            dice_pos = dice_pos[pos_index]
            dice = torch.cat([dice_pos, dice_neg])

        return torch.mean(dice).detach().item()

    def compute_med_dice(self, hyps, refs):
        # from benchX
        hyps, refs = torch.cat(hyps, dim=0), torch.cat(refs, dim=0)
        sample_num = refs.shape[0]
        dice = 0
        for i in range(sample_num):
            p = hyps[i]
            t = refs[i]
            t_sum = t.sum()
            p_sum = p.sum()

            if t_sum == 0:
                dice_instance = float(p_sum == 0)
                dice += dice_instance
            else:
                mask = t != 0
                p_not0 = p[mask]
                t_not0 = t[mask]
                inter = (p_not0 == t_not0).sum() * 2
                dice_instance = inter / (p_sum + t_sum)
                dice += dice_instance
        dice /= sample_num
        return dice

    def forward(self, batch):
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

        # ===== compute loss function (language modeling loss) ======
        # obtain input_embed

        # =================== compute disease classification logits =======================
        if self.args['knowledge_feat']:
            encoder_outputs = torch.cat([visual_latent, indication_latent], dim=1)
        else:
            encoder_outputs = visual_latent
        logit = self.segment(encoder_outputs, vision_feat[:, 1:, :])
        logit = logit.squeeze(dim=1)
        # batch['disease_labels'] used for change
        loss = self.loss_fn(logit, batch['mask'])

        prob = torch.sigmoid(logit)
        dice = self.get_dice(prob, batch['mask'])

        preds = (prob >= 0.5).long()
        return {
            'preds': preds,
            # 'targets': F.one_hot(batch['mask'].long(), num_classes=2).permute(0, 3, 1, 2),
            'targets': batch['mask'].long(),
            'old_dice': dice,
            'loss': loss
        }

    def training_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training-step
        """
        # Inference:
        batch_size = len(batch['image'])
        loss_dict = self(batch)

        self.log_dict({f'tra_step_loss': loss_dict['loss'].detach().cpu().item()}, on_step=True, on_epoch=False,
                      batch_size=batch_size, prog_bar=True, sync_dist=True)
        if batch_idx % self.args['log_every_n_steps'] == 0 or batch_idx + 1 == self.trainer.num_training_batches:
            self.log_once(
                f"Epoch {self.current_epoch}, training step {batch_idx}/{self.trainer.num_training_batches}, "
                f"loss: {loss_dict['loss'].detach().cpu().item()}, lr: {self.optimizers().param_groups[0]['lr']}")

        self.train_loss_metric['loss'].update(loss_dict['loss'].detach())
        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation-step
        """
        # Inference:
        batch_size = len(batch['image'])
        result = self(batch)

        # Logging:
        self.log_dict({f'val_step_loss': result['loss'].detach().cpu().item()}, on_epoch=False, on_step=True,
                      batch_size=batch_size, prog_bar=False, sync_dist=True)

        if batch_idx % self.args['log_every_n_steps'] == 0 or batch_idx + 1 == self.trainer.num_val_batches[0]:
            self.log_once(
                f"Epoch {self.current_epoch}, validation step {batch_idx}/{self.trainer.num_val_batches[0]}, "
                f"loss: {result['loss'].detach().cpu().item()}, lr: {self.optimizers().param_groups[0]['lr']}")

        # update metrics
        self.val_dice.update(result['preds'], result['targets'])
        self.old_dice_list.append(result['old_dice'])
        self.x_hyps.append(result['preds'].cpu().detach())
        self.x_refs.append(result['targets'].cpu().detach())

    def test_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#test-step
        """
        # Inference:
        batch_size = len(batch['image'])
        result = self(batch)

        # Logging:
        self.log_dict({f'test_step_loss': result['loss'].detach().cpu().item()}, on_epoch=False, on_step=True,
                      batch_size=batch_size, prog_bar=True, sync_dist=True)
        if batch_idx % self.args['log_every_n_steps'] == 0 or batch_idx + 1 == self.trainer.num_test_batches[0]:
            self.log_once(f"Epoch {self.current_epoch}, testing step {batch_idx}/{self.trainer.num_test_batches[0]}, "
                          f"loss: {result['loss'].detach().cpu().item()}")

        # update metrics
        self.test_dice.update(result['preds'], result['targets'])
        self.old_dice_list.append(result['old_dice'])

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

        avg_dice = self.val_dice.compute()
        self.val_dice.reset()

        # compute bench-x metrics
        x_dice = self.compute_med_dice(self.x_hyps, self.x_refs)
        self.x_hyps, self.x_refs = [], []

        cur_all_metrics = {
            'torch-dice': torch.round(avg_dice, decimals=3),
            'old-dice': round(sum(self.old_dice_list) / len(self.old_dice_list), 3),
            'x_dice': x_dice,
            'monitor': x_dice
        }
        self.old_dice_list = []

        self.log_dict({f'val_{k}': v for k, v in cur_all_metrics.items()}, prog_bar=True)

        if cur_all_metrics['monitor'] > self.val_best_scores["best_monitor"]:
            self.val_best_scores = {
                "best_epoch": self.current_epoch,
                'best_monitor': cur_all_metrics['monitor']
            }
            if self.args['save_best_model']:
                self.save_finetune_checkpoint('best')

        cur_metrics_item = (f'torch-dice: {cur_all_metrics["torch-dice"].cpu().detach().item()}, '
                            f'old-dice: {cur_all_metrics["old-dice"]}, x-dice: {x_dice}')
        self.log_once(
            "###############################################################\n"
            f"Epoch {self.current_epoch}, Best epoch {self.val_best_scores['best_epoch']}. Validation is over, metrics:"
            f"{cur_metrics_item}, lr: {self.optimizers().param_groups[0]['lr']}\n"
        )

    def on_test_epoch_end(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#on-test-epoch-end
        """
        avg_dice = self.test_dice.compute()
        self.test_dice.reset()
        cur_all_metrics = {
            'torch-dice': torch.round(avg_dice, decimals=3),
            'old-dice': round(sum(self.old_dice_list) / len(self.old_dice_list), 3)
        }
        self.old_dice_list = []

        self.log_dict({f'test_{k}': v for k, v in cur_all_metrics.items()}, prog_bar=True)

        cur_metrics_item = (f'torch-dice: {cur_all_metrics["torch-dice"].cpu().detach().item()}, '
                            f'old-dice: {cur_all_metrics["old-dice"]}')
        self.log_once(
            "###############################################################\n"
            f"Epoch {self.current_epoch}, test is over, metrics:"
            f"{cur_metrics_item}\n"
        )

    def save_finetune_checkpoint(self, status):
        state_dict = {}
        for name, para in self.named_parameters():
            if "image_encoder" not in name:
                state_dict[name] = para
        checkpoint = {
            'state_dict': state_dict,
            'optimizer_state': self.trainer.optimizers[0].state_dict(),
            'epoch': self.current_epoch
        }
        torch.save(checkpoint, f'{self.args["ckpt_dir"]}/best_model.pt')
        print(f"The {status} model is saved on epoch {self.current_epoch}!")


class DiseaseHead(nn.Module):
    def __init__(self, input_dim, n_classes=2, global_token='pool'):
        super().__init__()
        if global_token == 'pool':
            self.pool = LightSelfAttentionPooling(input_dim)
        else:  # average
            self.pool = None
        self.disease_head_2class = nn.Linear(input_dim, n_classes)

    def forward(self, x):  # x: (B, 196, 2560)
        if self.pool is not None:
            x = self.pool(x)
        else:
            x = torch.mean(x, dim=1)
        logits = self.disease_head_2class(x)  # (B, 2)
        return logits


class Decoder2D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 features=(512, 256, 128, 64),
                 patch_size=(16, 16)):
        super().__init__()
        self.final_dense = nn.Linear(in_channels, in_channels)
        self.decoder_1 = nn.Sequential(
            nn.Conv2d(in_channels, features[0], 3, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )
        self.decoder_2 = nn.Sequential(
            nn.Conv2d(features[0], features[1], 3, padding=1),
            nn.BatchNorm2d(features[1]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )
        self.decoder_3 = nn.Sequential(
            nn.Conv2d(features[1], features[2], 3, padding=1),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )
        self.decoder_4 = nn.Sequential(
            nn.Conv2d(features[2], features[3], 3, padding=1),
            nn.BatchNorm2d(features[3]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )

        self.final_out = nn.Conv2d(features[-1], out_channels, 3, padding=1)

    def forward(self, x):
        # x is (b, 128, 768)
        x = self.final_dense(x)
        x = rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                      p1=hh, p2=ww, h=hh, w=ww, c=hidden_size)
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        x = self.decoder_4(x)
        x = self.final_out(x)
        # mask is (b, 1, 518, 518)
        return x


class SegmentationHead(nn.Module):
    def __init__(self, hidden_size=768, input_channels=128,
                 num_tokens=1369, decoder_channels=(512, 256, 128, 64),
                 img_size=(518, 518)):
        super().__init__()
        self.img_size = img_size
        self.grid_size = int(num_tokens ** 0.5)  # 37x37

        #  (128 -> 1369)
        self.token_expand = nn.Conv1d(input_channels, num_tokens, kernel_size=1)
        # LayerNorm
        self.norm = nn.LayerNorm(hidden_size)

        # Decoder Blocks
        self.decoder_blocks = nn.ModuleList()
        in_ch = hidden_size
        for ch in decoder_channels:
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                )
            )
            in_ch = ch

        # Final Output Layer
        self.final_conv = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, x, ori_vision_feat):
        """
        x:    (b, 128, 768)
        ori_vision_feat: (b, 1369, 768)
        """
        # Step 1: expand tokens
        x = self.token_expand(x)          # (b, 1369, 768)

        # Step 2: Skip connection & norm
        x = self.norm(x + ori_vision_feat)                # (b, 1369, 768)

        # Step 4: Reshape为feature map
        x = rearrange(x, "b (h w) c -> b c h w", h=self.grid_size, w=self.grid_size)

        # Step 5: Decoder逐步上采样
        for block in self.decoder_blocks:
            x = block(x)

        # Step 6: 输出预测mask并调整尺寸
        x = self.final_conv(x)
        x = nn.functional.interpolate(x, size=self.img_size, mode="bilinear", align_corners=True)
        return x
