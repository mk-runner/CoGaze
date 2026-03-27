from typing import Dict

import torch
import transformers
import torchmetrics
from torch.cuda.amp import autocast
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.utilities import rank_zero_only
from transformers import (AutoModel, AutoConfig, AutoImageProcessor, GPT2TokenizerFast,
                          PretrainedConfig, AutoTokenizer, LlamaForCausalLM)

from models.conversation_v0702_llm import conv_srrg_v1001, SeparatorStyle
from models.perceiver_pytorch import Perceiver
from tools.metrics.chexbert import RadGraphMetrics, F1CheXbertMetrics
from tools.metrics.coco import COCOCaptionMetrics
from tools.metrics.report_logger import ReportLogger
from tools.metrics.metrics import delete_organ
from models.utils import *
from tools.dataset_ab import *


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
        self.conv = conv_srrg_v1001
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
            self.train_set = SRRGGenerationDataset(self.args, 'train')
            self.val_set = SRRGGenerationDataset(self.args, 'val')

            print("No. of training & validation examples: {} & {}.".format(
                self.train_set.__len__(), self.val_set.__len__()))
            self.log_once("No. of training & validation examples: {} & {}.".format(
                self.train_set.__len__(), self.val_set.__len__()))
        if stage == "test" or stage is None:
            self.test_set = SRRGGenerationDataset(self.args, 'test')
            print("No. of test examples: {}.".format(self.test_set.__len__()))
            self.log_once("No. of test examples: {}.".format(self.test_set.__len__()))

    def train_dataloader(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#train-dataloader
        """
        pad_token = self.tokenizer.pad_token
        collate_fn = SRRGGenerationLLMCollateFn(self.args, self.image_processor, pad_token)
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
        collate_fn = SRRGGenerationLLMCollateFn(self.args, self.image_processor, pad_token)
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
        collate_fn = SRRGGenerationLLMCollateFn(self.args, self.image_processor, pad_token)
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
        lang_parameters = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            lang_parameters.append(param)

        optimiser = torch.optim.AdamW(
            [{'params': lang_parameters, 'lr': self.args['learning_rate']}])
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
        device = img_embeds.device
        conv = self.conv.copy()
        conv.append_message(conv.roles[0], 'Image: <image>')
        prompt = conv.get_prompt()
        p_before = prompt.split('<image>')[0]
        # system embeds
        tokens = self.llm_tokenizer(p_before, return_tensors='pt').to(device)
        p_before_embeds = self.embed_tokens(tokens['input_ids']).expand(batch_size, -1, -1)

        # assistant special tag
        assistant = '<|start_header_id|>assistant<|end_header_id|>\n\n'
        special_tokens = self.llm_tokenizer(assistant, return_tensors='pt').to(device)
        # the first token is bos_token
        p_after_embeds = self.embed_tokens(special_tokens['input_ids'][:, 1:]).expand(batch_size, -1, -1)

        # system + <image> + assistant
        wrapped_img_embeds = torch.cat([
            p_before_embeds,
            img_embeds,
            p_after_embeds
        ], dim=1)
        wrapped_atts_img = torch.ones(wrapped_img_embeds.size()[:-1], dtype=torch.long).to(img_embeds.device)

        return wrapped_img_embeds, wrapped_atts_img

    def build_batch_prompt_simple(self, device, responses):
        # the fist token is <begin_of_text>
        tokens = self.llm_tokenizer(responses, padding=True, return_tensors='pt',
                                    max_length=self.args['max_length'], truncation=True).to(device)
        input_ids = tokens['input_ids'][:, 1:]  # delete the first token
        embeds = self.embed_tokens(input_ids)
        atts_img = tokens['attention_mask'][:, 1:]
        # initialize targets
        targets = input_ids.detach().to(device)
        targets = targets.masked_fill(
            targets == self.llm_tokenizer.pad_token_id, -100
        )  # mask padding

        return embeds, atts_img, targets

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
            p_after_embeds, after_atts, after_targets = self.build_batch_prompt_simple(device, batch['report'])
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
            stop_str = self.conv.sep if self.conv.sep_style not in {SeparatorStyle.TWO, SeparatorStyle.LLAMA_3,
                                                                    SeparatorStyle.MISTRAL} else self.conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.llm_tokenizer, p_before_embeds.shape[1])
            with torch.no_grad(), autocast(dtype=torch.float16):  # more suitable for inference phase
                output_ids = self.llm.generate(
                    inputs_embeds=p_before_embeds,
                    attention_mask=before_atts,
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
        reference_reports = self.obtain_reference_reports(batch['report'])  # remove special tokens

        if batch_idx % self.args['log_every_n_steps'] == 0 or batch_idx + 1 == self.trainer.num_val_batches[0]:
            self.log_once(
                f"Epoch {self.current_epoch}, validation step {batch_idx}/{self.trainer.num_val_batches[0]}")

        # # Log reports:
        dicom_ids = batch['dicom_id']
        self.val_report_logger.update(generated_reports, dicom_ids=dicom_ids, reference_reports=reference_reports)

        generated_reports = [delete_organ(item) for item in generated_reports]
        generated_reports = [text if len(text) > 0 else "There is no findings." for text in generated_reports]
        reference_reports = [delete_organ(item) for item in reference_reports]
        # print(generated_reports)
        # # Evaluate:
        self.val_f1chexbert_metrics.update(generated_reports, reference_reports, ids=dicom_ids)
        self.val_coco_metrics.update(generated_reports, reference_reports, ids=dicom_ids)
        self.val_radgraph_metrics.update(generated_reports, reference_reports, ids=dicom_ids)

    def test_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#test-step
        """
        generated_reports = self(batch, mode='test')

        reference_reports = self.obtain_reference_reports(batch['report'])  # remove special tokens

        if batch_idx % self.args['log_every_n_steps'] == 0 or batch_idx + 1 == self.trainer.num_test_batches[0]:
            self.log_once(
                f"Epoch {self.current_epoch}, test step {batch_idx}/{self.trainer.num_test_batches[0]}")

        generated_reports = [delete_organ(item) for item in generated_reports]
        generated_reports = [text if len(text) > 0 else "there is no finding." for text in generated_reports]
        reference_reports = [delete_organ(item) for item in reference_reports]
        # print(generated_reports)

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
        self.conv = conv_srrg_v1001
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
            self.train_set = SRRGGenerationDataset(self.args, 'train')
            self.val_set = SRRGGenerationDataset(self.args, 'val')

            print("No. of training & validation examples: {} & {}.".format(
                self.train_set.__len__(), self.val_set.__len__()))
            self.log_once("No. of training & validation examples: {} & {}.".format(
                self.train_set.__len__(), self.val_set.__len__()))
        if stage == "test" or stage is None:
            self.test_set = SRRGGenerationDataset(self.args, 'test')
            print("No. of test examples: {}.".format(self.test_set.__len__()))
            self.log_once("No. of test examples: {}.".format(self.test_set.__len__()))

    def train_dataloader(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#train-dataloader
        """
        pad_token = self.tokenizer.pad_token
        collate_fn = SRRGGenerationLLMCollateFn(self.args, self.image_processor, pad_token)
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
        collate_fn = SRRGGenerationLLMCollateFn(self.args, self.image_processor, pad_token)
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
        collate_fn = SRRGGenerationLLMCollateFn(self.args, self.image_processor, pad_token)
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
        device = img_embeds.device
        conv = self.conv.copy()
        conv.append_message(conv.roles[0], 'Image: <image>')
        prompt = conv.get_prompt()
        p_before = prompt.split('<image>')[0]
        # system embeds
        tokens = self.llm_tokenizer(p_before, return_tensors='pt').to(device)
        p_before_embeds = self.embed_tokens(tokens['input_ids']).expand(batch_size, -1, -1)

        # assistant special tag
        assistant = '<|start_header_id|>assistant<|end_header_id|>\n\n'
        special_tokens = self.llm_tokenizer(assistant, return_tensors='pt').to(device)
        # the first token is bos_token
        p_after_embeds = self.embed_tokens(special_tokens['input_ids'][:, 1:]).expand(batch_size, -1, -1)

        # system + <image> + assistant
        wrapped_img_embeds = torch.cat([
            p_before_embeds,
            img_embeds,
            p_after_embeds
        ], dim=1)
        wrapped_atts_img = torch.ones(wrapped_img_embeds.size()[:-1], dtype=torch.long).to(img_embeds.device)

        return wrapped_img_embeds, wrapped_atts_img

    def build_batch_prompt_simple(self, device, responses):
        # the fist token is <begin_of_text>
        tokens = self.llm_tokenizer(responses, padding=True, return_tensors='pt',
                                    max_length=self.args['max_length'], truncation=True).to(device)
        input_ids = tokens['input_ids'][:, 1:]  # delete the first token
        embeds = self.embed_tokens(input_ids)
        atts_img = tokens['attention_mask'][:, 1:]
        # initialize targets
        targets = input_ids.detach().to(device)
        targets = targets.masked_fill(
            targets == self.llm_tokenizer.pad_token_id, -100
        )  # mask padding

        return embeds, atts_img, targets

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
            p_after_embeds, after_atts, after_targets = self.build_batch_prompt_simple(device, batch['report'])
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
            stop_str = self.conv.sep if self.conv.sep_style not in {SeparatorStyle.TWO, SeparatorStyle.LLAMA_3,
                                                                    SeparatorStyle.MISTRAL} else self.conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.llm_tokenizer, p_before_embeds.shape[1])
            with torch.no_grad(), autocast(dtype=torch.float16):  # more suitable for inference phase
                output_ids = self.llm.generate(
                    inputs_embeds=p_before_embeds,
                    attention_mask=before_atts,
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
            self.train_loss_metric[key].update(loss.detach())
        # Update and log scores for each validation metric:
        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation-step
        """
        # Inference:
        generated_reports = self(batch, mode='val')
        reference_reports = self.obtain_reference_reports(batch['report'])  # remove special tokens

        if batch_idx % self.args['log_every_n_steps'] == 0 or batch_idx + 1 == self.trainer.num_val_batches[0]:
            self.log_once(
                f"Epoch {self.current_epoch}, validation step {batch_idx}/{self.trainer.num_val_batches[0]}")

        # # Log reports:
        dicom_ids = batch['dicom_id']
        self.val_report_logger.update(generated_reports, dicom_ids=dicom_ids, reference_reports=reference_reports)

        generated_reports = [delete_organ(item) for item in generated_reports]
        generated_reports = [text if len(text) > 0 else "There is no findings." for text in generated_reports]
        reference_reports = [delete_organ(item) for item in reference_reports]
        # print(generated_reports)
        # # Evaluate:
        self.val_f1chexbert_metrics.update(generated_reports, reference_reports, ids=dicom_ids)
        self.val_coco_metrics.update(generated_reports, reference_reports, ids=dicom_ids)
        self.val_radgraph_metrics.update(generated_reports, reference_reports, ids=dicom_ids)

    def test_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#test-step
        """
        generated_reports = self(batch, mode='test')

        reference_reports = self.obtain_reference_reports(batch['report'])  # remove special tokens

        if batch_idx % self.args['log_every_n_steps'] == 0 or batch_idx + 1 == self.trainer.num_test_batches[0]:
            self.log_once(
                f"Epoch {self.current_epoch}, test step {batch_idx}/{self.trainer.num_test_batches[0]}")

        generated_reports = [delete_organ(item) for item in generated_reports]
        generated_reports = [text if len(text) > 0 else "there is no finding." for text in generated_reports]
        reference_reports = [delete_organ(item) for item in reference_reports]
        # print(generated_reports)

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
