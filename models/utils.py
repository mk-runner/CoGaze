import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import StoppingCriteria
import matplotlib.pyplot as plt
from einops import repeat


def plot_eye_heatmaps(pred_heatmap: np.ndarray, gt_heatmap: np.ndarray):

    """
    绘制预测和真实的 eye heatmap。

    参数:
        pred_heatmap: numpy.ndarray, 预测heatmap (值范围0~1)
        gt_heatmap: numpy.ndarray, 真实heatmap (值范围0~1)
    """
    # 检查输入
    # pred_heatmap, gt_heatmap = pred_heatmap * 255.0, gt_heatmap * 255.0
    if pred_heatmap.shape != gt_heatmap.shape:
        raise ValueError("预测heatmap和真实heatmap的形状必须相同")

    pred = pred_heatmap[-1].reshape(37, 37)
    gts = gt_heatmap[-1].reshape(37, 37)
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # 绘制预测heatmap
    plt.imshow(pred, cmap='jet')
    # axes[0].set_title("Predicted Eye Heatmap")
    plt.axis("off")
    # plt.colorbar()
    plt.tight_layout()
    plt.savefig('my-figures/gt.png', dpi=600)
    plt.show()

    # 绘制真实heatmap
    plt.imshow(gts, cmap='jet')
    # axes[1].set_title("Ground Truth Eye Heatmap")
    plt.axis("off")
    # plt.colorbar()
    plt.tight_layout()
    plt.savefig('my-figures/pred.png', dpi=600)
    plt.show()
    print()


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, start_len):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = start_len

    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            truncated_output_ids = output_ids[0, -keyword_id.shape[0]:]
            if torch.equal(truncated_output_ids, keyword_id):
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        rms = norm / (x.shape[-1] ** 0.5)
        return x / (rms + self.eps) * self.weight


class SpatialSEWithGazeResidual(nn.Module):
    def __init__(self, patch_dim, reduction=16, use_residual=True, norm='rms'):
        super().__init__()
        self.fc1 = nn.Linear(1, patch_dim // reduction)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(patch_dim // reduction, patch_dim)
        self.sigmoid = nn.Sigmoid()
        self.use_residual = use_residual

        # Positional Embedding Norm and final output norm
        if norm == 'rms':
            self.norm = RMSNorm(patch_dim)
        else:
            self.norm = nn.LayerNorm(patch_dim)

    def forward(self, visual_feat, gaze_heatmap):
        """
        visual_feat: (B, N, D)
        gaze_heatmap: (B, N)
        """
        # Gaze-based spatial attention modulation
        h = gaze_heatmap.unsqueeze(-1)           # (B, N, 1)
        attn = self.fc1(h)
        attn = self.act(attn)
        attn = self.fc2(attn)
        attn = self.sigmoid(attn)                # (B, N, D)

        # Modulate visual features with residual or multiplicative attention
        if self.use_residual:
            enhanced = visual_feat * (1 + attn)
        else:
            enhanced = visual_feat * attn

        return self.norm(enhanced)               # (B, N, D)


class AddViewPositionalEmbedding(nn.Module):
    def __init__(self, dim, view_position_path, norm='rms', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vp2id = json.load(open(view_position_path))
        self.pos_embed = nn.Parameter(torch.randn(len(self.vp2id), 1, dim), requires_grad=True)
        if norm == 'rms':
            self.pos_norm = RMSNorm(dim)
        else:
            self.pos_norm = nn.LayerNorm(dim)

    def forward(self, hidden_state, view_positions):
        index = torch.tensor([self.vp2id[vp] for vp in view_positions])
        pos_embed = self.pos_embed[index]
        return self.pos_norm(hidden_state + pos_embed)


class MultiScaleGazeModulation(nn.Module):
    def __init__(self, dim, scales=(1, 2, 4), reduction=16, norm='rms'):
        super().__init__()
        self.mods = nn.ModuleList([
            SpatialSEWithGazeResidual(dim, reduction, norm=norm, use_residual=False)
            for _ in scales
        ])
        self.scales = scales
        self.weights = nn.Parameter(torch.ones(len(scales)), requires_grad=True)  # Learnable scale fusion weights
        if norm == 'rms':
            self.norm = RMSNorm(dim)
        else:
            self.norm = nn.LayerNorm(dim)

    def forward(self, visual_feat, gaze_heatmap):
        """
        visual_feat:  (B, N, D)
        gaze_heatmap: (B, 1, H, W)
        """
        B, N, D = visual_feat.shape
        H = W = int(N ** 0.5)  # expected N == 729 -> H=W=27; N = 1369, H=W=37

        fused = torch.zeros_like(visual_feat).to(visual_feat)
        weights = torch.softmax(self.weights, dim=0)  # Ensure weights sum to 1

        for i, scale in enumerate(self.scales):
            # Downsample then upsample gaze heatmap
            down = F.adaptive_avg_pool2d(gaze_heatmap, output_size=(H // scale, W // scale))
            gh = F.interpolate(down, size=(H, W), mode='bilinear', align_corners=False)  # back to (27, 27)
            gh = gh.reshape(B, -1)  # flatten to (B, N)

            # Modulate with corresponding block
            modulated = self.mods[i](visual_feat, gh)  # (B, N, D)
            fused = fused + weights[i] * modulated     # Weighted sum

        return self.norm(fused + visual_feat)  # (B, N, D)


class Adapter(nn.Module):
    def __init__(self, dim, out_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, out_dim),
        )

    def forward(self, x):
        return self.layers(x)


def weighted_kl_div(pred, target, weight=None, eps=1e-8):
    """
    pred: (B, 1, H, W) - predicted heatmap
    target: (B, 1, H, W) - ground truth heatmap
    weight: (B, 1, H, W) - optional weight map
    """
    pred = pred.clamp(min=eps)
    target = target.clamp(min=eps)

    log_ratio = torch.log(pred / target)
    kl = target * log_ratio  # (B, 1, H, W)

    if weight is not None:
        kl = kl * weight

    return kl.sum() / pred.size(0)  # average over batch


class SetToGridProjector(nn.Module):
    def __init__(self, embed_dim=768, grid_size=16):
        super().__init__()
        self.grid_size = grid_size
        self.query_grid = nn.Parameter(torch.randn(1, grid_size * grid_size, embed_dim))
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)

    def forward(self, mix_embed):  # (B, seq, D)
        B = mix_embed.size(0)
        query = self.query_grid.expand(B, -1, -1)         # (B, 16*16, D)
        out, _ = self.cross_attn(query, mix_embed, mix_embed)
        return out.view(B, self.grid_size, self.grid_size, -1).permute(0, 3, 1, 2)  # (B, D, 16, 16)


class CNNHeatmapDecoder128(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_dim, 128, kernel_size=3, padding=1),        # (B, 128, 16, 16)
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),            # (B, 128, 32, 32)

            nn.Conv2d(128, 64, kernel_size=3, padding=1),            # (B, 64, 32, 32)
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),            # (B, 64, 64, 64)

            nn.Conv2d(64, 32, kernel_size=3, padding=1),             # (B, 32, 64, 64)
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),            # (B, 32, 128, 128)

            nn.Conv2d(32, 1, kernel_size=1),                          # (B, 1, 128, 128)
        )

    def forward(self, x):
        return self.decoder(x)


class GazePredictionHead128(nn.Module):
    def __init__(self, embed_dim=768, grid_size=16):
        super().__init__()
        self.projector = SetToGridProjector(embed_dim, grid_size)
        self.decoder = CNNHeatmapDecoder128(embed_dim)

    def forward(self, mix_embed):  # (B, seq, D)
        x = self.projector(mix_embed)            # (B, D, 16, 16)
        heatmap = self.decoder(x)                # (B, 1, 128, 128)
        heatmap = F.softmax(heatmap.view(heatmap.size(0), -1), dim=-1).view_as(heatmap)
        return heatmap
    

class FusionContext(nn.Module):
    def __init__(self, dim, num_heads=12, batch_first=True):
        super().__init__()
        self.v2p_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=batch_first)
        self.p2v_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=batch_first)
        self.ln = nn.LayerNorm(dim)

    def forward(self, vision_feat, context_feat):
        v2p, _ = self.v2p_attention(vision_feat, context_feat, context_feat)
        p2v, _ = self.p2v_attention(context_feat, vision_feat, vision_feat)
        mix_feat = torch.cat([v2p, p2v], dim=1)
        return self.ln(mix_feat)


def compute_kl_divergence_loss(pred, target, eps=1e-8):
    """
    compute kl_divergence, since some row of target is zero (mask)
    Args:
        pred: tensor (M, N) (patch_num, sen_num) or (sen_num, patch_num)
        target: tensor (M, N) (patch_num, sen_num) or (sen_num, patch_num)
        eps: int, avoid zero

    Returns:
        kl_divergence_loss
    """
    row_sum = target.sum(dim=-1, keepdim=True)  # (M, 1)
    mask = (row_sum > eps).float()              # (M, 1)

    target_norm = target / (row_sum + eps)      # (M, N)
    log_pred = F.log_softmax(pred, dim=-1)      # (M, N)

    kl = F.kl_div(log_pred, target_norm, reduction='none')  # (M, N) if batchmean kl.sum(dim=-1).mean()
    kl = kl.sum(dim=-1)  # (M,)
    kl = kl * mask.squeeze(-1)  # mask zeros row
    return kl.sum() / (mask.sum() + eps)


def make_dynamic_mask(p, topk_ratio=None, eps=1e-8):
    mask = torch.zeros_like(p)
    if topk_ratio is not None and topk_ratio > 0:
        nonzero_count = (p > eps).sum(dim=-1)  # the number of non-zero elements for each row(M,)
        dynamic_topk = torch.clamp((nonzero_count.float() * topk_ratio).long(), min=1)  # convert zero into 1
        for i in range(p.size(0)):
            k = dynamic_topk[i].item()
            _, idx = p[i].topk(k, dim=-1)
            mask[i].scatter_(0, idx, 1.0)
    else:
        mask = (p > eps).to(p)
    return mask


def js_divergence_official(p, q, eps=1e-8):
    m = 0.5 * (p + q) + eps  # 避免 log(0)

    kl_pm = (p * (torch.log(p + eps) - torch.log(m))).sum(dim=-1)
    kl_qm = (q * (torch.log(q + eps) - torch.log(m))).sum(dim=-1)

    js = 0.5 * (kl_pm + kl_qm)

    # 防止 NaN/Inf
    return js


def compute_js_loss_dynamic(pred, target, topk_ratio=None, eps=1e-8):
    # obtain mask
    mask = make_dynamic_mask(target, topk_ratio, eps)

    # obtain mask pred and target
    target_mask = target * mask

    # obtain valid row
    valid_row = target_mask.sum(dim=-1) > eps
    if valid_row.sum() == 0:
        # 没有合法行，返回 0，保持梯度图
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    # remove illegal rows
    valid_target_mask = target_mask[valid_row]
    valid_pred_mask = pred[valid_row]

    # normalize
    p = valid_target_mask / (valid_target_mask.sum(dim=-1, keepdim=True) + eps)
    q = torch.softmax(valid_pred_mask, dim=-1)

    # compute js_divergence
    js = js_divergence_official(p, q, eps=eps)
    loss = js.mean()

    return loss


def bidirectional_js_loss_dynamic(t2i_logis, t2i_heatmap, topk_ratio=None, i2t_weight=0.8):
    # transcript -> image
    loss_t2i = compute_js_loss_dynamic(
        t2i_logis, t2i_heatmap, topk_ratio=topk_ratio
    )
    # # image -> transcript
    loss_i2t = compute_js_loss_dynamic(
        t2i_logis.t(), t2i_heatmap.t(), topk_ratio=topk_ratio
    )
    loss = i2t_weight * loss_t2i + (1 - i2t_weight) * loss_i2t

    return loss


def bidirectional_mask(t2i_logis, t2i_heatmap, topk_ratio=None, i2t_weight=0.8):
    t2i_heatmap_mask = (t2i_heatmap != 0).float()

    # transcript -> image
    loss_t2i = F.binary_cross_entropy_with_logits(
        t2i_logis, t2i_heatmap_mask
    )

    # image -> transcript
    loss_i2t = F.binary_cross_entropy_with_logits(
        t2i_logis.t(), t2i_heatmap_mask.t()
    )

    loss = i2t_weight * loss_t2i + (1 - i2t_weight) * loss_i2t

    return loss


class LightSelfAttentionPooling(nn.Module):
    def __init__(self, input_dim=2560, num_heads=8, dropout=0.):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))
        self.attn = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(input_dim)

        # FFN
        self.norm2 = nn.LayerNorm(input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),  # Expand
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, input_dim),  # Project back
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(input_dim)

    def forward(self, x):  # x: (B, 196, 2560)
        # x = self.reduce(x)  # (B, 196, 512)
        B = x.size(0)
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, 512)
        x = torch.cat([cls_token, x], dim=1)  # (B, 197, 512)

        # Self-attention
        x = self.norm1(x)
        attn_out, _ = self.attn(x[:, 0:1, :], x, x)  # (B, 1, 512)

        # FFN with residual
        ffn_input = self.norm2(attn_out + x[:, 0:1, :])  # (B, 1, 512)
        ffn_out = self.ffn(ffn_input)  # (B, 1, 512)
        output = ffn_input + ffn_out  # Residual connection
        output = self.norm3(output)
        return output.squeeze(1)  # (B, 512)


class TextProjectorMLP(nn.Module):
    """
    project report/sentence/knowledge data into a shared space
    (including adding type embedding and a projector)
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        """
        # best performance version
        Args:
            x (torch.Tensor): embedding (batch_size, seq_len, dim)
            type_id (int): 0 for report/sentence findings; 1 for clinical context

        Returns:
            torch.Tensor, shape (batch_size, seq_len, output_dim)
        """
        return self.layers(x)


class TextProjectorTypeId(nn.Module):
    """
    project report/sentence/knowledge data into a shared space
    (including adding type embedding and a projector)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, norm='rms'):
        super().__init__()

        # self.type_embed = nn.Parameter(torch.randn(2, 1, input_dim), requires_grad=True)
        # if norm == 'rms':
        #     self.norm = RMSNorm(input_dim)
        # else:
        #     self.norm = nn.LayerNorm(input_dim)

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, type_id=0):
        """
        # best performance version
        Args:
            x (torch.Tensor): embedding (batch_size, seq_len, dim)
            type_id (int): 0 for report/sentence findings; 1 for clinical context

        Returns:
            torch.Tensor, shape (batch_size, seq_len, output_dim)
        """
        # batch_size = x.shape[0]
        # type_embeddings = repeat(self.type_embed[type_id], 'n d -> b n d', b=batch_size)
        # x = self.norm(x + type_embeddings)
        # x = x.permute(0, 2, 1)
        # x = self.head(x)
        # return x.permute(0, 2, 1)
        return self.layers(x)


class TextProjectorConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim,
                      kernel_size=1, stride=1, padding=0),  # change the dimension
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, output_dim,
                      kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.head(x)
        return x.permute(0, 2, 1)


class ProjectorConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)

        self.head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim,
                      kernel_size=1, stride=1, padding=0),  # change the dimension
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, output_dim,
                      kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        x = self.head(x)
        return x.permute(0, 2, 1)


class VisionProjectorConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, view_position_path, norm='ln') -> None:
        super().__init__()
        self.add_pos_embed = AddViewPositionalEmbedding(input_dim, view_position_path, norm=norm)

        self.head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim,
                      kernel_size=1, stride=1, padding=0),  # change the dimension
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, output_dim,
                      kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x, view_positions):
        x = self.add_pos_embed(x, view_positions)  # [B, N, D]
        x = x.permute(0, 2, 1)
        x = self.head(x)
        return x.permute(0, 2, 1)


class VisionProjectorMLP(nn.Module):
    """
    # best performance at ge
    Project visual embedding into a shared space
    (including adding view positional embedding + eye-gaze heatmap + projector)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, view_position_path, norm='rms'):
        super().__init__()
        # view positional embedding
        self.add_pos_embed = AddViewPositionalEmbedding(input_dim, view_position_path, norm=norm)
        # eye-gaze heatmap
        # self.eye_modulate = MultiScaleGazeModulation(dim=input_dim, scales=[1, 2, 4], norm=norm)
        # projector
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, view_positions):
        x = self.add_pos_embed(x, view_positions)  # [B, N, D]
        return self.projector(x)


class VisionProjectorMa(nn.Module):
    """
    # original version in ma
    Project visual embedding into a shared space
    (including adding view positional embedding + eye-gaze heatmap + projector)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, view_position_path, norm='rms'):
        super().__init__()
        # view positional embedding
        self.add_pos_embed = AddViewPositionalEmbedding(input_dim, view_position_path, norm=norm)
        # eye-gaze heatmap
        self.eye_modulate = MultiScaleGazeModulation(dim=input_dim, scales=[1, 2, 4], norm=norm)
        # projector
        # self.projector = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, output_dim),
        # )
        self.head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1, stride=1, padding=0),  # change the dimension
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x, view_positions, gaze_heatmap=None, gaze_idx=None):
        x = self.add_pos_embed(x, view_positions)  # [B, N, D]

        if gaze_heatmap is not None:
            gaze_idx = torch.LongTensor(gaze_idx).to(x.device)
            # obtain samples with eye-gaze data
            visual_x = x[gaze_idx]  # [M, N, D]

            # add eye-gaze heatmap for patch embedding and CLS token not adjust
            patch_embed = self.eye_modulate(visual_x[:, 1:], gaze_heatmap)  # [M, N-1, D]
            image_embed = torch.cat([visual_x[:, :1, :], patch_embed], dim=1)  # [M, N, D]

            # construct new X tensor, insert image_embed into gaze_idx.
            x_updated = x.clone()
            x_updated.index_copy_(0, gaze_idx, image_embed)
            x = x_updated

        x = x.permute(0, 2, 1)
        x = self.head(x)
        return x.permute(0, 2, 1)
        # return self.projector(x)


class ReportDiseaseClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # the top 13 observations (4 statuses: positive, negative, uncertain, blank, 1, 0, 2, 3)
        self.disease_heads_4class = nn.ModuleList([
            nn.Linear(input_dim, 4) for _ in range(13)
        ])

        # the No findings observation (2 statuses: positive, negative, 1, 0)
        self.disease_head_2class = nn.Linear(input_dim, 2)

    def forward(self, x):  # x: (B, 196, 2560)
        logits_4class = [head(x) for head in self.disease_heads_4class]  # list of 13 (B, 4)
        logits_2class = self.disease_head_2class(x)  # (B, 2)
        # output list：13 (B, 4)，1 (B, 2)
        return logits_4class + [logits_2class]


class DiseaseClassifier(nn.Module):
    def __init__(self, input_dim, pool_type='avg'):
        super().__init__()
        self.id2status = {
            1: 'positive',
            2: 'negative',
            3: 'uncertain',
            0: 'blank'
        }

        self.diseases = [
            'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema',
            'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices', 'No Finding'
        ]

        if pool_type == 'self-attn':
            self.pool = LightSelfAttentionPooling(input_dim)
        else:
            self.pool = None
        # the top 13 observations (4 statuses: positive, negative, uncertain, blank, 1, 0, 2, 3)
        self.disease_heads_4class = nn.ModuleList([
            nn.Linear(input_dim, 4) for _ in range(13)
        ])

        # the No findings observation (2 statuses: positive, negative, 1, 0)
        self.disease_head_2class = nn.Linear(input_dim, 2)

    def forward(self, x):  # x: (B, 196, 2560)
        if self.pool is None:
            x = x.mean(dim=1)
        else:
            x = self.pool(x)
        logits_4class = [head(x) for head in self.disease_heads_4class]  # list of 13 (B, 4)
        logits_2class = self.disease_head_2class(x)  # (B, 2)
        # output list：13 (B, 4)，1 (B, 2)
        return logits_4class + [logits_2class]

    def get_prediction(self, logits):
        preds_4class = torch.argmax(torch.stack(logits[:13], dim=1), dim=-1)  # (B, 13)
        preds_2class = torch.argmax(logits[-1], dim=-1)  # (B)
        all_preds = torch.cat([preds_4class, preds_2class.unsqueeze(1)], dim=-1)

        return all_preds

    def get_prediction_content(self, logits):
        preds_4class = torch.argmax(torch.stack(logits[:13], dim=1), dim=-1)  # (B, 13)
        preds_2class = torch.argmax(logits[-1], dim=-1)  # (B)
        all_preds = torch.cat([preds_4class, preds_2class.unsqueeze(1)], dim=-1)

        predications = [
            {
                self.diseases[j]: self.id2status[all_preds[i, j].item()]
                for j in range(14)
                if all_preds[i, j].item() != 0
            }
            for i in range(all_preds.shape[0])
        ]
        return predications, all_preds


class MultiLabelDiseaseClassifier(nn.Module):
    def __init__(self, input_dim, pool_type='avg', num_diseases=14):
        super().__init__()
        self.id2status = {
            1: 'positive',
            0: 'negative',
        }

        self.diseases = [
            'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema',
            'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices', 'No Finding'
        ]

        if pool_type == 'self-attn':
            self.pool = LightSelfAttentionPooling(input_dim)
        else:
            self.pool = None
        self.classifier = nn.Linear(input_dim, num_diseases)

    def forward(self, x, return_probs=False):  # x: (B, 196, 2560)
        if self.pool is None:
            x = x.mean(dim=1)
        else:
            x = self.pool(x)
        logits = self.classifier(x)  # (b, num_diseases)
        if not return_probs:
            return logits
        else:
            return torch.sigmoid(logits)

    def get_prediction_content(self, probs):
        predications = []
        for i in range(probs.shape[0]):
            item = []
            for j in range(probs.shape[1]):
                if probs[i, j] >= 0.5:
                    item.append(self.diseases[j])
            predications.append(item)

        return predications


class MultiLabelMultiClassLoss(nn.Module):
    def __init__(self):
        super(MultiLabelMultiClassLoss, self).__init__()
        # 前13类是4分类任务，最后1类是2分类任务
        self.loss_fns = [nn.CrossEntropyLoss() for _ in range(14)]

    def forward(self, preds, targets):
        """
        preds: list of 14 tensors.
               - preds[0] to preds[12] shape: (B, 4)   -> 4-class logits
               - preds[13] shape: (B, 2)              -> 2-class logits
        targets: Tensor of shape (B, 14)
               - targets[:, 0:13] in [0,1,2,3], targets[:, 13] in [0,1]
        """
        loss = self.loss_fns[-1](preds[13], targets[:, 13])
        for i in range(13):
            loss += self.loss_fns[i](preds[i], targets[:, i])
        return loss / 14  # 取平均


def compute_class_frequency(ann_path):
    # obtain class_frequency
    ann_data = json.load(open(ann_path))['train']

    class_freqs = [[0, 0, 0, 0] for _ in range(13)]
    class_freqs.append([0, 0])
    for item in ann_data:
        for idx, sta in enumerate(item['disease_labels']):
            class_freqs[idx][sta] += 1
    return class_freqs


def compute_binary_multilabel_class_frequency(ann_path):
    # obtain class_frequency
    ann_data = json.load(open(ann_path))['train']

    class_freqs = np.zeros(14, dtype=np.int64)
    for item in ann_data:
        labels = np.array(item['disease_labels'])
        class_freqs += ((labels == 1) | (labels == 3)).astype(int)
    return class_freqs.tolist()


class LearnableWeightedCELoss(nn.Module):
    def __init__(self, num_classes, device, init_weights=None):
        super().__init__()
        if init_weights is None:
            init_weights = torch.ones(num_classes, device=device)
        self.class_weights = nn.Parameter(init_weights, requires_grad=True).to(device)

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)  # (B, C)

        # NLL Loss 自定义实现：按 weight 手动加权
        weights = F.softmax(self.class_weights, dim=0)  # normalized
        nll = F.nll_loss(log_probs, targets, reduction='none')  # (B,)
        sample_weights = weights[targets]  # 每个样本用其对应类的权重
        loss = (nll * sample_weights)
        return loss


class LearnableFocalLossWithPositiveBias(nn.Module):
    def __init__(self, class_freqs, device, gamma=2.0, positive_bias=2.0):
        """
        :param class_freqs: List[List[float]]，frequency for each disease（前13个为4类，最后1个为2类）
        :param gamma: Focal loss γ
        :param positive_bias: for 1 class (positive)
        """
        super().__init__()
        self.gamma = gamma
        self.positive_bias = positive_bias
        self.num_tasks = len(class_freqs)

        # 初始化 learnable class weights，基于 inverse frequency + bias
        self.weighted_ce_loss = []

        for freqs in class_freqs:
            freq_tensor = torch.tensor(freqs)
            inv_freq = 1.0 / (freq_tensor + 1e-6)  # avoid divided zero
            inv_freq = inv_freq / inv_freq.sum()

            # add positive bias for first 13 diseases
            if len(inv_freq) == 4:
                inv_freq[1] += self.positive_bias

            init_weight = inv_freq / inv_freq.sum()  # normalize
            ce_loss = LearnableWeightedCELoss(len(inv_freq), device, init_weight)
            self.weighted_ce_loss.append(ce_loss)

    def forward(self, preds, targets):
        """
        :param preds: List of logits tensors, each of shape (B, C)
        :param targets: Tensor of shape (B, 14)，ground truth
        :return: loss (scalar), predictions (B, 14)
        """
        total_loss = 0

        for i in range(self.num_tasks):
            logits = preds[i]  # (B, C)
            target = targets[:, i]  # (B,)
            ce_loss_func = self.weighted_ce_loss[i]

            # Focal loss 计算
            ce = ce_loss_func(logits, target)  # (B,)
            pt = torch.exp(-ce)
            focal_loss = ((1 - pt) ** self.gamma) * ce
            total_loss += focal_loss.mean()

        return total_loss / self.num_tasks




