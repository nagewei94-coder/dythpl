"""
losses.py
包含：AsymmetricLoss, AsymmetricLossOptimized, ASLSingleLabel,
      TPLoss, BaselineLoss, DyTHPLLoss
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from l2pprompt import Prompt  # 用于调用 selection_entropy_loss


# ============================================================
# ASL 系列
# ============================================================

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8,
                 disable_torch_grad_focal_loss=True):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        x = x.float()
        y = y.float()

        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


class AsymmetricLossOptimized(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8,
                 disable_torch_grad_focal_loss=False):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = \
            self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        x = x.float()
        y = y.float()

        self.targets = y
        self.anti_targets = 1 - y

        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(
                1 - self.xs_pos - self.xs_neg,
                self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets
            )
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()


class ASLSingleLabel(nn.Module):
    def __init__(self, gamma_pos=0, gamma_neg=4, eps: float = 0.1, reduction='mean'):
        super().__init__()
        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target):
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.zeros_like(inputs).scatter_(
            1, target.long().unsqueeze(1), 1)

        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                 self.gamma_pos * targets + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(
                self.eps / num_classes)

        loss = -self.targets_classes.mul(log_preds)
        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()
        return loss


# ============================================================
# TPLoss（原 TATHPL，保留不动）
# ============================================================

class TPLoss(torch.nn.Module):
    def __init__(self, base_criterion, sig_criterion, epoch):
        super().__init__()
        self.base_criterion = base_criterion
        self.sig_criterion = sig_criterion
        self.epoch = epoch

    def forward(self, outputs_list, labels_list):
        if isinstance(labels_list[1], torch.Tensor):
            croase_lable = labels_list[1]
        else:
            croase_lable = torch.stack(labels_list[1])

        sig_loss_0 = self.sig_criterion(
            outputs_list[1][0].squeeze(), croase_lable[:, 0].long())
        sig_loss_1 = self.sig_criterion(
            outputs_list[1][1].squeeze(), croase_lable[:, 1].long())

        if isinstance(labels_list[0], torch.Tensor):
            base_label = labels_list[0]
        else:
            base_label = torch.stack(labels_list[0])

        base_loss_final = self.base_criterion(outputs_list[0], base_label)
        weights = [1, 0.1, 0.1, 0.1, 0.1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        assert len(weights) >= 13
        loss = base_loss_final * weights[0] + sig_loss_0 * weights[1] + sig_loss_1 * weights[2]
        return loss, sig_loss_0, sig_loss_1, base_loss_final


# ============================================================
# BaselineLoss（纯 ViT，不使用任何 Topic 信息）
# ============================================================

class BaselineLoss(torch.nn.Module):
    def __init__(self, base_criterion):
        super().__init__()
        self.base_criterion = base_criterion

    def forward(self, outputs_list, labels_list):
        cls_logits = outputs_list[0]
        cls_targets = labels_list[0]
        base_loss = self.base_criterion(cls_logits, cls_targets)
        return base_loss, torch.tensor(0.0), torch.tensor(0.0), base_loss


import torch
import torch.nn as nn

class DyTHPLLoss(nn.Module):
    def __init__(self, cls_criterion, lambda_l2p=0.05, lambda_div=0.1, model=None):
        super().__init__()
        self.cls_criterion = cls_criterion 
        self.lambda_l2p = lambda_l2p     # L2P 匹配拉力权重
        self.lambda_div = lambda_div     # 多样性排斥力权重
        self.model = model               # 传入模型实体，用于获取分层 Prompt 池

    def forward(self, outputs, targets):
        """
        outputs: (logits, surrogate_losses, selected_keys, soft_probs_list)
        targets: [batch, num_classes]
        """
        # 1. 安全解包 (兼容不同数量的返回值)
        if isinstance(outputs, (tuple, list)):
            logits = outputs[0]
            surrogate_losses = outputs[1] if len(outputs) > 1 else []
        else:
            logits = outputs
            surrogate_losses = []

        # 2. 计算分类 Loss (ASL)
        cls_loss = self.cls_criterion(logits, targets.float())

        # 3. 计算 L2P Loss (拉近 Query 和被选中的 Key)
        if surrogate_losses and len(surrogate_losses) > 0:
            l2p_loss_sum = sum(surrogate_losses)
            l2p_final_loss = l2p_loss_sum * self.lambda_l2p
        else:
            l2p_final_loss = torch.tensor(0.0, device=logits.device)
            
        # 4. 【核心新增】计算池内多样性损失 (Intra-pool Diversity Loss)
        div_loss = torch.tensor(0.0, device=logits.device)
        
        # 只有在训练模式下，且传入了 model 时才计算多样性惩罚
        if self.training and self.model is not None:
            div_losses = []
            
            # 分别独立计算每一层的内部多样性 (Domain Isolation)
            if hasattr(self.model, 'pool_0') and self.model.pool_0 is not None:
                div_losses.append(self.model.pool_0.intra_pool_diversity_loss())
                
            if hasattr(self.model, 'pool_mid') and self.model.pool_mid is not None:
                div_losses.append(self.model.pool_mid.intra_pool_diversity_loss())
                
            if hasattr(self.model, 'pool_1') and self.model.pool_1 is not None:
                div_losses.append(self.model.pool_1.intra_pool_diversity_loss())
            
            # 求所有活跃池子的多样性损失均值，并乘以权重
            if len(div_losses) > 0:
                div_loss = sum(div_losses) / len(div_losses) * self.lambda_div

        # 5. 计算总损失
        total_loss = cls_loss + l2p_final_loss + div_loss
        
        # 返回 4 个值，兼容你的 train_one_epoch 打印逻辑
        return total_loss, cls_loss, l2p_final_loss, div_loss