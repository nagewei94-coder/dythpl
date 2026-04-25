import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import sys
import os
from pathlib import Path

from timm.models import create_model
from timm.utils import NativeScaler
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torch.cuda.amp import autocast
from sklearn.metrics import average_precision_score, multilabel_confusion_matrix, hamming_loss
from torchvision.transforms import AutoAugment, AutoAugmentPolicy, RandomErasing

# --- 引入项目组件 ---
import model_learn  # 注册模型
from losses import DyTHPLLoss 
from loss import AsymmetricLoss
# 引入 Corel5k 专用加载器 和 PartialModelEma
from helper_functions import Corel5k_DyTHPL, add_weight_decay, one_error, get_auc, micro_f1, macro_f1, PartialModelEma
import utils

# ============================================================
# Logger
# ============================================================
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ============================================================
# Args
# ============================================================
def get_args_parser():
    parser = argparse.ArgumentParser('Corel5k DyT-HPL Training', add_help=False)
    
    # 基础参数
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=60, type=int) # Corel5k可能需要多跑几轮
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR')
    parser.add_argument('--weight-decay', type=float, default=0.05) # 可以尝试 0.05
    parser.add_argument('--input-size', default=224, type=int)
    
    # 模型参数
    parser.add_argument('--model', default='dythpl_base_patch16_224', type=str)
    parser.add_argument('--num_classes', default=260, type=int) # Corel5k 是 260 类
    
    # Loss 参数
    parser.add_argument('--lambda-l2p', type=float, default=0.2, help='L2P查询损失权重')
    
    # 学习率策略
    parser.add_argument('--use-layerwise-lr', action='store_true', help='开启分层学习率')
    parser.add_argument('--warmup-pct', type=float, default=0.2)
    parser.add_argument('--vit-lr-scale', type=float, default=0.3) # 骨干学习率倍率
    parser.add_argument('--head-lr-scale', type=float, default=10.0) # 分类头学习率倍率
    
    # 路径与环境
    parser.add_argument('--output_dir', default='./output_corel5k_dythpl')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=4, type=int) # Windows 必须为 0
    parser.add_argument('--data_path', default='.', type=str)

    parser.add_argument('--warmup-path', default='./output_warmup/corel5k_warmup_keys.pth', type=str)
    
    return parser

# ============================================================
# Main
# ============================================================
def main(args):
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        sys.stdout = Logger(os.path.join(args.output_dir, 'log.txt'))

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # -----------------------------------------------------------------------
    # 【方案一增强版】训练集数据增强
    # 包含：Resize -> HorizontalFlip -> AutoAugment -> ToTensor -> Normalize -> RandomErasing
    # -----------------------------------------------------------------------
    print("\n[Data Augmentation] Enabled AutoAugment (ImageNet Policy) & RandomErasing.")
    train_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        
        # 1. 随机水平翻转
        transforms.RandomHorizontalFlip(p=0.5),
        
        # 2. 【新增】AutoAugment: 自动应用最佳的旋转、剪切、色彩变换组合
        transforms.AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        
        # 3. 【新增】RandomErasing: 随机遮挡，强迫模型关注物体局部特征
        # p=0.25: 有25%的概率执行遮挡
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
    ])

    # 验证集保持干净，只做 Resize 和 Normalize
    val_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集 (使用 Corel5k 加载器)
    print("\n" + "="*60)
    print("Loading Corel5k data...")
    print("="*60)
    
    dataset_train = Corel5k_DyTHPL(data_path=args.data_path, transform=train_transform, is_train=True)
    dataset_val = Corel5k_DyTHPL(data_path=args.data_path, transform=val_transform, is_train=False) 

    print(f"  ✓ Train: {len(dataset_train)} samples")
    print(f"  ✓ Val:   {len(dataset_val)} samples")

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # 创建模型
    print(f"\nCreating Model: {args.model} (num_classes={args.num_classes})")
    model = create_model(
        args.model,
        pretrained=True,
        num_classes=args.num_classes,
        # warmup_path=args.warmup_path # 如果需要预热，取消注释并确保 args 有这个参数
    )
    model.to(device)

    train_corel5k(model, train_loader, val_loader, args)

# ============================================================
# Training Loop
# ============================================================
def train_corel5k(model, train_loader, val_loader, args):
    # 使用 PartialEMA，避免平滑 Prompt Pool
    print(f"Initializing Partial EMA (decay={0.99})...")
    ema = PartialModelEma(model, decay=0.99, device=args.device)
    
    # 优化器设置
    if args.use_layerwise_lr:
        print("\n[Optimization] Using Layer-wise LR...")
        prompt_params = []
        vit_params = []
        head_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            if 'pool' in name: prompt_params.append(param)
            elif 'head' in name: head_params.append(param)
            else: vit_params.append(param)
        
        param_groups = [
            {'params': prompt_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
            {'params': vit_params, 'lr': args.lr * args.vit_lr_scale, 'weight_decay': args.weight_decay},
            {'params': head_params, 'lr': args.lr * args.head_lr_scale, 'weight_decay': args.weight_decay},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0)
        max_lrs = [args.lr, args.lr * args.vit_lr_scale, args.lr * args.head_lr_scale]
    else:
        parameters = add_weight_decay(model, args.weight_decay)
        optimizer = torch.optim.AdamW(params=parameters, lr=args.lr, weight_decay=0)
        max_lrs = args.lr

    scheduler = lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lrs, steps_per_epoch=len(train_loader),
        epochs=args.epochs, pct_start=args.warmup_pct
    )
    
    loss_scaler = NativeScaler()
    best_map = 0.0
    
    # Loss: Corel5k 标签稀疏，建议 gamma_neg=2
    cls_criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05)
    criterion = DyTHPLLoss(cls_criterion, lambda_l2p=args.lambda_l2p).to(args.device)

    print(f"\nStart training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.epochs):
        train_one_epoch(
            model, criterion, train_loader,
            optimizer, args.device, epoch,
            loss_scaler, scheduler, ema
        )
        
        mAP_score = validate_corel5k(val_loader, model, ema)
        
        if mAP_score > best_map:
            best_map = mAP_score
            if args.output_dir:
                torch.save({
                    'model': model.state_dict(),
                    'ema': ema.module.state_dict(),
                    'epoch': epoch,
                    'mAP': mAP_score
                }, Path(args.output_dir) / 'best_dythpl.pth')
                print(f'✓ Saved best model: mAP={mAP_score:.3f}')
        
        print(f'Epoch {epoch}: current={mAP_score:.3f}, best={best_map:.3f}\n')

    total_time = time.time() - start_time
    print(f'Training time {datetime.timedelta(seconds=int(total_time))}')

    # ==========================================
    # 【新增】训练结束后，加载最佳模型并打印详细 AP
    # ==========================================
    print("\n" + "="*60)
    print("--- Starting Final Evaluation on Best Model ---")
    print("="*60)
    
    best_path = Path(args.output_dir) / 'best_dythpl.pth'
    if best_path.exists():
        # 加载 checkpoint
        # 【修复】显式允许加载非 Tensor 数据 (如 numpy 标量)
        checkpoint = torch.load(best_path, map_location=args.device, weights_only=False)
        
        # 优先加载 EMA 权重 (通常 EMA 泛化更好)
        if 'ema' in checkpoint and checkpoint['ema'] is not None:
            print("Loading Best EMA weights for evaluation...")
            model.load_state_dict(checkpoint['ema'])
        else:
            print("Loading Best Regular weights for evaluation...")
            model.load_state_dict(checkpoint['model'])
            
        # 调用验证函数，开启详细打印模式
        validate_corel5k(val_loader, model, None, print_detailed_ap=True)
    else:
        print("Error: Best model file not found!")

# ============================================================
# One Epoch Logic
# ============================================================
def train_one_epoch(model, criterion, loader, optimizer, device, epoch, loss_scaler, scheduler, ema):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'Epoch: [{epoch}]'
    
    for images, targets in metric_logger.log_every(loader, 50, header):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        with autocast():
            outputs = model(images)
            # 适配 DyTHPLLoss 的 4 个返回值
            loss_tuple = criterion(outputs, targets)
            loss, cls_loss, l2p_loss = loss_tuple[0], loss_tuple[1], loss_tuple[2]
            # div_loss = loss_tuple[3] # 如果有的话
        
        if not np.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping")
            sys.exit(1)
            
        optimizer.zero_grad(set_to_none=True)
        loss_scaler(loss, optimizer, parameters=model.parameters())
        scheduler.step()
        
        if ema: ema.update(model)
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(cls_loss=cls_loss.item())
        metric_logger.update(l2p_loss=l2p_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

# ============================================================
# Corel5k Validation
# ============================================================
# 文件: corel5k_dythpl.py

@torch.no_grad()
def validate_corel5k(val_loader, model, ema_model, print_detailed_ap=False):
    print("Starting validation...")
    model.eval()
    if ema_model: ema_model.module.eval()
    
    preds_reg, preds_ema, targets_all = [], [], []
    
    for images, targets in val_loader:
        images = images.to(torch.device('cuda'), non_blocking=True)
        with autocast():
            # Regular
            out_reg = model(images)
            preds_reg.append(torch.sigmoid(out_reg).cpu())
            # EMA
            if ema_model:
                out_ema = ema_model.module(images)
                preds_ema.append(torch.sigmoid(out_ema).cpu())
        targets_all.append(targets.cpu())
        
    targets_np = torch.cat(targets_all).numpy()
    preds_reg_np = torch.cat(preds_reg).numpy()
    
    # 1. 计算 mAP
    def calc_map_sklearn(targs, preds):
        ap_list = []
        for i in range(targs.shape[1]):
            if np.sum(targs[:, i]) > 0:
                ap_list.append(average_precision_score(targs[:, i], preds[:, i]))
            else:
                ap_list.append(float('nan'))
        valid_aps = [ap for ap in ap_list if not np.isnan(ap)]
        mAP = np.mean(valid_aps) * 100 if valid_aps else 0.0
        return mAP, ap_list

    mAP_regular, ap_list_reg = calc_map_sklearn(targets_np, preds_reg_np)
    
    mAP_ema = 0.0
    if ema_model:
        preds_ema_np = torch.cat(preds_ema).numpy()
        mAP_ema, _ = calc_map_sklearn(targets_np, preds_ema_np)
    
    # 2. 计算其他所有指标 (使用 Regular 结果)
    preds_binary = np.where(preds_reg_np > 0.5, 1, 0)
    mcm = multilabel_confusion_matrix(targets_np, preds_binary)
    
    # 解包获取详细指标
    mif1_score, op_score, or_score = micro_f1(mcm)
    maf1_score, cp_score, cr_score = macro_f1(mcm)
    
    one_err = one_error(torch.tensor(targets_np), torch.tensor(preds_reg_np))
    auc_score = get_auc(targets_np, preds_reg_np)
    hl_score = hamming_loss(targets_np, preds_binary)
    
    # 3. 【修改】打印完整指标 (与 VOC 格式对齐)
    print("mAP Reg: {:.3f}, mAP EMA: {:.3f}, OF1: {:.3f}, OP: {:.3f}, OR: {:.3f}, CF1: {:.3f}, CP: {:.3f}, CR: {:.3f}, OneErr: {:.4f}, AUC: {:.4f}, HL: {:.4f}".format(
        mAP_regular, mAP_ema, 
        mif1_score, op_score, or_score,
        maf1_score, cp_score, cr_score,
        one_err, auc_score, hl_score
    ))
    
    # 4. 详细 AP 打印
    if print_detailed_ap:
        if hasattr(val_loader.dataset, 'VOC_CLASSES'):
            class_names = val_loader.dataset.VOC_CLASSES
            print("\n" + "="*40)
            print("Per-class Average Precision (AP)")
            print("="*40)
            count = 0
            for i, class_name in enumerate(class_names):
                ap = ap_list_reg[i]
                if not np.isnan(ap):
                    print(f"{class_name[:20]:<22}: {ap*100:6.2f}%  ", end="")
                    count += 1
                    if count % 2 == 0: print()
            print("\n" + "="*40)

    return max(mAP_regular, mAP_ema)

def calc_map(targs, preds):
    ap_list = []
    for i in range(targs.shape[1]):
        if np.sum(targs[:, i]) > 0:
            ap_list.append(average_precision_score(targs[:, i], preds[:, i]))
    return np.mean(ap_list) * 100 if ap_list else 0.0

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Corel5k DyT-HPL', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)