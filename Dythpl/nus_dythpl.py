import warnings
warnings.filterwarnings("ignore")

import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import sys
import os
import pandas as pd
from PIL import Image
from pathlib import Path

from timm.models import create_model
from timm.utils import NativeScaler
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy, RandomErasing
from torch.cuda.amp import autocast
from sklearn.metrics import average_precision_score, multilabel_confusion_matrix, hamming_loss
from safetensors.torch import load_file

# 引入项目组件
import model_learn  # 必须导入以注册 DyT-HPL 模型
from losses import AsymmetricLossOptimized, DyTHPLLoss
from helper_functions import (
    PartialModelEma, 
    add_weight_decay, 
    micro_f1, 
    macro_f1, 
    get_auc, 
    one_error
)
import utils


# ==========================================
# 1. 日志记录器
# ==========================================
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


# ==========================================
# 2. NUS-WIDE 数据集定义
# ==========================================
class NUSWIDEDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.classes = self.df.columns[2:-1].tolist()
        self.labels = self.df.drop(columns=['imageid', 'phase', 'num_label']).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = str(self.df.iloc[idx]['imageid']) + '.jpg'
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


# ==========================================
# 3. 参数配置 (新增 DyT-HPL 专属参数)
# ==========================================
def get_args_parser():
    parser = argparse.ArgumentParser('NUS-WIDE DyT-HPL Training', add_help=False)
    
    # 基础参数
    parser.add_argument('--batch-size', default=128, type=int) # 若显存爆了请改为 32
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr', type=float, default=2e-4, metavar='LR')
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--input-size', default=224, type=int)
    
    # 模型 (调用你注册好的大创模型)
    parser.add_argument('--model', default='dythpl_base_patch16_224', type=str)
    parser.add_argument('--num_classes', default=81, type=int)
    
    # DyT-HPL 专属参数
    parser.add_argument('--lambda-l2p', type=float, default=0.2)
    parser.add_argument('--lambda-div', type=float, default=0.1)
    parser.add_argument('--use-layerwise-lr', action='store_true', default=True)
    parser.add_argument('--vit-lr-scale', type=float, default=0.2)
    parser.add_argument('--head-lr-scale', type=float, default=10.0)
    parser.add_argument('--pool-config', default='full', type=str, 
                        choices=['baseline', 'single_deep', 'single_shallow', 'two', 'full'],
                        help='Prompt注入策略 (结构消融)')
    parser.add_argument('--warmup-path', default='', type=str,
                        help='离线聚类向量的权重路径 (可选)')
    
    # 路径配置
    parser.add_argument('--data_dir', default='/root/autodl-tmp/NUS_WIDE/data', type=str)
    parser.add_argument('--output_dir', default='./output_dythpl')
    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=8, type=int)

    return parser


# ==========================================
# 4. 主函数
# ==========================================
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

    # 保持与基线绝对一致的数据增强
    train_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("\nLoading NUS-WIDE data...")
    train_csv = os.path.join(args.data_dir, 'train.csv')
    test_csv = os.path.join(args.data_dir, 'test.csv')
    img_dir = os.path.join(args.data_dir, 'images')

    dataset_train = NUSWIDEDataset(csv_file=train_csv, img_dir=img_dir, transform=train_transform)
    dataset_val = NUSWIDEDataset(csv_file=test_csv, img_dir=img_dir, transform=val_transform)

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

    print(f"Creating DyT-HPL Model: {args.model}")
    model = create_model(
        args.model,
        pretrained=True, # 这里设为True，触发 model_learn 里加载本地权重的逻辑
        num_classes=args.num_classes,
        pool_config=args.pool_config,
        warmup_path=args.warmup_path
    )
    model.to(device)

    train_dythpl(model, train_loader, val_loader, args)


# ==========================================
# 5. 训练控制逻辑 (分层 LR + DyT-HPL Loss)
# ==========================================
def train_dythpl(model, train_loader, val_loader, args):
    print("Initializing Partial EMA for Prompt Learning...")
    ema = PartialModelEma(model, decay=0.9997, device=args.device)
    
    # 【核心】：分层学习率 (Layer-wise LR)
    if args.use_layerwise_lr:
        print("-> Using Layer-wise Learning Rate Setup.")
        prompt_params = []
        vit_params = []
        head_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            if 'pool' in name or 'prompt' in name: 
                prompt_params.append(param)
            elif 'head' in name: 
                head_params.append(param)
            else: 
                vit_params.append(param)
        
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
        epochs=args.epochs, pct_start=0.2
    )
    
    loss_scaler = NativeScaler()
    best_map = 0.0
    
    # 组合 Loss: ASL 分类 + L2P 拉力 + 池内多样性排斥力
    cls_criterion = AsymmetricLossOptimized(gamma_neg=4, gamma_pos=0, clip=0.05)
    criterion = DyTHPLLoss(
        cls_criterion=cls_criterion, 
        lambda_l2p=args.lambda_l2p, 
        lambda_div=args.lambda_div, 
        model=model # 传入模型以便计算内部的池排斥
    ).to(args.device)

    print(f"\nStart DyT-HPL training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.epochs):
        train_one_epoch(
            model, criterion, train_loader,
            optimizer, args.device, epoch,
            loss_scaler, scheduler, ema
        )
        
        mAP_score = validate_nuswide(val_loader, model, ema)
        
        if mAP_score > best_map:
            best_map = mAP_score
            if args.output_dir:
                torch.save({
                    'model': model.state_dict(),
                    'ema': ema.module.state_dict(),
                    'epoch': epoch,
                    'mAP': mAP_score
                }, Path(args.output_dir) / 'best_dythpl.pth')
                print(f'🌟 Saved best DyT-HPL model: mAP={mAP_score:.3f}')
        
        print(f'Epoch {epoch}: current={mAP_score:.3f}, best={best_map:.3f}\n')

    total_time = time.time() - start_time
    print(f'Training time {datetime.timedelta(seconds=int(total_time))}')


# ==========================================
# 6. 单个 Epoch 训练逻辑
# ==========================================
def train_one_epoch(model, criterion, loader, optimizer, device, epoch, loss_scaler, scheduler, ema):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'Epoch: [{epoch}]'
    
    for images, targets in metric_logger.log_every(loader, 100, header):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        with autocast():
            # DyT-HPL 前向传播返回 4 个值
            outputs = model(images)
            # DyTHPLLoss 会自动解包并计算 3 种 Loss
            loss_tuple = criterion(outputs, targets)
            loss, cls_loss, l2p_loss, div_loss = loss_tuple[0], loss_tuple[1], loss_tuple[2], loss_tuple[3]
        
        if not np.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping")
            sys.exit(1)
            
        optimizer.zero_grad(set_to_none=True)
        loss_scaler(loss, optimizer, parameters=model.parameters())
        scheduler.step()
        
        if ema: 
            ema.update(model)
        
        # 详细记录三大核心指标，方便在 log 里排查模型是在努力分类还是在拉扯 Prompt
        metric_logger.update(loss=loss.item())
        metric_logger.update(cls=cls_loss.item())
        metric_logger.update(l2p=l2p_loss.item())
        metric_logger.update(div=div_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


# ==========================================
# 7. 验证逻辑
# ==========================================
# ==========================================
# 7. 验证逻辑 (全指标精装版)
# ==========================================
@torch.no_grad()
def validate_nuswide(val_loader, model, ema_model, print_detailed_ap=False):
    model.eval()
    if ema_model: 
        ema_model.module.eval()
    
    print("Starting validation on NUS-WIDE...")
    preds_reg, preds_ema, targets_all = [], [], []
    
    for images, targets in val_loader:
        images = images.to(torch.device('cuda'), non_blocking=True)
        with autocast():
            # 推理阶段
            out_reg = model(images)
            preds_reg.append(torch.sigmoid(out_reg).cpu())
            if ema_model:
                out_ema = ema_model.module(images)
                preds_ema.append(torch.sigmoid(out_ema).cpu())
        targets_all.append(targets.cpu())
        
    targets_np = torch.cat(targets_all).numpy()
    preds_reg_np = torch.cat(preds_reg).numpy()
    
    # 1. 计算 mAP
    def calc_map(targs, preds):
        ap_list = []
        for i in range(targs.shape[1]):
            if np.sum(targs[:, i]) > 0:
                ap_list.append(average_precision_score(targs[:, i], preds[:, i]))
            else:
                ap_list.append(float('nan'))
        valid = [ap for ap in ap_list if not np.isnan(ap)]
        return (np.mean(valid) * 100 if valid else 0.0), ap_list

    mAP_reg, ap_list_reg = calc_map(targets_np, preds_reg_np)
    mAP_ema = 0.0
    if ema_model:
        preds_ema_np = torch.cat(preds_ema).numpy()
        mAP_ema, _ = calc_map(targets_np, preds_ema_np)
        
    # 2. 计算其他高级指标 (F1, P, R, AUC, HL等)
    preds_binary = (preds_reg_np > 0.5).astype(np.int32)
    mcm = multilabel_confusion_matrix(targets_np, preds_binary)
    mif1, op, or_ = micro_f1(mcm)
    maf1, cp, cr  = macro_f1(mcm)
    one_err       = one_error(torch.tensor(targets_np), torch.tensor(preds_reg_np))
    auc_score     = get_auc(targets_np, preds_reg_np)
    hl            = hamming_loss(targets_np, preds_binary)

    # 3. 豪华打印输出
    print(
        "mAP Reg: {:.3f}  mAP EMA: {:.3f} | "
        "OF1: {:.4f}  OP: {:.4f}  OR: {:.4f} | "
        "CF1: {:.4f}  CP: {:.4f}  CR: {:.4f} | "
        "OneErr: {:.4f}  AUC: {:.4f}  HL: {:.4f}".format(
            mAP_reg, mAP_ema, mif1, op, or_,
            maf1, cp, cr, one_err, auc_score, hl
        )
    )

    # 如果需要打印每个类别的详细 AP (需将 print_detailed_ap 设为 True)
    if print_detailed_ap and hasattr(val_loader.dataset, 'classes'):
        class_names = val_loader.dataset.classes
        print("\n" + "=" * 50 + "\nPer-class AP\n" + "=" * 50)
        count = 0
        for i, name in enumerate(class_names):
            ap = ap_list_reg[i]
            if not np.isnan(ap):
                print(f"{name[:20]:<22}: {ap * 100:6.2f}%  ", end="")
                count += 1
                if count % 2 == 0:
                    print()
        print("\n" + "=" * 50)

    return max(mAP_reg, mAP_ema)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('NUS-WIDE DyT-HPL', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)