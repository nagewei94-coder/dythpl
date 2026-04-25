# 文件名: coco_dythpl.py

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
from pathlib import Path

from timm.models import create_model
from timm.utils import NativeScaler
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy, RandomErasing
from torch.cuda.amp import autocast
from sklearn.metrics import average_precision_score, multilabel_confusion_matrix, hamming_loss

# 引入项目组件
import model_learn
from losses import DyTHPLLoss
from loss import AsymmetricLoss
from helper_functions import Coco_DyTHPL, add_weight_decay, one_error, get_auc, micro_f1, macro_f1, PartialModelEma
import utils

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

def get_args_parser():
    parser = argparse.ArgumentParser('MS-COCO DyT-HPL Training', add_help=False)
    
    # 基础参数
    parser.add_argument('--batch-size', default=32, type=int) # COCO建议大一点，显存不够改32
    parser.add_argument('--epochs', default=50, type=int)     # COCO通常需要多跑几轮
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR')
    parser.add_argument('--weight-decay', type=float, default=0.05) # 大数据集用标准 WD
    parser.add_argument('--input-size', default=224, type=int)
    
    # 模型
    parser.add_argument('--model', default='dythpl_base_patch16_224', type=str)
    parser.add_argument('--num_classes', default=80, type=int) # COCO 是 80 类
    
    # DyT-HPL 参数
    parser.add_argument('--lambda-l2p', type=float, default=0.2)
    parser.add_argument('--use-layerwise-lr', action='store_true', default=True)
    parser.add_argument('--vit-lr-scale', type=float, default=0.2) # 大数据可以让骨干多学点
    parser.add_argument('--head-lr-scale', type=float, default=10.0)
    
    # 路径 (你需要根据实际情况修改 data_coco)
    parser.add_argument('--data_coco', default='.', type=str)
    parser.add_argument('--output_dir', default='./output_coco_dythpl')
    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    
    return parser

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

    # 强力数据增强 (COCO 标配)
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

    print("\nLoading MS-COCO data...")
    # 路径拼接
    train_ann_file = os.path.join(args.data_coco, 'annotations', 'instances_train2014.json')
    val_ann_file = os.path.join(args.data_coco, 'annotations', 'instances_val2014.json')
    train_img_root = os.path.join(args.data_coco, 'train2014')
    val_img_root = os.path.join(args.data_coco, 'val2014')

    dataset_train = Coco_DyTHPL(root=train_img_root, annFile=train_ann_file, transform=train_transform)
    dataset_val = Coco_DyTHPL(root=val_img_root, annFile=val_ann_file, transform=val_transform)

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
        pretrained=True,
        num_classes=args.num_classes,
        # warmup_path=... # 如果跑了 COCO 的预热脚本，这里可以加
    )
    model.to(device)

    train_coco(model, train_loader, val_loader, args)

def train_coco(model, train_loader, val_loader, args):
    # 使用 Partial EMA
    print(f"Initializing Partial EMA...")
    ema = PartialModelEma(model, decay=0.9997, device=args.device)
    
    # 分层学习率
    if args.use_layerwise_lr:
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
        epochs=args.epochs, pct_start=0.2
    )
    
    loss_scaler = NativeScaler()
    best_map = 0.0
    
    # Loss: 标准 ASL，gamma=4 适合 COCO
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
        
        mAP_score = validate_coco(val_loader, model, ema)
        
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

# 训练循环 (和之前一样)
def train_one_epoch(model, criterion, loader, optimizer, device, epoch, loss_scaler, scheduler, ema):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'Epoch: [{epoch}]'
    
    for images, targets in metric_logger.log_every(loader, 100, header): # COCO batch多，打印频率调低
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        with autocast():
            outputs = model(images)
            loss_tuple = criterion(outputs, targets)
            loss, cls_loss, l2p_loss = loss_tuple[0], loss_tuple[1], loss_tuple[2]
        
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

# COCO 验证函数
@torch.no_grad()
def validate_coco(val_loader, model, ema_model):
    model.eval()
    if ema_model: ema_model.module.eval()
    
    preds_reg, preds_ema, targets_all = [], [], []
    
    for images, targets in val_loader:
        images = images.to(torch.device('cuda'), non_blocking=True)
        with autocast():
            out_reg = model(images)
            preds_reg.append(torch.sigmoid(out_reg).cpu())
            if ema_model:
                out_ema = ema_model.module(images)
                preds_ema.append(torch.sigmoid(out_ema).cpu())
        targets_all.append(targets.cpu())
        
    targets_np = torch.cat(targets_all).numpy()
    preds_reg_np = torch.cat(preds_reg).numpy()
    
    def calc_map_sklearn(targs, preds):
        ap_list = []
        for i in range(targs.shape[1]):
            # COCO 验证集应该都有正样本，但防万一
            if np.sum(targs[:, i]) > 0:
                ap_list.append(average_precision_score(targs[:, i], preds[:, i]))
        return np.mean(ap_list) * 100 if ap_list else 0.0

    mAP_regular = calc_map_sklearn(targets_np, preds_reg_np)
    mAP_ema = 0.0
    if ema_model:
        preds_ema_np = torch.cat(preds_ema).numpy()
        mAP_ema = calc_map_sklearn(targets_np, preds_ema_np)
    
    # 打印简要指标
    print(f"mAP Reg: {mAP_regular:.3f}, mAP EMA: {mAP_ema:.3f}")
    
    return max(mAP_regular, mAP_ema)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('COCO DyT-HPL', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)