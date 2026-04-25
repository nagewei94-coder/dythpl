import argparse
import torch
import numpy as np
import os
from tqdm import tqdm
from sklearn.cluster import KMeans
from timm.models import create_model
import model_learn  # 注册模型
from helper_functions import voc2007_DyTHPL, Corel5k_DyTHPL
import torchvision.transforms as transforms

def get_args():
    parser = argparse.ArgumentParser(description='Prompt Pool Warmup using K-Means (3 Layers)')
    parser.add_argument('--dataset', type=str, required=True, choices=['voc', 'corel5k'])
    parser.add_argument('--data_path', default='.', type=str)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output_path', default='./output_warmup', type=str)
    
    # 【修改】支持三个池子的大小配置
    parser.add_argument('--n_coarse', default=5, type=int)  # Pool 0
    parser.add_argument('--n_mid', default=20, type=int)    # Pool Mid (新增)
    parser.add_argument('--n_fine', default=50, type=int)   # Pool 1
    
    return parser.parse_args()

@torch.no_grad()
def extract_all_queries(loader, model, device):
    """提取所有训练图片的 Query 特征 (3层)"""
    print("Extracting features from frozen ViT...")
    model.eval()
    
    coarse_feats = []
    mid_feats = []
    fine_feats = []
    
    for images, _ in tqdm(loader):
        images = images.to(device)
        # 使用冻结模型提取 3 层特征
        # 注意：这里的层索引要和你 model_learn.py 里的一致 (6, 10)
        q_coarse, q_mid, q_fine = model.frozen_query_vit.extract_queries(
            images, layer_idx_mid=6, layer_idx_fine=10
        )
        coarse_feats.append(q_coarse.cpu().numpy())
        mid_feats.append(q_mid.cpu().numpy())
        fine_feats.append(q_fine.cpu().numpy())
        
    return np.concatenate(coarse_feats), np.concatenate(mid_feats), np.concatenate(fine_feats)

def run_kmeans(features, n_clusters, name):
    print(f"Running K-Means for {name} (K={n_clusters})...")
    # L2 归一化 (因为 L2P 使用余弦相似度)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / (norms + 1e-10)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(features)
    
    # 聚类中心即为初始化的 Keys
    centroids = kmeans.cluster_centers_
    # 再次归一化中心点
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)
    
    return torch.from_numpy(centroids).float()

def main():
    args = get_args()
    device = torch.device(args.device)
    os.makedirs(args.output_path, exist_ok=True)
    
    # 1. 准备数据
    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if args.dataset == 'voc':
        dataset = voc2007_DyTHPL(data_path=args.data_path, transform=transform, is_train=True)
        # num_classes 无所谓，只用来创建模型骨架
        num_classes = 20
    else:
        dataset = Corel5k_DyTHPL(data_path=args.data_path, transform=transform, is_train=True)
        num_classes = 260
        
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # 2. 准备模型
    print("Creating model backbone...")
    model = create_model('dythpl_base_patch16_224', pretrained=True, num_classes=num_classes)
    model.to(device)
    
    # 3. 提取特征 (3层)
    feats_coarse, feats_mid, feats_fine = extract_all_queries(loader, model, device)
    print(f"Features extracted:")
    print(f"  Coarse: {feats_coarse.shape}")
    print(f"  Mid:    {feats_mid.shape}")
    print(f"  Fine:   {feats_fine.shape}")
    
    # 4. K-Means 聚类 (3次)
    keys_coarse = run_kmeans(feats_coarse, args.n_coarse, "Coarse Pool")
    keys_mid = run_kmeans(feats_mid, args.n_mid, "Mid Pool")
    keys_fine = run_kmeans(feats_fine, args.n_fine, "Fine Pool")
    
    # 5. 保存结果
    save_file = os.path.join(args.output_path, f'{args.dataset}_warmup_keys.pth')
    torch.save({
        'pool_0_keys': keys_coarse,
        'pool_mid_keys': keys_mid, # 【新增】
        'pool_1_keys': keys_fine
    }, save_file)
    
    print(f"✅ Warmup keys saved to {save_file}")

if __name__ == '__main__':
    main()