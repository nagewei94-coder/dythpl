import torch
import torch.nn as nn
import torch.nn.functional as F

class Prompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean',
                 prompt_init='uniform', prompt_pool=False,
                 prompt_key=False, pool_size=None, top_k=None,
                 batchwise_prompt=False, prompt_key_init='uniform'):
        super().__init__()

        self.length        = length
        self.embed_dim     = embed_dim
        self.prompt_pool   = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init   = prompt_init
        self.prompt_key    = prompt_key
        self.pool_size     = pool_size
        self.top_k         = top_k
        self.batchwise_prompt = batchwise_prompt

        if self.prompt_pool:
            prompt_pool_shape = (pool_size, length, embed_dim)
            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)

        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """对给定的向量或矩阵进行 L2 归一化"""
        norm = torch.norm(x, p=2, dim=dim, keepdim=True)
        return x / (norm + epsilon)

    def forward(self, x_embed, frozen_query=None):
        out = dict()

        # 1. 确定 Query
        if frozen_query is not None:
            query = frozen_query
        elif self.embedding_key == 'cls' and x_embed.dim() == 3:
            query = x_embed[:, 0, :]
        else:
            query = x_embed

        # 2. 计算相似度并选择 Top-K
        query_norm = self.l2_normalize(query, dim=1)
        key_norm   = self.l2_normalize(self.prompt_key, dim=1)
        similarity = torch.matmul(query_norm, key_norm.t())

        _, idx = torch.topk(similarity, k=self.top_k, dim=1)

        if self.batchwise_prompt:
            idx = idx[0].unsqueeze(0).expand(x_embed.shape[0], -1)

        # 3. 提取对应的 Prompt Value
        batched_prompt_raw = self.prompt[idx]
        batch_size, top_k, length, c = batched_prompt_raw.shape
        batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c)

        # 4. 计算 L2P 代理匹配损失 (Surrogate Loss)
        batched_key_norm = key_norm[idx]
        sim              = (batched_key_norm * query_norm.unsqueeze(1)).sum(dim=-1)
        reduce_sim       = sim.sum() / batch_size

        # (可选保留) 提取 soft_probs 供后续分析或备用方案使用
        tau = 0.5
        soft_probs = F.softmax(similarity / tau, dim=1)

        # 5. 组装输出字典
        out['reduce_sim']         = 1.0 - reduce_sim
        out['selected_key']       = batched_key_norm
        out['total_prompt_len']   = batched_prompt.shape[1]
        out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)
        out['prompt_idx']         = idx
        out['soft_probs']         = soft_probs

        return out

    def init_from_vectors(self, keys_tensor):
        """使用外部离线聚类好的语义向量初始化 Key 和 Value"""
        with torch.no_grad():
            if self.prompt_key.shape == keys_tensor.shape:
                print(f"  -> Initializing Prompt Keys from external vectors "
                      f"(shape {keys_tensor.shape})")
                self.prompt_key.data.copy_(keys_tensor)

                if (self.prompt.shape[0] == keys_tensor.shape[0] and
                        self.prompt.shape[2] == keys_tensor.shape[1]):
                    print(f"  -> Also initializing Prompt Values from keys...")
                    # 必须使用 .clone()，防止显存地址复用导致后续梯度计算异常
                    expanded_keys = keys_tensor.unsqueeze(1).expand(
                        -1, self.length, -1
                    ).clone()
                    self.prompt.data.copy_(expanded_keys)
            else:
                print(f"Warning: Shape mismatch! "
                      f"Pool: {self.prompt_key.shape}, Init: {keys_tensor.shape}")

    # --------------------------------------------------------
    # 核心新增：池内多样性损失 (Intra-pool Diversity Loss)
    # --------------------------------------------------------
    def intra_pool_diversity_loss(self):
        """
        计算池内所有 Key 两两之间的余弦相似度的平方均值。
        强迫池子里的 Key 在高维空间中相互排斥，防止模式坍塌。
        """
        if not hasattr(self, 'prompt_key') or self.prompt_key is None:
            return torch.tensor(0.0, device=self.prompt.device)
            
        # 1. 对所有 Key 进行归一化
        key_norm   = F.normalize(self.prompt_key, dim=1)
        
        # 2. 计算 M x M 的相似度矩阵
        sim_matrix = torch.matmul(key_norm, key_norm.t())
        
        # 3. 提取上三角矩阵（排除对角线上自己与自己的相似度 1.0）
        M    = sim_matrix.shape[0]
        mask = torch.triu(
            torch.ones(M, M, device=sim_matrix.device), diagonal=1
        ).bool()
        
        # 4. 返回相似度的平方均值作为惩罚项
        return sim_matrix[mask].pow(2).mean()