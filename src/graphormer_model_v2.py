from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .graph_builder import EDGE_TEXT_IMG


class LearnableEdgeWeights(nn.Module):
    def __init__(self, num_edge_types: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.base_weights = nn.Parameter(torch.ones(num_edge_types))
        self.edge_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            ) for _ in range(num_edge_types)
        ])


class GraphormerLayerV2(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.res_scale = nn.Parameter(torch.ones(1))

    def forward(self, x, attn_bias, node_mask):
        B, N, _ = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5) + attn_bias
        key_mask = node_mask.unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(~key_mask, float("-inf"))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, self.d_model)
        out = self.dropout(self.out_proj(out))
        x = self.norm1(x + self.res_scale * out)
        x = self.norm2(x + self.res_scale * self.ff(x))
        return x


class ConflictDetector(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.text_proj = nn.Linear(d_model, d_model)
        self.image_proj = nn.Linear(d_model, d_model)
        self.conflict_net = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
    
    def forward(self, text_pool, image_pool):
        text_feat = self.text_proj(text_pool)
        image_feat = self.image_proj(image_pool)
        combined = torch.cat([text_feat, text_feat, image_feat, image_feat], dim=-1)
        return self.conflict_net(combined).squeeze(-1)


class GraphormerModelV2(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.15,
        modality_drop: float = 0.15,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.modality_drop = modality_drop

        self.node_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.layers = nn.ModuleList([
            GraphormerLayerV2(d_model, num_heads, dropout) for _ in range(num_layers)
        ])
        self.edge_type_emb = nn.Embedding(4, num_heads)
        self.edge_weight_learner = LearnableEdgeWeights(4, 32)
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        
        self.conflict_detector = ConflictDetector(d_model, dropout)
        self.final_norm = nn.LayerNorm(d_model)
        
        mlp_input = d_model * 7 + 1
        self.classifier = nn.Sequential(
            nn.Linear(mlp_input, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def _build_attention_bias(self, edge_index_list, edge_type_list, edge_weight_list, max_nodes, device):
        bias = torch.zeros(len(edge_index_list), self.num_heads, max_nodes, max_nodes, device=device)
        et_emb = self.edge_type_emb.weight
        for b, (edge_index, edge_type, edge_weight) in enumerate(zip(edge_index_list, edge_type_list, edge_weight_list)):
            if edge_index.numel() == 0:
                continue
            src, dst, et, ew = edge_index[0], edge_index[1], edge_type, edge_weight
            bias[b, :, src, dst] += et_emb[et].transpose(0, 1)
            text_img_mask = et == EDGE_TEXT_IMG
            if text_img_mask.any():
                bias[b, :, src[text_img_mask], dst[text_img_mask]] += ew[text_img_mask].unsqueeze(0)
        return bias

    def forward(self, node_feats, node_mask, text_mask, image_mask, global_indices, 
                edge_index_list, edge_type_list, edge_weight_list):
        device = node_feats.device
        max_nodes = node_feats.size(1)
        attn_bias = self._build_attention_bias(edge_index_list, edge_type_list, edge_weight_list, max_nodes, device)
        
        x = self.node_proj(node_feats)
        
        layer_globals = []
        for layer in self.layers:
            x = layer(x, attn_bias, node_mask)
            batch_indices = torch.arange(x.size(0), device=device)
            layer_global = x[batch_indices, global_indices]
            layer_globals.append(layer_global)
        
        x = self.final_norm(x)
        
        text_mask_f = text_mask.float()
        image_mask_f = image_mask.float()
        text_pool = (x * text_mask_f.unsqueeze(-1)).sum(dim=1) / text_mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        image_pool = (x * image_mask_f.unsqueeze(-1)).sum(dim=1) / image_mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        batch_indices = torch.arange(x.size(0), device=device)
        final_global = x[batch_indices, global_indices]
        
        conflict_logits = self.conflict_detector(text_pool, image_pool)
        
        combined = torch.cat(layer_globals + [text_pool, image_pool, final_global, conflict_logits.unsqueeze(-1)], dim=-1)
        logits = self.classifier(combined).squeeze(-1)
        
        return {
            "logits": logits,
            "conflict_logits": conflict_logits,
            "text_pool": text_pool,
            "image_pool": image_pool,
        }
