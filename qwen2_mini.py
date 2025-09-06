import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import time

class RotaryPositionEmbedding(nn.Module):
    """旋转位置编码，Qwen2中使用的位置编码方式"""
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # 计算频率
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # 生成位置索引
        positions = torch.arange(max_seq_len)
        self.register_buffer('positions', positions)
        
        # 预计算正弦和余弦
        self._precompute_sin_cos()
    
    def _precompute_sin_cos(self):
        # 计算频率和位置的乘积
        freqs = torch.einsum('i,j->ij', self.positions, self.inv_freq)
        # 计算正弦和余弦
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('sin_emb', emb.sin())
        self.register_buffer('cos_emb', emb.cos())
    
    def forward(self, x):
        # x形状: [batch_size, seq_len, dim]
        seq_len = x.size(1)
        # 提取对应长度的正弦和余弦值
        sin = self.sin_emb[:seq_len, :]
        cos = self.cos_emb[:seq_len, :]
        
        # 应用旋转位置编码
        x1 = x[..., :self.dim//2]  # 前半部分
        x2 = x[..., self.dim//2:]  # 后半部分
        rotated = torch.cat([
            x1 * cos[..., :self.dim//2] - x2 * sin[..., :self.dim//2],
            x2 * cos[..., :self.dim//2] + x1 * sin[..., :self.dim//2]
        ], dim=-1)
        return rotated

class Attention(nn.Module):
    """注意力机制模块"""
    def __init__(self, dim, num_heads=2):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # 确保头维度正确
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        
        # Q, K, V投影
        self.qkv_proj = nn.Linear(dim, dim * 3)
        # 输出投影
        self.out_proj = nn.Linear(dim, dim)
        
        # 旋转位置编码
        self.rotary_emb = RotaryPositionEmbedding(self.head_dim)
    
    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        
        # 计算Q, K, V
        qkv = self.qkv_proj(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # 分离为Q, K, V
        
        # 应用旋转位置编码
        q = self.rotary_emb(q.transpose(1, 2)).transpose(1, 2)
        k = self.rotary_emb(k.transpose(1, 2)).transpose(1, 2)
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        # 应用注意力
        attn_output = torch.matmul(attn_probs, v)
        
        # 拼接多头结果
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        
        # 输出投影
        return self.out_proj(attn_output)

class FeedForward(nn.Module):
    """前馈神经网络模块"""
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.activation = nn.GELU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer块"""
    def __init__(self, dim, num_heads=2):
        super().__init__()
        self.attention = Attention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.feed_forward = FeedForward(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        # 注意力残差连接
        x = x + self.attention(self.norm1(x))
        # 前馈网络残差连接
        x = x + self.feed_forward(self.norm2(x))
        return x

class Qwen2Mini(nn.Module):
    """轻量化Qwen2模型"""
    def __init__(self, vocab_size, dim=128, num_layers=2, num_heads=2, max_seq_len=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, dim)
        
        # Transformer层
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads) for _ in range(num_layers)
        ])
        
        # 最终层归一化
        self.norm = nn.LayerNorm(dim)
        
        # 输出层
        self.output = nn.Linear(dim, vocab_size)
    
    def forward(self, x, labels=None):
        # x形状: [batch_size, seq_len]
        x = self.embedding(x)  # [batch_size, seq_len, dim]
        
        # 通过Transformer层
        for layer in self.layers:
            x = layer(x)
        
        # 最终处理
        x = self.norm(x)
        logits = self.output(x)  # [batch_size, seq_len, vocab_size]
        
        # 如果提供了标签，计算损失
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1)
            )
            return logits, loss
        
        return logits
