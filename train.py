import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import pickle
from tqdm import tqdm
from dataset import create_exact_match_dataset, get_exact_match_dataloader

# 自动选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"训练设备：{device}（GPU加速已启用）" if torch.cuda.is_available() else "训练设备：CPU（建议使用GPU提升速度）")

class ExactMatchModel(nn.Module):
    """原诗精准匹配模型：简化结构但强化拟合能力，确保原诗记忆"""
    def __init__(self, vocab_size, embed_dim=256, num_heads=4, num_layers=3, max_seq_len=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        
        # 词嵌入层：原诗单字少，用较大嵌入维度提升区分度
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0
        )
        
        # 位置编码：严格适配原诗句式（5/7字每句）
        self.pos_encoding = self._create_poem_positional_encoding(embed_dim, max_seq_len)
        self.pos_encoding.requires_grad = False
        
        # Transformer Encoder：3层足够拟合原诗（避免过复杂导致泛化错误）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=1024,
            dropout=0.05,  # 极低dropout，确保原诗信息不丢失
            layer_norm_eps=1e-5,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # 输出层：无激活函数，直接映射（确保原诗字的概率最大化）
        self.final_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def _create_poem_positional_encoding(self, embed_dim, max_seq_len):
        """位置编码：适配原诗句式（每5/7字一个周期，增强句式记忆）"""
        position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        # 周期设置为5和7的最小公倍数35，适配两种句式
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-np.log(35.0) / embed_dim))
        pos_encoding = torch.zeros(max_seq_len, embed_dim, dtype=torch.float32)
        
        # 正弦余弦交替，周期35，贴合原诗句式
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        pos_encoding = pos_encoding.unsqueeze(0)
        return pos_encoding

    def forward(self, x):
        batch_size, seq_len = x.shape
        x_embed = self.embedding(x)
        x_embed = x_embed * np.sqrt(self.embed_dim)  # 归一化
        pos_encoding = self.pos_encoding[:, :seq_len, :].to(x.device)
        x_embed = x_embed + pos_encoding
        
        encoder_out = self.transformer_encoder(x_embed)
        encoder_out = self.final_norm(encoder_out)
        logits = self.output_proj(encoder_out)
        
        return logits

def train_exact_match_model(max_train_time=1200):  # 延长训练到20分钟，确保充分拟合
    """
    核心优化：
    1. 去除标签平滑，让模型精准学习原诗标签
    2. 低学习率+慢退火，确保收敛到全局最优
    3. 早停阈值0.4，确保损失足够低（原诗拟合充分）
    """
    # 1. 加载原诗精准匹配数据集
    dataset, vocab = create_exact_match_dataset(
        total_samples=20000,
        max_length=64
    )
    
    # 2. 保存词汇表（与模型严格对应）
    with open("exact_match_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    print(f"原诗匹配词汇表已保存：exact_match_vocab.pkl（大小：{len(vocab)}）")
    
    # 3. 创建数据加载器（CPU批次32，GPU批次64）
    batch_size = 64 if torch.cuda.is_available() else 32
    dataloader = get_exact_match_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    print(f"数据加载器配置：批次大小={batch_size}，总批次数={len(dataloader)}")
    
    # 4. 初始化原诗匹配模型
    model = ExactMatchModel(
        vocab_size=len(vocab),
        embed_dim=256,
        num_heads=4,
        num_layers=3,
        max_seq_len=64
    ).to(device)
    print("原诗精准匹配模型结构：3层Transformer Encoder + 256维嵌入（强化原诗拟合）")
    
    # 5. 优化器与损失函数（无标签平滑，精准拟合）
    optimizer = optim.AdamW(
        model.parameters(),
        lr=3e-4,  # 低学习率，缓慢收敛
        weight_decay=1e-6,  # 极低权重衰减，保护原诗参数
        betas=(0.9, 0.99)  # 大动量，稳定收敛
    )
    
    # 余弦退火调度器：周期长，确保充分收敛
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(dataloader) * 8,  # 每8个epoch退火一次
        eta_min=5e-6  # 最小学习率，后期精细调整
    )
    
    # 损失函数：无标签平滑，确保原诗标签精准
    criterion = nn.CrossEntropyLoss(
        ignore_index=0  # 仅忽略PAD，其他标签严格匹配
    )
    
    # 6. 训练参数（严格早停）
    train_losses = []
    best_loss = float("inf")
    no_improve_count = 0
    start_time = time.time()
    best_model_state = None
    loss_improve_threshold = 0.005  # 损失降低0.005以上才算改善，避免波动
    
    # 7. 开始训练（强化原诗拟合）
    print(f"\n训练启动（限时{max_train_time//60}分钟，目标损失<0.4）...")
    model.train()
    for epoch in range(60):  # 最大60个epoch，足够拟合原诗
        elapsed_time = time.time() - start_time
        if elapsed_time >= max_train_time:
            print(f"\n✅ 达到训练时间限制（{max_train_time}秒），停止训练")
            break
        
        epoch_loss = 0.0
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1:2d}/60 | 耗时：{elapsed_time:.0f}秒 | 最佳损失：{best_loss:.4f}"
        )
        
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            inputs = inputs.to(device, non_blocking=torch.cuda.is_available())
            labels = labels.to(device, non_blocking=torch.cuda.is_available())
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                labels.view(-1)
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # 低梯度裁剪，避免参数震荡
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            avg_batch_loss = epoch_loss / (batch_idx + 1)
            progress_bar.set_postfix({
                "batch_loss": f"{loss.item():.4f}",
                "avg_loss": f"{avg_batch_loss:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.6f}"
            })
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_epoch_loss)
        print(f"📊 Epoch {epoch+1:2d} 结束 | 平均损失：{avg_epoch_loss:.4f} | 耗时：{elapsed_time:.0f}秒")
        
        # 保存最佳模型（损失降低足够多）
        if avg_epoch_loss < best_loss - loss_improve_threshold:
            best_loss = avg_epoch_loss
            best_model_state = model.state_dict().copy()
            no_improve_count = 0
            print(f"🔧 最佳模型更新：当前损失{avg_epoch_loss:.4f} < 历史最佳{best_loss+loss_improve_threshold:.4f}")
        else:
            no_improve_count += 1
            # 早停条件：连续4次无改善，或损失<0.4（目标达成）
            if no_improve_count >= 4 or avg_epoch_loss < 0.4:
                if avg_epoch_loss < 0.4:
                    print(f"\n✅ 损失达到目标值（{avg_epoch_loss:.4f}<0.4），原诗拟合充分，停止训练")
                else:
                    print(f"\n✅ 损失连续4次无改善，停止训练")
                break
    
    # 8. 保存最佳模型（确保原诗匹配效果）
    if best_model_state is not None:
        torch.save(best_model_state, "exact_match_poem_model.pth")
        print(f"\n📁 原诗精准匹配模型已保存：exact_match_poem_model.pth（最佳损失：{best_loss:.4f}）")
    else:
        torch.save(model.state_dict(), "exact_match_poem_model.pth")
        print(f"\n📁 原诗精准匹配模型已保存：exact_match_poem_model.pth（当前损失：{avg_epoch_loss:.4f}）")
    
    # 9. 保存损失曲线
    with open("exact_match_losses.pkl", "wb") as f:
        pickle.dump(train_losses, f)
    
    # 10. 绘制损失曲线
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.plot(train_losses, color="#E74C3C", linewidth=2, marker="o", markersize=4)
        plt.axhline(y=0.4, color="#3498DB", linestyle="--", label="目标损失（0.4）")
        plt.title("原诗精准匹配模型训练损失曲线（20分钟限时）", fontsize=12)
        plt.xlabel("训练轮次（Epoch）", fontsize=10)
        plt.ylabel("平均损失值", fontsize=10)
        plt.legend()
        plt.grid(alpha=0.3, linestyle="--")
        plt.savefig("exact_match_loss_curve.png", dpi=100, bbox_inches="tight")
        print(f"📊 损失曲线已保存：exact_match_loss_curve.png")
    except ImportError:
        print("⚠️  未安装matplotlib，跳过损失曲线绘制")
    
    return model, vocab, train_losses

if __name__ == "__main__":
    model, vocab, losses = train_exact_match_model(max_train_time=1200)  # 20分钟训练
    print("\n🎉 原诗精准匹配模型训练完成！可运行 inference.py 体验诗词生成。")
