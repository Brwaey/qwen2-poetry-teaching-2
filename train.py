import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import pickle
from tqdm import tqdm
from dataset import create_enhanced_poetry_dataset, get_data_loader

# 自动选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"训练设备：{device}（GPU加速已启用）" if torch.cuda.is_available() else "训练设备：CPU（建议使用GPU提升速度）")

class EnhancedQwen2PoetryModel(nn.Module):
    """增强版Qwen2诗词模型：提升容量与表达能力"""
    def __init__(self, vocab_size, embed_dim=256, num_heads=4, num_layers=4, max_seq_len=64):
        super().__init__()
        self.embed_dim = embed_dim        # 提升嵌入维度（192→256）
        self.max_seq_len = max_seq_len    # 最大序列长度
        self.vocab_size = vocab_size      # 词汇表大小
        
        # 1. 词嵌入层（增加维度）
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0  # PAD_TOKEN的ID=0
        )
        
        # 2. 改进的位置编码（更适合诗词）
        self.pos_encoding = self._create_positional_encoding(embed_dim, max_seq_len)
        self.pos_encoding.requires_grad = False
        
        # 3. 增加Transformer层数（3→4）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,          # 增加注意力头数（3→4）
            dim_feedforward=1024,     # 增大前馈网络（768→1024）
            dropout=0.1,              # 降低dropout（0.15→0.1）
            layer_norm_eps=1e-5,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers     # 增加层数（3→4）
        )
        
        # 4. 输出层（增加中间层提升表达）
        self.final_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, vocab_size)
        )

    def _create_positional_encoding(self, embed_dim, max_seq_len):
        """改进的位置编码，更适合短文本诗词"""
        position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-np.log(10000.0) / embed_dim))
        pos_encoding = torch.zeros(max_seq_len, embed_dim, dtype=torch.float32)
        
        # 调整频率参数，更适合诗词的短序列特性
        pos_encoding[:, 0::2] = torch.sin(position * div_term * 0.5)
        pos_encoding[:, 1::2] = torch.cos(position * div_term * 0.5)
        pos_encoding = pos_encoding.unsqueeze(0)
        return pos_encoding

    def forward(self, x):
        """前向传播：增强特征提取能力"""
        batch_size, seq_len = x.shape
        
        # 1. 词嵌入 + 位置编码
        x_embed = self.embedding(x)
        x_embed = x_embed * np.sqrt(self.embed_dim)
        pos_encoding = self.pos_encoding[:, :seq_len, :].to(x.device)
        x_embed = x_embed + pos_encoding
        
        # 2. Transformer Encoder处理
        encoder_out = self.transformer_encoder(x_embed)
        
        # 3. 输出层
        encoder_out = self.final_norm(encoder_out)
        logits = self.output_proj(encoder_out)
        
        return logits

def train_model(max_train_time=600):  # 延长训练时间到10分钟
    """优化训练策略：提升收敛质量"""
    # 1. 加载增强版数据集（12000样本）
    dataset, vocab = create_enhanced_poetry_dataset(
        total_samples=12000,
        max_length=64
    )
    
    # 2. 保存词汇表
    with open("vocab_enhanced.pkl", "wb") as f:
        pickle.dump(vocab, f)
    print(f"词汇表已保存：vocab_enhanced.pkl（大小：{len(vocab)}）")
    
    # 3. 创建数据加载器
    batch_size = 64 if torch.cuda.is_available() else 32
    dataloader = get_data_loader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    print(f"数据加载器配置：批次大小={batch_size}，总批次数={len(dataloader)}")
    
    # 4. 初始化增强版模型
    model = EnhancedQwen2PoetryModel(
        vocab_size=len(vocab),
        embed_dim=256,
        num_heads=4,
        num_layers=4,
        max_seq_len=64
    ).to(device)
    print("模型结构：4层Transformer Encoder + 256维嵌入（增强版）")
    
    # 5. 优化器和损失函数（更精细的参数）
    optimizer = optim.AdamW(
        model.parameters(),
        lr=5e-4,                # 调整学习率（6e-4→5e-4，更稳定）
        weight_decay=1e-5,      # 降低权重衰减
        betas=(0.9, 0.98)
    )
    
    # 增加最大epoch到30
    num_epochs = 30
    total_training_steps = len(dataloader) * num_epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-4,
        total_steps=total_training_steps,
        pct_start=0.15,  # 增加热身比例
        anneal_strategy="cos",  # 余弦退火，更稳定
        final_div_factor=100
    )
    
    # 改进损失函数（降低标签平滑）
    criterion = nn.CrossEntropyLoss(
        ignore_index=0,
        label_smoothing=0.05  # 从0.1→0.05，更关注正确标签
    )
    
    # 6. 训练参数（更严格的早停）
    train_losses = []
    best_loss = float("inf")
    no_improve_count = 0
    start_time = time.time()
    patience = 3  # 恢复到3次，允许更多迭代
    
    # 7. 开始训练
    print(f"\n训练启动（限时{max_train_time//60}分钟）...")
    model.train()
    for epoch in range(num_epochs):
        elapsed_time = time.time() - start_time
        if elapsed_time >= max_train_time:
            print(f"\n✅ 达到训练时间限制（{max_train_time}秒），停止训练")
            break
        
        epoch_loss = 0.0
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1:2d}/{num_epochs} | 耗时：{elapsed_time:.0f}秒"
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        
        # 早停策略（更严格）
        if avg_epoch_loss < best_loss * 0.99:  # 要求损失降低1%以上才算改善
            best_loss = avg_epoch_loss
            no_improve_count = 0
            # 保存最佳模型
            torch.save(model.state_dict(), "qwen2_best_poetry_model.pth")
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"\n✅ 损失连续{patience}次无改善，提前停止")
                break
        
        # 目标损失值调整为更低（<0.6）
        if avg_epoch_loss < 0.6:
            print(f"\n✅ 损失达到目标值（<0.6），提前停止训练")
            break
    
    # 8. 保存最终模型
    torch.save(model.state_dict(), "qwen2_enhanced_poetry_model.pth")
    print(f"\n📁 最终模型已保存：qwen2_enhanced_poetry_model.pth")
    print(f"📁 最佳模型已保存：qwen2_best_poetry_model.pth")
    
    # 9. 保存损失曲线
    with open("train_losses_enhanced.pkl", "wb") as f:
        pickle.dump(train_losses, f)
    
    # 10. 绘制损失曲线
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.plot(train_losses, color="#2E86AB", linewidth=2, marker="o", markersize=4)
        plt.title("增强版诗词模型训练损失曲线", fontsize=12)
        plt.xlabel("训练轮次（Epoch）", fontsize=10)
        plt.ylabel("平均损失值", fontsize=10)
        plt.grid(alpha=0.3, linestyle="--")
        plt.savefig("train_loss_curve_enhanced.png", dpi=100, bbox_inches="tight")
        print(f"📊 损失曲线已保存：train_loss_curve_enhanced.png")
    except ImportError:
        print("⚠️  未安装matplotlib，跳过损失曲线绘制")
    
    return model, vocab, train_losses

if __name__ == "__main__":
    model, vocab, losses = train_model(max_train_time=600)  # 10分钟=600秒
    print("\n🎉 训练完成！可运行 inference.py 体验诗词生成。")
