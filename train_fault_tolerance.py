import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import pickle
from tqdm import tqdm
from dataset_fault_tolerance import create_fault_tolerance_dataset, get_fault_tolerance_dataloader

# 自动选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"微调设备：{device}（GPU加速已启用）" if torch.cuda.is_available() else "微调设备：CPU（建议使用GPU提升速度）")

def load_pretrained_model(model_path, vocab, embed_dim=256, num_heads=4, num_layers=3, max_seq_len=64):
    """加载预训练模型用于微调"""
    from train import ExactMatchModel
    
    # 初始化模型
    model = ExactMatchModel(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=max_seq_len
    ).to(device)
    
    # 加载预训练权重
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ 加载预训练模型成功：{model_path}")
        return model
    except FileNotFoundError:
        print(f"❌ 未找到预训练模型文件 {model_path}")
        print("请先运行原始train.py训练基础模型")
        return None

def fine_tune_fault_tolerance(pretrained_model_path="exact_match_poem_model.pth",
                             max_train_time=600,  # 微调时间10分钟
                             batch_size=32):
    """微调模型以增强容错能力"""
    # 1. 加载词汇表
    try:
        with open("exact_match_vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        print(f"✅ 加载词汇表成功：{len(vocab)}个标记")
    except FileNotFoundError:
        print("❌ 未找到词汇表文件 exact_match_vocab.pkl")
        print("请先运行原始train.py生成词汇表")
        return None
    
    # 2. 创建容错输入数据集
    dataset, vocab = create_fault_tolerance_dataset(
        original_poems_path="original_poems.pkl",
        vocab=vocab,
        total_samples=10000,
        max_length=64
    )
    if dataset is None:
        return None
    
    # 3. 创建数据加载器
    dataloader = get_fault_tolerance_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    print(f"数据加载器配置：批次大小={batch_size}，总批次数={len(dataloader)}")
    
    # 4. 加载预训练模型
    model = load_pretrained_model(pretrained_model_path, vocab)
    if model is None:
        return None
    
    # 5. 微调优化器与损失函数（使用较小的学习率）
    # 只微调最后几层，冻结前面的层
    for param in list(model.parameters())[:-10]:  # 冻结大部分参数
        param.requires_grad = False
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),  # 只优化可训练参数
        lr=5e-5,  # 微调学习率要小，避免破坏预训练知识
        weight_decay=1e-6
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(dataloader) * 4,
        eta_min=1e-6
    )
    
    criterion = nn.CrossEntropyLoss(
        ignore_index=0  # 忽略PAD
    )
    
    # 6. 微调参数
    train_losses = []
    best_loss = float("inf")
    no_improve_count = 0
    start_time = time.time()
    best_model_state = None
    loss_improve_threshold = 0.002  # 微调的改进阈值更小
    
    # 7. 开始微调
    print(f"\n容错输入微调启动（限时{max_train_time//60}分钟）...")
    model.train()
    for epoch in range(20):  # 微调轮次较少
        elapsed_time = time.time() - start_time
        if elapsed_time >= max_train_time:
            print(f"\n✅ 达到微调时间限制（{max_train_time}秒），停止训练")
            break
        
        epoch_loss = 0.0
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1:2d}/20 | 耗时：{elapsed_time:.0f}秒 | 最佳损失：{best_loss:.4f}"
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)  # 更严格的梯度裁剪
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
        
        # 保存最佳模型
        if avg_epoch_loss < best_loss - loss_improve_threshold:
            best_loss = avg_epoch_loss
            best_model_state = model.state_dict().copy()
            no_improve_count = 0
            print(f"🔧 最佳模型更新：当前损失{avg_epoch_loss:.4f} < 历史最佳{best_loss+loss_improve_threshold:.4f}")
        else:
            no_improve_count += 1
            # 早停条件更宽松
            if no_improve_count >= 5:
                print(f"\n✅ 损失连续5次无改善，停止微调")
                break
    
    # 8. 保存微调后的模型
    if best_model_state is not None:
        torch.save(best_model_state, "fault_tolerance_poem_model.pth")
        print(f"\n📁 容错输入模型已保存：fault_tolerance_poem_model.pth（最佳损失：{best_loss:.4f}）")
    else:
        torch.save(model.state_dict(), "fault_tolerance_poem_model.pth")
        print(f"\n📁 容错输入模型已保存：fault_tolerance_poem_model.pth（当前损失：{avg_epoch_loss:.4f}）")
    
    # 9. 保存损失曲线
    with open("fault_tolerance_losses.pkl", "wb") as f:
        pickle.dump(train_losses, f)
    
    # 10. 绘制损失曲线
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.plot(train_losses, color="#2ECC71", linewidth=2, marker="o", markersize=4)
        plt.title("容错输入模型微调损失曲线", fontsize=12)
        plt.xlabel("微调轮次（Epoch）", fontsize=10)
        plt.ylabel("平均损失值", fontsize=10)
        plt.grid(alpha=0.3, linestyle="--")
        plt.savefig("fault_tolerance_loss_curve.png", dpi=100, bbox_inches="tight")
        print(f"📊 损失曲线已保存：fault_tolerance_loss_curve.png")
    except ImportError:
        print("⚠️  未安装matplotlib，跳过损失曲线绘制")
    
    return model, vocab, train_losses

if __name__ == "__main__":
    model, vocab, losses = fine_tune_fault_tolerance()
    print("\n🎉 容错输入模型微调完成！可运行 inference_fault_tolerance.py 体验带容错能力的诗词生成。")
    