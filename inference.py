import torch
import jieba
import pickle
import random
from tqdm import tqdm
from train import EnhancedQwen2PoetryModel
from dataset import EnhancedVocab

# 特殊标记配置（与训练完全一致）
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
PAD_ID = 0    # PAD_TOKEN的ID
UNK_ID = 1    # UNK_TOKEN的ID
BOS_ID = 2    # BOS_TOKEN的ID
EOS_ID = 3    # EOS_TOKEN的ID

# 自动选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"生成设备：{device}")

def load_model_and_vocab(
    model_path="qwen2_best_poetry_model.pth",  # 加载最佳模型
    vocab_path="vocab_enhanced.pkl"
):
    """加载增强版模型和词汇表"""
    # 1. 加载词汇表
    try:
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        print(f"✅ 增强版词汇表加载成功：{len(vocab)}个标记")
    except FileNotFoundError:
        print(f"❌ 错误：未找到词汇表文件 {vocab_path}")
        print("请先运行 train.py 生成词汇表")
        return None, None
    
    # 2. 加载增强版模型
    model = EnhancedQwen2PoetryModel(
        vocab_size=len(vocab),
        embed_dim=256,
        num_heads=4,
        num_layers=4,
        max_seq_len=64
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # 推理模式
        print(f"✅ 增强版模型加载成功：{model_path}")
        return model, vocab
    except FileNotFoundError:
        print(f"❌ 错误：未找到模型文件 {model_path}")
        print("请先运行 train.py 训练模型")
        return None, vocab

def find_cutoff_index(cumulative_probs, top_p):
    """兼容低版本PyTorch的Top-P截断逻辑"""
    cum_probs_np = cumulative_probs.cpu().numpy()[0]
    cutoff_idx = len(cum_probs_np)
    for i in range(len(cum_probs_np)):
        if cum_probs_np[i] >= top_p:
            cutoff_idx = i
            break
    return max(1, cutoff_idx)

def generate_poem(
    model, vocab, input_text,
    max_length=48,        # 增加生成长度（32→48）
    temperature=0.25,     # 降低温度（0.35→0.25），更确定
    top_k=8,              # 增加候选词（6→8）
    top_p=0.95,           # 提高累积概率（0.92→0.95）
    repeat_penalty=1.2,   # 降低惩罚力度（1.5→1.2），避免过早停
    max_repeat=3          # 允许更多重复（2→3）
):
    """优化生成逻辑：确保生成完整、通顺的诗句"""
    if model is None or vocab is None:
        return "❌ 模型或词汇表未加载，无法生成诗词"
    
    # 1. 输入预处理
    input_tokens = [BOS_TOKEN] + vocab.tokenize(input_text)
    input_ids = vocab.convert_tokens_to_ids(input_tokens)
    
    # 输入长度检查
    if len(input_ids) >= max_length:
        return "❌ 输入过长，请控制在10字以内"
    if len(input_text.strip()) == 0:
        return "❌ 输入不能为空，请输入诗句开头"
    
    # 2. 初始化生成序列
    generated_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    generated_tokens = input_tokens.copy()
    
    # 3. 逐词生成诗词（增加最小生成长度）
    min_generate_length = max(8, 16 - len(input_ids))  # 至少生成8-16字
    with torch.no_grad():
        for step in range(max_length - len(input_ids)):
            seq_len = generated_ids.shape[1]
            
            # 模型预测
            logits = model(generated_ids)
            next_token_logits = logits[:, -1, :]
            
            # 4. 改进的重复惩罚（区分功能词和内容词）
            for token_id in generated_ids[0]:
                token = vocab.id_to_token.get(token_id, "")
                # 对内容词惩罚重，功能词（如"，"）惩罚轻
                if token_id not in [PAD_ID, UNK_ID, BOS_ID, EOS_ID] and len(token) > 0:
                    penalty = repeat_penalty if len(token) > 1 else 1.05
                    next_token_logits[0, token_id] /= penalty
            
            # 5. 温度调整
            next_token_logits = next_token_logits / temperature
            
            # 6. Top-P核采样
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = probs.cumsum(dim=-1)
            
            cutoff_idx = find_cutoff_index(cumulative_probs, top_p)
            sorted_logits = sorted_logits[:, :cutoff_idx+1]
            sorted_indices = sorted_indices[:, :cutoff_idx+1]
            
            # 7. Top-K过滤
            if top_k is not None and top_k < len(sorted_indices[0]):
                sorted_logits = sorted_logits[:, :top_k]
                sorted_indices = sorted_indices[:, :top_k]
            
            # 8. 重新计算概率
            probs = torch.softmax(sorted_logits, dim=-1)
            next_token_idx = torch.multinomial(probs, num_samples=1).item()
            next_token_id = sorted_indices[0, next_token_idx].item()
            
            # 9. 改进的重复检查（允许合理重复）
            if len(generated_ids[0]) >= max_repeat:
                last_tokens = generated_ids[0][-max_repeat:].tolist()
                if all(token == next_token_id for token in last_tokens):
                    if len(sorted_indices[0]) > 1:
                        next_token_id = sorted_indices[0, 1].item()
                    else:
                        common_tokens = [
                            tid for tid in range(len(vocab))
                            if tid not in generated_ids[0] and tid not in [PAD_ID, UNK_ID, BOS_ID, EOS_ID]
                        ]
                        if common_tokens:
                            next_token_id = random.choice(common_tokens)
            
            # 10. 改进的停止条件（确保生成足够长度）
            if next_token_id == EOS_ID:
                # 未达到最小长度则忽略EOS
                if step < min_generate_length:
                    # 选择概率第二高的词
                    if len(sorted_indices[0]) > 1:
                        next_token_id = sorted_indices[0, 1].item()
                    else:
                        next_token_id = random.choice([tid for tid in range(len(vocab)) 
                                                     if tid not in [PAD_ID, UNK_ID, BOS_ID, EOS_ID]])
                else:
                    break  # 达到最小长度则停止
            
            # 11. 添加到生成序列
            generated_ids = torch.cat([
                generated_ids,
                torch.tensor([[next_token_id]], dtype=torch.long).to(device)
            ], dim=1)
            generated_tokens.append(vocab.id_to_token[next_token_id])
    
    # 12. 改进的后处理（更智能的标点添加）
    final_tokens = [
        token for token in generated_tokens
        if token not in [BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN]
    ]
    final_text = "".join(final_tokens)
    
    # 提取续写部分
    续写_text = final_text[len(input_text):] if len(final_text) >= len(input_text) else ""
    full_text = input_text + 续写_text
    
    # 智能标点添加（基于诗句结构）
    punctuated_text = ""
    char_count = 0
    for i, char in enumerate(full_text):
        # 跳过已有的标点
        if char in ["，", "。", "！", "？"]:
            punctuated_text += char
            continue
            
        punctuated_text += char
        char_count += 1
        
        # 7字一句（唐诗常见格式）
        if char_count % 7 == 0 and i != len(full_text) - 1:
            # 检查下一个字符是否已经是标点
            if i + 1 < len(full_text) and full_text[i+1] not in ["，", "。"]:
                punctuated_text += "，"
        # 14字一结（两句一结）
        elif char_count % 14 == 0 and i != len(full_text) - 1:
            if i + 1 < len(full_text) and full_text[i+1] not in ["，", "。"]:
                punctuated_text += "。"
    
    # 确保最后有句号
    if len(punctuated_text) > 0 and punctuated_text[-1] not in ["。", "！", "？"]:
        punctuated_text += "。"
    
    return punctuated_text

def interactive_demo():
    """交互式诗词生成演示"""
    print("=" * 60)
    print("          📜 增强版基于Qwen2的中文诗词生成教学演示          ")
    print("=" * 60)
    print("📚 支持唐诗宋词续写，示例输入：")
    print("  - 床前明月光 → 疑是地上霜。举头望明月，低头思故乡。")
    print("  - 白日依山尽 → 黄河入海流。欲穷千里目，更上一层楼。")
    print("  - 国破山河在 → 城春草木深。感时花溅泪，恨别鸟惊心。")
    print("💡 输入 'exit' 退出演示，输入 'help' 查看帮助")
    print("=" * 60)
    
    # 加载模型和词汇表
    print("\n正在加载增强版模型和词汇表...")
    model, vocab = load_model_and_vocab()
    if model is None:
        print("❌ 演示启动失败，请先确保模型和词汇表已生成")
        return
    
    # 交互循环
    print("\n✅ 演示启动成功！请输入诗句开头：")
    while True:
        try:
            user_input = input("\n你输入的诗句开头：").strip()
            
            # 命令处理
            if user_input.lower() == "exit":
                print("👋 感谢使用，诗词生成演示结束！")
                break
            elif user_input.lower() == "help":
                print("📖 帮助说明：")
                print("  1. 输入诗句开头（1-10字），模型将续写完整诗词")
                print("  2. 输入 'exit' 退出演示，输入 'help' 查看帮助")
                print("  3. 示例：输入'床前明月光'，生成完整《静夜思》")
                continue
            
            # 生成诗词
            print("✍️  正在生成诗词...")
            result = generate_poem(model, vocab, user_input)
            
            # 显示结果
            print(f"\n🎯 生成结果：{result}")
            
        except KeyboardInterrupt:
            print("\n\n👋 手动中断，演示结束！")
            break
        except Exception as e:
            print(f"\n❌ 生成错误：{str(e)}，请重试")

if __name__ == "__main__":
    interactive_demo()
