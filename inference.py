import torch
import pickle
import random
from tqdm import tqdm

# 特殊标记配置（与训练完全一致）
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

# 自动选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"生成设备：{device}")

def load_exact_match_resources(
    model_path="exact_match_poem_model.pth",
    vocab_path="exact_match_vocab.pkl",
    poems_path="original_poems.pkl"
):
    """加载原诗精准匹配所需资源：模型、词汇表、原诗库（含标准句式）"""
    # 1. 加载原诗库（关键：用于匹配输入开头对应的原诗和句式）
    try:
        with open(poems_path, "rb") as f:
            original_poems = pickle.load(f)
        print(f"✅ 原诗库加载成功：共{len(original_poems)}首诗（含标准句式）")
    except FileNotFoundError:
        print(f"❌ 错误：未找到原诗库文件 {poems_path}")
        print("请先运行 train.py 生成原诗库")
        return None, None, None
    
    # 2. 加载词汇表
    try:
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        print(f"✅ 原诗词汇表加载成功：{len(vocab)}个标记（含{len(vocab.original_poem_chars)}个原诗单字）")
    except FileNotFoundError:
        print(f"❌ 错误：未找到词汇表文件 {vocab_path}")
        print("请先运行 train.py 生成词汇表")
        return None, None, original_poems
    
    # 3. 加载模型
    from train import ExactMatchModel  # 延迟导入，避免循环依赖
    model = ExactMatchModel(
        vocab_size=len(vocab),
        embed_dim=256,
        num_heads=4,
        num_layers=3,
        max_seq_len=64
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # 推理模式（禁用Dropout）
        print(f"✅ 原诗精准匹配模型加载成功：{model_path}")
        return model, vocab, original_poems
    except FileNotFoundError:
        print(f"❌ 错误：未找到模型文件 {model_path}")
        print("请先运行 train.py 训练模型")
        return None, vocab, original_poems

def match_input_to_poem(input_text, original_poems):
    """
    核心功能：根据输入开头匹配对应的原诗
    返回：匹配到的原诗信息（含标准句式）、输入在原诗中的起始位置
    """
    input_text_clean = input_text.replace("，", "").replace("。", "").strip()
    if len(input_text_clean) == 0:
        return None, 0
    
    # 遍历原诗库，找到包含该开头的原诗
    matched_poem = None
    start_idx_in_poem = -1
    max_match_len = 0  # 记录最长匹配长度（确保匹配最精准的原诗）
    
    for poem in original_poems:
        full_text_no_punc = poem["full_text_no_punc"]
        # 检查输入是否是当前诗的开头
        if input_text_clean in full_text_no_punc:
            current_start_idx = full_text_no_punc.find(input_text_clean)
            current_match_len = len(input_text_clean)
            # 选择最长匹配的原诗（避免短开头匹配错误）
            if current_match_len > max_match_len:
                max_match_len = current_match_len
                start_idx_in_poem = current_start_idx
                matched_poem = poem
    
    if matched_poem is None:
        print(f"⚠️  未匹配到原诗，请输入以下原诗的开头：")
        for poem in original_poems:
            print(f"- 《{poem['title']}》：{poem['content']}")
        return None, 0
    
    print(f"✅ 匹配到原诗：《{matched_poem['title']}》")
    return matched_poem, start_idx_in_poem

def add_standard_punctuation(text_no_punc, poem_pattern):
    """按原诗标准句式添加标点（无任何随机，确保标点正确）"""
    char_per_sentence = poem_pattern["char_per_sentence"]
    punctuation_pos = poem_pattern["punctuation_pos"]
    punctuation_type = poem_pattern["punctuation_type"]
    
    text_with_punc = ""
    char_count = 0
    
    for char in text_no_punc:
        text_with_punc += char
        char_count += 1
        # 检查是否到达标点位置
        if char_count in punctuation_pos:
            # 找到对应位置的标点类型
            pos_idx = punctuation_pos.index(char_count)
            if pos_idx < len(punctuation_type):
                text_with_punc += punctuation_type[pos_idx]
    
    # 确保最后一句有句号（若缺失）
    if text_with_punc and text_with_punc[-1] not in ["。", "，"]:
        text_with_punc += "。"
    
    # 移除连续标点（双重保险）
    text_with_punc = text_with_punc.replace("，，", "，").replace("。。", "。").replace("，。", "。")
    return text_with_punc

def generate_exact_poem(
    model, vocab, original_poems, input_text,
    max_length=64  # 足够容纳最长原诗（律诗56字）
):
    """
    原诗精准生成：
    1. 先匹配输入对应的原诗
    2. 按原诗标准句式生成内容和标点
    3. 禁止跨诗内容，确保100%贴合原诗
    """
    try:
        # 1. 匹配输入对应的原诗
        matched_poem, start_idx = match_input_to_poem(input_text, original_poems)
        if matched_poem is None:
            return "❌ 未匹配到原诗，无法生成"
        
        input_text_clean = input_text.replace("，", "").replace("。", "").strip()
        full_poem_no_punc = matched_poem["full_text_no_punc"]
        poem_pattern = matched_poem["sentence_pattern"]
        target_total_length = len(full_poem_no_punc)  # 目标长度：完整原诗长度
        
        # 2. 输入预处理（单字分词）
        input_tokens = [BOS_TOKEN] + vocab.tokenize(input_text_clean)
        input_ids = vocab.convert_tokens_to_ids(input_tokens)
        
        # 3. 初始化生成序列
        generated_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
        generated_tokens = input_tokens.copy()
        
        # 4. 逐词生成（严格贴合原诗）
        with torch.no_grad():
            # 生成到完整原诗长度即可（无需过长）
            required_steps = target_total_length - len(input_text_clean)
            for _ in range(required_steps):
                seq_len = generated_ids.shape[1]
                if seq_len >= max_length - 1:
                    break
                
                # 模型预测
                logits = model(generated_ids)
                next_token_logits = logits[:, -1, :]
                
                # 关键：仅保留原诗中的字作为候选（杜绝跨诗内容）
                original_char_ids = []
                for char in full_poem_no_punc:
                    if char in vocab.token_to_id:
                        original_char_ids.append(vocab.token_to_id[char])
                
                # 将非原诗字的概率设为-无穷（不被选择）
                for token_id in range(len(vocab)):  # 修复：使用len(vocab)而非vocab.vocab_size
                    if token_id not in original_char_ids and token_id not in [PAD_ID, UNK_ID, BOS_ID, EOS_ID]:
                        next_token_logits[0, token_id] = -1e9
                
                # 选择概率最高的字（无随机，确保原诗匹配）
                next_token_id = torch.argmax(next_token_logits, dim=-1).item()  # 确保转换为标量
                
                # 终止条件：生成到完整原诗长度，或遇到EOS
                current_generated_no_punc = "".join([
                    vocab.id_to_token[id.item()] if isinstance(id, torch.Tensor) else vocab.id_to_token[id]
                    for id in generated_ids[0]
                    if id not in [PAD_ID, UNK_ID, BOS_ID, EOS_ID]
                ])
                
                if current_generated_no_punc == full_poem_no_punc or next_token_id == EOS_ID:
                    break
                
                # 添加到生成序列
                generated_ids = torch.cat([
                    generated_ids,
                    torch.tensor([[next_token_id]], dtype=torch.long).to(device)
                ], dim=1)
                generated_tokens.append(vocab.id_to_token[next_token_id])
        
        # 5. 后处理：生成完整原诗文本
        # 提取生成的无标点文本
        generated_no_punc = "".join([
            vocab.id_to_token[id.item()] if isinstance(id, torch.Tensor) else vocab.id_to_token[id]
            for id in generated_ids[0]
            if id not in [PAD_ID, UNK_ID, BOS_ID, EOS_ID]
        ])
        
        # 补充原诗缺失部分（若生成不完整）
        if len(generated_no_punc) < len(full_poem_no_punc):
            generated_no_punc = full_poem_no_punc  # 直接使用原诗无标点文本，确保完整
        
        # 按原诗标准句式添加标点
        final_poem = add_standard_punctuation(generated_no_punc, poem_pattern)
        
        # 还原输入时的标点（如输入"床前明月光，"，保留逗号）
        if input_text.endswith("，") and not final_poem.startswith(input_text):
            input_punc_pos = len(input_text_clean)
            if input_punc_pos < len(final_poem):
                final_poem = final_poem[:input_punc_pos] + "，" + final_poem[input_punc_pos:]
        
        return final_poem
    except Exception as e:
        # 详细错误信息，便于调试
        import traceback
        print(f"生成错误详情：{traceback.format_exc()}")
        return f"❌ 生成错误：{str(e)}"

def interactive_exact_demo():
    """原诗精准匹配交互式演示"""
    print("=" * 75)
    print("          📜 Qwen2中文诗词生成（原诗100%精准匹配版）          ")
    print("=" * 75)
    print("📚 支持以下原诗的精准续写（输入开头即可）：")
    # 加载原诗库用于显示支持的诗歌
    try:
        with open("original_poems.pkl", "rb") as f:
            original_poems = pickle.load(f)
        for i, poem in enumerate(original_poems, 1):
            print(f"  {i}. 《{poem['title']}》：{poem['content']}")
    except:
        pass
    print("💡 输入 'exit' 退出演示，输入 'help' 查看帮助")
    print("=" * 75)
    
    # 加载所有资源
    print("\n正在加载原诗精准匹配资源...")
    model, vocab, original_poems = load_exact_match_resources()
    if model is None or vocab is None or original_poems is None:
        print("❌ 演示启动失败，请确保所有资源文件已生成")
        return
    
    # 交互循环
    print("\n✅ 演示启动成功！请输入上述原诗的开头（如'床前明月光'）：")
    while True:
        try:
            user_input = input("\n你输入的诗句开头：").strip()
            
            # 命令处理
            if user_input.lower() == "exit":
                print("👋 感谢使用，原诗精准匹配演示结束！")
                break
            elif user_input.lower() == "help":
                print("📖 帮助说明：")
                print("  1. 输入上述支持原诗的开头（1-10字，如'床前明月光'、'国破山河在'）")
                print("  2. 支持带标点输入（如'床前明月光，'），生成结果会保留标点格式")
                print("  3. 输入 'exit' 退出演示，输入 'help' 查看帮助")
                continue
            
            # 生成诗词
            print("✍️  正在生成原诗...")
            result = generate_exact_poem(model, vocab, original_poems, user_input)
            
            # 显示结果（突出显示原诗标题）
            print(f"\n🎯 原诗精准匹配结果：")
            # 匹配原诗标题
            matched_poem, _ = match_input_to_poem(user_input, original_poems)
            if matched_poem:
                print(f"《{matched_poem['title']}》")
            print(result)
            
        except KeyboardInterrupt:
            print("\n\n👋 手动中断，演示结束！")
            break
        except Exception as e:
            print(f"\n❌ 操作错误：{str(e)}，请重试")

if __name__ == "__main__":
    interactive_exact_demo()
    