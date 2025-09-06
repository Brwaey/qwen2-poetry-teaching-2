import torch
import pickle
import editdistance  # 用于计算字符串编辑距离
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

def load_fault_tolerance_resources(
    model_path="fault_tolerance_poem_model.pth",
    vocab_path="exact_match_vocab.pkl",
    poems_path="original_poems.pkl"
):
    """加载容错输入模型所需资源"""
    # 1. 加载原诗库
    try:
        with open(poems_path, "rb") as f:
            original_poems = pickle.load(f)
        print(f"✅ 原诗库加载成功：共{len(original_poems)}首诗（含标准句式）")
    except FileNotFoundError:
        print(f"❌ 错误：未找到原诗库文件 {poems_path}")
        return None, None, None
    
    # 2. 加载词汇表（修复属性名拼写错误）
    try:
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        # 修复：original_poems_chars → original_poem_chars
        print(f"✅ 原诗词汇表加载成功：{len(vocab)}个标记（含{len(vocab.original_poem_chars)}个原诗单字）")
    except FileNotFoundError:
        print(f"❌ 错误：未找到词汇表文件 {vocab_path}")
        return None, None, original_poems
    
    # 3. 加载模型
    from train import ExactMatchModel
    model = ExactMatchModel(
        vocab_size=len(vocab),
        embed_dim=256,
        num_heads=4,
        num_layers=3,
        max_seq_len=64
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"✅ 容错输入模型加载成功：{model_path}")
        return model, vocab, original_poems
    except FileNotFoundError:
        print(f"❌ 错误：未找到模型文件 {model_path}")
        print("请先运行 train_fault_tolerance.py 训练模型")
        return None, vocab, original_poems
    

def find_best_match(input_text, original_poems, max_edit_distance=2):
    """
    寻找与输入文本最匹配的原诗开头（容忍一定错误）
    使用编辑距离衡量字符串相似度
    """
    input_text_clean = input_text.replace("，", "").replace("。", "").strip()
    if len(input_text_clean) == 0:
        return None, 0, 0
    
    best_match_poem = None
    best_start_idx = -1
    min_edit_dist = float('inf')
    best_match_text = ""
    
    for poem in original_poems:
        # 检查诗中的所有可能开头
        for start_text in poem["all_starts"]:
            # 计算编辑距离（衡量字符串相似度）
            dist = editdistance.eval(input_text_clean, start_text)
            
            # 优先选择长度相近且编辑距离小的匹配
            length_diff = abs(len(input_text_clean) - len(start_text))
            combined_score = dist + length_diff * 0.5  # 综合评分
            
            # 更新最佳匹配
            if combined_score < min_edit_dist and dist <= max_edit_distance:
                min_edit_dist = combined_score
                best_match_poem = poem
                best_match_text = start_text
                # 找到在完整无标点文本中的位置
                best_start_idx = poem["full_text_no_punc"].find(start_text)
    
    if best_match_poem is not None:
        print(f"✅ 容错匹配：输入 '{input_text}' 匹配到原诗开头 '{best_match_text}'（编辑距离：{min_edit_dist}）")
        return best_match_poem, best_start_idx, min_edit_dist
    else:
        print(f"⚠️  未找到足够相似的原诗开头（最大允许编辑距离：{max_edit_distance}）")
        return None, 0, 0

def add_standard_punctuation(text_no_punc, poem_pattern):
    """按原诗标准句式添加标点"""
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
            pos_idx = punctuation_pos.index(char_count)
            if pos_idx < len(punctuation_type):
                text_with_punc += punctuation_type[pos_idx]
    
    # 确保最后一句有句号
    if text_with_punc and text_with_punc[-1] not in ["。", "，"]:
        text_with_punc += "。"
    
    # 移除连续标点
    text_with_punc = text_with_punc.replace("，，", "，").replace("。。", "。").replace("，。", "。")
    return text_with_punc

def generate_with_fault_tolerance(
    model, vocab, original_poems, input_text,
    max_length=64,
    max_edit_distance=2  # 最大允许编辑距离
):
    """带容错能力的诗词生成"""
    try:
        # 1. 容错匹配原诗
        matched_poem, start_idx, edit_dist = find_best_match(
            input_text, original_poems, max_edit_distance
        )
        if matched_poem is None:
            return "❌ 未找到足够相似的原诗，无法生成"
        
        input_text_clean = input_text.replace("，", "").replace("。", "").strip()
        full_poem_no_punc = matched_poem["full_text_no_punc"]
        poem_pattern = matched_poem["sentence_pattern"]
        target_total_length = len(full_poem_no_punc)
        
        # 2. 输入预处理（使用匹配到的正确开头进行处理）
        correct_start = matched_poem["all_starts"][0]  # 取匹配到的诗的第一个正确开头
        for start in matched_poem["all_starts"]:
            if full_poem_no_punc.find(start) == start_idx:
                correct_start = start
                break
                
        input_tokens = [BOS_TOKEN] + vocab.tokenize(input_text_clean)
        input_ids = vocab.convert_tokens_to_ids(input_tokens)
        
        # 3. 初始化生成序列
        generated_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
        
        # 4. 逐词生成
        with torch.no_grad():
            required_steps = target_total_length - len(input_text_clean)
            for _ in range(required_steps):
                seq_len = generated_ids.shape[1]
                if seq_len >= max_length - 1:
                    break
                
                # 模型预测
                logits = model(generated_ids)
                next_token_logits = logits[:, -1, :]
                
                # 仅保留原诗中的字作为候选
                original_char_ids = [vocab.token_to_id[char] for char in full_poem_no_punc 
                                    if char in vocab.token_to_id]
                
                # 排除非原诗字
                for token_id in range(len(vocab)):
                    if token_id not in original_char_ids and token_id not in [PAD_ID, UNK_ID, BOS_ID, EOS_ID]:
                        next_token_logits[0, token_id] = -1e9
                
                # 选择概率最高的字
                next_token_id = torch.argmax(next_token_logits, dim=-1).item()
                
                # 终止条件
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
        
        # 5. 后处理
        generated_no_punc = "".join([
            vocab.id_to_token[id.item()] if isinstance(id, torch.Tensor) else vocab.id_to_token[id]
            for id in generated_ids[0]
            if id not in [PAD_ID, UNK_ID, BOS_ID, EOS_ID]
        ])
        
        # 补充原诗缺失部分
        if len(generated_no_punc) < len(full_poem_no_punc):
            generated_no_punc = full_poem_no_punc
        
        # 添加标点
        final_poem = add_standard_punctuation(generated_no_punc, poem_pattern)
        
        # 还原输入时的标点
        if input_text.endswith("，") and not final_poem.startswith(input_text):
            input_punc_pos = len(input_text_clean)
            if input_punc_pos < len(final_poem):
                final_poem = final_poem[:input_punc_pos] + "，" + final_poem[input_punc_pos:]
        
        return final_poem
    except Exception as e:
        import traceback
        print(f"生成错误详情：{traceback.format_exc()}")
        return f"❌ 生成错误：{str(e)}"

def interactive_fault_tolerance_demo():
    """容错输入交互式演示"""
    print("=" * 80)
    print("          📜 Qwen2中文诗词生成（容错输入增强版）          ")
    print("=" * 80)
    print("📚 支持带错别字、漏字的诗句开头（容错输入），例如：")
    print("  - 输入 '床前明月广' → 应匹配 '床前明月光' 并生成完整《静夜思》")
    print("  - 输入 '白日依山' → 应匹配 '白日依山尽' 并生成完整《登鹳雀楼》")
    print("  - 输入 '国破山河再' → 应匹配 '国破山河在' 并生成完整《春望》")
    print("💡 输入 'exit' 退出演示，输入 'help' 查看帮助")
    print("=" * 80)
    
    # 加载所有资源
    print("\n正在加载容错输入模型资源...")
    model, vocab, original_poems = load_fault_tolerance_resources()
    if model is None or vocab is None or original_poems is None:
        print("❌ 演示启动失败，请确保所有资源文件已生成")
        return
    
    # 交互循环
    print("\n✅ 演示启动成功！请输入带可能错误的诗句开头（如'床前明月广'）：")
    while True:
        try:
            user_input = input("\n你输入的诗句开头：").strip()
            
            # 命令处理
            if user_input.lower() == "exit":
                print("👋 感谢使用，容错输入诗词生成演示结束！")
                break
            elif user_input.lower() == "help":
                print("📖 帮助说明：")
                print("  1. 可以输入带错别字的开头（如'床前明月广'）")
                print("  2. 可以输入不完整的开头（如'白日依山'）")
                print("  3. 可以输入带多余字的开头（如'床前的明月光'）")
                print("  4. 输入 'exit' 退出演示，输入 'help' 查看帮助")
                continue
            
            # 生成诗词
            print("✍️  正在容错匹配并生成原诗...")
            result = generate_with_fault_tolerance(model, vocab, original_poems, user_input)
            
            # 显示结果
            print(f"\n🎯 容错生成结果：")
            matched_poem, _, _ = find_best_match(user_input, original_poems)
            if matched_poem:
                print(f"《{matched_poem['title']}》")
            print(result)
            
        except KeyboardInterrupt:
            print("\n\n👋 手动中断，演示结束！")
            break
        except Exception as e:
            print(f"\n❌ 操作错误：{str(e)}，请重试")

if __name__ == "__main__":
    interactive_fault_tolerance_demo()
    