import jieba
import numpy as np
import torch
import pickle
import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 特殊标记定义（与主项目保持一致）
PAD_TOKEN = "<pad>"  # ID=0
UNK_TOKEN = "<unk>"  # ID=1
BOS_TOKEN = "<bos>"  # ID=2
EOS_TOKEN = "<eos>"  # ID=3

class FaultToleranceDataset(Dataset):
    """容错输入数据集：包含错误输入和正确输出的匹配样本"""
    def __init__(self, data, vocab, max_length=64):
        self.data = data
        self.vocab = vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # 错误的输入（带错别字或漏字）
        error_input = item["error_input"]
        # 正确的目标输出
        correct_target = item["correct_target"]
        # 正确的原诗标题
        poem_title = item["poem_title"]
        
        # 单字分词
        input_tokens = [BOS_TOKEN] + self._tokenize(error_input)
        target_tokens = self._tokenize(correct_target.replace("，", "").replace("。", "")) + [EOS_TOKEN]
        full_tokens = input_tokens + target_tokens
        
        # 截断过长序列
        if len(full_tokens) > self.max_length:
            full_tokens = full_tokens[:self.max_length]
            if full_tokens[-1] != EOS_TOKEN:
                full_tokens[-1] = EOS_TOKEN
        
        # 构建输入和标签
        x_tokens = full_tokens[:-1]
        y_tokens = full_tokens[1:]
        
        # 填充到固定长度
        pad_len = self.max_length - 1 - len(x_tokens)
        if pad_len > 0:
            x_tokens += [PAD_TOKEN] * pad_len
            y_tokens += [PAD_TOKEN] * pad_len
        
        x = self.vocab.convert_tokens_to_ids(x_tokens)
        y = self.vocab.convert_tokens_to_ids(y_tokens)
        
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
    
    def _tokenize(self, text):
        """文本分词（保留标点但在处理时忽略）"""
        return [c for c in text if c.strip()]

def generate_error_variations(correct_text, error_rate=0.2, max_errors=2):
    """
    为正确文本生成带错误的变体
    错误类型包括：错别字、漏字、多字
    """
    if len(correct_text) <= 1:
        return [correct_text]  # 太短的文本不生成错误
    
    variations = []
    text_chars = list(correct_text)
    num_chars = len(text_chars)
    
    # 计算最大错误数（取比例和固定值的最小值）
    max_possible_errors = min(int(num_chars * error_rate), max_errors)
    if max_possible_errors < 1:
        max_possible_errors = 1
    
    # 生成不同错误类型的变体
    for num_errors in range(1, max_possible_errors + 1):
        # 错别字变体
        if num_chars > num_errors:
            error_positions = random.sample(range(num_chars), num_errors)
            error_chars = []
            for i in range(num_chars):
                if i in error_positions:
                    # 替换为发音或字形相近的错别字
                    char = text_chars[i]
                    error_char = get_similar_char(char)
                    error_chars.append(error_char if error_char else char)
                else:
                    error_chars.append(text_chars[i])
            variations.append(''.join(error_chars))
        
        # 漏字变体
        if num_chars > num_errors:
            error_positions = sorted(random.sample(range(num_chars), num_errors))
            error_chars = []
            for i in range(num_chars):
                if i not in error_positions:
                    error_chars.append(text_chars[i])
            variations.append(''.join(error_chars))
        
        # 多字变体（添加无关字）
        if num_chars + num_errors <= 10:  # 避免太长
            error_positions = sorted(random.sample(range(num_chars + num_errors), num_errors))
            error_chars = []
            added = 0
            for i in range(num_chars + num_errors):
                if i in error_positions:
                    # 添加一个无关但常见的字
                    error_chars.append(random.choice(['的', '了', '是', '在', '有']))
                    added += 1
                else:
                    original_idx = i - added
                    if original_idx < num_chars:
                        error_chars.append(text_chars[original_idx])
            variations.append(''.join(error_chars))
    
    # 去重并确保包含原始文本作为对照
    variations = list(set(variations))
    return variations

def get_similar_char(char):
    """返回与给定汉字发音或字形相近的错别字"""
    similar_chars = {
        '床': ['床', '庄', '广'],
        '前': ['前', '剪', '箭'],
        '明': ['明', '名', '鸣'],
        '月': ['月', '曰', '目'],
        '光': ['光', '广', '先'],
        '疑': ['疑', '凝', '颖'],
        '是': ['是', '事', '市'],
        '地': ['地', '的', '他'],
        '上': ['上', '尚', '土'],
        '霜': ['霜', '箱', '相'],
        '举': ['举', '兴', '誉'],
        '头': ['头', '斗', '兴'],
        '望': ['望', '忘', '王'],
        '低': ['低', '底', '氐'],
        '思': ['思', '恩', '想'],
        '故': ['故', '古', '固'],
        '乡': ['乡', '香', '相'],
        '国': ['国', '图', '固'],
        '破': ['破', '坡', '波'],
        '山': ['山', '出', '仙'],
        '河': ['河', '何', '可'],
        '在': ['在', '再', '存'],
        '白': ['白', '百', '自'],
        '日': ['日', '曰', '目'],
        '依': ['依', '衣', '伊'],
        '山': ['山', '出', '仙'],
        '尽': ['尽', '近', '进'],
        '黄': ['黄', '皇', '簧'],
        '入': ['入', '人', '八'],
        '海': ['海', '悔', '梅'],
        '流': ['流', '留', '刘']
    }
    return random.choice(similar_chars.get(char, [char])) if char in similar_chars else None

def create_fault_tolerance_dataset(original_poems_path="original_poems.pkl", 
                                  vocab=None, 
                                  total_samples=10000, 
                                  max_length=64):
    """创建容错输入数据集"""
    print("开始创建【容错输入数据集】...")
    
    # 加载原诗库
    try:
        with open(original_poems_path, "rb") as f:
            original_poems = pickle.load(f)
        print(f"✅ 加载原诗库成功：共{len(original_poems)}首诗")
    except FileNotFoundError:
        print(f"❌ 未找到原诗库文件 {original_poems_path}")
        print("请先运行原始train.py生成原诗库")
        return None, None
    
    # 如果未提供词汇表，则创建新的
    if vocab is None:
        from dataset import ExactMatchVocab
        vocab = ExactMatchVocab()
        vocab.build_from_original_poems(original_poems)
    
    # 生成错误样本
    data = []
    for _ in tqdm(range(total_samples), desc="生成容错训练样本"):
        # 随机选择一首原诗
        selected_poem = random.choice(original_poems)
        poem_title = selected_poem["title"]
        full_content = selected_poem["content"]
        full_text_no_punc = selected_poem["full_text_no_punc"]
        all_starts = selected_poem["all_starts"]
        
        # 随机选择一个正确的开头
        correct_start = random.choice(all_starts)
        if len(correct_start) < 2:  # 跳过太短的开头，难以生成错误变体
            correct_start = all_starts[min(4, len(all_starts)-1)]  # 选择稍长的开头
        
        # 生成该开头的错误变体
        error_starts = generate_error_variations(correct_start)
        
        # 找到正确的目标文本
        start_idx = full_text_no_punc.find(correct_start)
        if start_idx == -1:
            start_idx = 0
        
        # 计算目标文本在带标点原诗中的位置
        punc_count_before = sum(1 for c in full_content[:full_content.find(correct_start)] if c in ["，", "。"])
        target_start_idx_in_content = len(correct_start) + punc_count_before
        correct_target = full_content[target_start_idx_in_content:]
        
        # 添加错误样本
        for error_start in error_starts:
            if len(error_start) > 0 and len(correct_target) > 0:
                data.append({
                    "error_input": error_start,
                    "correct_input": correct_start,
                    "correct_target": correct_target,
                    "poem_title": poem_title
                })
                if len(data) >= total_samples:
                    break
        
        if len(data) >= total_samples:
            break
    
    # 创建数据集实例
    dataset = FaultToleranceDataset(data, vocab, max_length)
    print(f"数据集创建完成：{len(dataset)}个容错样本，词汇表大小：{len(vocab)}")
    
    # 保存容错数据集
    with open("fault_tolerance_dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)
    print("容错数据集已保存：fault_tolerance_dataset.pkl")
    
    return dataset, vocab

def get_fault_tolerance_dataloader(dataset, batch_size=64, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True if torch.cuda.is_available() else False,
        num_workers=2 if torch.cuda.is_available() else 0,
        drop_last=True
    )
    