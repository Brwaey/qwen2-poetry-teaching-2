import jieba
import numpy as np
import torch
import pickle  # 添加pickle导入
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm

# 特殊标记定义（全项目统一）
PAD_TOKEN = "<pad>"  # ID=0
UNK_TOKEN = "<unk>"  # ID=1
BOS_TOKEN = "<bos>"  # ID=2
EOS_TOKEN = "<eos>"  # ID=3

class ExactMatchVocab:
    """原诗专用词汇表：强制保留所有原诗单字，无低频过滤"""
    def __init__(self):
        self.token_to_id = {
            PAD_TOKEN: 0,
            UNK_TOKEN: 1,
            BOS_TOKEN: 2,
            EOS_TOKEN: 3
        }
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.original_poem_chars = set()  # 存储所有原诗单字，确保无遗漏
    
    def add_original_poem_chars(self, poem_text):
        """添加原诗单字到词汇表（确保所有原诗字都在词汇表中）"""
        chars = [c for c in poem_text if c.strip() and c not in ["，", "。"]]
        for char in chars:
            self.original_poem_chars.add(char)
            if char not in self.token_to_id:
                new_id = len(self.token_to_id)
                self.token_to_id[char] = new_id
                self.id_to_token[new_id] = char
    
    def build_from_original_poems(self, original_poems):
        """从原诗库构建词汇表（仅保留原诗单字+特殊标记）"""
        print("构建原诗专用词汇表（强制保留所有原诗单字）...")
        for poem in original_poems:
            self.add_original_poem_chars(poem["content"])
            self.add_original_poem_chars(poem["full_text_no_punc"])
        
        print(f"词汇表构建完成：共{len(self.token_to_id)}个标记（含{len(self.original_poem_chars)}个原诗单字）")
        return self
    
    def tokenize(self, text):
        """中文单字分词（原诗以单字为单位，避免jieba分词误差）"""
        return [c for c in text if c.strip() and c not in ["，", "。"]]
    
    def convert_tokens_to_ids(self, tokens):
        """确保原诗单字都能正确转换为ID（无UNK）"""
        ids = []
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                # 异常处理：若出现非原诗字，用UNK但打印警告
                print(f"⚠️  非原诗字 '{token}'，替换为UNK")
                ids.append(self.token_to_id[UNK_TOKEN])
        return ids
    
    def convert_ids_to_tokens(self, ids):
        return [self.id_to_token.get(id, UNK_TOKEN) for id in ids]
    
    def __len__(self):
        return len(self.token_to_id)

class ExactMatchDataset(Dataset):
    """原诗精准匹配数据集：确保每个样本的输入→目标都来自同一首诗"""
    def __init__(self, data, vocab, max_length=64):
        self.data = data
        self.vocab = vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item["input"]  # 来自某首诗的开头
        target_text = item["target"]  # 该诗对应的后续内容（带正确标点）
        poem_title = item["poem_title"]  # 记录所属诗歌，确保不跨诗
        
        # 单字分词（避免jieba分词错误）
        input_tokens = [BOS_TOKEN] + self.vocab.tokenize(input_text)
        target_tokens = self.vocab.tokenize(target_text.replace("，", "").replace("。", "")) + [EOS_TOKEN]
        full_tokens = input_tokens + target_tokens
        
        # 截断过长序列（确保不超过max_length）
        if len(full_tokens) > self.max_length:
            full_tokens = full_tokens[:self.max_length]
            # 确保最后一个token是EOS（避免截断导致目标不完整）
            if full_tokens[-1] != EOS_TOKEN:
                full_tokens[-1] = EOS_TOKEN
        
        # 构建输入和标签（严格对应原诗）
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

def create_exact_match_dataset(total_samples=20000, max_length=64):
    """
    核心优化：
    1. 原诗库添加“标准句式”和“所有可能开头”，确保输入→原诗一一对应
    2. 每个样本都来自同一首诗，避免跨诗混杂
    3. 目标文本带标准标点，让模型学习正确标点位置
    """
    print("开始创建【原诗精准匹配数据集】...")
    
    # ---------------------- 原诗库（带标准句式和所有可能开头）----------------------
    original_poems = [
        # 李白《静夜思》
        {
            "title": "静夜思",
            "content": "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
            "full_text_no_punc": "床前明月光疑是地上霜举头望明月低头思故乡",
            "sentence_pattern": {
                "char_per_sentence": 7,
                "punctuation_pos": [7, 14, 21],
                "punctuation_type": ["，", "。", "，", "。"]
            },
            "all_starts": [
                "床", "床前", "床前明", "床前明月", "床前明月光",
                "疑", "疑是", "疑是地", "疑是地上", "疑是地上霜",
                "举", "举头", "举头望", "举头望明", "举头望明月",
                "低", "低头", "低头思", "低头思乡", "低头思故乡"
            ]
        },
        # 李白《早发白帝城》
        {
            "title": "早发白帝城",
            "content": "朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。",
            "full_text_no_punc": "朝辞白帝彩云间千里江陵一日还两岸猿声啼不住轻舟已过万重山",
            "sentence_pattern": {
                "char_per_sentence": 7,
                "punctuation_pos": [7, 14, 21],
                "punctuation_type": ["，", "。", "，", "。"]
            },
            "all_starts": [
                "朝", "朝辞", "朝辞白", "朝辞白帝", "朝辞白帝彩", "朝辞白帝彩云", "朝辞白帝彩云间",
                "千", "千里", "千里江", "千里江陵", "千里江陵一", "千里江陵一日", "千里江陵一日还",
                "两", "两岸", "两岸猿", "两岸猿声", "两岸猿声啼", "两岸猿声啼不", "两岸猿声啼不住",
                "轻", "轻舟", "轻舟已", "轻舟已过", "轻舟已过万", "轻舟已过万重", "轻舟已过万重山"
            ]
        },
        # 杜甫《春望》
        {
            "title": "春望",
            "content": "国破山河在，城春草木深。感时花溅泪，恨别鸟惊心。烽火连三月，家书抵万金。白头搔更短，浑欲不胜簪。",
            "full_text_no_punc": "国破山河在城春草木深感时花溅泪恨别鸟惊心烽火连三月家书抵万金白头搔更短浑欲不胜簪",
            "sentence_pattern": {
                "char_per_sentence": 5,
                "punctuation_pos": [5, 10, 15, 20, 25, 30, 35],
                "punctuation_type": ["，", "。", "，", "。", "，", "。", "，", "。"]
            },
            "all_starts": [
                "国", "国破", "国破山", "国破山河", "国破山河在",
                "城", "城春", "城春草", "城春草木", "城春草木深",
                "感", "感时", "感时花", "感时花溅", "感时花溅泪",
                "恨", "恨别", "恨别鸟", "恨别鸟惊", "恨别鸟惊心",
                "烽", "烽火", "烽火连", "烽火连三", "烽火连三月",
                "家", "家书", "家书抵", "家书抵万", "家书抵万金",
                "白", "白头", "白头搔", "白头搔更", "白头搔更短",
                "浑", "浑欲", "浑欲不", "浑欲不胜", "浑欲不胜簪"
            ]
        },
        # 王之涣《登鹳雀楼》
        {
            "title": "登鹳雀楼",
            "content": "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
            "full_text_no_punc": "白日依山尽黄河入海流欲穷千里目更上一层楼",
            "sentence_pattern": {
                "char_per_sentence": 5,
                "punctuation_pos": [5, 10, 15],
                "punctuation_type": ["，", "。", "，", "。"]
            },
            "all_starts": [
                "白", "白日", "白日依", "白日依山", "白日依山尽",
                "黄", "黄河", "黄河入", "黄河入海", "黄河入海流",
                "欲", "欲穷", "欲穷千", "欲穷千里", "欲穷千里目",
                "更", "更上", "更上一", "更上一层", "更上一层楼"
            ]
        },
        # 杜牧《山行》
        {
            "title": "山行",
            "content": "远上寒山石径斜，白云生处有人家。停车坐爱枫林晚，霜叶红于二月花。",
            "full_text_no_punc": "远上寒山石径斜白云生处有人家停车坐爱枫林晚霜叶红于二月花",
            "sentence_pattern": {
                "char_per_sentence": 7,
                "punctuation_pos": [7, 14, 21],
                "punctuation_type": ["，", "。", "，", "。"]
            },
            "all_starts": [
                "远", "远上", "远上寒", "远上寒山", "远上寒山石", "远上寒山石径", "远上寒山石径斜",
                "白", "白云", "白云生", "白云生处", "白云生处有", "白云生处有人", "白云生处有人家",
                "停", "停车", "停车坐", "停车坐爱", "停车坐爱枫", "停车坐爱枫林", "停车坐爱枫林晚",
                "霜", "霜叶", "霜叶红", "霜叶红于", "霜叶红于二", "霜叶红于二月", "霜叶红于二月花"
            ]
        },
        # 杜甫《绝句》
        {
            "title": "绝句",
            "content": "两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船。",
            "full_text_no_punc": "两个黄鹂鸣翠柳一行白鹭上青天窗含西岭千秋雪门泊东吴万里船",
            "sentence_pattern": {
                "char_per_sentence": 7,
                "punctuation_pos": [7, 14, 21],
                "punctuation_type": ["，", "。", "，", "。"]
            },
            "all_starts": [
                "两", "两个", "两个黄", "两个黄鹂", "两个黄鹂鸣", "两个黄鹂鸣翠", "两个黄鹂鸣翠柳",
                "一", "一行", "一行白", "一行白鹭", "一行白鹭上", "一行白鹭上青", "一行白鹭上青天",
                "窗", "窗含", "窗含西", "窗含西岭", "窗含西岭千", "窗含西岭千秋", "窗含西岭千秋雪",
                "门", "门泊", "门泊东", "门泊东吴", "门泊东吴万", "门泊东吴万里", "门泊东吴万里船"
            ]
        }
    ]
    
    # ---------------------- 生成20000个样本（100%来自原诗，无跨诗）----------------------
    data = []
    for _ in tqdm(range(total_samples), desc="生成原诗精准匹配样本"):
        # 随机选择一首原诗
        selected_poem = random.choice(original_poems)
        poem_title = selected_poem["title"]
        full_content = selected_poem["content"]  # 带标准标点的完整原诗
        full_text_no_punc = selected_poem["full_text_no_punc"]  # 无标点全文
        all_starts = selected_poem["all_starts"]
        
        # 随机选择一个开头（确保来自当前诗）
        input_start = random.choice(all_starts)
        # 找到该开头在无标点全文中的位置
        start_idx = full_text_no_punc.find(input_start)
        if start_idx == -1:
            # 异常处理：若开头未找到（理论上不会发生），重新选择
            input_start = all_starts[0]
            start_idx = 0
        
        # 目标文本：从开头后到完整原诗结束（带标准标点）
        # 计算目标文本在带标点原诗中的位置
        # 先统计开头前的标点数量（用于修正带标点文本的索引）
        punc_count_before = sum(1 for c in full_content[:full_content.find(input_start)] if c in ["，", "。"])
        target_start_idx_in_content = len(input_start) + punc_count_before
        target_text = full_content[target_start_idx_in_content:]
        
        # 过滤无效样本（确保目标文本不为空）
        if len(input_start) > 0 and len(target_text) > 0:
            data.append({
                "input": input_start,
                "target": target_text,
                "poem_title": poem_title,
                "poem_pattern": selected_poem["sentence_pattern"]  # 记录句式，用于生成时标点
            })
    
    # ---------------------- 构建原诗专用词汇表 ----------------------
    vocab = ExactMatchVocab()
    vocab.build_from_original_poems(original_poems)
    
    # ---------------------- 创建数据集实例 ----------------------
    dataset = ExactMatchDataset(data, vocab, max_length)
    print(f"数据集创建完成：{len(dataset)}个样本（100%来自原诗），词汇表大小：{len(vocab)}")
    
    # 保存原诗库（用于生成时匹配句式）
    with open("original_poems.pkl", "wb") as f:
        pickle.dump(original_poems, f)
    print("原诗库已保存：original_poems.pkl（含标准句式）")
    
    return dataset, vocab

def get_exact_match_dataloader(dataset, batch_size=64, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True if torch.cuda.is_available() else False,
        num_workers=2 if torch.cuda.is_available() else 0,
        drop_last=True
    )
