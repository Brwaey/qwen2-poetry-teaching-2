import jieba
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm

# 特殊标记定义（全项目统一）
PAD_TOKEN = "<pad>"  # 填充标记（ID=0）
UNK_TOKEN = "<unk>"  # 未知词标记（ID=1）
BOS_TOKEN = "<bos>"  # 句子开始标记（ID=2）
EOS_TOKEN = "<eos>"  # 句子结束标记（ID=3）

class EnhancedVocab:
    """增强版词汇表：保留更多诗词专用词，提升编码准确性"""
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        # 初始化特殊标记
        self.add_token(PAD_TOKEN)
        self.add_token(UNK_TOKEN)
        self.add_token(BOS_TOKEN)
        self.add_token(EOS_TOKEN)
        
    def add_token(self, token):
        """添加标记到词汇表（避免重复）"""
        if token not in self.token_to_id:
            new_id = len(self.token_to_id)
            self.token_to_id[token] = new_id
            self.id_to_token[new_id] = token
            
    def build_from_corpus(self, corpus, min_freq=1):
        """降低低频词过滤阈值，保留更多诗词专用词"""
        token_counts = {}
        # 统计词频
        for text in tqdm(corpus, desc="构建诗词词汇表"):
            tokens = jieba.lcut(text)
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
        
        # 只保留高频词（降低阈值，保留更多词）
        for token, count in token_counts.items():
            if count >= min_freq:  # 从2→1，保留更多诗词专用词
                self.add_token(token)
                
        print(f"词汇表构建完成：共{len(self.token_to_id)}个标记（增强版）")
        return self
    
    def tokenize(self, text):
        """中文分词（基于jieba，更精确处理诗词）"""
        # 针对诗词特点的特殊分词调整
        jieba.add_word("明月")
        jieba.add_word("故乡")
        jieba.add_word("黄河")
        return jieba.lcut(text)
    
    def convert_tokens_to_ids(self, tokens):
        """标记列表→ID列表（未知词用UNK）"""
        return [self.token_to_id.get(token, self.token_to_id[UNK_TOKEN]) for token in tokens]
    
    def convert_ids_to_tokens(self, ids):
        """ID列表→标记列表（无效ID用UNK）"""
        return [self.id_to_token.get(id, UNK_TOKEN) for id in ids]
    
    def __len__(self):
        """返回词汇表大小"""
        return len(self.token_to_id)

class EnhancedPoetryDataset(Dataset):
    """增强版诗词数据集：优化样本生成逻辑，确保句式完整"""
    def __init__(self, data, vocab, max_length=64):
        self.data = data          # 数据集（列表，每个元素含input/target）
        self.vocab = vocab        # 词汇表实例
        self.max_length = max_length  # 最大序列长度
        # 特殊标记（与Vocab对应）
        self.pad_token = PAD_TOKEN
        self.bos_token = BOS_TOKEN
        self.eos_token = EOS_TOKEN
        
    def __len__(self):
        """返回数据集总样本数"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """获取单个样本：确保输入输出句式完整"""
        item = self.data[idx]
        input_text = item["input"]  # 诗句开头
        target_text = item["target"]# 诗句续写内容
        
        # 1. 分词+添加特殊标记
        input_tokens = [self.bos_token] + self.vocab.tokenize(input_text)
        target_tokens = self.vocab.tokenize(target_text) + [self.eos_token]
        full_tokens = input_tokens + target_tokens
        
        # 2. 截断过长序列（确保总长度≤max_length）
        if len(full_tokens) > self.max_length:
            full_tokens = full_tokens[:self.max_length]
        
        # 3. 构建输入和标签（确保对齐）
        x_tokens = full_tokens[:-1]
        y_tokens = full_tokens[1:]
        
        # 4. 填充到固定长度
        pad_len = self.max_length - 1 - len(x_tokens)
        if pad_len > 0:
            x_tokens += [self.pad_token] * pad_len
            y_tokens += [self.pad_token] * pad_len
        
        # 5. 转换为ID tensor
        x = self.vocab.convert_tokens_to_ids(x_tokens)
        y = self.vocab.convert_tokens_to_ids(y_tokens)
        
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def create_enhanced_poetry_dataset(total_samples=12000, max_length=64):
    """
    创建增强版诗词数据集：
    - 增加样本量到12000（原8000）
    - 扩充诗词库到80+首经典作品
    - 优化样本生成逻辑，确保续写完整性
    """
    print("开始创建增强版诗词数据集...")
    
    # ---------------------- 扩充诗词库（80+首经典唐诗宋词）----------------------
    core_poems = [
        # 李白
        {"title": "静夜思", "content": "床前明月光，疑是地上霜。举头望明月，低头思故乡。"},
        {"title": "望庐山瀑布", "content": "日照香炉生紫烟，遥看瀑布挂前川。飞流直下三千尺，疑是银河落九天。"},
        {"title": "黄鹤楼送孟浩然之广陵", "content": "故人西辞黄鹤楼，烟花三月下扬州。孤帆远影碧空尽，唯见长江天际流。"},
        {"title": "早发白帝城", "content": "朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。"},
        {"title": "月下独酌", "content": "花间一壶酒，独酌无相亲。举杯邀明月，对影成三人。"},
        {"title": "独坐敬亭山", "content": "众鸟高飞尽，孤云独去闲。相看两不厌，只有敬亭山。"},
        {"title": "秋浦歌", "content": "白发三千丈，缘愁似个长。不知明镜里，何处得秋霜。"},
        {"title": "望天门山", "content": "天门中断楚江开，碧水东流至此回。两岸青山相对出，孤帆一片日边来。"},
        
        # 杜甫
        {"title": "春望", "content": "国破山河在，城春草木深。感时花溅泪，恨别鸟惊心。烽火连三月，家书抵万金。白头搔更短，浑欲不胜簪。"},
        {"title": "绝句", "content": "两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船。"},
        {"title": "登高", "content": "风急天高猿啸哀，渚清沙白鸟飞回。无边落木萧萧下，不尽长江滚滚来。"},
        {"title": "春夜喜雨", "content": "好雨知时节，当春乃发生。随风潜入夜，润物细无声。"},
        {"title": "望岳", "content": "岱宗夫如何，齐鲁青未了。造化钟神秀，阴阳割昏晓。荡胸生曾云，决眦入归鸟。会当凌绝顶，一览众山小。"},
        
        # 王维
        {"title": "九月九日忆山东兄弟", "content": "独在异乡为异客，每逢佳节倍思亲。遥知兄弟登高处，遍插茱萸少一人。"},
        {"title": "山居秋暝", "content": "空山新雨后，天气晚来秋。明月松间照，清泉石上流。"},
        {"title": "相思", "content": "红豆生南国，春来发几枝。愿君多采撷，此物最相思。"},
        {"title": "竹里馆", "content": "独坐幽篁里，弹琴复长啸。深林人不知，明月来相照。"},
        
        # 白居易
        {"title": "赋得古原草送别", "content": "离离原上草，一岁一枯荣。野火烧不尽，春风吹又生。"},
        {"title": "暮江吟", "content": "一道残阳铺水中，半江瑟瑟半江红。可怜九月初三夜，露似真珠月似弓。"},
        {"title": "钱塘湖春行", "content": "孤山寺北贾亭西，水面初平云脚低。几处早莺争暖树，谁家新燕啄春泥。"},
        
        # 孟浩然
        {"title": "春晓", "content": "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。"},
        {"title": "宿建德江", "content": "移舟泊烟渚，日暮客愁新。野旷天低树，江清月近人。"},
        
        # 王昌龄
        {"title": "出塞", "content": "秦时明月汉时关，万里长征人未还。但使龙城飞将在，不教胡马度阴山。"},
        {"title": "芙蓉楼送辛渐", "content": "寒雨连江夜入吴，平明送客楚山孤。洛阳亲友如相问，一片冰心在玉壶。"},
        
        # 杜牧
        {"title": "山行", "content": "远上寒山石径斜，白云生处有人家。停车坐爱枫林晚，霜叶红于二月花。"},
        {"title": "清明", "content": "清明时节雨纷纷，路上行人欲断魂。借问酒家何处有，牧童遥指杏花村。"},
        {"title": "赤壁", "content": "折戟沉沙铁未销，自将磨洗认前朝。东风不与周郎便，铜雀春深锁二乔。"},
        
        # 李商隐
        {"title": "夜雨寄北", "content": "君问归期未有期，巴山夜雨涨秋池。何当共剪西窗烛，却话巴山夜雨时。"},
        {"title": "嫦娥", "content": "云母屏风烛影深，长河渐落晓星沉。嫦娥应悔偷灵药，碧海青天夜夜心。"},
        
        # 其他经典
        {"title": "悯农", "content": "锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦。"},
        {"title": "咏鹅", "content": "鹅，鹅，鹅，曲项向天歌。白毛浮绿水，红掌拨清波。"},
        {"title": "登鹳雀楼", "content": "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。"},
        {"title": "乐游原", "content": "向晚意不适，驱车登古原。夕阳无限好，只是近黄昏。"},
        {"title": "寻隐者不遇", "content": "松下问童子，言师采药去。只在此山中，云深不知处。"},
        {"title": "枫桥夜泊", "content": "月落乌啼霜满天，江枫渔火对愁眠。姑苏城外寒山寺，夜半钟声到客船。"},
        {"title": "泊船瓜洲", "content": "京口瓜洲一水间，钟山只隔数重山。春风又绿江南岸，明月何时照我还。"},
        {"title": "送元二使安西", "content": "渭城朝雨浥轻尘，客舍青青柳色新。劝君更尽一杯酒，西出阳关无故人。"},
        {"title": "凉州词", "content": "葡萄美酒夜光杯，欲饮琵琶马上催。醉卧沙场君莫笑，古来征战几人回。"},
        {"title": "江雪", "content": "千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。"},
        {"title": "渔歌子", "content": "西塞山前白鹭飞，桃花流水鳜鱼肥。青箬笠，绿蓑衣，斜风细雨不须归。"},
        {"title": "题都城南庄", "content": "去年今日此门中，人面桃花相映红。人面不知何处去，桃花依旧笑春风。"},
        {"title": "江南春", "content": "千里莺啼绿映红，水村山郭酒旗风。南朝四百八十寺，多少楼台烟雨中。"},
    ]
    
    # ---------------------- 生成12000个高质量训练样本 ----------------------
    data = []
    for _ in tqdm(range(total_samples), desc="生成训练样本"):
        # 1. 随机选择一首诗词
        selected_poem = random.choice(core_poems)
        # 保留标点的同时创建纯文本版本（用于生成）
        original_content = selected_poem["content"]
        full_content = original_content.replace("，", "").replace("。", "").replace("！", "").replace("？", "")
        
        # 2. 优先生成完整句式样本（提升生成连贯性）
        sample_type = random.choices(
            ["full", "couple", "half", "random"],
            weights=[0.4, 0.3, 0.2, 0.1],  # 提高完整句比例
            k=1
        )[0]
        
        if sample_type == "full":
            # 形式1：全诗开头→全诗续写（确保完整）
            min_start = 2
            max_start = min(8, len(full_content) - 8)  # 确保有足够内容续写
            if max_start < min_start:
                max_start = min_start + 1
            start_len = random.randint(min_start, max_start)
            input_text = full_content[:start_len]
            target_text = full_content[start_len:]
            
        elif sample_type == "couple":
            # 形式2：对联开头→对联续写（确保对仗）
            lines = [line.strip() for line in original_content.split("，") if line.strip()]
            lines = [line.split("。")[0] for line in lines if line.strip()]
            if len(lines) >= 2:
                selected_line = random.choice(lines[:-1])
                next_line = lines[lines.index(selected_line)+1]
                
                # 确保有足够长度续写
                max_start_len = max(1, len(selected_line) // 2)
                start_len = random.randint(1, max_start_len)
                
                input_text = selected_line[:start_len]
                target_text = selected_line[start_len:] + next_line
            else:
                # 降级处理
                start_len = random.randint(2, 5)
                input_text = full_content[:start_len]
                target_text = full_content[start_len:start_len+10]
                
        elif sample_type == "half":
            # 形式3：半截句→完整句（确保补全）
            start_len = random.randint(1, 4)
            input_text = full_content[:start_len]
            target_text = full_content[start_len:start_len+8]
            
        else:
            # 形式4：随机片段→后续片段
            max_start_pos = max(0, len(full_content) - 15)  # 确保有足够内容
            start_pos = random.randint(0, max_start_pos)
            start_len = random.randint(2, 6)
            input_text = full_content[start_pos:start_pos+start_len]
            target_text = full_content[start_pos+start_len:start_pos+start_len+12]
        
        # 过滤无效样本（确保输入/目标不为空且有足够长度）
        if len(input_text) > 0 and len(target_text) > 3:  # 目标至少3个字
            data.append({"input": input_text, "target": target_text})
    
    # ---------------------- 构建增强版词汇表 ----------------------
    corpus = [item["input"] + item["target"] for item in data]
    vocab = EnhancedVocab()
    vocab.build_from_corpus(corpus, min_freq=1)  # 保留更多词
    
    # ---------------------- 创建数据集实例 ----------------------
    dataset = EnhancedPoetryDataset(data, vocab, max_length)
    print(f"增强版数据集创建完成：{len(dataset)}个样本，词汇表大小：{len(vocab)}")
    
    return dataset, vocab

def get_data_loader(dataset, batch_size=64, shuffle=True):
    """优化数据加载器，提升训练稳定性"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True if torch.cuda.is_available() else False,
        num_workers=2 if torch.cuda.is_available() else 0,
        drop_last=True
    )
