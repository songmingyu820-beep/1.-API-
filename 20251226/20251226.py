import re
from transformers import BertTokenizer, BertModel
import torch
import jieba.posseg as pseg  # 用于词性标注（可选，增强转喻分析）

# ====================== 1. 读取对话文件 ======================
file_path = "dialog metonymy.txt"  # 文件和代码同目录时，直接写文件名
try:
    with open(file_path, "r", encoding="utf-8") as f:
        dialog_lines = f.readlines()  # 逐行读取文件
except FileNotFoundError:
    print(f"❌ 错误：找不到文件 {file_path}！请检查文件路径和名称～")
    exit()  # 找不到文件则终止程序


# ====================== 2. 文本清洗：去除冗杂符号 ======================
def clean_dialog_text(raw_text):
    """
    清洗逻辑：去除括号内内容（如(喂.)、(wèi.)、[干ha,]）、标点、多余空格
    可根据你txt里的实际符号，调整正则表达式～
    """
    # 去除圆括号及内部所有内容（如 (喂.) → 空）
    text = re.sub(r"\([^)]*\)", "", raw_text)
    # 去除方括号及内部所有内容（如 [干ha,] → 空）
    text = re.sub(r"\[[^]]*\]", "", text)
    # 去除中文/英文标点（可根据需求调整，若需保留标点则删除这行）
    text = text.replace("，", "").replace("。", "").replace(",", "").replace(".", "")
    # 去除首尾空白和换行符
    text = text.strip()
    return text


# 清洗所有对话行，过滤空行
cleaned_dialogs = []
for line in dialog_lines:
    cleaned_line = clean_dialog_text(line)
    if cleaned_line:  # 只保留非空行
        cleaned_dialogs.append(cleaned_line)

# ====================== 3. 加载BERT中文预训练模型 ======================
print("🔌 正在加载BERT模型...（首次加载较慢，请耐心等待）")
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")  # 中文分词器
model = BertModel.from_pretrained("bert-base-chinese")  # 中文BERT模型
print("✅ BERT模型加载完成！")

# ====================== 4. 对每条对话进行BERT编码+转喻分析 ======================
for idx, dialog in enumerate(cleaned_dialogs, start=1):
    # ---------- 4.1 BERT编码：获取语义表示 ----------
    # 对文本分词+编码（return_tensors='pt'表示输出PyTorch张量）
    inputs = tokenizer(
        dialog,
        return_tensors="pt",
        padding="max_length",  # 填充到最大长度（保证批次内长度一致）
        truncation=True,  # 超过max_length则截断
        max_length=128  # 控制输入序列长度（根据需求调整）
    )

    # 前向传播，获取模型输出（with torch.no_grad() 不计算梯度，节省显存）
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state  # 最后一层的隐藏状态（语义信息核心）

    # ---------- 4.2 打印基础信息 ----------
    print(f"\n\n======= 第{idx}条对话分析 =======")
    print(f"【清洗前原文】：{line}")  # 原始行（带冗杂符号）
    print(f"【清洗后文本】：{dialog}")  # 清洗后的文本
    print(f"【BERT编码维度】：{last_hidden_state.shape}")  # 形状为 (1, 序列长度, 768)（batch_size=1, 隐藏层维度768）

    # 提取句子的[CLS]向量（用于分类任务的典型句子表示）
    cls_embedding = last_hidden_state[:, 0, :]  # [CLS] token对应的隐藏状态
    print(f"【[CLS]向量维度】：{cls_embedding.shape}")  # 形状为 (1, 768)

    # ---------- 4.3 转喻分析：自定义逻辑（示例：词性标注+长度启发式判断） ----------
    # 🔹 词性标注：看是否有“动词→名词”等转喻常见模式（如“打”是动词，“电话”是名词）
    words = pseg.cut(dialog)
    pos_tags = [(word.word, word.flag) for word in words]
    print(f"【词性标注结果】：{pos_tags}")

    # 🔹 启发式规则（示例：短文本更可能有转喻，仅作演示）
    if len(dialog) < 5:
        print("⚠️ 提示：该对话较短，可能存在转喻表达的可能，需结合领域知识验证～")

    # 🔹 更专业的分析：需结合「转喻标注数据集」微调BERT，或用领域词典匹配
    # （这部分属于进阶内容，需额外准备数据和代码，此处仅提供思路～）

print("\n\n🎉 对话清洗与BERT编码完成！可根据需求扩展转喻分析逻辑～")
import torch
from transformers import BertTokenizer, BertModel
import jieba.posseg as pseg

# ===================== 1. 加载BERT模型与分词器（复用已有编码逻辑） =====================
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')

# ===================== 2. 定义转喻分析规则（可按需扩展） =====================
def analyze_metonymy(dialog):
    """分析单条对话的转喻可能性，返回分析结果"""
    # ---------- 2.1 BERT编码：获取[CLS]向量（保留你原有的编码逻辑） ----------
    inputs = tokenizer(dialog, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    cls_embedding = last_hidden_state[:, 0, :]  # [CLS] token对应隐藏状态，形状(1,768)

    # ---------- 2.2 中文分词+词性标注（jieba.posseg） ----------
    words = pseg.cut(dialog)
    pos_tags = [(word.word, word.flag) for word in words]  # 格式：[(词1, 词性1), (词2, 词性2), ...]
    pos_seq = [flag for _, flag in pos_tags]  # 仅提取词性序列（用于规则匹配）

    # ---------- 2.3 转喻分析规则：多维度启发式判断 ----------
    possible_metonymy = False  # 标记是否可能存在转喻
    reasons = []               # 记录转喻依据

    # 🔹 规则1：长度启发式（短文本更易出现转喻，示例阈值=5）
    dialog_len = len(dialog)
    if dialog_len < 5:
        possible_metonymy = True
        reasons.append(f"文本长度较短（{dialog_len}字），符合转喻「短文本倾向」特征")

    # 🔹 规则2：词性模式（动词→名词的转喻常见模式，如“打→电话”“开→车”）
    # 👉 预定义**常见转喻动词-名词对**（需根据研究领域补充！）
    verb_noun_metonym_pairs = {
        "打": ["电话", "篮球"],  # 打电话（动作代行为）、打篮球（动作代运动）
        "开": ["车", "灯"],     # 开车（动作代交通工具）、开灯（动作代设备）
        "喝": ["酒", "水"],     # 喝酒（动作代饮品）
        "看": ["电视", "书"],   # 看电视（动作代媒介）
        "买": ["东西", "票"],   # 买东西（动作代商品）
        "吃": ["饭", "面"]      # 吃饭（动作代餐食）
    }
    # 遍历词性序列，匹配「动词→名词」模式
    for i in range(len(pos_tags)-1):
        word_i, pos_i = pos_tags[i]
        word_j, pos_j = pos_tags[i+1]
        # 前一词为动词（v开头），后一词为名词（n开头）
        if pos_i.startswith('v') and pos_j.startswith('n'):
            if word_i in verb_noun_metonym_pairs:
                if word_j in verb_noun_metonym_pairs[word_i]:
                    possible_metonymy = True
                    reasons.append(f"存在动词-名词转喻模式：「{word_i}({pos_i}) → {word_j}({pos_j})」，属于常见转喻搭配")

    # 🔹 规则3：领域词典（若有专业领域转喻，如医疗/科技，需自定义！示例留空）
    domain_metonym_dict = {}  # 示例：{"AI": "人工智能", "CT": "计算机断层扫描"}
    for word, _ in pos_tags:
        if word in domain_metonym_dict:
            possible_metonymy = True
            reasons.append(f"存在领域特定转喻：「{word}」代指「{domain_metonym_dict[word]}」")
            break  # 单个词匹配即标记

    # ---------- 2.4 整理分析结果 ----------
    result = {
        "dialog": dialog,          # 原始对话
        "is_metonymy": possible_metonymy,  # 是否可能含转喻
        "reasons": reasons,        # 转喻依据
        "cls_embedding_shape": cls_embedding.shape,  # [CLS]向量维度
        "pos_tags": pos_tags       # 词性标注结果
    }
    return result


# ===================== 3. 读取对话文档并批量分析 =====================
# 读取文件（确保dialog metonymy.txt存在，编码为UTF-8，每行一条对话）
with open("dialog metonymy.txt", "r", encoding="utf-8") as f:
    dialogs = [line.strip() for line in f if line.strip()]  # 过滤空行

# 逐条分析对话
for dialog in dialogs:
    analysis = analyze_metonymy(dialog)
    print(f"\n=== 分析对话：{analysis['dialog']} ===")
    print(f"[是否可能含转喻]：{analysis['is_metonymy']}")
    print(f"[分析依据]：{'；'.join(analysis['reasons']) if analysis['reasons'] else '无明确转喻依据'}")
    print(f"[词性标注结果]：{analysis['pos_tags']}")
    print(f"[CLS向量维度]：{analysis['cls_embedding_shape']}")