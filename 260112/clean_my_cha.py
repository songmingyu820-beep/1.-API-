import os
import re
import glob
from opencc import OpenCC

# 1. 初始化繁转简工具
cc = OpenCC('t2s')


def clean_text(text):
    """清理单行对话中的 CHILDES 符号并转为简体"""
    # 去掉方括号内容 [= xxx] [?] [+ xxx]
    text = re.sub(r'\[.*?\]', '', text)
    # 去掉圆括号内容 (..)
    text = re.sub(r'\(.*?\)', '', text)
    # 去掉尖括号 < >
    text = re.sub(r'[<>]', '', text)
    # 去掉特殊符号如 &, + , #, /
    text = re.sub(r'[\&\+#/]', '', text)
    # 繁体转简体
    text = cc.convert(text)
    # 仅保留汉字和基本标点（去掉拼音、希腊字母、特殊代码）
    text = "".join(re.findall(r'[\u4e00-\u9fa5，。？！；：“”]', text))
    return text.strip()


def batch_process_childes(input_dir):
    """批量清洗指定路径下的 .cha 文件"""

    # 定义输出文件夹：在原文件夹同级目录下创建 cleaned_txt
    output_dir = os.path.join(os.path.dirname(input_dir), "cleaned_txt")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f">>> 已创建输出文件夹: {output_dir}")

    # 获取所有 .cha 文件
    files = glob.glob(os.path.join(input_dir, "*.cha"))

    if not files:
        print(f"错误：在路径 {input_dir} 中没找到 .cha 文件！请检查路径是否正确。")
        return

    print(f">>> 找到 {len(files)} 个文件，准备开始清洗...\n")

    for file_path in files:
        file_name = os.path.basename(file_path)
        output_file_path = os.path.join(output_dir, file_name.replace('.cha', '.txt'))

        cleaned_lines = []

        # 使用 utf-8 编码读取
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # 只保留以 * 开头的对话行
                if line.startswith('*'):
                    try:
                        # 提取说话人标签 (如 CHI, MOT, EXP)
                        speaker = line[1:4].upper()
                        # 提取对话文本
                        raw_content = line.split(':', 1)[-1]
                        # 清洗文本
                        pure_content = clean_text(raw_content)

                        if pure_content:
                            cleaned_lines.append(f"{speaker}: {pure_content}")
                    except Exception:
                        continue

        # 保存结果
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(cleaned_lines))

        print(f"成功清洗: {file_name}")

    print(f"\n>>> 全部完成！清洗后的文件存放在: {output_dir}")


# ==================== 执行区域 ====================
# 使用 r"" 这种原始字符串格式，防止 Windows 路径的反斜杠报错
my_path = r"C:\Users\smy\Desktop\partcha"

batch_process_childes(my_path)
# =================================================
import os
from openai import OpenAI

# 1. API 配置
client = OpenAI(
    api_key="sk-dc48e456ec1a46a18abec7d2f1e57a18",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

input_dir = r"C:\Users\smy\PycharmProjects\PythonProject\260112\cleaned_txt"
files = sorted([f for f in os.listdir(input_dir) if f.endswith('.txt')])
target_file = os.path.join(input_dir, files[0])

# 2. 深度认知分析指令 (System Prompt)
# 我们不再提供关键词，而是提供“语义范式”
SYSTEM_PROMPT = """你是一个认知语言学和对话分析专家。你的任务是分析语料中“状态代行事（State-for-Action）”的转喻共鸣。

任务定义：
1. 识别状态句：寻找一方（通常是成人）陈述某种物理、心理、环境或生物状态的句子（例如：饿了、脏了、快完了、位置不够了、这个坏了）。
2. 分析共鸣：观察另一方（通常是孩子）的下一句回应，判断其是否通过转喻思维跳跃到了“核心动作（Action/CORE）”。

分类标准：
- [转喻共鸣]：回应直接针对状态背后的“动作”或“解决方案”（如：陈述“地脏”，回应“扫扫”）。
- [字面共鸣]：回应仅停留于确认或重复该状态（如：陈述“地脏”，回应“脏”或“嗯”）。
- [非共鸣回复]：回复完全跳出了该场景。

输出要求：
请仅提取并输出符合“状态陈述”特征的对话片段。如果一段对话中没有状态陈述，请忽略。
格式如下：
场景序号：[N]
对话原文：
人名A: [状态陈述句]
人名B: [回应句]
认知分析：[简述为何这是状态代行事，以及孩子是否完成了动作跳跃]
最终分类：[转喻共鸣/字面共鸣/非共鸣回复]
"""


def analyze_chunk(chunk_text):
    """调用 API 分析一小块对话内容"""
    try:
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"请扫描以下对话块，识别并分析其中的“状态代行事”现象：\n\n{chunk_text}"}
            ],
            temperature=0.0  # 绝对严谨，不胡思乱想
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"发生错误: {e}"


# 3. 读取语料并分块处理
with open(target_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 我们每 10 行作为一个 Chunk 传给 AI，重叠 2 行以防状态和回应被切断在两个块中
chunk_size = 10
overlap = 2
print(f"--- 正在智能扫描文档: {files[0]} ---\n")

for i in range(0, len(lines), chunk_size - overlap):
    chunk = "".join(lines[i: i + chunk_size])
    if not chunk.strip():
        continue

    # 打印进度提示
    print(f"[进度] 正在扫描第 {i} 至 {i + chunk_size} 行...")

    result = analyze_chunk(chunk)

    # 如果 AI 发现了转喻现象（即输出中包含“场景序号”），则打印
    if "场景序号" in result:
        print("\n" + "=" * 50)
        print(result)
        print("=" * 50 + "\n")

print("\n全案分析完毕。")
import os
import re
import pandas as pd
from openai import OpenAI

# ================= 配置区域 =================
API_KEY = "sk-dc48e456ec1a46a18abec7d2f1e57a18"
INPUT_DIR = r"C:\Users\smy\PycharmProjects\PythonProject\260112\cleaned_txt"
# 结果保存路径（默认保存在输入文件夹的同级目录下）
OUTPUT_MD = os.path.join(os.path.dirname(INPUT_DIR), "分析结果_可读版.md")
OUTPUT_EXCEL = os.path.join(os.path.dirname(INPUT_DIR), "分析结果_统计版.xlsx")
# ===========================================

client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def analyze_chunk(chunk_text):
    system_prompt = """你是一个认知语言学专家。请识别对话中的“状态代行事”现象。
    输出必须严格包含：【场景原文】、【认知分析】、【分类结果】这三个标签。
    分类结果只能是：转喻共鸣、字面共鸣、非共鸣。"""

    try:
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk_text}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"错误: {e}"


def parse_result_to_dict(result_text):
    """将 AI 的文本输出解析为字典，方便存入 Excel"""
    try:
        # 使用正则提取分类结果
        res = re.search(r"分类结果[:：]\s*(.*)", result_text)
        category = res.group(1).strip() if res else "未知"
        return {"原始分析": result_text, "最终分类": category}
    except:
        return {"原始分析": result_text, "最终分类": "解析失败"}


def main():
    all_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.txt')])
    if not all_files: return

    target_file = os.path.join(INPUT_DIR, all_files[0])
    with open(target_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    all_results_for_excel = []

    # 准备写入 Markdown
    with open(OUTPUT_MD, 'w', encoding='utf-8') as md_file:
        md_file.write(f"# 状态代行事转喻分析报告\n\n分析文档: {all_files[0]}\n\n---\n")

        chunk_size = 10
        step = 8
        for i in range(0, len(lines), step):
            chunk = "".join(lines[i: i + chunk_size])
            if not chunk.strip(): continue

            print(f"进度: 正在处理第 {i} 行...")
            result = analyze_chunk(chunk)

            # 如果 AI 发现了有效场景
            if "场景" in result or "分析" in result:
                # 1. 写入 Markdown
                md_file.write(f"### 场景 ID: {i}\n")
                md_file.write(result + "\n\n---\n")

                # 2. 存入 Excel 数据列表
                parsed = parse_result_to_dict(result)
                all_results_for_excel.append({
                    "行号": i,
                    "完整内容": parsed["原始分析"],
                    "分类": parsed["最终分类"]
                })

    # 3. 生成 Excel
    if all_results_for_excel:
        df = pd.DataFrame(all_results_for_excel)
        df.to_excel(OUTPUT_EXCEL, index=False)
        print(f"\n>>> 分析完成！")
        print(f"可读版报告（Markdown）: {OUTPUT_MD}")
        print(f"统计版表格（Excel）: {OUTPUT_EXCEL}")


if __name__ == "__main__":
    main()