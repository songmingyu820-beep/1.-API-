import os
import re
import pandas as pd
from openai import OpenAI

# ================= 配置区域 =================
API_KEY = "sk-dc48e456ec1a46a18abec7d2f1e57a18"
INPUT_DIR = r"C:\Users\smy\Desktop\cleaned_txt"
# ===========================================

client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def parse_result_to_dict(result_text):
    """
    解析 AI 输出。
    寻找指令中定义的“最终分类：[...]”
    """
    try:
        # 匹配指令要求的“最终分类：[...]”格式
        # 兼容中文冒号、英文冒号、加粗星号和空格
        pattern = r"最终分类[:：]\s*\[?(.*?)\]?$"
        # 如果最后一行找不到，就在全文找
        res = re.findall(pattern, result_text, re.MULTILINE)
        if res:
            category = res[-1].strip().replace("*", "").replace("[", "").replace("]", "")
            return {"原始分析": result_text, "分类": category}

        # 备选方案：如果格式稍微走样，寻找关键词
        for cat in ["转喻共鸣", "字面共鸣", "非共鸣回复"]:
            if cat in result_text.split("最终分类")[-1]:
                return {"原始分析": result_text, "分类": cat}

        return {"原始分析": result_text, "分类": "未知"}
    except:
        return {"原始分析": result_text, "分类": "解析异常"}


def analyze_chunk(chunk_text, scene_id):
    """
    使用你指定的指令进行分析
    """
    system_prompt = f"""你是一个认知语言学和对话分析专家。你的任务是分析语料中“状态代行事（State-for-Action）”的转喻共鸣。

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
场景序号：[{scene_id}]
对话原文：
人名A: [状态陈述句]
人名B: [回应句]
认知分析：[简述为何这是状态代行事，以及孩子是否完成了动作跳跃]
最终分类：[转喻共鸣/字面共鸣/非共鸣回复]"""

    try:
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"请分析以下对话块：\n\n{chunk_text}"}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"API 错误: {e}"


def main():
    # 1. 获取并排序文件
    all_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.txt')])

    if len(all_files) < 3:
        print(f"文件夹中只有 {len(all_files)} 个文件，无法找到第3个。")
        return

    target_filename = all_files[2]  # 选取第3个文件
    target_path = os.path.join(INPUT_DIR, target_filename)

    # 2. 设置输出路径
    output_base = target_filename.replace('.txt', '')
    output_md = os.path.join(os.path.dirname(INPUT_DIR), f"分析报告_{output_base}.md")
    output_excel = os.path.join(os.path.dirname(INPUT_DIR), f"分析结果_{output_base}.xlsx")

    print(f"--- 正在开始分析第3个文件: {target_filename} ---")

    with open(target_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    final_data_list = []

    # 3. 写入 Markdown 并处理数据
    with open(output_md, 'w', encoding='utf-8') as md_file:
        md_file.write(f"# 状态代行事转喻分析报告\n分析文件: {target_filename}\n\n---\n")

        chunk_size = 10
        step = 8  # 保持重叠以防切断对话

        for i in range(0, len(lines), step):
            chunk = "".join(lines[i: i + chunk_size])
            if not chunk.strip(): continue

            print(f"处理中: 第 {i} 行...")
            result = analyze_chunk(chunk, i)

            # 只有 AI 识别出“对话原文”才记录（即过滤掉废话）
            if "对话原文" in result:
                # 记录到 Markdown
                md_file.write(result + "\n\n---\n")

                # 解析并记录到 Excel 数据列表
                parsed = parse_result_to_dict(result)
                final_data_list.append({
                    "行号": i,
                    "完整分析": parsed["原始分析"],
                    "最终分类": parsed["分类"]
                })

    # 4. 生成 Excel
    if final_data_list:
        df = pd.DataFrame(final_data_list)
        df.to_excel(output_excel, index=False)
        print(f"\n>>> 分析完毕！")
        print(f"结果预览（Excel）已保存: {output_excel}")
        print(f"详细报告（Markdown）已保存: {output_md}")
    else:
        print("\n未发现符合条件的“状态代行事”场景。")


if __name__ == "__main__":
    main()