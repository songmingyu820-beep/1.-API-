import jieba
import dashscope
from dashscope import Generation
import os

# 设置API密钥（需要先申请）
dashscope.api_key = "sk-dc48e456ec1a46a18abec7d2f1e57a18"  # 记得替换成你的真实API密钥


def simple_metonymy_analysis(text_file="me.txt"):
    """简化版转喻分析 - 专门针对你的me.txt文件"""

    # 检查文件是否存在
    if not os.path.exists(text_file):
        print(f"错误：找不到文件 {text_file}")
        print("请确保me.txt文件与Python脚本在同一目录下")
        return

    try:
        # 读取文本文件
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()

        # 检查文件是否为空
        if not text.strip():
            print("警告：me.txt文件是空的")
            return

        print("=" * 50)
        print("转喻分析开始")
        print("=" * 50)
        print(f"分析文本预览: {text[:200]}...")
        print(f"文本总长度: {len(text)} 字符")
        print("=" * 50)

        # 使用千问模型进行分析
        prompt = f"""
        请分析以下中文文本中的转喻现象。转喻是一种修辞手法，用一个相关的事物代替另一个事物。

        文本内容：
        「{text}」

        请完成以下分析任务：
        1. 找出文本中所有的转喻用法
        2. 说明每个转喻属于哪种类型（如：部分代整体、容器代内容、地点代机构等）
        3. 分析转喻的修辞效果
        4. 评价转喻使用的恰当性

        请用清晰的中文回答。
        """

        print("正在调用千问模型进行分析...")

        response = Generation.call(
            model='qwen-plus',
            prompt=prompt,
            max_tokens=1000,
            temperature=0.3
        )

        if response.status_code == 200:
            print("\n分析结果:")
            print("=" * 50)
            print(response.output.text)
            print("=" * 50)

            # 保存分析结果到文件
            with open("分析结果.txt", "w", encoding="utf-8") as result_file:
                result_file.write("转喻分析结果\n")
                result_file.write("=" * 50 + "\n")
                result_file.write(f"分析文件: {text_file}\n")
                result_file.write(f"文本长度: {len(text)} 字符\n\n")
                result_file.write("分析结果:\n")
                result_file.write(response.output.text)

            print("分析结果已保存到 '分析结果.txt' 文件")

        else:
            print(f"分析失败: {response.message}")

    except Exception as e:
        print(f"读取或分析过程中出现错误: {e}")


# 使用示例
if __name__ == "__main__":
    # 直接调用，默认读取me.txt文件
    simple_metonymy_analysis()
