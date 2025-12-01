import os

# 设置使用国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import pipeline
import warnings

warnings.filterwarnings('ignore')  # 忽略警告信息


def chinese_sentiment_analysis():
    """专门针对中文的情感分析"""

    try:
        # 读取你的文本文件
        with open("me.txt", "r", encoding="utf-8") as f:
            text = f.read().strip()

        print(f"📖 分析文本: {text}")
        print("=" * 60)

        # 使用专门的中文情感分析模型
        print("🔧 加载中文情感分析模型...")

        # 明确指定中文情感分析模型
        classifier = pipeline(
            "sentiment-analysis",
            model="uer/roberta-base-finetuned-jd-full-chinese",  # 中文情感分析模型
            tokenizer="uer/roberta-base-finetuned-jd-full-chinese"
        )

        print("🤖 分析中...")
        result = classifier(text)[0]

        print("✅ 分析结果:")
        print(f"情感倾向: {result['label']}")
        print(f"置信度: {result['score']:.3f}")

        # 中文标签解释
        if result['label'] == 'positive':
            print("情感解读: 积极正面 😊")
        elif result['label'] == 'negative':
            print("情感解读: 消极负面 😔")
        else:
            print("情感解读: 中性 😐")

    except Exception as e:
        print(f"❌ 分析失败: {e}")
        # 如果transformers失败，使用备选方案
        use_fallback_solution()


def use_fallback_solution():
    """备选方案：使用snownlp"""
    print("\n🔄 尝试使用备选方案...")

    try:
        from snownlp import SnowNLP

        with open("me.txt", "r", encoding="utf-8") as f:
            text = f.read().strip()

        s = SnowNLP(text)
        score = s.sentiments

        if score > 0.6:
            sentiment = "积极"
            emoji = "😊"
        elif score < 0.4:
            sentiment = "消极"
            emoji = "😔"
        else:
            sentiment = "中性"
            emoji = "😐"

        print(f"{emoji} 情感分析结果 (SnownLP):")
        print(f"情感得分: {score:.3f}")
        print(f"情感倾向: {sentiment}")
        print(f"关键词: {', '.join(s.keywords(3))}")

    except Exception as e:
        print(f"❌ 备选方案也失败: {e}")
        use_simple_analysis()


def use_simple_analysis():
    """最简单的规则分析"""
    print("\n📋 使用简单规则分析...")

    try:
        with open("me.txt", "r", encoding="utf-8") as f:
            text = f.read().strip()

        positive_words = ['好', '开心', '喜欢', '优秀', '美好', '高兴', '满意', '棒', '温暖', '幸福']
        negative_words = ['坏', '伤心', '讨厌', '糟糕', '痛苦', '失望', '不满', '差', '冷', '寂寞']

        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)

        if positive_count > negative_count:
            print("情感倾向: 积极 😊")
        elif negative_count > positive_count:
            print("情感倾向: 消极 😔")
        else:
            print("情感倾向: 中性 😐")

        print(f"积极词数量: {positive_count}")
        print(f"消极词数量: {negative_count}")

    except Exception as e:
        print(f"❌ 所有方法都失败: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("中文情感分析程序")
    print("=" * 60)
    chinese_sentiment_analysis()
    import matplotlib.pyplot as plt


    def plot_sentiment_results():
        """用图表显示分析结果"""

        # 你的数据
        stars = 4
        confidence = 0.422
        max_stars = 5

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # 左图：星级评分
        ax1.barh(['星级评分'], [stars], color='gold', alpha=0.7)
        ax1.set_xlim(0, max_stars)
        ax1.set_xlabel('星数')
        ax1.set_title('情感星级评分')
        ax1.text(stars / 2, 0, f'{stars}星', ha='center', va='center', color='black', fontsize=12)

        # 右图：置信度
        ax2.barh(['置信度'], [confidence * 100], color='lightblue', alpha=0.7)
        ax2.set_xlim(0, 100)
        ax2.set_xlabel('百分比 (%)')
        ax2.set_title('分析置信度')
        ax2.text(confidence * 50, 0, f'{confidence * 100:.1f}%', ha='center', va='center', color='black', fontsize=12)

        # 添加总体评价
        plt.figtext(0.5, 0.01,
                    f'情感倾向: 中性 | 可靠性: {"较低" if confidence < 0.5 else "一般"}',
                    ha='center', fontsize=12, style='italic')

        plt.tight_layout()
        plt.show()


    # 运行
    plot_sentiment_results()


    def beautiful_text_display():
        """用字符画创建漂亮的文本显示"""

        stars = 4
        confidence = 0.422

        print("🎨" + "=" * 50 + "🎨")
        print("          情感分析可视化报告")
        print("🎨" + "=" * 50 + "🎨")
        print()

        # 星级显示
        print("🌟 情感强度评分")
        print("   " + "★" * stars + "☆" * (5 - stars))
        print(f"   {stars}星 (满分5星)")
        print()

        # 置信度进度条（美化版）
        print("📊 分析置信度")
        bar_length = 30
        filled = int(confidence * bar_length)

        # 用不同字符创建渐变效果
        bar = "█" * (filled - 2) + "▓" * 1 + "░" * (bar_length - filled)
        print(f"   [{bar}]")
        print(f"   {confidence * 100:.1f}% 把握程度")
        print()

        # 可靠性评价
        if confidence > 0.7:
            reliability_emoji = "✅"
            reliability_text = "高可靠性"
        elif confidence > 0.5:
            reliability_emoji = "⚠️"
            reliability_text = "一般可靠性"
        else:
            reliability_emoji = "❓"
            reliability_text = "较低可靠性"

        print(f"{reliability_emoji} 可靠性评价: {reliability_text}")
        print()

        # 使用建议
        print("💡 使用建议:")
        if stars >= 4 and confidence > 0.5:
            print("   情感分析结果较为可靠，可以采纳")
        elif confidence < 0.4:
            print("   情感复杂，建议结合人工判断")
        else:
            print("   分析结果可供参考")

        print()
        print("🎨" + "=" * 50 + "🎨")


    # 运行文本美化版
    beautiful_text_display()