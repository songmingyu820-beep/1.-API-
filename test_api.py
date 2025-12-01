from qwen_client import QwenClient


def test_connection():
    """测试API连接"""
    try:
        client = QwenClient()
        print("✅ API连接测试成功！")

        # 测试简单对话
        test_prompt = "请用一句话介绍你自己"
        print(f"测试问题: {test_prompt}")

        response = client.chat(test_prompt)
        print(f"模型回复: {response}")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


if __name__ == "__main__":
    test_connection()