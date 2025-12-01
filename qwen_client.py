import dashscope
from config import Config


class QwenClient:
    def __init__(self):
        # 验证配置
        Config.validate_config()

        # 设置API密钥
        dashscope.api_key = Config.DASHSCOPE_API_KEY
        self.model_name = Config.MODEL_NAME
        print("✅ Qwen客户端初始化成功！")

    def chat(self, prompt, history=None):
        """
        与Qwen-plus对话
        """
        if history is None:
            history = []

        try:
            messages = history + [{"role": "user", "content": prompt}]

            response = dashscope.Generation.call(
                model=self.model_name,
                messages=messages,
                result_format='message',
                max_tokens=Config.MAX_TOKENS,
                temperature=Config.TEMPERATURE
            )

            if response.status_code == 200:
                return response.output.choices[0]['message']['content']
            else:
                return f"❌ 请求失败: {response.code} - {response.message}"

        except Exception as e:
            return f"❌ 发生错误: {str(e)}"

    def stream_chat(self, prompt, history=None):
        """
        流式输出（逐字显示）
        """
        if history is None:
            history = []

        try:
            messages = history + [{"role": "user", "content": prompt}]

            responses = dashscope.Generation.call(
                model=self.model_name,
                messages=messages,
                result_format='message',
                stream=True,
                max_tokens=Config.MAX_TOKENS,
                temperature=Config.TEMPERATURE
            )

            print("Qwen: ", end="", flush=True)
            full_response = ""

            for response in responses:
                if response.status_code == 200:
                    chunk = response.output.choices[0]['message']['content']
                    print(chunk, end="", flush=True)
                    full_response += chunk
                else:
                    print(f"\n❌ 流式请求失败: {response.code}")
                    break

            print()  # 换行
            return full_response

        except Exception as e:
            error_msg = f"❌ 流式对话错误: {str(e)}"
            print(error_msg)
            return error_msg


def main():
    try:
        client = QwenClient()

        print("=== 🤖 Qwen-plus 智能助手 ===")
        print("输入内容开始对话，输入以下命令:")
        print("  'quit' - 退出程序")
        print("  'stream' - 切换流式输出")
        print("  'clear' - 清空对话历史")
        print("-" * 40)

        conversation_history = []
        use_stream = False  # 默认非流式

        while True:
            user_input = input("\n👤 你: ").strip()

            if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                print("再见！👋")
                break

            elif user_input.lower() == 'stream':
                use_stream = not use_stream
                mode = "流式输出" if use_stream else "一次性输出"
                print(f"已切换为{mode}模式")
                continue

            elif user_input.lower() == 'clear':
                conversation_history.clear()
                print("🗑️ 对话历史已清空")
                continue

            if not user_input:
                continue

            print("🤖 Qwen: ", end="", flush=True)

            if use_stream:
                response = client.stream_chat(user_input, conversation_history)
            else:
                response = client.chat(user_input, conversation_history)
                print(response)

            # 更新对话历史
            conversation_history.extend([
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": response}
            ])

            # 限制历史长度
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]

    except ValueError as e:
        print(f"配置错误: {e}")
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"意外错误: {e}")


if __name__ == "__main__":
    main()