import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class Config:
    # API配置
    DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
    MODEL_NAME = "qwen-plus"

    # 请求配置
    MAX_TOKENS = 1500
    TEMPERATURE = 0.7

    @classmethod
    def validate_config(cls):
        """检查配置是否完整"""
        if not cls.DASHSCOPE_API_KEY:
            raise ValueError("请检查.env文件中的DASHSCOPE_API_KEY配置")
        return True