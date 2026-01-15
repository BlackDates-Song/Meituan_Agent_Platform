"""
大模型API调用封装层
支持 DeepSeek, OpenAI, Qwen 等多种大模型API
"""

import os
import sys
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union
import logging
from pathlib import Path
import json

# 导入配置文件
try:
    from backend.config import (
        DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL,
        DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL,
        OPENAI_API_KEY, OPENAI_BASE_URL
    )
except ImportError:
    # 如果导入失败，使用默认值
    DEEPSEEK_API_KEY = "sk-58c039d11c8c463d9111bf6ccf11cfa1"
    DEEPSEEK_BASE_URL = "https://api.deepseek.com"
    DASHSCOPE_API_KEY = "sk-763dc0c810da46e7809ae61a1223e880"
    DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    OPENAI_API_KEY = ""
    OPENAI_BASE_URL = "https://api.openai.com/v1"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """大模型客户端基类"""
    
    def __init__(self, api_key: Optional[str] = None, 
                 base_url: Optional[str] = None,
                 model: Optional[str] = None):
        """
        初始化LLM客户端
        
        Args:
            api_key: API密钥
            base_url: API基础URL
            model: 模型名称
        """
        self.api_key = api_key or self._get_api_key_from_env()
        self.base_url = base_url
        self.model = model or self._get_default_model()
        
    @abstractmethod
    def _get_api_key_from_env(self) -> str:
        """从环境变量获取API密钥"""
        pass
        
    @abstractmethod
    def _get_default_model(self) -> str:
        """获取默认模型名称"""
        pass
        
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        聊天补全
        
        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            **kwargs: 其他参数（temperature, max_tokens等）
            
        Returns:
            模型回复内容
        """
        pass
        
    def generate(self, prompt: str, **kwargs) -> str:
        """
        简单生成（将prompt包装为chat消息）
        
        Args:
            prompt: 提示词
            **kwargs: 其他参数
            
        Returns:
            模型回复
        """
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)


class DeepSeekClient(BaseLLMClient):
    """DeepSeek API 客户端"""
    
    def __init__(self, api_key: Optional[str] = None, 
                 model: str = "deepseek-chat"):
        """
        初始化 DeepSeek 客户端
        
        Args:
            api_key: DeepSeek API密钥
            model: 模型名称（默认 deepseek-chat）
        """
        super().__init__(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            model=model
        )
        
    def _get_api_key_from_env(self) -> str:
        """从配置文件或环境变量获取DeepSeek API密钥"""
        # 优先使用配置文件
        if DEEPSEEK_API_KEY:
            return DEEPSEEK_API_KEY
        # 其次使用环境变量
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            logger.warning("未设置 DEEPSEEK_API_KEY（请在 backend/config.py 或环境变量中设置）")
        return api_key
        
    def _get_default_model(self) -> str:
        """获取默认模型"""
        return "deepseek-chat"
        
    def chat(self, messages: List[Dict[str, str]], 
             temperature: float = 0.7,
             max_tokens: int = 2000,
             stream: bool = False,
             **kwargs) -> str:
        """
        DeepSeek 聊天补全
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大生成token数
            stream: 是否流式输出
            **kwargs: 其他参数
            
        Returns:
            模型回复
        """
        try:
            from openai import OpenAI
            
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                **kwargs
            )
            
            if stream:
                # 流式输出
                full_response = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        print(content, end="", flush=True)
                print()  # 换行
                return full_response
            else:
                return response.choices[0].message.content
                
        except ImportError:
            logger.error("请安装 openai 库: pip install openai")
            raise
        except Exception as e:
            logger.error(f"DeepSeek API 调用失败: {str(e)}")
            raise


class QwenClient(BaseLLMClient):
    """阿里通义千问 API 客户端"""
    
    def __init__(self, api_key: Optional[str] = None,
                 model: str = "qwen-turbo"):
        """
        初始化 Qwen 客户端
        
        Args:
            api_key: DashScope API密钥
            model: 模型名称
        """
        super().__init__(
            api_key=api_key,
            model=model
        )
        
    def _get_api_key_from_env(self) -> str:
        """从配置文件或环境变量获取Qwen API密钥"""
        # 优先使用配置文件
        if DASHSCOPE_API_KEY:
            return DASHSCOPE_API_KEY
        # 其次使用环境变量
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            logger.warning("未设置 DASHSCOPE_API_KEY（请在 backend/config.py 或环境变量中设置）")
        return api_key
        
    def _get_default_model(self) -> str:
        """获取默认模型"""
        return "qwen-turbo"
        
    def chat(self, messages: List[Dict[str, str]],
             temperature: float = 0.7,
             max_tokens: int = 2000,
             stream: bool = False,
             **kwargs) -> str:
        """
        Qwen 聊天补全
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大生成token数
            stream: 是否流式输出
            **kwargs: 其他参数
            
        Returns:
            模型回复
        """
        try:
            import dashscope
            from dashscope import Generation
            
            dashscope.api_key = self.api_key
            
            response = Generation.call(
                model=self.model,
                messages=messages,
                result_format='message',
                stream=stream,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            if stream:
                full_response = ""
                for chunk in response:
                    if chunk.output and chunk.output.choices:
                        content = chunk.output.choices[0].message.content
                        full_response += content
                        print(content, end="", flush=True)
                print()
                return full_response
            else:
                return response.output.choices[0].message.content
                
        except ImportError:
            logger.error("请安装 dashscope 库: pip install dashscope")
            raise
        except Exception as e:
            logger.error(f"Qwen API 调用失败: {str(e)}")
            raise


class OpenAIClient(BaseLLMClient):
    """OpenAI API 客户端"""
    
    def __init__(self, api_key: Optional[str] = None,
                 model: str = "gpt-3.5-turbo"):
        """
        初始化 OpenAI 客户端
        
        Args:
            api_key: OpenAI API密钥
            model: 模型名称
        """
        super().__init__(
            api_key=api_key,
            base_url="https://api.openai.com/v1",
            model=model
        )
        
    def _get_api_key_from_env(self) -> str:
        """从配置文件或环境变量获取OpenAI API密钥"""
        # 优先使用配置文件
        if OPENAI_API_KEY:
            return OPENAI_API_KEY
        # 其次使用环境变量
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("未设置 OPENAI_API_KEY（请在 backend/config.py 或环境变量中设置）")
        return api_key
        
    def _get_default_model(self) -> str:
        """获取默认模型"""
        return "gpt-3.5-turbo"
        
    def chat(self, messages: List[Dict[str, str]],
             temperature: float = 0.7,
             max_tokens: int = 2000,
             stream: bool = False,
             **kwargs) -> str:
        """
        OpenAI 聊天补全
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大生成token数
            stream: 是否流式输出
            **kwargs: 其他参数
            
        Returns:
            模型回复
        """
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=self.api_key)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                **kwargs
            )
            
            if stream:
                full_response = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        print(content, end="", flush=True)
                print()
                return full_response
            else:
                return response.choices[0].message.content
                
        except ImportError:
            logger.error("请安装 openai 库: pip install openai")
            raise
        except Exception as e:
            logger.error(f"OpenAI API 调用失败: {str(e)}")
            raise


class LLMFactory:
    """LLM工厂类 - 统一创建不同的LLM客户端"""
    
    @staticmethod
    def create_client(provider: str = "deepseek", **kwargs) -> BaseLLMClient:
        """
        创建LLM客户端
        
        Args:
            provider: 提供商 ('deepseek', 'qwen', 'openai')
            **kwargs: 传递给客户端的参数
            
        Returns:
            LLM客户端实例
        """
        providers = {
            'deepseek': DeepSeekClient,
            'qwen': QwenClient,
            'openai': OpenAIClient,
        }
        
        if provider.lower() not in providers:
            raise ValueError(f"不支持的提供商: {provider}. 支持的: {list(providers.keys())}")
            
        client_class = providers[provider.lower()]
        return client_class(**kwargs)


class LLMService:
    """LLM服务类 - 封装常用的业务功能"""
    
    def __init__(self, client: BaseLLMClient):
        """
        初始化LLM服务
        
        Args:
            client: LLM客户端
        """
        self.client = client
        
    def generate_recommendation_reason(self, user_profile: Dict,
                                      item_info: Dict) -> str:
        """
        生成推荐理由
        
        Args:
            user_profile: 用户画像
            item_info: 物品信息
            
        Returns:
            推荐理由文本
        """
        prompt = f"""
你是一个外卖推荐助手。请根据用户画像和商家/菜品信息，生成简短的推荐理由（1-2句话）。

用户画像：
{json.dumps(user_profile, ensure_ascii=False, indent=2)}

商家/菜品信息：
{json.dumps(item_info, ensure_ascii=False, indent=2)}

请生成推荐理由：
"""
        
        return self.client.generate(prompt, temperature=0.7, max_tokens=200)
        
    def summarize_comments(self, comments: List[str]) -> str:
        """
        评论摘要生成
        
        Args:
            comments: 评论列表
            
        Returns:
            摘要文本
        """
        prompt = f"""
请对以下用户评论进行摘要，提取主要的优点和缺点（用简短的关键词）。

评论：
{chr(10).join(f"{i+1}. {comment}" for i, comment in enumerate(comments[:20]))}

请生成摘要：
"""
        
        return self.client.generate(prompt, temperature=0.5, max_tokens=300)
        
    def answer_question(self, question: str, context: str) -> str:
        """
        基于上下文回答问题
        
        Args:
            question: 用户问题
            context: 上下文信息
            
        Returns:
            回答
        """
        messages = [
            {
                "role": "system",
                "content": "你是一个智能外卖助手，帮助用户解答关于商家和菜品的问题。请基于提供的信息回答问题。"
            },
            {
                "role": "user",
                "content": f"上下文信息：\n{context}\n\n问题：{question}"
            }
        ]
        
        return self.client.chat(messages, temperature=0.7, max_tokens=500)
        
    def generate_operation_advice(self, merchant_data: Dict) -> str:
        """
        为商家生成运营建议
        
        Args:
            merchant_data: 商家数据
            
        Returns:
            运营建议
        """
        prompt = f"""
你是一个外卖平台的运营顾问。请根据商家的运营数据，提供具体可行的改进建议。

商家数据：
{json.dumps(merchant_data, ensure_ascii=False, indent=2)}

请生成运营建议（包括问题诊断和具体措施）：
"""
        
        return self.client.generate(prompt, temperature=0.7, max_tokens=1000)
        
    def sentiment_analysis(self, text: str) -> str:
        """
        情感分析
        
        Args:
            text: 待分析文本
            
        Returns:
            情感分类（正面/负面/中性）和置信度
        """
        prompt = f"""
请对以下文本进行情感分析，判断是正面、负面还是中性，并给出置信度。

文本：{text}

请以JSON格式输出：
{{"sentiment": "正面/负面/中性", "confidence": 0.95}}
"""
        
        return self.client.generate(prompt, temperature=0.3, max_tokens=100)


def main():
    """主函数 - 示例用法"""
    logger.info("=" * 60)
    logger.info("LLM API 封装层测试")
    logger.info("=" * 60)
    
    # 创建DeepSeek客户端
    try:
        client = LLMFactory.create_client('deepseek')
        service = LLMService(client)
        
        # 测试简单对话
        logger.info("\n测试1: 简单对话")
        response = client.generate("你好，请介绍一下你自己。")
        logger.info(f"回复: {response}")
        
        # 测试推荐理由生成
        logger.info("\n测试2: 生成推荐理由")
        user_profile = {
            "avg_price": "36-49元",
            "favorite_period": "午餐",
            "order_count": 15
        }
        item_info = {
            "name": "川菜馆",
            "score": 4.8,
            "popular_dishes": ["水煮鱼", "回锅肉"]
        }
        
        reason = service.generate_recommendation_reason(user_profile, item_info)
        logger.info(f"推荐理由: {reason}")
        
        logger.info("\n✓ LLM API 测试完成")
        
    except Exception as e:
        logger.error(f"测试失败: {str(e)}")
        logger.info("\n提示: 请确保设置了相应的API密钥环境变量")
        logger.info("  - DEEPSEEK_API_KEY (DeepSeek)")
        logger.info("  - DASHSCOPE_API_KEY (Qwen)")
        logger.info("  - OPENAI_API_KEY (OpenAI)")


if __name__ == "__main__":
    main()
