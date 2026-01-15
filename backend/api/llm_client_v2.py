"""
优化版 LLM 客户端 - 添加缓存和性能监控
"""

import os
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import sys

# 导入缓存和监控工具
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.cache import cache_llm_response, monitor_performance


class BaseLLMClient(ABC):
    """LLM客户端基类"""
    
    @abstractmethod
    @cache_llm_response
    @monitor_performance("llm_chat")
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """聊天接口"""
        pass
    
    @abstractmethod  
    @cache_llm_response
    @monitor_performance("llm_generate")
    def generate(self, prompt: str, **kwargs) -> str:
        """生成接口"""
        pass


class DeepSeekClient(BaseLLMClient):
    """DeepSeek API客户端（带缓存）"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key not found")
        
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com"
            )
        except ImportError:
            raise ImportError("请安装openai库: pip install openai")
    
    @cache_llm_response
    @monitor_performance("deepseek_chat")
    def chat(self, messages: List[Dict[str, str]], 
             model: str = "deepseek-chat",
             temperature: float = 0.7,
             max_tokens: int = 1000,
             **kwargs) -> str:
        """聊天接口（带缓存）"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"DeepSeek调用失败: {str(e)}"
    
    @cache_llm_response
    @monitor_performance("deepseek_generate")
    def generate(self, prompt: str, **kwargs) -> str:
        """生成接口（带缓存）"""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)


class QwenClient(BaseLLMClient):
    """通义千问API客户端（带缓存）"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DashScope API key not found")
        
        try:
            import dashscope
            dashscope.api_key = self.api_key
            self.dashscope = dashscope
        except ImportError:
            raise ImportError("请安装dashscope库: pip install dashscope")
    
    @cache_llm_response
    @monitor_performance("qwen_chat")
    def chat(self, messages: List[Dict[str, str]], 
             model: str = "qwen-plus",
             **kwargs) -> str:
        """聊天接口（带缓存）"""
        try:
            from dashscope import Generation
            response = Generation.call(
                model=model,
                messages=messages,
                result_format='message',
                **kwargs
            )
            if response.status_code == 200:
                return response.output.choices[0].message.content
            else:
                return f"Qwen调用失败: {response.message}"
        except Exception as e:
            return f"Qwen调用失败: {str(e)}"
    
    @cache_llm_response
    @monitor_performance("qwen_generate")
    def generate(self, prompt: str, **kwargs) -> str:
        """生成接口（带缓存）"""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)


class OpenAIClient(BaseLLMClient):
    """OpenAI API客户端（带缓存）"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found")
        
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        except ImportError:
            raise ImportError("请安装openai库: pip install openai")
    
    @cache_llm_response
    @monitor_performance("openai_chat")
    def chat(self, messages: List[Dict[str, str]], 
             model: str = "gpt-3.5-turbo",
             temperature: float = 0.7,
             max_tokens: int = 1000,
             **kwargs) -> str:
        """聊天接口（带缓存）"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"OpenAI调用失败: {str(e)}"
    
    @cache_llm_response
    @monitor_performance("openai_generate")
    def generate(self, prompt: str, **kwargs) -> str:
        """生成接口（带缓存）"""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)


class LLMFactory:
    """LLM客户端工厂"""
    
    @staticmethod
    def create_client(provider: str = "auto") -> Optional[BaseLLMClient]:
        """创建LLM客户端（自动选择或指定）"""
        
        if provider == "auto":
            # 自动检测可用的API
            if os.getenv("DEEPSEEK_API_KEY"):
                provider = "deepseek"
            elif os.getenv("DASHSCOPE_API_KEY"):
                provider = "qwen"
            elif os.getenv("OPENAI_API_KEY"):
                provider = "openai"
            else:
                print("警告: 未找到任何LLM API密钥")
                return None
        
        try:
            if provider == "deepseek":
                return DeepSeekClient()
            elif provider == "qwen":
                return QwenClient()
            elif provider == "openai":
                return OpenAIClient()
            else:
                raise ValueError(f"不支持的provider: {provider}")
        except Exception as e:
            print(f"LLM客户端创建失败: {e}")
            return None


class LLMService:
    """LLM业务服务层（带缓存优化）"""
    
    def __init__(self, client: BaseLLMClient):
        self.client = client
    
    @cache_llm_response
    @monitor_performance("generate_recommendation_reason")
    def generate_recommendation_reason(self, user_info: Dict[str, Any], 
                                      item_info: Dict[str, Any]) -> str:
        """生成推荐理由（带缓存）"""
        prompt = f"""为用户推荐商家，生成简短推荐理由（50字以内）。

用户信息：{user_info}
商家信息：{item_info}

请生成一句话推荐理由："""

        return self.client.generate(prompt, max_tokens=100, temperature=0.7)
    
    @cache_llm_response
    @monitor_performance("parse_search_intent")
    def parse_search_intent(self, query: str) -> Dict[str, Any]:
        """解析搜索意图（带缓存）"""
        prompt = f"""解析用户搜索意图，提取关键信息（JSON格式）：

查询：{query}

返回格式：
{{"cuisine": "菜系", "location": "位置", "price_range": "价格", "requirements": ["需求列表"]}}
"""

        response = self.client.generate(prompt, max_tokens=200, temperature=0.3)
        
        try:
            import json
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return {"raw_query": query}
    
    @cache_llm_response
    @monitor_performance("summarize_comments")  
    def summarize_comments(self, comments: List[str]) -> str:
        """评论摘要（带缓存）"""
        comments_text = "\n".join([f"- {c}" for c in comments[:20]])
        
        prompt = f"""总结以下用户评论，提取核心观点（150字以内）：

{comments_text}

请生成摘要："""

        return self.client.generate(prompt, max_tokens=300, temperature=0.5)
    
    @cache_llm_response
    @monitor_performance("answer_question")
    def answer_question(self, question: str, context: Any = None) -> str:
        """回答问题（带缓存）"""
        context_str = f"\n\n背景信息：\n{context}" if context else ""
        
        prompt = f"""请简洁专业地回答用户问题（200字以内）：

问题：{question}{context_str}

回答："""

        return self.client.generate(prompt, max_tokens=400, temperature=0.7)
    
    @cache_llm_response
    @monitor_performance("generate_operation_advice")
    def generate_operation_advice(self, merchant_data: Dict[str, Any]) -> str:
        """生成运营建议（带缓存）"""
        prompt = f"""作为外卖运营顾问，请为商家提供运营建议：

商家数据：
{merchant_data}

请从以下角度提供3-5条具体建议（每条50字以内）：
1. 菜品优化
2. 定价策略
3. 营销推广
4. 服务提升

建议："""

        return self.client.generate(prompt, max_tokens=600, temperature=0.7)


if __name__ == "__main__":
    print("="*60)
    print("优化版LLM客户端测试")
    print("="*60)
    
    # 创建客户端
    client = LLMFactory.create_client()
    
    if client:
        service = LLMService(client)
        
        # 测试缓存效果
        print("\n第一次调用（无缓存）：")
        import time
        start = time.time()
        result1 = service.generate_recommendation_reason(
            {"user_id": "123"},
            {"name": "川菜馆", "score": 4.5}
        )
        print(f"耗时: {time.time() - start:.2f}秒")
        print(f"结果: {result1}")
        
        print("\n第二次调用（命中缓存）：")
        start = time.time()
        result2 = service.generate_recommendation_reason(
            {"user_id": "123"},
            {"name": "川菜馆", "score": 4.5}
        )
        print(f"耗时: {time.time() - start:.2f}秒")
        print(f"结果: {result2}")
        
        # 查看性能统计
        print("\n性能统计：")
        from utils.cache import perf_monitor, llm_cache
        
        stats = perf_monitor.get_stats()
        for op, metrics in stats.items():
            print(f"\n{op}:")
            for key, val in metrics.items():
                if 'time' in key:
                    print(f"  {key}: {val:.4f}秒")
                else:
                    print(f"  {key}: {val}")
        
        print("\n缓存统计：")
        cache_stats = llm_cache.get_stats()
        for key, val in cache_stats.items():
            print(f"  {key}: {val}")
    else:
        print("未找到可用的LLM API密钥")
