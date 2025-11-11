#!/usr/bin/env python3
"""
LLM客户端适配器 - 支持多提供商API调用
"""

import os
import time
from typing import Dict, Any, Optional, List
import requests
from openai import OpenAI
import json

class LLMAdapter:
    """LLM客户端适配器，支持多提供商"""
    
    def __init__(self, provider: str, config: Dict[str, Any]):
        self.provider = provider
        self.config = config
        self.client = self._initialize_client()
        
    def _initialize_client(self):
        """初始化客户端"""
        if self.provider == "openai":
            return OpenAI(api_key=self.config.get('api_key'))
        elif self.provider == "deepseek":
            return OpenAI(
                api_key=self.config.get('api_key'),
                base_url=self.config.get('base_url', 'https://api.deepseek.com/v1')
            )
        else:
            raise ValueError(f"不支持的LLM提供商: {self.provider}")
    
    def call(self, prompt: str, max_tokens: int = None, temperature: float = None, 
             system: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """调用LLM接口"""
        
        # 使用配置默认值或参数值
        max_tokens = max_tokens or self.config.get('max_tokens', 1000)
        temperature = temperature or self.config.get('temperature', 0.1)
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.get('model'),
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            content = response.choices[0].message.content
            return {
                "text": content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "model": response.model,
                "finish_reason": response.choices[0].finish_reason
            }
            
        except Exception as e:
            return {
                "text": f"API调用错误: {str(e)}",
                "error": True,
                "error_message": str(e)
            }
    
    def call_with_retry(self, prompt: str, max_retries: int = 3, **kwargs) -> Dict[str, Any]:
        """带重试机制的调用"""
        for attempt in range(max_retries):
            try:
                result = self.call(prompt, **kwargs)
                if not result.get('error'):
                    return result
            except Exception as e:
                if attempt == max_retries - 1:
                    return {
                        "text": f"所有重试失败: {str(e)}",
                        "error": True,
                        "error_message": str(e)
                    }
                time.sleep(2 ** attempt)  # 指数退避
        return {"text": "未知错误", "error": True}