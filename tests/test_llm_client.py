#!/usr/bin/env python3
"""
LLM客户端测试
测试LLM适配器的基本功能和多提供商支持
"""

import pytest
import os
import sys
from unittest.mock import Mock, patch
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.llm_client import LLMAdapter

class TestLLMClient:
    """LLM客户端测试类"""
    
    def setup_method(self):
        """测试方法前的设置"""
        self.config = {
            'provider': 'deepseek',
            'api_key': 'test_api_key',
            'model': 'test-model',
            'temperature': 0.1,
            'max_tokens': 1000,
            'base_url': 'https://api.test.com/v1'
        }
    
    def test_initialization(self):
        """测试客户端初始化"""
        adapter = LLMAdapter('deepseek', self.config)
        assert adapter.provider == 'deepseek'
        assert adapter.config == self.config
    
    @patch('src.llm.llm_client.OpenAI')
    def test_deepseek_client_initialization(self, mock_openai):
        """测试DeepSeek客户端初始化"""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        adapter = LLMAdapter('deepseek', self.config)
        
        # 验证OpenAI客户端是否正确初始化
        mock_openai.assert_called_once_with(
            api_key='test_api_key',
            base_url='https://api.test.com/v1'
        )
    
    @patch('src.llm.llm_client.OpenAI')
    def test_successful_api_call(self, mock_openai):
        """测试成功的API调用"""
        # 模拟响应
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '测试响应内容'
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 200
        mock_response.usage.total_tokens = 300
        mock_response.model = 'test-model'
        mock_response.choices[0].finish_reason = 'stop'
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        adapter = LLMAdapter('deepseek', self.config)
        result = adapter.call('测试提示词')
        
        # 验证结果
        assert result['text'] == '测试响应内容'
        assert result['usage']['prompt_tokens'] == 100
        assert result['model'] == 'test-model'
        assert not result.get('error')
    
    @patch('src.llm.llm_client.OpenAI')
    def test_api_call_with_error(self, mock_openai):
        """测试API调用错误处理"""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception('API错误')
        mock_openai.return_value = mock_client
        
        adapter = LLMAdapter('deepseek', self.config)
        result = adapter.call('测试提示词')
        
        # 验证错误处理
        assert result['error'] == True
        assert 'API错误' in result['text']
    
    def test_unsupported_provider(self):
        """测试不支持的提供商"""
        with pytest.raises(ValueError, match='不支持的LLM提供商'):
            LLMAdapter('unsupported_provider', self.config)
    
    @patch('src.llm.llm_client.OpenAI')
    def test_call_with_retry_success(self, mock_openai):
        """测试带重试的成功调用"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '重试测试响应'
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 100
        mock_response.usage.total_tokens = 150
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        adapter = LLMAdapter('deepseek', self.config)
        result = adapter.call_with_retry('测试提示词', max_retries=3)
        
        assert result['text'] == '重试测试响应'
        assert not result.get('error')
    
    @patch('src.llm.llm_client.OpenAI')
    @patch('time.sleep')  # 模拟sleep以避免实际等待
    def test_call_with_retry_failure(self, mock_sleep, mock_openai):
        """测试带重试的失败调用"""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception('持续错误')
        mock_openai.return_value = mock_client
        
        adapter = LLMAdapter('deepseek', self.config)
        result = adapter.call_with_retry('测试提示词', max_retries=2)
        
        assert result['error'] == True
        assert '所有重试失败' in result['text']
        # 验证重试次数
        assert mock_client.chat.completions.create.call_count == 2
    
    def test_config_validation(self):
        """测试配置验证"""
        # 测试缺少必需配置
        invalid_config = {'provider': 'deepseek'}  # 缺少api_key等
        
        with pytest.raises(Exception):
            adapter = LLMAdapter('deepseek', invalid_config)
            adapter.call('测试提示词')

if __name__ == "__main__":
    pytest.main([__file__, '-v'])