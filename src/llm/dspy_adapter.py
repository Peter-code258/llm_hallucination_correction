#!/usr/bin/env python3
"""
DSPy LM 适配器 - 把项目现有的 LLMAdapter（基于 OpenAI SDK 的 DeepSeek/OpenAI 客户端）
包装为 DSPy 的 dspy.LM 接口，使 DSPy Module / Optimizer 可直接复用现有 API 配置。

设计要点：
1. 不破坏 LLMAdapter 的对外接口，orchestrator 等模块继续用原来的 .call / .call_with_retry。
2. 暴露一个 get_dspy_lm() 工厂方法，返回 dspy.LM 实例，并自动 dspy.configure() 注册为全局 LM。
3. 复用现有 config 的 provider/api_key/model/base_url/temperature/max_tokens。
"""

import os
from typing import Any, Dict, Optional

import dspy

# 复用项目已有的 LLMAdapter
from src.llm.llm_client import LLMAdapter


# provider -> OpenAI 兼容 model 字符串映射
# DSPy 的 dspy.LM 通过 model 字符串前缀路由到不同 backend：
#   "openai/xxx"   -> OpenAI 官方
#   "deepseek/xxx" -> 走 OpenAI client 兼容协议（DeepSeek 用 openai base_url）
_PROVIDER_TO_DSPY_MODEL_PREFIX = {
    "openai": "openai",
    "deepseek": "openai",  # DeepSeek 走 openai 兼容接口，仅 base_url 不同
    "anthropic": "anthropic",
}


class DSPyLLMAdapter:
    """
    把 LLMAdapter 配置包装成 dspy.LM。
    优先用 dspy.LM 原生 OpenAI 兼容协议，避免双重封装导致的 token 计数错误。
    """

    def __init__(self, llm_adapter: LLMAdapter):
        self.llm_adapter = llm_adapter
        self._dspy_lm: Optional[dspy.LM] = None

    def _build_dspy_lm(self) -> dspy.LM:
        cfg = self.llm_adapter.config
        provider = self.llm_adapter.provider
        prefix = _PROVIDER_TO_DSPY_MODEL_PREFIX.get(provider, "openai")

        # 拼出 dspy.LM 期望的 model 字符串：prefix/model_name
        model_str = f"{prefix}/{cfg.get('model', 'gpt-3.5-turbo')}"

        lm_kwargs: Dict[str, Any] = {
            "model": model_str,
            "api_key": cfg.get("api_key") or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            "api_base": cfg.get("base_url"),  # DeepSeek 自定义 base_url
            "model_type": "chat",
            "temperature": cfg.get("temperature", 0.1),
            "max_tokens": cfg.get("max_tokens", 1000),
            "num_retries": 3,
        }
        # 移除值为 None 的字段，避免 dspy.LM 报错
        lm_kwargs = {k: v for k, v in lm_kwargs.items() if v is not None}
        return dspy.LM(**lm_kwargs)

    def get_dspy_lm(self, configure_global: bool = True) -> dspy.LM:
        """获取（并可选地全局注册）一个 dspy.LM 实例。"""
        if self._dspy_lm is None:
            self._dspy_lm = self._build_dspy_lm()
            if configure_global:
                dspy.configure(lm=self._dspy_lm)
        return self._dspy_lm


def configure_dspy_from_config(config: Dict[str, Any]) -> dspy.LM:
    """
    便利函数：从项目 config 字典直接配置 DSPy 全局 LM。
    给 scripts/optimize_prompts.py 使用。
    """
    llm_adapter = LLMAdapter(config["llm"]["provider"], config["llm"])
    wrapper = DSPyLLMAdapter(llm_adapter)
    return wrapper.get_dspy_lm(configure_global=True)
