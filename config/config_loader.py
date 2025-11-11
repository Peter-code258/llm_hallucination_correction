#!/usr/bin/env python3
"""
配置加载器 - 处理配置文件加载和验证
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """加载配置文件"""
    config_file = Path(config_path)
    
    if not config_file.exists():
        # 创建默认配置
        return _create_default_config(config_file)
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 处理环境变量替换
    config = _resolve_environment_variables(config)
    
    return config

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """验证配置完整性"""
    errors = []
    warnings = []
    
    # 必需配置项检查
    required_sections = ['llm', 'vector_db', 'retrieval']
    for section in required_sections:
        if section not in config:
            errors.append(f"缺少必需配置段: {section}")
    
    # LLM配置验证
    if 'llm' in config:
        llm_config = config['llm']
        if not llm_config.get('api_key'):
            errors.append("LLM API密钥未配置")
        if not llm_config.get('model'):
            warnings.append("未指定LLM模型，使用默认值")
    
    # 向量数据库验证
    if 'vector_db' in config:
        db_config = config['vector_db']
        if not db_config.get('db_path'):
            warnings.append("向量数据库路径未指定，使用默认路径")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }

def _create_default_config(config_file: Path) -> Dict[str, Any]:
    """创建默认配置文件"""
    config_dir = config_file.parent
    config_dir.mkdir(parents=True, exist_ok=True)
    
    default_config = {
        'llm': {
            'provider': 'deepseek',
            'api_key': '${DEEPSEEK_API_KEY}',
            'model': 'deepseek-chat',
            'base_url': 'https://api.deepseek.com/v1',
            'temperature': 0.1,
            'max_tokens': 1000,
            'timeout': 30
        },
        'vector_db': {
            'embedding_model': 'BAAI/bge-base-en',
            'db_path': './data/vector_db',
            'collection_name': 'knowledge_base'
        },
        'retrieval': {
            'similarity_threshold': 0.7,
            'max_retrieved_docs': 5,
            'chunk_size': 1000,
            'chunk_overlap': 200
        },
        'verification': {
            'confidence_threshold': 0.8,
            'max_verification_attempts': 3
        },
        'intent': {
            'supported_intents': ['事实查询', '比较查询', '方法查询', '观点查询'],
            'default_intent': '事实查询'
        },
        'system': {
            'log_level': 'INFO',
            'max_concurrent_requests': 5,
            'cache_ttl': 3600
        }
    }
    
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ 已创建默认配置文件: {config_file}")
    return default_config

def _resolve_environment_variables(config: Dict[str, Any]) -> Dict[str, Any]:
    """解析环境变量占位符"""
    def resolve_value(value):
        if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            env_var = value[2:-1]
            return os.getenv(env_var, value)
        return value
    
    def traverse_dict(obj):
        if isinstance(obj, dict):
            return {k: traverse_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [traverse_dict(item) for item in obj]
        else:
            return resolve_value(obj)
    
    return traverse_dict(config)

def save_config(config: Dict[str, Any], config_path: str):
    """保存配置到文件"""
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

def get_config_template() -> Dict[str, Any]:
    """获取配置模板"""
    return {
        'llm': {
            'provider': 'deepseek|openai|anthropic',
            'api_key': 'YOUR_API_KEY',
            'model': '模型名称',
            'base_url': 'API基础URL',
            'temperature': 0.1,
            'max_tokens': 1000
        },
        'vector_db': {
            'embedding_model': '嵌入模型名称',
            'db_path': './data/vector_db',
            'collection_name': 'knowledge_base'
        },
        'retrieval': {
            'similarity_threshold': 0.7,
            'max_retrieved_docs': 5
        }
    }