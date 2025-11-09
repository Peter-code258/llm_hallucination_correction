import re
from typing import Dict, Any, List
from .llm_client import LLMAdapter

class IntentClassifier:
    """查询意图分类器"""
    
    def __init__(self, llm_adapter: LLMAdapter, config: Dict[str, Any]):
        self.llm = llm_adapter
        self.config = config
        self.supported_intents = config.get('supported_intents', [])
        self.default_intent = config.get('default_intent', '事实查询')
        
        # 意图模板映射
        self.intent_templates = {
            "事实查询": "找到关于[{}]的具体事实",
            "比较查询": "找到[{}]的对比信息", 
            "方法查询": "找到如何[{}]的方法或步骤",
            "观点查询": "找到关于[{}]的不同观点或评价"
        }
    
    def classify_intent(self, query: str) -> str:
        """分类查询意图"""
        prompt = self._build_intent_classification_prompt(query)
        
        response = self.llm.call_with_retry(
            prompt=prompt,
            max_tokens=100,
            temperature=0.1
        )
        
        if response.get('error'):
            return self.default_intent
        
        intent = self._parse_intent_response(response['text'])
        return intent if intent in self.supported_intents else self.default_intent
    
    def _build_intent_classification_prompt(self, query: str) -> str:
        """构建意图分类提示词"""
        return f"""
        请分析以下用户查询的意图类型，从以下选项中选择最匹配的一个：
        
        可选意图类型：
        - 事实查询：寻求具体事实、数据、定义、属性等客观信息
        - 比较查询：比较两个或多个实体、概念、方法的异同点
        - 方法查询：寻求操作流程、解决方案、实施步骤、操作方法
        - 观点查询：收集多方意见、评价、争议观点、不同立场
        
        查询内容："{query}"
        
        请只返回意图类型的名称，不要添加其他解释。
        """
    
    def _parse_intent_response(self, response: str) -> str:
        """解析意图分类响应"""
        # 清理响应文本
        cleaned = response.strip()
        
        # 尝试匹配支持的意图
        for intent in self.supported_intents:
            if intent in cleaned:
                return intent
        
        # 使用正则表达式提取
        patterns = [
            r'意图[：:]?\s*(\S+)',
            r'^(\S+查询)',
            r'类型[：:]?\s*(\S+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, cleaned)
            if match:
                candidate = match.group(1)
                if candidate in self.supported_intents:
                    return candidate
        
        return self.default_intent
    
    def generate_retrieval_prompt(self, query: str, intent: str) -> str:
        """根据意图生成检索提示词"""
        if intent == "比较查询":
            # 特殊处理比较查询
            compared_entities = self._extract_comparison_entities(query)
            if len(compared_entities) >= 2:
                entities_str = "和".join(compared_entities[:2])
                return self.intent_templates[intent].format(entities_str)
        
        return self.intent_templates[intent].format(query)
    
    def _extract_comparison_entities(self, query: str) -> List[str]:
        """提取比较查询中的实体"""
        prompt = f"""
        从以下比较查询中提取被比较的主要实体（通常为2个核心实体）：
        查询："{query}"
        
        请只返回实体名称，用逗号分隔，不要添加其他内容。
        """
        
        response = self.llm.call_with_retry(
            prompt=prompt,
            max_tokens=100,
            temperature=0.1
        )
        
        if response.get('error'):
            return [query]  # 回退到使用整个查询
        
        entities = [e.strip() for e in response['text'].split(',')]
        return [e for e in entities if e]  # 过滤空实体