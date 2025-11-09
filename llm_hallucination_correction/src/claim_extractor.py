import re
import json
from typing import List, Dict, Any
from .llm_client import LLMAdapter

class ClaimExtractor:
    """从文本中提取原子声明的模块"""
    
    def __init__(self, llm_adapter: LLMAdapter):
        self.llm = llm_adapter
    
    def extract_claims(self, text: str) -> List[Dict[str, Any]]:
        """从文本中提取原子声明"""
        if not text or len(text.strip()) < 10:  # 文本过短
            return [{"text": text, "confidence": 1.0, "atomic": True}]
        
        prompt = self._build_claim_extraction_prompt(text)
        
        response = self.llm.call_with_retry(
            prompt=prompt,
            max_tokens=500,
            temperature=0.1
        )
        
        if response.get('error'):
            return self._fallback_extraction(text)
        
        return self._parse_claims_response(response['text'], text)
    
    def _build_claim_extraction_prompt(self, text: str) -> str:
        """构建声明提取提示词"""
        return f"""
        任务：将下面的文本分解为独立的真实性陈述（原子断言）。
        
        要求：
        1. 每个陈述应该是原子性的，表达一个完整的事实或观点
        2. 为每个陈述编号，格式为：[CLAIM_1]: 陈述内容
        3. 确保分解后的陈述覆盖原文的所有重要信息
        4. 保持陈述的客观性和准确性
        
        文本内容：
        {text}
        
        请按以下格式输出：
        [CLAIM_1]: 第一个原子陈述
        [CLAIM_2]: 第二个原子陈述
        ...
        """
    
    def _parse_claims_response(self, response: str, original_text: str) -> List[Dict[str, Any]]:
        """解析声明提取响应"""
        claims = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 匹配 [CLAIM_n]: 陈述内容 格式
            match = re.match(r'\[CLAIM_(\d+)\]:\s*(.+)', line)
            if match:
                claim_num = int(match.group(1))
                claim_text = match.group(2).strip()
                
                claims.append({
                    "id": f"claim_{claim_num}",
                    "text": claim_text,
                    "confidence": 0.9,  # 默认置信度
                    "atomic": True,
                    "original_position": self._estimate_position(claim_text, original_text)
                })
        
        # 如果没有提取到声明，使用回退方法
        if not claims:
            return self._fallback_extraction(original_text)
        
        return claims
    
    def _fallback_extraction(self, text: str) -> List[Dict[str, Any]]:
        """声明提取失败时的回退方法"""
        # 简单按句子分割
        sentences = re.split(r'[.!?。！？]+', text)
        claims = []
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) > 10:  # 过滤过短的句子
                claims.append({
                    "id": f"claim_{i+1}",
                    "text": sentence,
                    "confidence": 0.7,
                    "atomic": False,  # 非原子性声明
                    "original_position": i
                })
        
        return claims if claims else [{"text": text, "confidence": 1.0, "atomic": True}]
    
    def _estimate_position(self, claim_text: str, original_text: str) -> int:
        """估计声明在原文中的位置"""
        if claim_text in original_text:
            return original_text.index(claim_text)
        return 0