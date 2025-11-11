#!/usr/bin/env python3
"""
证据验证器 - 基于检索证据验证声明真实性
"""

import json
import re
from typing import Dict, Any, List
from src.llm.llm_client import LLMAdapter

class EvidenceVerifier:
    """证据验证器 - 基于检索结果验证声明真实性"""
    
    def __init__(self, llm_adapter: LLMAdapter, config: Dict[str, Any]):
        self.llm = llm_adapter
        self.config = config
        self.confidence_threshold = config.get('confidence_threshold', 0.8)
    
    def verify_claim(self, claim: str, evidence_snippets: List[Dict], 
                   query: str, intent: str) -> Dict[str, Any]:
        """验证单个声明的真实性"""
        
        prompt = self._build_verification_prompt(claim, evidence_snippets, query, intent)
        
        response = self.llm.call_with_retry(
            prompt=prompt,
            max_tokens=800,
            temperature=0.1
        )
        
        if response.get('error'):
            return self._create_error_result(claim, response['error_message'])
        
        return self._parse_verification_response(response['text'], claim, intent)
    
    def _build_verification_prompt(self, claim: str, evidence_snippets: List[Dict], 
                                 query: str, intent: str) -> str:
        """构建验证提示词"""
        
        evidence_text = self._format_evidence_snippets(evidence_snippets)
        
        return f"""
        作为事实核查专家，请基于提供的证据验证以下声明的真实性。
        
        查询意图：{intent}
        原始查询："{query}"
        需要验证的声明："{claim}"
        
        相关证据片段：
        {evidence_text}
        
        请按以下JSON格式输出验证结果：
        {{
            "verdict": "SUPPORTED|CONTRADICTED|PARTIALLY_SUPPORTED|UNVERIFIED",
            "confidence": 0.0-1.0,
            "supporting_evidence": [
                {{
                    "text": "证据文本",
                    "source": "来源",
                    "relevance_score": 0.0-1.0
                }}
            ],
            "contradicting_evidence": [
                {{
                    "text": "矛盾证据文本", 
                    "source": "来源",
                    "contradiction_score": 0.0-1.0
                }}
            ],
            "reasoning": "详细的推理过程",
            "intent_specific_analysis": "针对{intent}意图的特别分析"
        }}
        
        验证指南：
        - SUPPORTED: 有明确证据支持该声明
        - CONTRADICTED: 有明确证据反驳该声明  
        - PARTIALLY_SUPPORTED: 部分证据支持，但有不准确或夸大之处
        - UNVERIFIED: 缺乏足够证据进行判断
        
        请确保推理过程详细且基于证据。
        """
    
    def _format_evidence_snippets(self, snippets: List[Dict]) -> str:
        """格式化证据片段"""
        if not snippets:
            return "暂无相关证据片段"
        
        formatted = []
        for i, snippet in enumerate(snippets, 1):
            formatted.append(
                f"[证据{i}] 来源: {snippet.get('source', '未知')} "
                f"(相关性: {snippet.get('similarity', 0):.3f})\n"
                f"{snippet.get('text', '')}\n"
            )
        
        return "\n".join(formatted)
    
    def _parse_verification_response(self, response: str, claim: str, intent: str) -> Dict[str, Any]:
        """解析验证响应"""
        try:
            # 尝试提取JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
            else:
                # 尝试直接解析整个响应
                result = json.loads(response)
        except json.JSONDecodeError:
            return self._create_fallback_result(claim, intent, response)
        
        # 验证必需字段
        required_fields = ['verdict', 'confidence']
        for field in required_fields:
            if field not in result:
                return self._create_fallback_result(claim, intent, response)
        
        # 标准化结果
        result['claim'] = claim
        result['intent'] = intent
        result['timestamp'] = self._get_timestamp()
        
        # 确保列表字段存在
        for list_field in ['supporting_evidence', 'contradicting_evidence']:
            if list_field not in result or not isinstance(result[list_field], list):
                result[list_field] = []
        
        return result
    
    def _create_error_result(self, claim: str, error_message: str) -> Dict[str, Any]:
        """创建错误结果"""
        return {
            "verdict": "UNVERIFIED",
            "confidence": 0.0,
            "claim": claim,
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "reasoning": f"验证过程中发生错误: {error_message}",
            "intent_specific_analysis": "",
            "error": True,
            "timestamp": self._get_timestamp()
        }
    
    def _create_fallback_result(self, claim: str, intent: str, raw_response: str) -> Dict[str, Any]:
        """创建回退结果"""
        return {
            "verdict": "UNVERIFIED",
            "confidence": 0.3,
            "claim": claim,
            "intent": intent,
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "reasoning": f"无法解析验证响应，原始响应: {raw_response[:200]}...",
            "intent_specific_analysis": "响应格式错误，无法进行意图特定分析",
            "fallback": True,
            "timestamp": self._get_timestamp()
        }
    
    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def batch_verify(self, claims: List[Dict], evidence_map: Dict, query: str, intent: str) -> List[Dict[str, Any]]:
        """批量验证多个声明"""
        verification_results = []
        
        for claim_info in claims:
            claim_id = claim_info['id']
            claim_text = claim_info['text']
            
            if claim_id in evidence_map:
                evidence_info = evidence_map[claim_id]
                evidence_snippets = evidence_info['evidence']
                
                # 执行验证
                verification_result = self.verify_claim(
                    claim_text, evidence_snippets, query, intent
                )
                
                # 添加声明ID和证据信息
                verification_result['claim_id'] = claim_id
                verification_result['evidence_count'] = len(evidence_snippets)
                
                verification_results.append(verification_result)
            else:
                # 如果没有检索到证据，创建未验证结果
                verification_results.append({
                    "claim_id": claim_id,
                    "claim": claim_text,
                    "verdict": "UNVERIFIED",
                    "confidence": 0.0,
                    "evidence_count": 0,
                    "reasoning": "未能检索到相关证据进行验证",
                    "supporting_evidence": [],
                    "contradicting_evidence": []
                })
        
        return verification_results