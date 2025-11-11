#!/usr/bin/env python3
"""
答案纠正器 - 基于验证结果修正回答
"""

from typing import List, Dict, Any
from src.llm.llm_client import LLMAdapter

class AnswerCorrector:
    """答案纠正器 - 基于验证结果生成修正后的答案"""
    
    def __init__(self, llm_adapter: LLMAdapter):
        self.llm = llm_adapter
        
        # 意图特定的纠正模板
        self.correction_templates = {
            "事实查询": self._get_factual_correction_template(),
            "比较查询": self._get_comparison_correction_template(),
            "方法查询": self._get_method_correction_template(),
            "观点查询": self._get_opinion_correction_template()
        }
    
    def correct_answer(self, original_answer: str, verifications: List[Dict], 
                      query: str, intent: str) -> Dict[str, Any]:
        """生成纠正后的答案"""
        
        template = self.correction_templates.get(intent, self._get_factual_correction_template())
        verification_summary = self._prepare_verification_summary(verifications)
        
        prompt = template.format(
            query=query,
            intent=intent,
            original_answer=original_answer,
            verification_summary=verification_summary
        )
        
        response = self.llm.call_with_retry(
            prompt=prompt,
            max_tokens=1200,
            temperature=0.3
        )
        
        if response.get('error'):
            corrected_text = f"纠正过程出错: {response.get('error_message', '未知错误')}"
        else:
            corrected_text = response['text']
        
        return {
            "corrected_answer": corrected_text,
            "original_answer": original_answer,
            "query": query,
            "intent": intent,
            "verifications_count": len(verifications),
            "supported_claims": len([v for v in verifications if v.get('verdict') == 'SUPPORTED']),
            "contradicted_claims": len([v for v in verifications if v.get('verdict') == 'CONTRADICTED']),
            "correction_metadata": response.get('usage', {})
        }
    
    def _prepare_verification_summary(self, verifications: List[Dict]) -> str:
        """准备验证结果摘要"""
        if not verifications:
            return "没有进行声明验证或验证全部失败。"
        
        summary_parts = []
        for i, verification in enumerate(verifications, 1):
            claim = verification.get('claim', f'声明{i}')
            verdict = verification.get('verdict', 'UNVERIFIED')
            confidence = verification.get('confidence', 0)
            reasoning = verification.get('reasoning', '无详细推理')
            
            summary_parts.append(f"声明{i}: {claim}")
            summary_parts.append(f"验证结果: {verdict} (置信度: {confidence:.2f})")
            
            # 添加证据摘要
            supporting_evidence = verification.get('supporting_evidence', [])
            contradicting_evidence = verification.get('contradicting_evidence', [])
            
            if supporting_evidence:
                summary_parts.append(f"支持证据: {len(supporting_evidence)}条")
            if contradicting_evidence:
                summary_parts.append(f"矛盾证据: {len(contradicting_evidence)}条")
            
            summary_parts.append(f"推理: {reasoning[:100]}..." if len(reasoning) > 100 else f"推理: {reasoning}")
            summary_parts.append("---")
        
        return "\n".join(summary_parts)
    
    def _get_factual_correction_template(self) -> str:
        """事实查询的纠正模板"""
        return """
        作为事实核查专家，请根据验证结果重新生成一个准确的事实性答案。

        查询意图：{intent} - 事实查询
        原始查询："{query}"
        原始答案：{original_answer}

        验证结果摘要：
        {verification_summary}

        请生成修正后的答案，要求：
        1. 严格基于验证证据，不添加未经证实的信息
        2. 保持客观、准确、简洁的专业风格
        3. 如果某些声明无法验证或存在矛盾，请明确说明局限性
        4. 优先使用支持度高的证据，对矛盾证据进行说明
        5. 保持答案的完整性和连贯性

        修正后的答案应直接回答原始查询，同时体现验证过程的严谨性。

        修正后的答案：
        """
    
    def _get_comparison_correction_template(self) -> str:
        """比较查询的纠正模板"""
        return """
        作为比较分析专家，请根据验证结果重新生成一个全面准确的比较性答案。

        查询意图：{intent} - 比较查询  
        原始查询："{query}"
        原始答案：{original_answer}

        验证结果摘要：
        {verification_summary}

        请生成修正后的比较分析，要求：
        1. 确保比较维度的全面性和平衡性，避免偏颇
        2. 基于证据提供具体的对比点和数据支持
        3. 客观呈现各方的优势和劣势
        4. 如果某些比较点缺乏充分证据，请谨慎表述并说明不确定性
        5. 提供有依据的结论或建议

        修正后的比较分析应具有结构化的对比框架和基于证据的判断。

        修正后的比较分析：
        """
    
    def _get_method_correction_template(self) -> str:
        """方法查询的纠正模板"""
        return """
        作为方法指导专家，请根据验证结果重新生成一个可操作的方法指南。

        查询意图：{intent} - 方法查询
        原始查询："{query}"
        原始答案：{original_answer}

        验证结果摘要：
        {verification_summary}

        请生成修正后的方法指南，要求：
        1. 确保步骤的可行性、正确性和安全性
        2. 提供清晰、有序的操作指引
        3. 基于验证证据调整或完善有问题的步骤
        极速模式4. 包含必要的注意事项和常见问题解决方案
        5. 如果某些方法步骤缺乏充分验证，请注明其不确定性

        修正后的方法指南应具有实用性和可操作性。

        修正后的方法指南：
        """
    
    def _get_opinion_correction_template(self) -> str:
        """观点查询的纠正模板"""
        return """
        作为观点综述专家，请根据验证结果重新生成一个平衡客观的观点综述。

        查询意图：{intent} - 观点查询
        原始查询："{query}"
        原始答案：{original_answer}

        验证结果摘要：
        {verification_summary}

        请生成修正后的观点综述，要求：
        1. 全面、平衡地呈现不同的观点立场和论据
        2. 基于证据客观表述各方观点，避免主观偏向
        3. 明确区分事实性内容和观点性内容
        4. 如果某些观点缺乏充分证据支持，请说明其争议性
        5. 提供基于证据的综合分析或趋势判断

        修正后的观点综述应体现多元视角和客观分析。

        修正后的观点综述：
        """
    
    def generate_correction_report(self, original_answer: str, corrected_answer: str, 
                                 verifications: List[Dict], query: str, intent: str) -> Dict[str, Any]:
        """生成纠正报告"""
        # 计算纠正效果指标
        supported_count = len([v for v in verifications if v.get('verdict') == 'SUPPORTED'])
        contradicted_count = len([v for v in verifications if v.get('verdict') == 'CONTRADICTED'])
        total_claims = len(verifications)
        
        support_ratio = supported_count / total_claims if total_claims > 0 else 0
        contradiction_ratio = contradicted_count / total_claims if total_claims > 0 else 0
        
        # 分析改进点
        improvements = self._analyze_improvements(original_answer, corrected_answer, verifications)
        
        return {
            "original_answer": original_answer,
            "corrected_answer": corrected_answer,
            "query": query,
            "intent": intent,
            "verification_stats": {
                "total_claims": total_claims,
                "supported_claims": supported_count,
                "contradicted_claims": contradicted_count,
                "support_ratio": support_ratio,
                "contradiction_ratio": contradiction_ratio
            },
            "improvements": improvements,
            "correction_effectiveness": self._assess_effectiveness(original_answer, corrected_answer)
        }
    
    def _analyze_improvements(self, original: str, corrected: str, verifications: List[Dict]) -> List[str]:
        """分析改进点"""
        improvements = []
        
        # 基于验证结果分析改进
        for verification in verifications:
            if verification.get('verdict') == 'CONTRADICTED':
                claim = verification.get('claim', '')
                improvements.append(f"纠正了不准确的声明: {claim[:50]}...")
        
        # 简单文本比较
        if len(corrected) > len(original) * 1.2:
            improvements.append("增加了更多证据支持的细节")
        elif len(corrected) < len(original) * 0.8:
            improvements.append("移除了未经证实的冗余信息")
        
        return improvements if improvements else ["保持了回答的核心内容，提高了准确性"]
    
    def _assess_effectiveness(self, original: str, corrected: str) -> str:
        """评估纠正效果"""
        # 简单评估逻辑
        original_length = len(original)
        corrected_length = len(corrected)
        
        if corrected_length > original_length * 1.3:
            return "显著增强：添加了大量证据支持的内容"
        elif corrected_length > original_length * 1.1:
            return "适度改进：增加了关键证据和细节"
        elif corrected_length < original_length * 0.9:
            return "精简优化：移除了不确定和冗余内容"
        else:
            return "质量提升：保持了简洁性同时提高了准确性"