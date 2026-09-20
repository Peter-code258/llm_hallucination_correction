#!/usr/bin/env python3
"""
Prompt模板管理 - 集中管理所有提示词模板

支持加载 DSPy 优化后的 instruction 与 few-shot demos：
- 当 data/optimized_prompts/<name>.json 存在时，对应模板会被优化版本替换。
- 找不到优化文件时自动回退到原手工模板。
- 通过 self.optimized_prompts 字典访问已加载的优化 prompt（含 instructions + demos）。
"""

import os
from typing import Any, Dict, Optional


# 优化 prompt 文件名 <-> 模板名 映射
_OPTIMIZED_NAME_MAP = {
    "initial_answer": "initial_answer",
    "intent_classification": "intent_classification",
    "claim_extraction": "claim_extraction",
    "fact_verification": "fact_verification",
    "hallucination_detection": "hallucination_detection",
    "answer_correction": "answer_correction",
    # 4 个意图变体（按意图分别优化）
    "answer_correction_factual": "answer_correction_factual",
    "answer_correction_comparison": "answer_correction_comparison",
    "answer_correction_method": "answer_correction_method",
    "answer_correction_opinion": "answer_correction_opinion",
}

# intent 字符串 -> answer_correction 变体 name
_INTENT_TO_VARIANT = {
    "事实查询": "answer_correction_factual",
    "比较查询": "answer_correction_comparison",
    "方法查询": "answer_correction_method",
    "观点查询": "answer_correction_opinion",
}


class PromptTemplates:
    """Prompt模板管理器"""
    
    # ==================== 初始回答生成模板 ====================
    INITIAL_ANSWER_TEMPLATE = """
    请直接回答以下问题，不需要进行事实核查或验证，提供您认为最合适的答案。

    问题: {question}
    
    请提供详细、全面的回答，包括所有相关信息和背景知识：
    """
    
    # ==================== 意图分类模板 ====================
    INTENT_CLASSIFICATION_TEMPLATE = """
    你是一个专业的查询意图分类器。你的任务是根据用户查询的内容，准确判断其意图类型。
    
    ## 分类标准
    - **事实查询**: 寻求具体事实、数据、定义、属性等客观信息
    - **比较查询**: 比较两个或多个实体、概念、方法的异同点  
    - **方法查询**: 寻求操作流程、解决方案、实施步骤、操作方法
    - **观点查询**: 收集多方意见、评价、争议观点、不同立场
    
    ## 分类规则
    1. 如果查询包含"比较"、"对比"、"区别"、"哪个更好"等关键词，归类为比较查询
    2. 如果查询包含"如何"、"怎样"、"步骤"、"方法"等关键词，归类为方法查询  
    3. 如果查询包含"观点"、"看法"、"评价"、"争议"等关键词，归类为观点查询
    4. 其他情况默认为事实查询
    
    ## 输出格式
    只需返回意图类型的名称，不要添加任何解释。
    
    当前查询: "{query}"
    意图类型:
    """
    
    # ==================== 声明提取模板 ====================
    CLAIM_EXTRACTION_TEMPLATE = """
    任务：将下面的文本分解为独立的真实性陈述（原子断言）。
    
    需要提取的文本: "{text}"
    
    提取结果:
    """
    
    # ==================== 事实验证模板 ====================
    FACT_VERIFICATION_TEMPLATE = """
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
                "source": "来源名称",
                "relevance_score": 0.0-1.0
            }}
        ],
        "contradicting_evidence": [
            {{
                "text": "矛盾证据文本", 
                "source": "来源名称",
                "contradiction_score": 0.0-1.0
            }}
        ],
        "reasoning": "详细的推理过程",
        "intent_specific_analysis": "针对查询意图的特别分析"
    }}
    """
    
    # ==================== 幻觉检测模板 ====================
    HALLUCINATION_DETECTION_TEMPLATE = """
    作为幻觉检测专家，请分析以下AI回答是否存在幻觉（虚构、不准确或缺乏证据支持的内容）。
    
    ## 检测标准
    - **事实性幻觉**: 陈述与可验证事实不符
    - **逻辑性幻觉**: 推理过程存在矛盾或不合逻辑
    - **证据性幻觉**: 缺乏可靠证据支持的关键声明
    - **一致性幻觉**: 与已知信息或上下文不一致
    
    ## 分析材料
    原始问题: "{question}"
    AI初始回答: "{initial_answer}"
    验证后回答: "{verified_answer}"
    支持证据: "{evidence}"
    
    ## 检测要求
    请按以下JSON格式输出检测结果：
    {{
        "has_hallucination": true|false,
        "hallucination_type": "FACTUAL|LOGICAL|EVIDENTIAL|CONSISTENCY|MIXED|NONE",
        "confidence": 0.0-1.0,
        "affected_sections": [
            {{
                "text": "存在幻觉的文本片段",
                "type": "幻觉类型",
                "severity": "LOW|MEDIUM|HIGH",
                "correction": "建议修正内容"
            }}
        ],
        "comparison_analysis": {{
            "initial_answer_quality": "评估初始回答质量",
            "verification_impact": "验证过程带来的改进",
            "key_differences": "主要差异点分析",
            "overall_improvement": "整体改善程度评估"
        }},
        "recommendations": [
            "改进建议1",
            "改进建议2"
        ]
    }}
    """
    
    # ==================== 答案纠正模板 ====================
    CORRECTION_TEMPLATES = {
        "事实查询": """
        作为事实核查专家，请根据验证结果重新生成一个准确的事实性答案。
        
        查询意图：{intent} - 事实查询
        原始查询："{query}"
        初始答案：{initial_answer}
        验证结果摘要：{verification_summary}
        
        修正后的答案：
        """,
        
        "比较查询": """
        作为比较分析专家，请根据验证结果重新生成一个全面准确的比较性答案。
        
        查询意图：{intent} - 比较查询  
        原始查询："{query}"
        初始答案：{initial_answer}
        验证结果摘要：{verification_summary}
        
        修正后的比较分析：
        """,
        
        "方法查询": """
        作为方法指导专家，请根据验证结果重新生成一个可操作的方法指南。
        
        查询意图：{intent} - 方法查询
        原始查询："{query}"
        初始答案：{initial_answer}
        验证结果摘要：{verification_summary}
        
        修正后的方法指南：
        """,
        
        "观点查询": """
        作为观点综述专家，请根据验证结果重新生成一个平衡客观的观点综述。
        
        查询意图：{intent} - 观点查询
        原始查询："{query}"
        初始答案：{initial_answer}
        验证结果摘要：{verification_summary}
        
        修正后的观点综述：
        """
    }
    
    def __init__(self, optimized_dir: Optional[str] = None):
        """
        参数：
            optimized_dir: DSPy 优化后 prompt 的存储目录。默认 data/optimized_prompts。
                           传 None 表示完全使用原手工模板（兼容旧行为）。
        """
        # 优化后的 prompt 数据：name -> {"instructions": str, "demos": list}
        self.optimized_prompts: Dict[str, Dict[str, Any]] = {}
        # 各模板对应的"槽位变量名 -> 模板占位符"映射，便于把优化 instruction 拼成最终 prompt
        self._slot_map = {
            "initial_answer": {"question": "question"},
            "intent_classification": {"query": "query"},
            "claim_extraction": {"text": "text"},
            "fact_verification": {
                "intent": "intent",
                "query": "query",
                "claim": "claim",
                "evidence_text": "evidence_text",
            },
            "hallucination_detection": {
                "question": "question",
                "initial_answer": "initial_answer",
                "verified_answer": "verified_answer",
                "evidence": "evidence",
            },
            # answer_correction 走单独的 get_correction_prompt
        }
        if optimized_dir is None:
            optimized_dir = os.environ.get(
                "OPTIMIZED_PROMPTS_DIR", "./data/optimized_prompts"
            )
        self.optimized_dir = optimized_dir
        self._load_optimized_prompts()

    def _load_optimized_prompts(self) -> None:
        """从 optimized_dir 加载所有 .json 优化 prompt。失败则保持空字典，走原模板。"""
        if not os.path.isdir(self.optimized_dir):
            return
        for name in _OPTIMIZED_NAME_MAP:
            path = os.path.join(self.optimized_dir, f"{name}.json")
            if not os.path.exists(path):
                continue
            try:
                import json
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "instructions" in data:
                    self.optimized_prompts[name] = data
            except Exception:
                # 加载失败不影响主流程，继续用原模板
                continue

    def _render_optimized(self, name: str, slots: Dict[str, Any]) -> Optional[str]:
        """
        用优化后的 instruction 渲染最终 prompt。
        instruction 中通常已经包含任务说明，把 slots 以 key: value 形式附加在末尾。
        返回 None 表示该模板没有优化版本，调用方应回退到原模板。
        """
        data = self.optimized_prompts.get(name)
        if not data:
            return None
        instruction = data.get("instructions", "")
        demos = data.get("demos", [])

        parts: list = []
        if instruction:
            parts.append(instruction.strip())
        # few-shot demos
        if demos:
            parts.append("\n## 示例")
            for i, d in enumerate(demos, 1):
                # demo 是 dict，按 key=value 串起来
                kv = "\n".join(f"{k}: {v}" for k, v in d.items() if v is not None)
                parts.append(f"### 示例{i}\n{kv}")
        # 当前输入
        if slots:
            parts.append("\n## 当前输入")
            parts.extend(f"{k}: {v}" for k, v in slots.items())
        return "\n\n".join(parts)

    def get_initial_answer_prompt(self, question: str) -> str:
        """获取初始回答生成提示词"""
        rendered = self._render_optimized(
            "initial_answer", {"question": question}
        )
        if rendered:
            return rendered
        return self.INITIAL_ANSWER_TEMPLATE.format(question=question)

    def get_intent_classification_prompt(self, query: str) -> str:
        """获取意图分类提示词"""
        rendered = self._render_optimized(
            "intent_classification", {"query": query}
        )
        if rendered:
            return rendered
        return self.INTENT_CLASSIFICATION_TEMPLATE.format(query=query)

    def get_claim_extraction_prompt(self, text: str) -> str:
        """获取声明提取提示词"""
        rendered = self._render_optimized(
            "claim_extraction", {"text": text}
        )
        if rendered:
            return rendered
        return self.CLAIM_EXTRACTION_TEMPLATE.format(text=text)

    def get_fact_verification_prompt(self, intent: str, query: str, claim: str, evidence_text: str) -> str:
        """获取事实验证提示词"""
        rendered = self._render_optimized(
            "fact_verification",
            {
                "intent": intent,
                "query": query,
                "claim": claim,
                "evidence_text": evidence_text,
            },
        )
        if rendered:
            return rendered
        return self.FACT_VERIFICATION_TEMPLATE.format(
            intent=intent,
            query=query,
            claim=claim,
            evidence_text=evidence_text
        )

    def get_hallucination_detection_prompt(self, question: str, initial_answer: str,
                                         verified_answer: str, evidence: str) -> str:
        """获取幻觉检测提示词"""
        rendered = self._render_optimized(
            "hallucination_detection",
            {
                "question": question,
                "initial_answer": initial_answer,
                "verified_answer": verified_answer,
                "evidence": evidence,
            },
        )
        if rendered:
            return rendered
        return self.HALLUCINATION_DETECTION_TEMPLATE.format(
            question=question,
            initial_answer=initial_answer,
            verified_answer=verified_answer,
            evidence=evidence
        )

    def get_correction_prompt(self, intent: str, query: str, initial_answer: str, verification_summary: str) -> str:
        """获取答案纠正提示词"""
        slots = {
            "intent": intent,
            "query": query,
            "initial_answer": initial_answer,
            "verification_summary": verification_summary,
        }
        # 1) 优先用按意图优化的变体（answer_correction_factual / comparison / method / opinion）
        variant_name = _INTENT_TO_VARIANT.get(intent)
        if variant_name:
            rendered = self._render_optimized(variant_name, slots)
            if rendered:
                return rendered
        # 2) 回退到通用 answer_correction 单一优化版本
        rendered = self._render_optimized("answer_correction", slots)
        if rendered:
            return rendered
        # 3) 最终回退到原手工模板（4 意图变体）
        template = self.CORRECTION_TEMPLATES.get(intent, self.CORRECTION_TEMPLATES["事实查询"])
        return template.format(
            intent=intent,
            query=query,
            initial_answer=initial_answer,
            verification_summary=verification_summary
        )