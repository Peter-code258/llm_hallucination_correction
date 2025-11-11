#!/usr/bin/env python3
"""
Prompt模板管理 - 集中管理所有提示词模板
"""

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
    
    def get_initial_answer_prompt(self, question: str) -> str:
        """获取初始回答生成提示词"""
        return self.INITIAL_ANSWER_TEMPLATE.format(question=question)
    
    def get_intent_classification_prompt(self, query: str) -> str:
        """获取意图分类提示词"""
        return self.INTENT_CLASSIFICATION_TEMPLATE.format(query=query)
    
    def get_claim_extraction_prompt(self, text: str) -> str:
        """获取声明提取提示词"""
        return self.CLAIM_EXTRACTION_TEMPLATE.format(text=text)
    
    def get_fact_verification_prompt(self, intent: str, query: str, claim: str, evidence_text: str) -> str:
        """获取事实验证提示词"""
        return self.FACT_VERIFICATION_TEMPLATE.format(
            intent=intent,
            query=query,
            claim=claim,
            evidence_text=evidence_text
        )
    
    def get_hallucination_detection_prompt(self, question: str, initial_answer: str, 
                                         verified_answer: str, evidence: str) -> str:
        """获取幻觉检测提示词"""
        return self.HALLUCINATION_DETECTION_TEMPLATE.format(
            question=question,
            initial_answer=initial_answer,
            verified_answer=verified_answer,
            evidence=evidence
        )
    
    def get_correction_prompt(self, intent: str, query: str, initial_answer: str, verification_summary: str) -> str:
        """获取答案纠正提示词"""
        template = self.CORRECTION_TEMPLATES.get(intent, self.CORRECTION_TEMPLATES["事实查询"])
        return template.format(
            intent=intent,
            query=query,
            initial_answer=initial_answer,
            verification_summary=verification_summary
        )