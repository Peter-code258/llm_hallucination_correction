#!/usr/bin/env python3
"""
DSPy Signature 定义 - 将 6 个手工 prompt 模板重构为可被 DSPy 自动优化的程序化契约。

每个 Signature 用类语法声明输入/输出字段及 docstring，DSPy 会据此自动构造 instruction
和 few-shot 示例，再由 MIPROv2 / BootstrapFewShotWithMetric 在数据集上自动搜索最优 prompt。
"""

import dspy
from typing import List


# ==================== Signature 定义 ====================

class InitialAnswerSignature(dspy.Signature):
    """根据用户问题直接生成一个详细、全面的回答（不做事实核查）。"""
    question: str = dspy.InputField(desc="用户提出的原始问题")
    answer: str = dspy.OutputField(desc="对问题的详细、全面回答")


class IntentClassificationSignature(dspy.Signature):
    """将用户查询分类为以下四类意图之一：事实查询、比较查询、方法查询、观点查询。

    分类规则：
    - 含"比较/对比/区别/哪个更好"等关键词 -> 比较查询
    - 含"如何/怎样/步骤/方法"等关键词     -> 方法查询
    - 含"观点/看法/评价/争议"等关键词     -> 观点查询
    - 其余默认为事实查询
    """
    query: str = dspy.InputField(desc="用户的查询文本")
    intent: str = dspy.OutputField(desc="意图类型名称（仅返回名称，不加解释）")


class ClaimExtractionSignature(dspy.Signature):
    """将一段文本分解为独立、原子化的真实性陈述（atomic claims）。"""
    text: str = dspy.InputField(desc="待提取声明的文本")
    claims: List[str] = dspy.OutputField(desc="从文本中抽出的原子断言列表")


class FactVerificationSignature(dspy.Signature):
    """作为事实核查专家，基于证据验证声明的真实性。

    verdict 取值：SUPPORTED / CONTRADICTED / PARTIALLY_SUPPORTED / UNVERIFIED
    """
    intent: str = dspy.InputField(desc="查询意图（事实/比较/方法/观点查询）")
    query: str = dspy.InputField(desc="用户原始查询")
    claim: str = dspy.InputField(desc="需要被验证的单条声明")
    evidence_text: str = dspy.InputField(desc="检索到的相关证据片段文本")
    verdict: str = dspy.OutputField(desc="SUPPORTED|CONTRADICTED|PARTIALLY_SUPPORTED|UNVERIFIED")
    confidence: float = dspy.OutputField(desc="置信度 0.0-1.0")
    reasoning: str = dspy.OutputField(desc="详细推理过程")
    intent_specific_analysis: str = dspy.OutputField(desc="针对查询意图的特别分析")


class HallucinationDetectionSignature(dspy.Signature):
    """作为幻觉检测专家，分析 AI 回答是否存在幻觉（事实性/逻辑性/证据性/一致性）。

    hallucination_type 取值：FACTUAL / LOGICAL / EVIDENTIAL / CONSISTENCY / MIXED / NONE
    """
    question: str = dspy.InputField(desc="原始问题")
    initial_answer: str = dspy.InputField(desc="AI 初始回答")
    verified_answer: str = dspy.InputField(desc="验证后回答")
    evidence: str = dspy.InputField(desc="支持证据文本")
    has_hallucination: bool = dspy.OutputField(desc="是否存在幻觉")
    hallucination_type: str = dspy.OutputField(desc="FACTUAL|LOGICAL|EVIDENTIAL|CONSISTENCY|MIXED|NONE")
    confidence: float = dspy.OutputField(desc="置信度 0.0-1.0")
    reasoning: str = dspy.OutputField(desc="分析推理过程")


class AnswerCorrectionSignature(dspy.Signature):
    """根据验证结果和查询意图，重新生成一个修正后的、准确的答案。

    intent 决定重写风格：事实查询重准确性 / 比较查询重全面 / 方法查询重可操作性 / 观点查询重平衡。
    """
    intent: str = dspy.InputField(desc="查询意图（事实/比较/方法/观点查询）")
    query: str = dspy.InputField(desc="用户原始查询")
    initial_answer: str = dspy.InputField(desc="待修正的初始答案")
    verification_summary: str = dspy.InputField(desc="验证结果摘要")
    corrected_answer: str = dspy.OutputField(desc="修正后的答案")


# ==================== 答案纠正意图变体 ====================
# 把 4 种意图各拆为独立 Signature，便于分别优化 instruction + demos。
# 与原 AnswerCorrectionSignature 的字段一致，仅 docstring 强化意图重写风格。

class AnswerCorrectionFactualSignature(dspy.Signature):
    """作为事实核查专家，根据验证结果重新生成一个准确的事实性答案。

    要求：
    - 事实陈述必须有证据支持
    - 数值/日期/人物等关键信息以验证结果为准
    - 不输出未经证据验证的内容
    """
    intent: str = dspy.InputField(desc="查询意图：事实查询")
    query: str = dspy.InputField(desc="用户原始查询")
    initial_answer: str = dspy.InputField(desc="待修正的初始答案")
    verification_summary: str = dspy.InputField(desc="验证结果摘要")
    corrected_answer: str = dspy.OutputField(desc="修正后的事实性答案")


class AnswerCorrectionComparisonSignature(dspy.Signature):
    """作为比较分析专家，根据验证结果重新生成一个全面、平衡的比较性答案。

    要求：
    - 涵盖被比较实体的关键差异与相似点
    - 比较维度与证据一致，不偏袒任一方
    - 显式标出证据支持/反对的维度
    """
    intent: str = dspy.InputField(desc="查询意图：比较查询")
    query: str = dspy.InputField(desc="用户原始查询")
    initial_answer: str = dspy.InputField(desc="待修正的初始答案")
    verification_summary: str = dspy.InputField(desc="验证结果摘要")
    corrected_answer: str = dspy.OutputField(desc="修正后的比较分析")


class AnswerCorrectionMethodSignature(dspy.Signature):
    """作为方法指导专家，根据验证结果重新生成一个可操作的方法指南。

    要求：
    - 步骤清晰可执行，每步都可被验证
    - 不输出无法验证或无证据支持的工具/接口
    - 标出每步的验证依据
    """
    intent: str = dspy.InputField(desc="查询意图：方法查询")
    query: str = dspy.InputField(desc="用户原始查询")
    initial_answer: str = dspy.InputField(desc="待修正的初始答案")
    verification_summary: str = dspy.InputField(desc="验证结果摘要")
    corrected_answer: str = dspy.OutputField(desc="修正后的方法指南")


class AnswerCorrectionOpinionSignature(dspy.Signature):
    """作为观点综述专家，根据验证结果重新生成一个平衡、客观的观点综述。

    要求：
    - 完整呈现主要立场及其证据支持
    - 不带模型自己的判断或偏见
    - 每个观点须可追溯到证据
    """
    intent: str = dspy.InputField(desc="查询意图：观点查询")
    query: str = dspy.InputField(desc="用户原始查询")
    initial_answer: str = dspy.InputField(desc="待修正的初始答案")
    verification_summary: str = dspy.InputField(desc="验证结果摘要")
    corrected_answer: str = dspy.OutputField(desc="修正后的观点综述")


# ==================== Signature 注册表 ====================
# 供 prompt_optimizer.py 使用，便于遍历优化。
# 答案纠正保留单一通用入口 "answer_correction"（向后兼容），
# 同时新增 4 个意图变体入口供按意图精细优化。
SIGNATURE_REGISTRY = {
    "initial_answer": InitialAnswerSignature,
    "intent_classification": IntentClassificationSignature,
    "claim_extraction": ClaimExtractionSignature,
    "fact_verification": FactVerificationSignature,
    "hallucination_detection": HallucinationDetectionSignature,
    "answer_correction": AnswerCorrectionSignature,
    "answer_correction_factual": AnswerCorrectionFactualSignature,
    "answer_correction_comparison": AnswerCorrectionComparisonSignature,
    "answer_correction_method": AnswerCorrectionMethodSignature,
    "answer_correction_opinion": AnswerCorrectionOpinionSignature,
}


# intent 字符串 <-> answer_correction 变体 name 映射
INTENT_TO_CORRECTION_VARIANT = {
    "事实查询": "answer_correction_factual",
    "比较查询": "answer_correction_comparison",
    "方法查询": "answer_correction_method",
    "观点查询": "answer_correction_opinion",
}
