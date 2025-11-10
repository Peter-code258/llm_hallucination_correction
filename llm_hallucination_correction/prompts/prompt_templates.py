"""
大语言模型幻觉检测与纠正系统 - Prompt模板整合文件

本文件整合了系统中使用的所有Prompt模板，包括：
- 意图分类模板
- 声明提取模板  
- 事实验证模板
- 答案纠正模板
"""

class PromptTemplates:
    """Prompt模板管理器 - 集中管理所有提示词模板"""
    
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
    
    ## 示例
    用户查询: "机器学习和深度学习的区别是什么？"
    输出: 比较查询
    
    用户查询: "如何安装Python？"
    输出: 方法查询
    
    用户查询: "人工智能的未来发展前景"
    输出: 观点查询
    
    用户查询: "太阳的直径是多少？"
    输出: 事实查询
    
    当前查询: "{query}"
    意图类型:
    """
    
    # ==================== 声明提取模板 ====================
    CLAIM_EXTRACTION_TEMPLATE = """
    任务：将下面的文本分解为独立的真实性陈述（原子断言）。
    
    ## 提取要求
    1. **原子性**: 每个陈述应该是独立的，不能包含多个事实
    2. **完整性**: 覆盖原文的所有重要信息点
    3. **客观性**: 保持陈述的客观准确，不改变原意
    4. **编号格式**: 使用[CLAIM_1]: 陈述内容 的格式
    
    ## 输出格式
    [CLAIM_1]: 第一个原子陈述
    [CLAIM_2]: 第二个原子陈述
    [CLAIM_3]: 第三个原子陈述
    ...
    
    ## 处理规则
    - 如果文本过短或无法分解，返回原始文本作为一个声明
    - 忽略问候语、重复内容和无关信息
    - 将复杂句子拆分为简单的原子陈述
    
    ## 示例
    原文: "Python是一种高级编程语言，由Guido van Rossum在1991年创建。它具有简单易学的语法，广泛应用于Web开发和数据分析。"
    
    提取结果:
    [CLAIM_1]: Python是一种高级编程语言
    [CLAIM_2]: Python由Guido van Rossum创建
    [CLAIM_3]: Python在1991年创建
    [CLAIM_4]: Python具有简单易学的语法
    [CLAIM_5]: Python广泛应用于Web开发和数据分析
    
    需要提取的文本: "{text}"
    
    提取结果:
    """
    
    # ==================== 事实验证模板 ====================
    FACT_VERIFICATION_TEMPLATE = """
    作为事实核查专家，请基于提供的证据验证以下声明的真实性。
    
    ## 验证标准
    - **SUPPORTED**: 有明确、可靠的证据完全支持该声明
    - **CONTRADICTED**: 有明确证据反驳或否定该声明
    - **PARTIALLY_SUPPORTED**: 部分证据支持，但存在不准确或夸大之处
    - **UNVERIFIED**: 缺乏足够证据进行判断
    
    ## 证据评估原则
    1. **权威性**: 优先考虑权威来源的证据
    2. **时效性**: 关注证据的时间相关性
    3. **一致性**: 多个来源的一致性增加可信度
    4. **相关性**: 证据与声明的直接相关程度
    
    ## 输出格式(json)
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
    "reasoning": "详细的推理分析过程",
    "intent_specific_analysis": "针对查询意图的特别分析"
    }}
    ## 意图特定指南
- **事实查询**: 重点关注数据的准确性和来源可靠性
- **比较查询**: 确保比较的全面性和维度的一致性
- **方法查询**: 验证步骤的可行性和安全性
- **观点查询**: 关注观点表述的平衡性和代表性

查询意图：{intent}
原始查询："{query}"
需要验证的声明："{claim}"

相关证据片段：
{evidence_text}

请按指定JSON格式输出验证结果：
"""

# ==================== 答案纠正模板 ====================
CORRECTION_TEMPLATES = {
    "事实查询": """
    作为事实核查专家，请根据验证结果重新生成一个准确的事实性答案。
    
    查询意图：{intent} - 事实查询
    原始查询："{query}"
    原始答案：{original_answer}
    
    验证结果摘要：
    {verification_summary}
    
    ## 纠正要求
    1. **证据驱动**: 严格基于验证证据，不添加未经证实的信息
    2. **客观中立**: 保持专业客观的表述风格
    3. **完整性**: 确保答案完整回答原始查询
    4. **可读性**: 保持语言流畅，结构清晰
    
    ## 质量检查清单
    - [ ] 是否基于验证证据？
    - [ ] 是否回答了原始查询？
    - [ ] 是否保持客观中立？
    - [ ] 是否标注了不确定性？
    - [ ] 语言是否清晰流畅？
    
    修正后的答案：
    """,
    
    "比较查询": """
    作为比较分析专家，请根据验证结果重新生成一个全面准确的比较性答案。
    
    查询意图：{intent} - 比较查询  
    原始查询："{query}"
    原始答案：{original_answer}
    
    验证结果摘要：
    {verification_summary}
    
    ## 纠正要求
    1. **平衡对比**: 建立结构化的对比框架，平衡呈现各方优劣
    2. **证据支撑**: 基于证据提供具体的对比点和数据支持
    3. **客观判断**: 基于证据提供有依据的结论或建议
    4. **谨慎表述**: 对缺乏充分证据的比较点注明不确定性
    
    修正后的比较分析：
    """,
    
    "方法查询": """
    作为方法指导专家，请根据验证结果重新生成一个可操作的方法指南。
    
    查询意图：{intent} - 方法查询
    原始查询："{query}"
    原始答案：{original_answer}
    
    验证结果摘要：
    {verification_summary}
    
    ## 纠正要求
    1. **可行性**: 确保步骤的可行性、正确性和安全性
    2. **清晰指引**: 提供清晰、有序的操作指引
    3. **完善调整**: 基于验证证据调整或完善有问题的步骤
    4. **注意事项**: 包含必要的注意事项和常见问题解决方案
    
    修正后的方法指南：
    """,
    
    "观点查询": """
    作为观点综述专家，请根据验证结果重新生成一个平衡客观的观点综述。
    
    查询意图：{intent} - 观点查询
    原始查询："{query}"
    原始答案：{original_answer}
    
    验证结果摘要：
    {verification_summary}
    
    ## 纠正要求
    1. **全面呈现**: 全面、平衡地呈现不同的观点立场和论据
    2. **客观表述**: 基于证据客观表述各方观点，避免主观偏向
    3. **明确区分**: 明确区分事实性内容和观点性内容
    4. **争议说明**: 对缺乏充分证据支持的观点说明其争议性
    
    修正后的观点综述：
    """
}

# ==================== 检索增强模板 ====================
RETRIEVAL_AUGMENTED_TEMPLATE = """
基于检索到的相关知识，请对以下内容进行增强和完善。

原始内容：{original_content}
检索到的相关知识：
{retrieved_knowledge}

## 增强要求
1. **信息整合**: 将检索到的相关信息有机整合到原始内容中
2. **准确性**: 确保整合的信息准确无误
3. **连贯性**: 保持内容的连贯性和流畅性
4. **价值提升**: 通过知识整合提升内容的实用价值

增强后的内容：
"""

# ==================== 自我修正模板 ====================
SELF_REVISION_TEMPLATE = """
上下文：下面是模型最初的回答（包含声明标注）和对每条声明的验证结果。

原始回答：{original_answer}

验证结果：
{verification_results}

## 修正指南
请根据验证结果，重新生成一个修正后的答案，确保：
1. **忠于证据**: 严格基于验证证据进行修正
2. **纠正错误**: 修正所有被验证为错误或不确定的声明
3. **保持优点**: 保留原始回答中正确的部分
4. **增强表达**: 提升答案的清晰度和专业性

修正后的答案：
"""

@classmethod
def get_intent_classification_prompt(cls, query: str) -> str:
    """获取意图分类提示词"""
    return cls.INTENT_CLASSIFICATION_TEMPLATE.format(query=query)

@classmethod
def get_claim_extraction_prompt(cls, text: str) -> str:
    """获取声明提取提示词"""
    return cls.CLAIM_EXTRACTION_TEMPLATE.format(text=text)

@classmethod
def get_fact_verification_prompt(cls, intent: str, query: str, claim: str, evidence_text: str) -> str:
    """获取事实验证提示词"""
    return cls.FACT_VERIFICATION_TEMPLATE.format(
        intent=intent,
        query=query,
        claim=claim,
        evidence_text=evidence_text
    )

@classmethod
def get_correction_prompt(cls, intent: str, query: str, original_answer: str, verification_summary: str) -> str:
    """获取答案纠正提示词"""
    template = cls.CORRECTION_TEMPLATES.get(intent, cls.CORRECTION_TEMPLATES["事实查询"])
    return template.format(
        intent=intent,
        query=query,
        original_answer=original_answer,
        verification_summary=verification_summary
    )

@classmethod
def get_retrieval_augmented_prompt(cls, original_content: str, retrieved_knowledge: str) -> str:
    """获取检索增强提示词"""
    return cls.RETRIEVAL_AUGMENTED_TEMPLATE.format(
        original_content=original_content,
        retrieved_knowledge=retrieved_knowledge
    )

@classmethod
def get_self_revision_prompt(cls, original_answer: str, verification_results: str) -> str:
    """获取自我修正提示词"""
    return cls.SELF_REVISION_TEMPLATE.format(
        original_answer=original_answer,
        verification_results=verification_results
    )

@classmethod
def list_available_templates(cls) -> dict:
    """列出所有可用的模板"""
    return {
        "intent_classification": "意图分类模板",
        "claim_extraction": "声明提取模板",
        "fact_verification": "事实验证模板",
        "correction": "答案纠正模板",
        "retrieval_augmented": "检索增强模板",
        "self_revision": "自我修正模板"
    }


class TemplateValidator:
    """模板验证器 - 验证模板格式和完整性"""

@staticmethod
def validate_template(template: str, required_params: list) -> bool:
    """验证模板是否包含所有必需参数"""
    try:
        # 检查模板是否可以安全格式化
        template.format(**{param: "test" for param in required_params})
        return True
    except KeyError as e:
        print(f"模板缺少必需参数: {e}")
        return False
    except Exception as e:
        print(f"模板格式错误: {e}")
        return False

@staticmethod
def extract_template_variables(template: str) -> list:
    """提取模板中的所有变量"""
    import re
    variables = re.findall(r'\{(\w+)\}', template)
    return list(set(variables))  # 去重


# ==================== 使用示例 ====================
def usage_example():

    """使用示例"""

# 初始化模板管理器
templates = PromptTemplates()

# 示例1: 意图分类
query = "比较Python和Java在机器学习中的应用"
intent_prompt = templates.get_intent_classification_prompt(query)
print("=== 意图分类提示词 ===")
print(intent_prompt[:200] + "...")
print()

# 示例2: 声明提取
text = "Python是一种高级编程语言，由Guido van Rossum在1991年创建。"
claim_prompt = templates.get_claim_extraction_prompt(text)
print("=== 声明提取提示词 ===")
print(claim_prompt[:200] + "...")
print()

# 示例3: 答案纠正
correction_prompt = templates.get_correction_prompt(
    intent="比较查询",
    query=query,
    original_answer="Python比Java更好",
    verification_summary="验证结果摘要..."
)
print("=== 答案纠正提示词 ===")
print(correction_prompt[:200] + "...")
print()

# 列出所有可用模板
available_templates = templates.list_available_templates()
print("=== 可用模板列表 ===")
for key, description in available_templates.items():
    print(f"- {key}: {description}")


if __name__ == "__main__":
    # 运行使用示例
    usage_example()

# 验证模板完整性
validator = TemplateValidator()
template = PromptTemplates.INTENT_CLASSIFICATION_TEMPLATE
variables = validator.extract_template_variables(template)
print(f"\n=== 模板变量分析 ===")
print(f"提取到的变量: {variables}")
print(f"验证结果: {validator.validate_template(template, variables)}")
"""
#写主程里
from prompt_templates import PromptTemplates

# 快速使用
templates = PromptTemplates()

# 获取意图分类提示词
prompt = templates.get_intent_classification_prompt("你的查询内容")

# 获取答案纠正提示词
correction_prompt = templates.get_correction_prompt(
    intent="事实查询",
    query="原始问题",
    original_answer="需要验证的答案",
    verification_summary="验证结果摘要"
)
"""