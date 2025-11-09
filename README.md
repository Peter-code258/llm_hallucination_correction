# llm_hallucination_correction
基于检索增强生成(RAG)的LLM幻觉检测与纠正一体化框架，通过意图分类、证据检索、结构化验证和意图感知纠正，实现对大语言模型生成内容的幻觉检测与自动修正。
llm_hallucination_correction/
├── src/                          # 源代码目录
│   ├── __init__.py
│   ├── llm_client.py            # LLM客户端适配器
│   ├── retriever.py             # 向量检索模块
│   ├── intent_classifier.py     # 意图分类器
│   ├── claim_extractor.py       # 声明提取模块
│   ├── evidence_verifier.py     # 证据验证器
│   ├── corrector.py             # 纠正模块
│   └── orchestrator.py          # 流程协调器
├── prompts/                     # 提示词模板
│   ├── intent_classification.md
│   ├── claim_extraction.md
│   ├── verification.md
│   └── correction.md
├── config/                      # 配置文件
│   └── config.yaml
├── tests/                       # 测试代码
│   ├── test_intent_classifier.py
│   ├── test_verifier.py
│   └── test_full_pipeline.py
├── data/                        # 数据目录
│   └── knowledge_base/          # 知识库文件
├── requirements.txt             # 依赖列表
├── main.py                      # 主入口文件
└── README.md                    # 项目说明
