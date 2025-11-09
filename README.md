# 大语言模型幻觉检测与纠正系统

基于检索增强生成(RAG)的LLM幻觉检测与纠正一体化框架，通过意图分类、证据检索、结构化验证和意图感知纠正，实现对大语言模型生成内容的自动检测与修正。

## 核心特性

- 🎯 **意图感知**: 自动识别查询意图，针对性处理
- 🔍 **证据驱动**: 基于权威知识库进行事实核查
- ✅ **结构化验证**: 标准化的声明验证流程
- ✏️ **智能纠正**: 意图特定的答案重写策略
- 📊 **可解释性**: 完整的验证轨迹和纠正依据
- 🚀 **高性能**: 模块化设计，支持批量处理

## 系统架构
用户查询 → 意图分类 → 声明提取 → 证据检索 → 声明验证 → 答案纠正 → 最终答案
## 快速开始

### 环境要求

- Python 3.8+
- 至少8GB内存
- 网络连接（用于API调用）

### 安装步骤

1. **克隆项目**
bash
git clone https://github.com/your-org/llm-hallucination-correction.git
cd llm-hallucination-correction
2. **安装依赖**
bash
pip install -r requirements.txt
3. **配置环境变量**
bash
cp .env.example .env
编辑 .env 文件，配置API密钥等参数
4. **初始化知识库**
python
from src.orchestrator import EvidenceEnhancedCorrectionOrchestrator
from config_loader import load_config
config = load_config()
orchestrator = EvidenceEnhancedCorrectionOrchestrator(config)
添加示例知识文档
documents = [
"大语言模型幻觉是指生成不真实内容的现象。",
"检索增强生成(RAG)可以减轻幻觉问题。",
# 更多文档...
]
orchestrator.add_knowledge_documents(documents)
5. **运行测试**
bash
pytest tests/ -v
### 基本使用
python
from src.orchestrator import EvidenceEnhancedCorrectionOrchestrator
from config_loader import load_config
初始化系统
config = load_config()
orchestrator = EvidenceEnhancedCorrectionOrchestrator(config)
执行纠正
query = "什么是机器学习？"
original_answer = "机器学习是人工智能的一个分支，它能让计算机通过经验自动改进..."
result = orchestrator.process_correction(query, original_answer)
if result['success']:
print(f"纠正后的答案: {result['corrected_answer']}")
print(f"支持率: {result['analysis_results']['correction_summary']['support_ratio']:.2%}")
### API服务

启动RESTful API服务：
bash
python src/api_server.py
API文档访问: http://localhost:8000/docs

## 配置说明

### 主要配置项

- `llm.provider`: LLM服务提供商 (openai/deepseek)
- `llm.api_key`: API密钥
- `vector_db.embedding_model`: 嵌入模型路径
- `retrieval.similarity_threshold`: 检索相似度阈值

### 性能调优

- 调整`MAX_CONCURRENT_REQUESTS`控制并发数
- 配置`BATCH_SIZE`优化批量处理
- 设置`CACHE_TTL`启用结果缓存

## 项目结构

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
├── monitoring/                  # 监控配置
│   └── prometheus.yml           #监控和指标配置
└── README.md                    # 项目说明
## 开发指南

### 添加新的意图类型

1. 在`intent_classifier.py`中扩展支持列表
2. 在`corrector.py`中添加对应的纠正模板
3. 更新提示词模板文件

### 自定义验证规则

修改`evidence_verifier.py`中的验证逻辑，或扩展`verification.md`模板。

### 性能监控

系统内置Prometheus指标，可通过Grafana可视化监控数据。

## 贡献指南

我们欢迎各种形式的贡献！请阅读[CONTRIBUTING.md](docs/CONTRIBUTING.md)了解详情。

## 许可证

本项目采用MIT许可证。详见[LICENSE](LICENSE)文件。

## 引用

如果您在研究中使用了本项目，请引用：
bibtex
@software{llm_hallucination_correction2024,
title = {大语言模型幻觉检测与纠正系统},

## 技术支持

下次写(ง •_•)ง
