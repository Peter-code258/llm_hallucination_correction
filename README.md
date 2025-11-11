# 大语言模型幻觉检测与纠正系统

基于检索增强生成(RAG)的LLM幻觉检测与纠正一体化框架，通过意图分类、证据检索、结构化验证和意图感知纠正，实现对大语言模型生成内容的自动检测与修正。

## 核心特性

- 🎯 **意图感知**: 自动识别查询意图，针对性处理
- 🔍 **证据驱动**: 基于权威知识库进行事实核查
- ✅ **结构化验证**: 标准化的声明验证流程
- ✏️ **智能纠正**: 意图特定的答案重写策略
- 📊 **可解释性**: 完整的验证轨迹和纠正依据
- 🚀 **高性能**: 模块化设计，支持批量处理

## 项目结构
```
llm-hallucination-correction/
├── src/                          # 源代码目录
│   ├── core/                     # 核心流程控制
│   │   ├── __init__.py
│   │   └── orchestrator.py       # 流程协调器（主控制器）
│   ├── llm/                      # LLM相关模块
│   │   ├── __init__.py
│   │   ├── llm_client.py         # LLM客户端适配器（多提供商支持）
│   │   └── prompt_templates.py   # Prompt模板管理系统
│   ├── retrieval/                # 检索模块
│   │   ├── __init__.py
│   │   └── vector_retriever.py   # 向量检索器（基于ChromaDB）
│   ├── verification/             # 验证模块
│   │   ├── __init__.py
│   │   ├── intent_classifier.py  # 意图分类器
│   │   ├── claim_extractor.py    # 声明提取器
│   │   └── evidence_verifier.py  # 证据验证器
│   └── correction/              # 纠正模块
│       ├── __init__.py
│       └── answer_corrector.py   # 答案纠正器
├── config/                       # 配置文件目录
│   ├── __init__.py
│   └── config.yaml              # 主配置文件
├── data/                         # 数据目录
│   ├── knowledge_base/          # 知识库文档（原始文本）
│   └── vector_db/               # 向量数据库存储
├── tests/                       # 测试代码目录
│   ├── __init__.py
│   ├── test_llm_client.py       # LLM客户端测试
│   ├── test_retrieval.py        # 检索模块测试
│   └── test_full_pipeline.py   # 完整流程测试
├── docs/                        # 文档目录
│   ├── api.md                  # API接口文档
│   └── deployment.md           # 部署指南
├── scripts/                     # 脚本目录
│   ├── setup_knowledge_base.py # 知识库初始化脚本
│   └── batch_processing.py     # 批量处理脚本
├── main.py                      # 主入口文件
├── requirements.txt             # Python依赖列表
├── .env.example                 # 环境变量示例文件
├── .gitignore                  # Git忽略规则
└── README.md                   # 项目说明文档
```

## 🚀 快速开始

环境要求

 - Python: 3.8 或更高版本
 - 内存: 至少 8GB RAM
 - 存储: 至少 10GB 可用空间
 - 网络: 稳定的互联网连接（用于API调用）

## 安装步骤

克隆项目仓库 
```
git clone https://github.com/your-org/llm-hallucination-correction.git
cd llm-hallucination-correction
```
创建虚拟环境（推荐）
```
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
安装依赖包
pip install -r requirements.txt
配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入您的API密钥和其他配置
初始化知识库
python scripts/setup_knowledge_base.py
基本使用
命令行交互模式
python main.py --interactive
处理单个查询
python main.py --query "机器学习的基本概念是什么？"
批量处理文件中的查询
python main.py --file queries.txt --export json
检查系统状态
python main.py --status
```

## ⚙️ 配置说明

主要配置项:

配置文件位于 config/config.yaml，主要包含以下部分：

LLM提供商配置
```
llm:
  provider: "deepseek"           # 或 "openai", "anthropic"
  api_key: "${DEEPSEEK_API_KEY}" # 从环境变量读取
  model: "deepseek-chat"
  base_url: "https://api.deepseek.com/v1"
  temperature: 0.1
  max_tokens: 1000
```

向量数据库配置
```
vector_db:
  embedding_model: "BAAI/bge-base-en"
  db_path: "./data/vector_db"
  collection_name: "knowledge_base"
```

检索配置
```
retrieval:
  similarity_threshold: 0.7
  max_retrieved_docs: 5
```

验证配置
```
verification:
  confidence_threshold: 0.8
  max_verification_attempts: 3
```

环境变量配置

创建 .env文件并配置以下变量：

API密钥配置
```
DEEPSEEK_API_KEY=your_deepseek_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

系统配置
```
LOG_LEVEL=INFO
MAX_CONCURRENT_REQUESTS=5
REQUEST_TIMEOUT=30
```

知识库配置
```
KNOWLEDGE_BASE_PATH=./data/knowledge_base
VECTOR_DB_PATH=./data/vector_db
```

## 🧩 核心模块详解

 - LLM客户端适配器 (src/llm/llm_client.py),支持多提供商（DeepSeek、OpenAI、Anthropic等）

统一的API调用接口

内置重试机制和错误处理

使用示例：

```
from src.llm.llm_client import LLMAdapter
llm = LLMAdapter("deepseek", config)
response = llm.call("你的提示词", max_tokens=500)
```

 - Prompt模板管理系统 (src/llm/prompt_templates.py)
   
集中管理所有提示词模板

支持意图分类、声明提取、事实验证、答案纠正等场景

意图特定的模板定制

使用示例：
```
from src.llm.prompt_templates import PromptTemplates

templates = PromptTemplates()
prompt = templates.get_intent_classification_prompt("你的查询")
```

 - 向量检索器 (src/retrieval/vector_retriever.py)
   
基于ChromaDB的向量相似性检索

支持多种嵌入模型（Sentence Transformers）

可配置的相似度阈值和结果数量

使用示例：
```
from src.retrieval.vector_retriever import VectorRetriever

retriever = VectorRetriever(config)
results = retriever.search("查询文本", n_results=5)
```
 - 流程协调器 (src/core/orchestrator.py)

整合所有模块的完整工作流

状态管理和错误处理

性能监控和日志记录

使用示例：
```
from src.core.orchestrator import EvidenceEnhancedCorrectionOrchestrator

orchestrator = EvidenceEnhancedCorrectionOrchestrator(config)
result = orchestrator.process_query("你的查询")
```

# 📊 API接口
单次查询处理
```
POST /api/process
Content-Type: application/json

{
  "query": "机器学习的基本概念是什么？",
  "context": "可选上下文信息"
}
响应示例：
{
  "success": true,
  "query": "机器学习的基本概念是什么？",
  "results": {
    "initial_answer": "初始回答内容...",
    "corrected_answer": "纠正后回答内容...",
    "intent": "事实查询",
    "verifications": [...],
    "hallucination_analysis": {...}
  },
  "processing_metadata": {...}
}
批量处理接口
POST /api/batch-process
Content-Type: application/json

{
  "queries": ["查询1", "查询2", "查询3"],
  "context": "共享上下文信息"
}
```
# 🧪 测试与验证
运行单元测试 
```
# 运行所有测试
python -m pytest tests/

# 运行特定测试模块
python -m pytest tests/test_llm_client.py -v

# 带覆盖率报告
python -m pytest --cov=src tests/
测试数据准备
创建测试查询文件 test_queries.txt：
什么是机器学习？
Python和Java的主要区别是什么？
如何学习深度学习？
人工智能的未来发展前景
性能基准测试
系统包含性能测试脚本，可评估处理速度和资源使用：
python tests/performance_benchmark.py --queries 100 --workers 4
```
# 🚢 部署指南
本地部署

安装依赖：```pip install -r requirements.txt```

配置环境变量

初始化知识库

启动服务：```python main.py --interactive或使用WSGI服务器```

Docker部署
```
FROM python:3.9-slim

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

EXPOSE 8000
CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8000"]
```

构建和运行：
```
docker build -t llm-hallucination-correction .
docker run -p 8000:8000 llm-hallucination-correction
```
云部署建议

AWS: 使用EC2或Lambda + S3存储知识库

Azure: Azure Functions + Blob Storage

GCP: Cloud Functions + Cloud Storage

建议配置：至少4核CPU，8GB内存，50GB存储

# 📈 性能指标

系统在标准硬件配置下的性能表现：

单次查询处理时间: 2-5秒（取决于查询复杂度和证据检索时间）

并发处理能力: 支持5-10个并发查询

准确率: 85-95%（基于验证结果置信度）

幻觉检测率: 可识别80%以上的常见幻觉类型

# 🤝 贡献指南
我们欢迎社区贡献！请遵循以下步骤：

Fork本项目

创建特性分支：```git checkout -b feature/AmazingFeature```

提交更改：```git commit -m 'Add AmazingFeature'```

推送到分支：```git push origin feature/AmazingFeature```

提交Pull Request

开发规范

遵循PEP 8代码风格

编写适当的单元测试

更新相关文档

使用类型提示（Type Hints）

添加新功能

在相应模块中实现新功能

添加测试用例

更新配置模板（如需要）

文档化新功能的使用方法

# 📝 常见问题解答
Q: 如何添加新的LLM提供商？

A: 在 src/llm/llm_client.py中的 LLMAdapter类添加新提供商的支持，实现相应的API调用逻辑。


Q: 如何扩展知识库？

A: 将文档放入 data/knowledge_base/目录，然后运行 python scripts/setup_knowledge_base.py重新初始化向量数据库。


Q: 如何处理大量并发请求？

A: 调整配置中的 max_concurrent_requests参数，并考虑使用异步处理或部署多个实例。


Q: 如何自定义验证规则？

A: 修改 src/verification/evidence_verifier.py中的验证逻辑，或调整置信度阈值配置。


# 📜 许可证 

本项目采用 MIT 许可证 - 详见 LICENSE文件。

# 🏆 致谢

感谢以下开源项目的贡献：

ChromaDB：向量数据库解决方案

Sentence Transformers：文本嵌入模型

FastAPI：高性能API框架

Pytest：测试框架

# 🔄 版本历史

v1.0.0​ (2024-03-20)

初始版本发布

支持DeepSeek和OpenAI提供商

完整的幻觉检测与纠正流程

基础API接口

注意: 本项目仍在积极开发中，API和配置格式可能发生变化。建议定期查看更新日志和文档。

下次写(d •_•)d
