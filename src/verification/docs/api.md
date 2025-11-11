# API接口文档

## 概述
本文档描述了大语言模型幻觉检测与纠正系统的API接口，包括单次查询处理、批量处理和系统状态检查等功能。

## 基础信息
- **基础URL**: `http://localhost:8000` (默认)
- **内容类型**: `application/json`
- **认证方式**: API密钥（通过Header传递）

## 接口列表

### 1. 健康检查
检查系统是否正常运行。

**端点**: `GET /health`

**响应**:
项目文档与脚本完善
根据您的需求，我为您完善了项目中的文档文件和脚本代码。以下是完整的文件内容：
1. API接口文档 (docs/api.md)
# API接口文档

## 概述
本文档描述了大语言模型幻觉检测与纠正系统的API接口，包括单次查询处理、批量处理和系统状态检查等功能。

## 基础信息
- **基础URL**: `http://localhost:8000` (默认)
- **内容类型**: `application/json`
- **认证方式**: API密钥（通过Header传递）

## 接口列表

### 1. 健康检查
检查系统是否正常运行。

**端点**: `GET /health`

**响应**:

"""
{{
"status": "healthy",
"timestamp": "2024-03-20T10:30:00Z",
"version": "1.0.0"
}}
"""

### 2. 系统状态
获取系统详细状态信息。

**端点**: `GET /system/status`

**响应**:

"""
{{
"status": "running",
"components": {
"llm_client": "active",
"vector_retriever": "active",
"intent_classifier": "active",
"evidence_verifier": "active",
"answer_corrector": "active"
},
"timestamp": "2024-03-20T10:30:00Z"
}}
"""

### 3. 单次查询处理
处理单个查询并返回纠正后的答案。

**端点**: `POST /api/process`

**请求体**:

"""
{{
"query": "机器学习的基本概念是什么？",
"context": "可选上下文信息",
"config_overrides": {
"max_tokens": 1000,
"temperature": 0.1
}
}}
"""

**响应**:

"""
{{
"success": true,
"query": "机器学习的基本概念是什么？",
"results": {
"initial_answer": "机器学习是人工智能的一个分支...",
"corrected_answer": "机器学习是人工智能的一个分支，专注于...",
"intent": "事实查询",
"verifications": [
{
"claim": "机器学习是人工智能的一个分支",
"verdict": "SUPPORTED",
"confidence": 0.95,
"supporting_evidence": [...]
}
],
"hallucination_analysis": {
"has_hallucination": false,
"confidence": 0.92
}
},
"processing_metadata": {
"total_duration": 3.2,
"steps_completed": ["initial_answer", "intent_classification", "claim_extraction", "evidence_retrieval", "verification", "correction"],
"timestamp": "2024-03-20T10:30:00Z"
}
}}
"""

### 4. 批量查询处理
批量处理多个查询。

**端点**: `POST /api/batch-process`

**请求体**:

"""
{{
"queries": [
"查询1",
"查询2",
"查询3"
],
"context": "共享上下文信息",
"priority": "normal"
}}
"""

**响应**:

"""
{{
"batch_id": "batch_20240320103000",
"total_queries": 3,
"successful": 3,
"failed": 0,
"results": [
{
"query": "查询1",
"success": true,
"result": {...}
},
{
"query": "查询2",
"success": true,
"result": {...}
}
],
"processing_metadata": {
"total_duration": 8.7,
"average_duration_per_query": 2.9,
"timestamp": "2024-03-20T10:30:00Z"
}
}}
"""

### 5. 知识库管理
管理系统的知识库文档。

**端点**: `POST /api/knowledge/add`

**请求体**:
"""
{{
"documents": [
{
"content": "文档内容1",
"metadata": {"source": "wiki", "type": "definition"}
},
{
"content": "文档内容2",
"metadata": {"source": "paper", "type": "technical"}
}
]
}}
"""

**响应**:
"""
{{
"success": true,
"added_documents": 2,
"total_documents": 150,
"timestamp": "2024-03-20T10:30:00Z"
}}
"""

## 错误处理
所有API接口使用标准HTTP状态码：

- `200`: 成功
- `400`: 请求参数错误
- `500`: 服务器内部错误

错误响应格式：
{{
"error": "错误描述",
"code": "错误代码",
"details": "详细错误信息"
}}

## 速率限制
- 默认速率限制：每分钟60次请求
- 批量处理：每分钟5次批量请求

## 客户端示例

### Python客户端

"""
import requests
import json
class HallucinationCorrectionClient:
def init(self, base_url="http://localhost:8000", api_key=None):
self.base_url = base_url
self.headers = {
"Content-Type": "application/json",
"Authorization": f"Bearer {api_key}" if api_key else None
}
def process_query(self, query, context=None):
    payload = {"query": query}
    if context:
        payload["context"] = context
    
    response = requests.post(
        f"{self.base_url}/api/process",
        headers=self.headers,
        json=payload
    )
    return response.json()

def batch_process(self, queries, context=None):
    payload = {"queries": queries}
    if context:
        payload["context"] = context
    
    response = requests.post(
        f"{self.base_url}/api/batch-process",
        headers=self.headers,
        json=payload
    )
    return response.json()
"""

"""
使用示例
client = HallucinationCorrectionClient(api_key="your_api_key")
result = client.process_query("机器学习是什么？")
print(result)
"""