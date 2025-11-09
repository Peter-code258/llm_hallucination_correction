"""
RESTful API服务模块
提供HTTP接口供其他系统调用
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import json
import asyncio
from datetime import datetime
import logging

from .orchestrator import EvidenceEnhancedCorrectionOrchestrator

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义数据模型
class CorrectionRequest(BaseModel):
    query: str
    original_answer: str
    config_overrides: Optional[Dict[str, Any]] = None

class BatchCorrectionRequest(BaseModel):
    requests: List[CorrectionRequest]
    priority: str = "normal"  # low, normal, high

class CorrectionResponse(BaseModel):
    request_id: str
    status: str  # pending, processing, completed, failed
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

class HealthCheckResponse(BaseModel):
    status: str
    timestamp: datetime
    system_info: Dict[str, Any]

# 创建FastAPI应用
app = FastAPI(
    title="大语言模型幻觉检测与纠正API",
    description="基于RAG的LLM幻觉检测与纠正系统RESTful API",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
orchestrator = None
task_queue = asyncio.Queue()
processing_tasks = {}

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    global orchestrator
    try:
        # 这里应该加载配置
        from .config_loader import load_config
        config = load_config()
        orchestrator = EvidenceEnhancedCorrectionOrchestrator(config)
        logger.info("✅ 系统初始化完成")
    except Exception as e:
        logger.error(f"❌ 系统初始化失败: {e}")
        raise

@app.get("/")
async def root():
    """根端点"""
    return {
        "message": "大语言模型幻觉检测与纠正系统API",
        "version": "1.0.0",
        "status": "运行中"
    }

@app.get("/health")
async def health_check():
    """健康检查端点"""
    try:
        status = orchestrator.get_system_status() if orchestrator else {"status": "未初始化"}
        return HealthCheckResponse(
            status="healthy" if status.get("status") == "正常运行" else "unhealthy",
            timestamp=datetime.now(),
            system_info=status
        )
    except Exception as e:
        return HealthCheckResponse(
            status="unhealthy",
            timestamp=datetime.now(),
            system_info={"error": str(e)}
        )

@app.post("/correct", response_model=CorrectionResponse)
async def correct_single(request: CorrectionRequest, background_tasks: BackgroundTasks):
    """单个查询纠正端点"""
    request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(request)}"
    
    try:
        if not orchestrator:
            raise HTTPException(status_code=503, detail="系统未就绪")
        
        # 执行纠正
        result = orchestrator.process_correction(request.query, request.original_answer)
        
        response = CorrectionResponse(
            request_id=request_id,
            status="completed",
            result=result,
            created_at=datetime.now(),
            completed_at=datetime.now()
        )
        
        logger.info(f"✅ 请求 {request_id} 处理完成")
        return response
        
    except Exception as e:
        logger.error(f"❌ 请求 {request_id} 处理失败: {e}")
        return CorrectionResponse(
            request_id=request_id,
            status="failed",
            error=str(e),
            created_at=datetime.now()
        )

@app.post("/correct/batch", response_model=Dict[str, Any])
async def correct_batch(request: BatchCorrectionRequest):
    """批量纠正端点"""
    batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="系统未就绪")
    
    results = []
    for i, req in enumerate(request.requests):
        try:
            result = orchestrator.process_correction(req.query, req.original_answer)
            results.append({
                "index": i,
                "success": True,
                "result": result
            })
        except Exception as e:
            results.append({
                "index": i,
                "success": False,
                "error": str(e)
            })
    
    return {
        "batch_id": batch_id,
        "total_requests": len(request.requests),
        "successful": len([r for r in results if r["success"]]),
        "failed": len([r for r in results if not r["success"]]),
        "results": results
    }

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """获取任务状态"""
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return processing_tasks[task_id]

@app.get("/system/status")
async def get_system_status():
    """获取系统状态"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="系统未就绪")
    
    return orchestrator.get_system_status()

@app.post("/knowledge/add")
async def add_knowledge_documents(documents: List[str], metadatas: Optional[List[Dict]] = None):
    """向知识库添加文档"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="系统未就绪")
    
    try:
        orchestrator.add_knowledge_documents(
            documents, 
            metadatas or [{}] * len(documents)
        )
        return {"message": f"成功添加 {len(documents)} 个文档到知识库"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加文档失败: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "src.api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )