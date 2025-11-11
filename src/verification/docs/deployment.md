# 部署指南

## 环境要求

### 硬件要求
- **内存**: 至少8GB RAM
- **存储**: 至少10GB可用空间
- **CPU**: 4核以上推荐

### 软件要求
- **操作系统**: Linux, macOS, Windows (WSL2推荐用于Windows)
- **Python**: 3.8或更高版本
- **包管理**: pip 20.0以上

## 本地部署

### 1. 克隆项目

"""
git clone https://github.com/your-org/llm-hallucination-correction.git

cd llm-hallucination-correction
"""
### 2. 创建虚拟环境
"""
使用venv

python -m venv venv

source venv/bin/activate # Linux/macOS

或

venv\Scripts\activate # Windows

使用conda（可选）

conda create -n llm-hallucination python=3.9

conda activate llm-hallucination
"""
### 3. 安装依赖
"""
pip install -r requirements.txt
"""
### 4. 环境配置
"""
复制环境变量模板

cp .env.example .env

编辑.env文件，配置API密钥和其他设置

DEEPSEEK_API_KEY=your_actual_api_key

OPENAI_API_KEY=your_actual_api_key
"""
### 5. 初始化知识库
"""
python scripts/setup_knowledge_base.py
"""
### 6. 启动服务
"""
开发模式

python main.py --interactive

或作为API服务启动

python main.py --host 0.0.0.0 --port 8000
"""
## Docker部署

### 1. 构建Docker镜像
"""
Dockerfile

FROM python:3.9-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8000"]
"""
"""
docker build -t llm-hallucination-correction .
"""
### 2. 运行容器
"""
docker run -d \

-p 8000:8000 \

-v $(pwd)/data:/app/data \

-v $(pwd)/logs:/app/logs \

--env-file .env \

--name llm-correction \

llm-hallucination-correction
"""
### 3. 使用Docker Compose
"""
docker-compose.yml

version: '3.8'

services:

llm-correction:

build: .

ports:

"8000:8000"

volumes:

./data:/app/data

./logs:/app/logs

env_file:

.env

restart: unless-stopped
"""
"""
docker-compose up -d
"""
## 云平台部署

### AWS部署
1. 创建EC2实例（推荐t3.medium或更大）
2. 安装Docker和Docker Compose
3. 克隆项目并配置环境变量
4. 使用Docker Compose启动服务

### Azure部署
1. 创建Azure Container Instance
2. 构建并推送Docker镜像到Azure Container Registry
3. 部署容器实例

### Google Cloud部署
1. 创建Google Cloud Run服务
2. 构建并推送Docker镜像
3. 部署到Cloud Run

## 生产环境配置

### 1. 环境变量配置
"""
API配置

DEEPSEEK_API_KEY=your_production_api_key

OPENAI_API_KEY=your_backup_api_key

系统配置

LOG_LEVEL=INFO

MAX_CONCURRENT_REQUESTS=10

REQUEST_TIMEOUT=30

数据库配置

VECTOR_DB_PATH=/app/data/vector_db

KNOWLEDGE_BASE_PATH=/app/data/knowledge_base

性能配置

WORKER_COUNT=4

MAX_BATCH_SIZE=100

CACHE_ENABLED=true
"""
### 2. 反向代理配置（Nginx）
"""
/etc/nginx/sites-available/llm-correction

server {

listen 80;

server_name your-domain.com;

location / {

    proxy_pass http://localhost:8000;

    proxy_set_header Host $host;

    proxy_set_header X-Real-IP $remote_addr;

    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

}
"""
# 静态文件服务（可选）
"""
location /static/ {
    alias /app/static/;
}
"""
### 3. 系统服务配置（Systemd）
"""
/etc/systemd/system/llm-correction.service

[Unit]

Description=LLM Hallucination Correction Service

After=network.target

[Service]

User=llm-user

Group=llm-group

WorkingDirectory=/app/llm-hallucination-correction

EnvironmentFile=/app/llm-hallucination-correction/.env

ExecStart=/app/llm-hallucination-correction/venv/bin/python main.py --host 0.0.0.0 --port 8000

Restart=always

RestartSec=5

[Install]

WantedBy=multi-user.target
"""
## 监控与维护

### 1. 日志管理
- 日志位置：`logs/system.log`
- 日志轮转：使用logrotate配置
- 监控：集成Prometheus/Grafana

### 2. 性能监控
"""
监控CPU和内存使用

top -p $(pgrep -f "python main.py")

监控API响应时间

curl -o /dev/null -s -w "%{time_total}\n" http://localhost:8000/health
"""
### 3. 备份策略
- 定期备份向量数据库：`data/vector_db`
- 备份知识库文档：`data/knowledge_base`
- 备份配置文件：`config/`

## 故障排除

### 常见问题
1. **API密钥错误**：检查.env文件中的API密钥配置
2. **内存不足**：增加系统内存或减少并发请求数
3. **向量数据库损坏**：删除并重新初始化向量数据库

### 获取帮助
- 查看日志：`tail -f logs/system.log`
- 社区支持：GitHub Issues
- 文档：本项目README.md