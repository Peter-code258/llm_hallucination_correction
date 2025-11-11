#!/usr/bin/env python3
"""
知识库初始化脚本
用于加载文档、生成嵌入向量并初始化向量数据库
"""

import os
import sys
import json
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Any

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.vector_retriever import VectorRetriever
from src.config.config_loader import load_config

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/knowledge_setup.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def load_sample_documents() -> List[Dict[str, Any]]:
    """加载示例文档"""
    sample_docs = [
        {
            'content': '机器学习是人工智能的一个分支，专注于算法和统计模型，使计算机能够从数据中学习而不需要明确编程。',
            'metadata': {'source': 'wikipedia', 'type': 'definition', 'topic': 'machine_learning'}
        },
        {
            'content': '深度学习是基于神经网络架构的机器学习方法，能够自动学习特征表示，在图像识别和自然语言处理中表现出色。',
            'metadata': {'source': 'research_paper', 'type': 'technical', 'topic': 'deep_learning'}
        },
        {
            'content': 'Python是一种高级编程语言，由Guido van Rossum在1991年创建，具有简单易学的语法特点。',
            'metadata': {'source': 'python_org', 'type': 'definition', 'topic': 'programming'}
        },
        {
            'content': '神经网络受人脑结构启发，由相互连接的节点（神经元）组成，能够通过训练学习复杂模式。',
            'metadata': {'source': 'textbook', 'type': 'technical', 'topic': 'neural_networks'}
        },
        {
            'content': '自然语言处理（NLP）是人工智能的一个领域，专注于计算机和人类语言之间的交互。',
            'metadata': {'source': 'academic', 'type': 'definition', 'topic': 'nlp'}
        },
        {
            'content': 'Transformer架构是自然语言处理中的突破性技术，基于自注意力机制，能够高效处理序列数据。',
            'metadata': {'source': 'research_paper', 'type': 'technical', 'topic': 'transformer'}
        },
        {
            'content': '大语言模型（LLM）是基于Transformer架构的大规模神经网络，能够生成类似人类的文本。',
            'metadata': {'source': 'tech_blog', 'type': 'definition', 'topic': 'llm'}
        },
        {
            'content': '幻觉是指大语言模型生成不准确、虚构或缺乏证据支持的信息的现象。',
            'metadata': {'source': 'research_paper', 'type': 'definition', 'topic': 'hallucination'}
        },
        {
            'content': '检索增强生成（RAG）通过结合检索系统和生成模型，减少幻觉并提高回答准确性。',
            'metadata': {'source': 'academic', 'type': 'technical', 'topic': 'rag'}
        },
        {
            'content': 'BERT是Google开发的Transformer模型，通过双向编码器表示实现上下文理解。',
            'metadata': {'source': 'research_paper', 'type': 'technical', 'topic': 'bert'}
        }
    ]
    return sample_docs

def load_documents_from_directory(directory_path: str) -> List[Dict[str, Any]]:
    """从目录加载文档"""
    docs = []
    directory = Path(directory_path)
    
    if not directory.exists():
        logger.warning(f"文档目录不存在: {directory_path}")
        return []
    
    supported_extensions = ['.txt', '.md', '.json']
    
    for file_path in directory.rglob('*'):
        if file_path.suffix.lower() in supported_extensions:
            try:
                if file_path.suffix.lower() == '.json':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = json.load(f)
                        if isinstance(content, list):
                            docs.extend(content)
                        else:
                            docs.append(content)
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            docs.append({
                                'content': content,
                                'metadata': {
                                    'source': file_path.name,
                                    'type': 'document',
                                    'file_path': str(file_path)
                                }
                            })
            except Exception as e:
                logger.error(f"读取文件失败 {file_path}: {e}")
    
    return docs

def initialize_knowledge_base(config: Dict[str, Any], documents: List[Dict[str, Any]]) -> bool:
    """初始化知识库"""
    try:
        # 初始化向量检索器
        retriever = VectorRetriever(config['vector_db'])
        logger.info("✅ 向量检索器初始化成功")
        
        # 准备文档内容
        documents_content = []
        metadatas = []
        
        for doc in documents:
            documents_content.append(doc['content'])
            metadatas.append(doc.get('metadata', {}))
        
        # 添加文档到向量数据库
        retriever.add_documents(documents_content, metadatas)
        logger.info(f"✅ 成功添加 {len(documents)} 个文档到知识库")
        
        # 验证知识库状态
        stats = retriever.get_collection_stats()
        logger.info(f"📊 知识库统计: {stats['count']} 个文档")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 知识库初始化失败: {e}")
        return False

def main():
    """主函数"""
    global logger
    logger = setup_logging()
    
    logger.info("🚀 开始初始化知识库")
    
    try:
        # 加载配置
        config = load_config()
        logger.info("✅ 配置文件加载成功")
        
        # 确定文档来源
        knowledge_base_dir = Path('data/knowledge_base')
        documents = []
        
        if knowledge_base_dir.exists() and any(knowledge_base_dir.iterdir()):
            # 从目录加载文档
            logger.info("📁 从知识库目录加载文档")
            documents = load_documents_from_directory('data/knowledge_base')
        else:
            # 使用示例文档
            logger.info("📝 使用示例文档初始化")
            documents = load_sample_documents()
            
            # 创建知识库目录并保存示例文档
            knowledge_base_dir.mkdir(parents=True, exist_ok=True)
            sample_file = knowledge_base_dir / 'sample_documents.json'
            with open(sample_file, 'w', encoding='utf-8') as f:
                json.dump(documents, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 示例文档已保存到: {sample_file}")
        
        if not documents:
            logger.error("❌ 没有可用的文档")
            return False
        
        logger.info(f"📄 找到 {len(documents)} 个文档")
        
        # 初始化知识库
        success = initialize_knowledge_base(config, documents)
        
        if success:
            logger.info("🎉 知识库初始化完成")
            return True
        else:
            logger.error("❌ 知识库初始化失败")
            return False
            
    except Exception as e:
        logger.error(f"❌ 初始化过程出错: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)