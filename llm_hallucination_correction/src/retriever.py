import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
import os

class VectorRetriever:
    """基于向量数据库的检索器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_model = None
        self.client = None
        self.collection = None
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化嵌入模型和向量数据库"""
        # 初始化嵌入模型
        self.embedding_model = SentenceTransformer(
            self.config.get('embedding_model', 'BAAI/bge-base-en')
        )
        
        # 初始化ChromaDB
        db_path = self.config.get('db_path', './data/vector_db')
        os.makedirs(db_path, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=db_path)
        
        # 获取或创建集合
        collection_name = self.config.get('collection_name', 'knowledge_base')
        try:
            self.collection = self.client.get_collection(collection_name)
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "LLM幻觉检测知识库"}
            )
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None):
        """添加文档到知识库"""
        if not documents:
            return
        
        if metadatas is None:
            metadatas = [{}] * len(documents)
        
        # 生成嵌入向量
        embeddings = self.embedding_model.encode(documents).tolist()
        
        # 生成文档ID
        doc_ids = [f"doc_{i}" for i in range(len(documents))]
        
        # 添加到集合
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=doc_ids
        )
    
    def search(self, query: str, n_results: int = 5, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """检索相关文档"""
        # 生成查询嵌入
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # 执行检索
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        # 处理结果
        snippets = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0], 
            results['metadatas'][0], 
            results['distances'][0]
        )):
            similarity = 1 - distance  # 转换为相似度分数
            if similarity >= similarity_threshold:
                snippets.append({
                    "text": doc,
                    "source": metadata.get('source', 'unknown'),
                    "similarity": similarity,
                    "rank": i + 1
                })
        
        return snippets
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        return {
            "count": self.collection.count(),
            "name": self.collection.name
        }