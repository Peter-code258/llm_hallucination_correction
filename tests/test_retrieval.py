#!/usr/bin/env python3
"""
检索模块测试 - 向量检索器功能测试
测试VectorRetriever类的初始化、文档添加、搜索和性能
"""

import pytest
import sys
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.vector_retriever import VectorRetriever

class TestVectorRetriever:
    """向量检索器测试类"""
    
    def setup_method(self):
        """测试方法前的设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'embedding_model': 'BAAI/bge-base-en',
            'db_path': self.temp_dir,
            'collection_name': 'test_collection'
        }
    
    def teardown_method(self):
        """测试方法后的清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('src.retrieval.vector_retriever.SentenceTransformer')
    @patch('src.retrieval.vector_retriever.chromadb')
    def test_initialization(self, mock_chroma, mock_sentence_transformer):
        """测试检索器初始化"""
        # 模拟嵌入模型
        mock_embedding = Mock()
        mock_embedding.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_sentence_transformer.return_value = mock_embedding
        
        # 模拟ChromaDB客户端
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_chroma.PersistentClient.return_value = mock_client
        
        # 初始化检索器
        retriever = VectorRetriever(self.config)
        
        # 验证初始化
        assert retriever.config == self.config
        assert retriever.embedding_model is not None
        assert retriever.client is not None
        assert retriever.collection is not None
        
        # 验证方法调用
        mock_sentence_transformer.assert_called_once_with('BAAI/bge-base-en')
        mock_chroma.PersistentClient.assert_called_once_with(path=self.temp_dir)
        mock_client.get_collection.assert_called_once_with('test_collection')
    
    @patch('src.retrieval.vector_retriever.SentenceTransformer')
    @patch('src.retrieval.vector_retriever.chromadb')
    def test_add_documents(self, mock_chroma, mock_sentence_transformer):
        """测试添加文档功能"""
        # 模拟嵌入模型
        mock_embedding = Mock()
        mock_embedding.encode.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_sentence_transformer.return_value = mock_embedding
        
        # 模拟ChromaDB
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_client.create_collection.return_value = mock_collection
        mock_chroma.PersistentClient.return_value = mock_client
        
        # 初始化检索器
        retriever = VectorRetriever(self.config)
        
        # 准备测试文档
        documents = [
            '机器学习是人工智能的一个分支',
            '深度学习基于神经网络架构'
        ]
        metadatas = [
            {'source': 'wiki', 'type': 'definition'},
            {'source': 'research', 'type': 'technical'}
        ]
        
        # 添加文档
        retriever.add_documents(documents, metadatas)
        
        # 验证方法调用
        mock_embedding.encode.assert_called_once_with(documents)
        mock_collection.add.assert_called_once()
        
        # 验证调用参数
        call_args = mock_collection.add.call_args
        assert 'embeddings' in call_args.kwargs
        assert 'documents' in call_args.kwargs
        assert 'metadatas' in call_args.kwargs
        assert len(call_args.kwargs['documents']) == 2
    
    @patch('src.retrieval.vector_retriever.SentenceTransformer')
    @patch('src.retrieval.vector_retriever.chromadb')
    def test_search_functionality(self, mock_chroma, mock_sentence_transformer):
        """测试搜索功能"""
        # 模拟嵌入模型
        mock_embedding = Mock()
        mock_embedding.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_sentence_transformer.return_value = mock_embedding
        
        # 模拟ChromaDB响应
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.query.return_value = {
            'documents': [['机器学习是AI分支', '深度学习基于神经网络']],
            'metadatas': [[{'source': 'wiki'}, {'source': 'research'}]],
            'distances': [[0.1, 0.2]]
        }
        mock_client.get_collection.return_value = mock_collection
        mock_chroma.PersistentClient.return_value = mock_client
        
        # 初始化检索器
        retriever = VectorRetriever(self.config)
        
        # 执行搜索
        query = "什么是机器学习"
        results = retriever.search(query, n_results=2)
        
        # 验证结果
        assert len(results) == 2
        assert all('text' in result for result in results)
        assert all('source' in result for result in results)
        assert all('similarity' in result for result in results)
        
        # 验证方法调用
        mock_embedding.encode.assert_called_with([query])
        mock_collection.query.assert_called_once()
    
    @patch('src.retrieval.vector_retriever.SentenceTransformer')
    @patch('src.retrieval.vector_retriever.chromadb')
    def test_search_with_similarity_threshold(self, mock_chroma, mock_sentence_transformer):
        """测试带相似度阈值的搜索"""
        # 模拟嵌入模型
        mock_embedding = Mock()
        mock_embedding.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_sentence_transformer.return_value = mock_embedding
        
        # 模拟ChromaDB响应（包含低相似度结果）
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.query.return_value = {
            'documents': [['相关文档', '不相关文档']],
            'metadatas': [[{'source': 'wiki'}, {'source': 'blog'}]],
            'distances': [[0.1, 0.8]]  # 第二个结果相似度低
        }
        mock_client.get_collection.return_value = mock_collection
        mock_chroma.PersistentClient.return_value = mock_client
        
        # 初始化检索器
        retriever = VectorRetriever(self.config)
        
        # 执行搜索（高相似度阈值）
        results = retriever.search("测试查询", n_results=2, similarity_threshold=0.7)
        
        # 验证只有高相似度结果返回
        assert len(results) == 1
        assert results[0]['similarity'] >= 0.7
    
    @patch('src.retrieval.vector_retriever.SentenceTransformer')
    @patch('src.retrieval.vector_retriever.chromadb')
    def test_empty_search_results(self, mock_chroma, mock_sentence_transformer):
        """测试空搜索结果处理"""
        # 模拟嵌入模型
        mock_embedding = Mock()
        mock_embedding.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_sentence_transformer.return_value = mock_embedding
        
        # 模拟空响应
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.query.return_value = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        mock_client.get_collection.return_value = mock_collection
        mock_chroma.PersistentClient.return_value = mock_client
        
        # 初始化检索器
        retriever = VectorRetriever(self.config)
        
        # 执行搜索
        results = retriever.search("不存在的查询")
        
        # 验证空结果
        assert len(results) == 0
    
    @patch('src.retrieval.vector_retriever.SentenceTransformer')
    @patch('src.retrieval.vector_retriever.chromadb')
    def test_get_collection_stats(self, mock_chroma, mock_sentence_transformer):
        """测试获取集合统计信息"""
        # 模拟嵌入模型
        mock_embedding = Mock()
        mock_embedding.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_sentence_transformer.return_value = mock_embedding
        
        # 模拟集合统计
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.count.return_value = 10
        mock_client.get_collection.return_value = mock_collection
        mock_chroma.PersistentClient.return_value = mock_client
        
        # 初始化检索器
        retriever = VectorRetriever(self.config)
        
        # 获取统计信息
        stats = retriever.get_collection_stats()
        
        # 验证统计信息
        assert 'count' in stats
        assert 'name' in stats
        assert stats['count'] == 10
        
        # 验证方法调用
        mock_collection.count.assert_called_once()
    
    @patch('src.retrieval.vector_retriever.SentenceTransformer')
    @patch('src.retrieval.vector_retriever.chromadb')
    def test_error_handling(self, mock_chroma, mock_sentence_transformer):
        """测试错误处理"""
        # 模拟嵌入模型
        mock_embedding = Mock()
        mock_embedding.encode.side_effect = Exception("嵌入模型错误")
        mock_sentence_transformer.return_value = mock_embedding
        
        # 初始化检索器
        retriever = VectorRetriever(self.config)
        
        # 测试添加文档时的错误处理
        with pytest.raises(Exception, match="嵌入模型错误"):
            retriever.add_documents(["测试文档"])
    
    @patch('src.retrieval.vector_retriever.SentenceTransformer')
    @patch('src.retrieval.vector_retriever.chromadb')
    def test_performance_benchmark(self, mock_chroma, mock_sentence_transformer):
        """测试性能基准"""
        import time
        from unittest.mock import patch
        
        # 模拟嵌入模型
        mock_embedding = Mock()
        mock_embedding.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_sentence_transformer.return_value = mock_embedding
        
        # 模拟ChromaDB
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.query.return_value = {
            'documents': [['文档1', '文档2', '文档3']],
            'metadatas': [[{'source': 'wiki'}, {'source': 'research'}, {'source': 'blog'}]],
            'distances': [[0.1, 0.2, 0.3]]
        }
        mock_client.get_collection.return_value = mock_collection
        mock_chroma.PersistentClient.return_value = mock_client
        
        # 初始化检索器
        retriever = VectorRetriever(self.config)
        
        # 性能测试：多次搜索
        start_time = time.time()
        for i in range(5):  # 执行5次搜索
            results = retriever.search(f"测试查询{i}")
            assert len(results) == 3
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / 5
        
        # 验证性能在合理范围内（模拟环境下应该很快）
        assert avg_time < 1.0  # 平均每次搜索小于1秒
        
        print(f"性能测试: 平均搜索时间 {avg_time:.4f}秒")
    
    @patch('src.retrieval.vector_retriever.SentenceTransformer')
    @patch('src.retrieval.vector_retriever.chromadb')
    def test_batch_operations(self, mock_chroma, mock_sentence_transformer):
        """测试批量操作"""
        # 模拟嵌入模型
        mock_embedding = Mock()
        mock_embedding.encode.return_value = [[0.1, 0.2, 0.3]] * 10  # 10个文档的嵌入
        mock_sentence_transformer.return_value = mock_embedding
        
        # 模拟ChromaDB
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_chroma.PersistentClient.return_value = mock_client
        
        # 初始化检索器
        retriever = VectorRetriever(self.config)
        
        # 批量添加文档
        documents = [f"文档{i}" for i in range(10)]
        metadatas = [{'source': 'batch', 'index': i} for i in range(10)]
        
        retriever.add_documents(documents, metadatas)
        
        # 验证批量添加
        mock_embedding.encode.assert_called_once_with(documents)
        mock_collection.add.assert_called_once()
        
        # 验证添加的文档数量
        call_args = mock_collection.add.call_args
        assert len(call_args.kwargs['documents']) == 10
        assert len(call_args.kwargs['metadatas']) == 10

if __name__ == "__main__":
    pytest.main([__file__, "-v"])