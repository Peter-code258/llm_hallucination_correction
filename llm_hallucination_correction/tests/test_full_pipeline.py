#!/usr/bin/env python3
"""
完整流程测试脚本 - 大语言模型幻觉检测与纠正系统

本脚本测试从意图分类到答案纠正的完整流程，验证系统各模块的集成功能。
"""

import os
import sys
import yaml
import json
import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orchestrator import EvidenceEnhancedCorrectionOrchestrator
from src.llm_client import LLMAdapter
from src.retriever import VectorRetriever
from src.intent_classifier import IntentClassifier
from src.claim_extractor import ClaimExtractor
from src.evidence_verifier import EvidenceVerifier
from src.corrector import IntentAwareCorrector

class TestFullPipeline:
    """完整流程测试类"""
    
    @classmethod
    def setup_class(cls):
        """测试类初始化"""
        print("\n" + "="*60)
        print("🧪 开始完整流程测试")
        print("="*60)
        
        # 加载测试配置
        cls.test_config = cls._load_test_config()
        
        # 创建测试数据目录
        os.makedirs('tests/test_data', exist_ok=True)
        
    def setup_method(self):
        """每个测试方法前的设置"""
        self.start_time = datetime.now()
        print(f"\n⏰ 开始测试: {self._testMethodName}")
    
    def teardown_method(self):
        """每个测试方法后的清理"""
        duration = (datetime.now() - self.start_time).total_seconds()
        print(f"✅ 测试完成: {self._testMethodName} (耗时: {duration:.2f}s)")
    
    @staticmethod
    def _load_test_config():
        """加载测试配置"""
        config_path = 'config/test_config.yaml'
        if not os.path.exists(config_path):
            # 创建基础测试配置
            test_config = {
                'llm': {
                    'provider': 'mock',
                    'api_key': 'test_key',
                    'model': 'test-model',
                    'temperature': 0.1,
                    'max_tokens': 500
                },
                'vector_db': {
                    'embedding_model': 'BAAI/bge-base-en',
                    'db_path': 'tests/test_data/vector_db',
                    'collection_name': 'test_knowledge_base'
                },
                'retrieval': {
                    'similarity_threshold': 0.7,
                    'max_retrieved_docs': 5
                },
                'verification': {
                    'confidence_threshold': 0.8,
                    'max_verification_attempts': 3
                },
                'intent': {
                    'supported_intents': ['事实查询', '比较查询', '方法查询', '观点查询'],
                    'default_intent': '事实查询'
                }
            }
            
            os.makedirs('config', exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(test_config, f, default_flow_style=False)
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _create_mock_llm_response(self, response_text: str):
        """创建模拟LLM响应"""
        return {
            'text': response_text,
            'usage': {
                'prompt_tokens': 100,
                'completion_tokens': 200,
                'total_tokens': 300
            },
            'model': 'test-model',
            'finish_reason': 'stop'
        }
    
    def _setup_mock_llm(self):
        """设置模拟LLM"""
        mock_llm = Mock(spec=LLMAdapter)
        
        # 模拟意图分类响应
        mock_llm.call_with_retry.return_value = self._create_mock_llm_response('事实查询')
        
        return mock_llm
    
    def test_01_system_initialization(self):
        """测试系统初始化"""
        print("🔧 测试系统初始化...")
        
        with patch('src.llm_client.LLMAdapter') as mock_llm_class:
            mock_llm_instance = self._setup_mock_llm()
            mock_llm_class.return_value = mock_llm_instance
            
            # 初始化系统
            orchestrator = EvidenceEnhancedCorrectionOrchestrator(self.test_config)
            
            # 验证系统状态
            status = orchestrator.get_system_status()
            assert status['components_initialized'] == True
            assert 'timestamp' in status
            print("✅ 系统初始化验证通过")
    
    def test_02_intent_classification(self):
        """测试意图分类功能"""
        print("🎯 测试意图分类...")
        
        test_cases = [
            {
                'query': '什么是机器学习？',
                'expected_intent': '事实查询',
                'description': '事实性查询'
            },
            {
                'query': 'Python和Java哪个更好？',
                'expected_intent': '比较查询', 
                'description': '比较性查询'
            },
            {
                'query': '如何学习深度学习？',
                'expected_intent': '方法查询',
                'description': '方法性查询'
            },
            {
                'query': '大家对人工智能的看法是什么？',
                'expected_intent': '观点查询',
                'description': '观点性查询'
            }
        ]
        
        with patch('src.llm_client.LLMAdapter') as mock_llm_class:
            mock_llm_instance = Mock()
            
            for i, test_case in enumerate(test_cases, 1):
                # 设置当前测试用例的模拟响应
                mock_llm_instance.call_with_retry.return_value = self._create_mock_llm_response(
                    test_case['expected_intent']
                )
                mock_llm_class.return_value = mock_llm_instance
                
                # 测试意图分类
                classifier = IntentClassifier(mock_llm_instance, self.test_config['intent'])
                detected_intent = classifier.classify_intent(test_case['query'])
                
                assert detected_intent == test_case['expected_intent']
                print(f"✅ 测试用例 {i}: {test_case['description']} - 通过")
    
    def test_03_claim_extraction(self):
        """测试声明提取功能"""
        print("🔍 测试声明提取...")
        
        test_texts = [
            {
                'input': 'Python是一种高级编程语言，由Guido van Rossum在1991年创建。它具有简单易学的语法。',
                'expected_claims': 3,
                'description': '多事实文本'
            },
            {
                'input': '机器学习是人工智能的重要分支。',
                'expected_claims': 1, 
                'description': '单事实文本'
            },
            {
                'input': '深度学习基于神经网络，能够自动学习特征表示，在图像识别和自然语言处理中表现出色。',
                'expected_claims': 3,
                'description': '复杂技术描述'
            }
        ]
        
        with patch('src.llm_client.LLMAdapter') as mock_llm_class:
            mock_llm_instance = Mock()
            
            for i, test_case in enumerate(test_texts, 1):
                # 模拟声明提取响应
                mock_response = """
                [CLAIM_1]: Python是一种高级编程语言
                [CLAIM_2]: Python由Guido van Rossum创建  
                [CLAIM_3]: Python在1991年创建
                [CLAIM_4]: Python具有简单易学的语法
                """
                mock_llm_instance.call_with_retry.return_value = self._create_mock_llm_response(mock_response)
                mock_llm_class.return_value = mock_llm_instance
                
                # 测试声明提取
                extractor = ClaimExtractor(mock_llm_instance)
                claims = extractor.extract_claims(test_case['input'])
                
                assert len(claims) >= test_case['expected_claims']
                assert all('text' in claim for claim in claims)
                assert all('confidence' in claim for claim in claims)
                print(f"✅ 测试用例 {i}: {test_case['description']} - 提取到 {len(claims)} 个声明")
    
    def test_04_evidence_retrieval(self):
        """测试证据检索功能"""
        print("📚 测试证据检索...")
        
        # 模拟向量数据库
        with patch('src.retriever.SentenceTransformer') as mock_embedding, \
             patch('src.retriever.chromadb') as mock_chroma:
            
            # 设置模拟嵌入模型
            mock_embedding_instance = Mock()
            mock_embedding_instance.encode.return_value = [[0.1, 0.2, 0.3]]  # 模拟嵌入向量
            mock_embedding.return_value = mock_embedding_instance
            
            # 设置模拟向量数据库
            mock_collection = Mock()
            mock_collection.query.return_value = {
                'documents': [['模拟证据文本1', '模拟证据文本2']],
                'metadatas': [[{'source': 'wiki'}, {'source': 'paper'}]],
                'distances': [[0.1, 0.2]]
            }
            mock_client = Mock()
            mock_client.get_collection.return_value = mock_collection
            mock_chroma.PersistentClient.return_value = mock_client
            
            # 初始化检索器
            retriever = VectorRetriever(self.test_config['vector_db'])
            
            # 测试检索
            query = "什么是机器学习？"
            results = retriever.search(query, n_results=2)
            
            assert len(results) == 2
            assert all('text' in result for result in results)
            assert all('source' in result for result in results)
            assert all('similarity' in result for result in results)
            print("✅ 证据检索功能验证通过")
    
    def test_05_claim_verification(self):
        """测试声明验证功能"""
        print("✅ 测试声明验证...")
        
        test_claims = [
            {
                'claim': 'Python是一种编程语言',
                'evidence': [
                    {'text': 'Python是高级编程语言', 'source': 'wiki', 'similarity': 0.9},
                    {'text': 'Python用于软件开发', 'source': 'docs', 'similarity': 0.8}
                ],
                'expected_verdict': 'SUPPORTED'
            },
            {
                'claim': 'Python是编译型语言', 
                'evidence': [
                    {'text': 'Python是解释型语言', 'source': 'official', 'similarity': 0.95}
                ],
                'expected_verdict': 'CONTRADICTED'
            }
        ]
        
        with patch('src.llm_client.LLMAdapter') as mock_llm_class:
            mock_llm_instance = Mock()
            
            for i, test_case in enumerate(test_claims, 1):
                # 模拟验证响应
                mock_response = json.dumps({
                    "verdict": test_case['expected_verdict'],
                    "confidence": 0.9,
                    "supporting_evidence": test_case['evidence'] if test_case['expected_verdict'] == 'SUPPORTED' else [],
                    "contradicting_evidence": test_case['evidence'] if test_case['expected_verdict'] == 'CONTRADICTED' else [],
                    "reasoning": "基于证据的推理过程",
                    "intent_specific_analysis": "事实查询的特别分析"
                })
                mock_llm_instance.call_with_retry.return_value = self._create_mock_llm_response(mock_response)
                mock_llm_class.return_value = mock_llm_instance
                
                # 测试声明验证
                verifier = EvidenceVerifier(mock_llm_instance, self.test_config['verification'])
                verification_result = verifier.verify_claim(
                    test_case['claim'], test_case['evidence'], "测试查询", "事实查询"
                )
                
                assert verification_result['verdict'] == test_case['expected_verdict']
                assert 'confidence' in verification_result
                assert 'reasoning' in verification_result
                print(f"✅ 验证测试 {i}: {test_case['claim']} - {test_case['expected_verdict']}")
    
    def test_06_answer_correction(self):
        """测试答案纠正功能"""
        print("✏️ 测试答案纠正...")
        
        test_scenario = {
            'query': '什么是Python？',
            'original_answer': 'Python是一种编译型编程语言，由Java创始人创建于2000年。',
            'verifications': [
                {
                    'claim': 'Python是一种编译型编程语言',
                    'verdict': 'CONTRADICTED',
                    'confidence': 0.95,
                    'reasoning': 'Python是解释型语言而非编译型语言',
                    'supporting_evidence': [],
                    'contradicting_evidence': [
                        {'text': 'Python是解释型语言', 'source': 'official', 'relevance_score': 0.9}
                    ]
                },
                {
                    'claim': 'Python由Java创始人创建',
                    'verdict': 'CONTRADICTED', 
                    'confidence': 0.98,
                    'reasoning': 'Python由Guido van Rossum创建，与Java无关',
                    'supporting_evidence': [],
                    'contradicting_evidence': [
                        {'text': 'Python由Guido van Rossum创建', 'source': 'wiki', 'relevance_score': 0.95}
                    ]
                },
                {
                    'claim': 'Python创建于2000年',
                    'verdict': 'CONTRADICTED',
                    'confidence': 0.9,
                    'reasoning': 'Python最初发布于1991年',
                    'supporting_evidence': [],
                    'contradicting_evidence': [
                        {'text': 'Python最初于1991年发布', 'source': 'history', 'relevance_score': 0.85}
                    ]
                }
            ]
        }
        
        with patch('src.llm_client.LLMAdapter') as mock_llm_class:
            mock_llm_instance = Mock()
            
            # 模拟纠正响应
            corrected_answer = """Python是一种高级编程语言，由Guido van Rossum在1991年创建。它是解释型语言，具有简单易学的语法特点，广泛应用于Web开发、数据分析、人工智能等领域。"""
            
            mock_llm_instance.call_with_retry.return_value = self._create_mock_llm_response(corrected_answer)
            mock_llm_class.return_value = mock_llm_instance
            
            # 测试答案纠正
            corrector = IntentAwareCorrector(mock_llm_instance)
            correction_result = corrector.correct_answer(
                test_scenario['original_answer'],
                test_scenario['verifications'],
                test_scenario['query'],
                '事实查询'
            )
            
            assert 'corrected_answer' in correction_result
            assert len(correction_result['corrected_answer']) > 0
            assert correction_result['supported_claims'] == 0  # 所有声明都被反驳
            assert correction_result['contradicted_claims'] == 3
            print("✅ 答案纠正功能验证通过")
    
    def test_07_full_pipeline_integration(self):
        """测试完整流程集成"""
        print("🔗 测试完整流程集成...")
        
        # 测试用例
        test_case = {
            'query': '比较Python和Java在机器学习中的应用',
            'original_answer': """Python是机器学习的唯一选择，Java完全不适合机器学习。
            Python有TensorFlow和PyTorch等强大库，而Java没有任何机器学习库。
            实际上所有数据科学家都只用Python，Java在机器学习领域毫无用处。"""
        }
        
        # 使用模拟对象测试完整流程
        with patch('src.llm_client.LLMAdapter') as mock_llm, \
             patch('src.retriever.SentenceTransformer') as mock_embedding, \
             patch('src.retriever.chromadb') as mock_chroma:
            
            # 设置模拟LLM响应链
            mock_llm_instance = Mock()
            
            # 意图分类响应
            mock_llm_instance.call_with_retry.side_effect = [
                self._create_mock_llm_response('比较查询'),  # 意图分类
                self._create_mock_llm_response("""
                [CLAIM_1]: Python是机器学习的唯一选择
                [CLAIM_2]: Java完全不适合机器学习
                [CLAIM_3]: Python有TensorFlow和PyTorch等强大库
                [CLAIM_4]: Java没有任何机器学习库
                [CLAIM_5]: 所有数据科学家都只用Python
                [CLAIM_6]: Java在机器学习领域毫无用处
                """),  # 声明提取
                self._create_mock_llm_response(json.dumps({
                    "verdict": "PARTIALLY_SUPPORTED",
                    "confidence": 0.7,
                    "supporting_evidence": [{"text": "Python在机器学习中很流行", "source": "survey", "relevance_score": 0.8}],
                    "contradicting_evidence": [{"text": "Java也有机器学习库如Weka", "source": "docs", "contradiction_score": 0.6}],
                    "reasoning": "Python确实更流行但Java也有应用",
                    "intent_specific_analysis": "需要更平衡的比较"
                })),  # 声明验证1
                # ... 更多验证响应
                self._create_mock_llm_response("""Python和Java在机器学习中各有优势。Python凭借丰富的库生态在研究和快速原型中更受欢迎，而Java在企业级应用和大规模系统中仍有价值。两者并非互斥，而是根据场景选择。""")  # 答案纠正
            ]
            
            # 设置模拟向量数据库
            mock_embedding_instance = Mock()
            mock_embedding_instance.encode.return_value = [[0.1, 0.2, 0.3]]
            mock_embedding.return_value = mock_embedding_instance
            
            mock_collection = Mock()
            mock_collection.query.return_value = {
                'documents': [['证据文本1', '证据文本2']],
                'metadatas': [[{'source': 'source1'}, {'source': 'source2'}]],
                'distances': [[0.1, 0.2]]
            }
            mock_client = Mock()
            mock_client.get_collection.return_value = mock_collection
            mock_chroma.PersistentClient.return_value = mock_client
            
            # 初始化协调器
            orchestrator = EvidenceEnhancedCorrectionOrchestrator(self.test_config)
            
            # 执行完整流程
            result = orchestrator.process_correction(test_case['query'], test_case['original_answer'])
            
            # 验证结果
            assert result['success'] == True
            assert 'corrected_answer' in result
            assert len(result['corrected_answer']) > 0
            assert result['detected_intent'] == '比较查询'
            assert 'processing_metadata' in result
            assert 'analysis_results' in result
            
            print("✅ 完整流程集成测试通过")
            print(f"📊 处理统计: {result['processing_metadata']}")
    
    def test_08_error_handling(self):
        """测试错误处理机制"""
        print("🛡️ 测试错误处理...")
        
        error_scenarios = [
            {
                'description': 'LLM API调用失败',
                'mock_behavior': lambda m: setattr(m.call_with_retry, 'side_effect', Exception("API连接失败")),
                'expected_error': True
            },
            {
                'description': '空查询处理',
                'mock_behavior': lambda m: setattr(m.call_with_retry, 'return_value', self._create_mock_llm_response('')),
                'expected_error': False  # 应该能处理空响应
            },
            {
                'description': '无效JSON响应',
                'mock_behavior': lambda m: setattr(m.call_with_retry, 'return_value', self._create_mock_llm_response('无效的JSON格式')),
                'expected_error': False  # 应该有fallback处理
            }
        ]
        
        for scenario in error_scenarios:
            with patch('src.llm_client.LLMAdapter') as mock_llm_class:
                mock_llm_instance = Mock()
                scenario['mock_behavior'](mock_llm_instance)
                mock_llm_class.return_value = mock_llm_instance
                
                try:
                    orchestrator = EvidenceEnhancedCorrectionOrchestrator(self.test_config)
                    result = orchestrator.process_correction("测试查询", "测试答案")
                    
                    if scenario['expected_error']:
                        assert result['success'] == False
                        assert 'error' in result
                    else:
                        assert 'corrected_answer' in result
                    
                    print(f"✅ 错误场景处理: {scenario['description']} - 通过")
                    
                except Exception as e:
                    if scenario['expected_error']:
                        print(f"✅ 错误场景处理: {scenario['description']} - 正确抛出异常")
                    else:
                        print(f"❌ 错误场景处理: {scenario['description']} - 意外异常: {e}")
                        raise
    
    def test_09_performance_benchmark(self):
        """性能基准测试"""
        print("⚡ 性能基准测试...")
        
        # 简单的性能测试（不涉及真实API调用）
        test_cases = [
            {'query': '简单查询', 'answer': '简短答案'},
            {'query': '中等复杂度查询', 'answer': '包含多个事实的中等长度答案'},
            {'query': '复杂技术比较查询', 'answer': '涉及多个概念和比较的长篇技术分析答案'}
        ]
        
        performance_results = []
        
        with patch('src.llm_client.LLMAdapter') as mock_llm, \
             patch('src.retriever.SentenceTransformer') as mock_embedding, \
             patch('src.retriever.chromadb') as mock_chroma:
            
            # 设置快速响应的模拟
            mock_llm_instance = Mock()
            mock_llm_instance.call_with_retry.return_value = self._create_mock_llm_response('测试响应')
            
            mock_embedding_instance = Mock()
            mock_embedding_instance.encode.return_value = [[0.1, 0.2, 0.3]]
            mock_embedding.return_value = mock_embedding_instance
            
            mock_collection = Mock()
            mock_collection.query.return_value = {
                'documents': [['证据1', '证据2']],
                'metadatas': [[{'source': 's1'}, {'source': 's2'}]],
                'distances': [[0.1, 0.2]]
            }
            mock_client = Mock()
            mock_client.get_collection.return_value = mock_collection
            mock_chroma.PersistentClient.return_value = mock_client
            
            orchestrator = EvidenceEnhancedCorrectionOrchestrator(self.test_config)
            
            for test_case in test_cases:
                start_time = datetime.now()
                
                result = orchestrator.process_correction(test_case['query'], test_case['answer'])
                
                duration = (datetime.now() - start_time).total_seconds()
                performance_results.append({
                    'test_case': test_case['query'],
                    'duration': duration,
                    'success': result['success']
                })
            
            # 输出性能结果
            print("📈 性能测试结果:")
            for result in performance_results:
                status = "✅ 成功" if result['success'] else "❌ 失败"
                print(f"  {result['test_case']}: {result['duration']:.2f}s - {status}")
            
            # 验证平均处理时间在合理范围内（模拟环境下应该很快）
            avg_duration = sum(r['duration'] for r in performance_results) / len(performance_results)
            assert avg_duration < 5.0  # 模拟环境下应该很快
            print(f"✅ 平均处理时间: {avg_duration:.2f}s - 在合理范围内")
    
    def test_10_data_persistence(self):
        """测试数据持久化"""
        print("💾 测试数据持久化...")
        
        # 测试配置持久化
        config_path = 'tests/test_data/test_config.json'
        test_data = {
            'timestamp': datetime.now().isoformat(),
            'test_cases': [
                {'query': '测试1', 'answer': '答案1'},
                {'query': '测试2', 'answer': '答案2'}
            ],
            'metadata': {'version': '1.0.0'}
        }
        
        # 保存测试数据
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        
        # 读取并验证
        with open(config_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data['test_cases'] == test_data['test_cases']
        assert loaded_data['metadata'] == test_data['metadata']
        print("✅ 数据持久化测试通过")

def generate_test_report():
    """生成测试报告"""
    report = {
        'test_suite': '完整流程测试',
        'timestamp': datetime.now().isoformat(),
        'total_tests': 10,
        'test_categories': [
            '系统初始化',
            '意图分类', 
            '声明提取',
            '证据检索',
            '声明验证',
            '答案纠正',
            '流程集成',
            '错误处理',
            '性能基准',
            '数据持久化'
        ],
        'environment': {
            'python_version': sys.version,
            'platform': sys.platform,
            'working_directory': os.getcwd()
        }
    }
    
    report_path = 'tests/test_reports/full_pipeline_report.json'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📊 测试报告已生成: {report_path}")

if __name__ == "__main__":
    # 直接运行测试
    pytest.main([
        __file__,
        '-v',  # 详细输出
        '--tb=short',  # 简短的traceback
        '-x'  # 遇到第一个失败就停止
    ])
    
    # 生成测试报告