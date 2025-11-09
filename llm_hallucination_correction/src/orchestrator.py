import time
import json
from datetime import datetime
from typing import Dict, Any, List
from .intent_classifier import IntentClassifier
from .claim_extractor import ClaimExtractor
from .evidence_verifier import EvidenceVerifier
from .corrector import IntentAwareCorrector
from .retriever import VectorRetriever
from .llm_client import LLMAdapter

class EvidenceEnhancedCorrectionOrchestrator:
    """证据增强的纠错协调器 - 主流程控制器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_adapter = None
        self.vector_retriever = None
        self.intent_classifier = None
        self.claim_extractor = None
        self.evidence_verifier = None
        self.corrector = None
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化所有组件"""
        # 初始化LLM适配器
        self.llm_adapter = LLMAdapter(
            self.config['llm']['provider'],
            self.config['llm']
        )
        
        # 初始化向量检索器
        self.vector_retriever = VectorRetriever(self.config['vector_db'])
        
        # 初始化各个处理器
        self.intent_classifier = IntentClassifier(self.llm_adapter, self.config['intent'])
        self.claim_extractor = ClaimExtractor(self.llm_adapter)
        self.evidence_verifier = EvidenceVerifier(self.llm_adapter, self.config['verification'])
        self.corrector = IntentAwareCorrector(self.llm_adapter)
        
        print("✅ 系统组件初始化完成")
    
    def process_correction(self, query: str, original_answer: str) -> Dict[str, Any]:
        """执行完整的证据增强纠正流程"""
        start_time = time.time()
        execution_steps = {}
        
        try:
            # 步骤1: 意图分类
            intent_start = time.time()
            intent = self.intent_classifier.classify_intent(query)
            execution_steps['intent_classification'] = {
                'duration': time.time() - intent_start,
                'result': intent
            }
            print(f"📊 检测到意图: {intent}")
            
            # 步骤2: 声明提取
            extraction_start = time.time()
            claims = self.claim_extractor.extract_claims(original_answer)
            execution_steps['claim_extraction'] = {
                'duration': time.time() - extraction_start,
                'claims_count': len(claims),
                'claims': [claim['text'] for claim in claims]
            }
            print(f"🔍 提取到 {len(claims)} 个声明")
            
            # 步骤3: 证据检索
            retrieval_start = time.time()
            evidence_map = self._retrieve_evidence_for_claims(query, claims, intent)
            execution_steps['evidence_retrieval'] = {
                'duration': time.time() - retrieval_start,
                'evidence_count': sum(len(info['evidence']) for info in evidence_map.values())
            }
            print("📚 证据检索完成")
            
            # 步骤4: 声明验证
            verification_start = time.time()
            verifications = self._verify_claims(claims, evidence_map, query, intent)
            execution_steps['claim_verification'] = {
                'duration': time.time() - verification_start,
                'verifications_count': len(verifications),
                'supported_count': len([v for v in verifications if v.get('verdict') == 'SUPPORTED']),
                'contradicted_count': len([v for v in verifications if v.get('verdict') == 'CONTRADICTED'])
            }
            print("✅ 声明验证完成")
            
            # 步骤5: 答案纠正
            correction_start = time.time()
            correction_result = self.corrector.correct_answer(original_answer, verifications, query, intent)
            execution_steps['answer_correction'] = {
                'duration': time.time() - correction_start
            }
            print("✏️ 答案纠正完成")
            
            # 汇总结果
            total_duration = time.time() - start_time
            
            result = {
                "success": True,
                "query": query,
                "detected_intent": intent,
                "original_answer": original_answer,
                "corrected_answer": correction_result['corrected_answer'],
                "processing_metadata": {
                    "total_duration": total_duration,
                    "timestamp": datetime.now().isoformat(),
                    "execution_steps": execution_steps
                },
                "analysis_results": {
                    "extracted_claims": claims,
                    "evidence_retrieval": evidence_map,
                    "claim_verifications": verifications,
                    "correction_summary": {
                        "total_claims": len(claims),
                        "supported_claims": correction_result['supported_claims'],
                        "contradicted_claims": correction_result['contradicted_claims'],
                        "support_ratio": correction_result['supported_claims'] / len(claims) if claims else 0
                    }
                }
            }
            
            print(f"🎯 处理完成! 总耗时: {total_duration:.2f}秒")
            return result
            
        except Exception as e:
            error_result = {
                "success": False,
                "query": query,
                "original_answer": original_answer,
                "error": {
                    "message": str(e),
                    "type": type(e).__name__,
                    "timestamp": datetime.now().isoformat()
                },
                "processing_metadata": {
                    "total_duration": time.time() - start_time,
                    "timestamp": datetime.now().isoformat()
                }
            }
            print(f"❌ 处理失败: {str(e)}")
            return error_result
    
    def _retrieve_evidence_for_claims(self, query: str, claims: List[Dict], intent: str) -> Dict[str, Any]:
        """为每个声明检索相关证据"""
        evidence_map = {}
        
        for i, claim_info in enumerate(claims):
            claim_text = claim_info['text']
            claim_id = claim_info['id']
            
            # 生成针对性的检索查询
            retrieval_query = self._build_retrieval_query(query, claim_text, intent)
            
            # 执行检索
            evidence_snippets = self.vector_retriever.search(
                retrieval_query, 
                n_results=self.config['retrieval']['max_retrieved_docs'],
                similarity_threshold=self.config['retrieval']['similarity_threshold']
            )
            
            evidence_map[claim_id] = {
                "claim": claim_text,
                "retrieval_query": retrieval_query,
                "evidence": evidence_snippets,
                "retrieval_timestamp": datetime.now().isoformat()
            }
        
        return evidence_map
    
    def _build_retrieval_query(self, original_query: str, claim: str, intent: str) -> str:
        """构建检索查询"""
        # 使用意图分类器生成检索prompt
        retrieval_prompt = self.intent_classifier.generate_retrieval_prompt(original_query, intent)
        
        # 结合具体声明增强检索针对性
        enhanced_query = f"{retrieval_prompt} 具体验证声明: {claim}"
        return enhanced_query
    
    def _verify_claims(self, claims: List[Dict], evidence_map: Dict, query: str, intent: str) -> List[Dict]:
        """验证所有声明"""
        verifications = []
        
        for claim_info in claims:
            claim_id = claim_info['id']
            claim_text = claim_info['text']
            
            if claim_id in evidence_map:
                evidence_info = evidence_map[claim_id]
                evidence_snippets = evidence_info['evidence']
                
                # 执行验证
                verification_result = self.evidence_verifier.verify_claim(
                    claim_text, evidence_snippets, query, intent
                )
                
                # 添加声明ID和证据信息
                verification_result['claim_id'] = claim_id
                verification_result['evidence_count'] = len(evidence_snippets)
                
                verifications.append(verification_result)
            else:
                # 如果没有检索到证据，创建未验证结果
                verifications.append({
                    "claim_id": claim_id,
                    "claim": claim_text,
                    "verdict": "UNVERIFIED",
                    "confidence": 0.0,
                    "evidence_count": 0,
                    "reasoning": "未能检索到相关证据进行验证",
                    "supporting_evidence": [],
                    "contradicting_evidence": []
                })
        
        return verifications
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态信息"""
        try:
            db_stats = self.vector_retriever.get_collection_stats()
            return {
                "status": "正常运行",
                "vector_db": db_stats,
                "components_initialized": True,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": f"异常: {str(e)}",
                "components_initialized": False,
                "timestamp": datetime.now().isoformat()
            }
    
    def add_knowledge_documents(self, documents: List[str], metadatas: List[Dict] = None):
        """向知识库添加文档"""
        self.vector_retriever.add_documents(documents, metadatas)
        print(f"✅ 已向知识库添加 {len(documents)} 个文档")