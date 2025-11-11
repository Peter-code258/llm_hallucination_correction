#!/usr/bin/env python3
"""
流程协调器 - 主流程控制器
整合所有模块，实现完整的幻觉检测与纠正流程
"""

import time
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from src.llm.llm_client import LLMAdapter
from src.llm.prompt_templates import PromptTemplates
from src.retrieval.vector_retriever import VectorRetriever
from src.verification.intent_classifier import IntentClassifier
from src.verification.claim_extractor import ClaimExtractor
from src.verification.evidence_verifier import EvidenceVerifier
from src.correction.answer_corrector import AnswerCorrector

class EvidenceEnhancedCorrectionOrchestrator:
    """证据增强的纠错协调器 - 主流程控制器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logging()
        self.components = self._initialize_components()
        self.logger.info("✅ 系统组件初始化完成")
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志系统"""
        logging.basicConfig(
            level=getattr(logging, self.config.get('system', {}).get('log_level', 'INFO')),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _initialize_components(self) -> Dict[str, Any]:
        """初始化所有组件"""
        llm_config = self.config['llm']
        
        # 初始化LLM适配器
        llm_adapter = LLMAdapter(llm_config['provider'], llm_config)
        
        # 初始化各模块
        components = {
            'llm_adapter': llm_adapter,
            'templates': PromptTemplates(),
            'vector_retriever': VectorRetriever(self.config['vector_db']),
            'intent_classifier': IntentClassifier(llm_adapter, self.config['intent']),
            'claim_extractor': ClaimExtractor(llm_adapter),
            'evidence_verifier': EvidenceVerifier(llm_adapter, self.config['verification']),
            'answer_corrector': AnswerCorrector(llm_adapter)
        }
        
        return components
    
    def process_query(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """处理查询的完整流程"""
        start_time = time.time()
        execution_steps = {}
        results = {}
        
        try:
            self.logger.info(f"🔍 开始处理查询: {query}")
            
            # 步骤1: 生成初始回答
            initial_answer, step_meta = self._generate_initial_answer(query, context)
            execution_steps['initial_answer_generation'] = step_meta
            results['initial_answer'] = initial_answer
            self.logger.info(f"📝 生成初始答案 (长度: {len(initial_answer)} 字符)")
            
            # 步骤2: 意图分类
            intent, step_meta = self._classify_intent(query)
            execution_steps['intent_classification'] = step_meta
            results['intent'] = intent
            self.logger.info(f"🎯 检测到意图: {intent}")
            
            # 步骤3: 声明提取
            claims, step_meta = self._extract_claims(initial_answer)
            execution_steps['claim_extraction'] = step_meta
            results['claims'] = claims
            self.logger.info(f"🔍 提取到 {len(claims)} 个声明")
            
            # 步骤4: 证据检索
            evidence_map, step_meta = self._retrieve_evidence(query, claims, intent)
            execution_steps['evidence_retrieval'] = step_meta
            results['evidence_map'] = evidence_map
            self.logger.info("📚 证据检索完成")
            
            # 步骤5: 声明验证
            verifications, step_meta = self._verify_claims(claims, evidence_map, query, intent)
            execution_steps['claim_verification'] = step_meta
            results['verifications'] = verifications
            self.logger.info(f"✅ 完成声明验证: {len(verifications)} 个声明")
            
            # 步骤6: 答案纠正
            corrected_answer, step_meta = self._correct_answer(
                initial_answer, verifications, query, intent
            )
            execution_steps['answer_correction'] = step_meta
            results['corrected_answer'] = corrected_answer
            self.logger.info(f"✏️ 答案纠正完成")
            
            # 步骤7: 幻觉检测
            hallucination_analysis, step_meta = self._detect_hallucinations(
                query, initial_answer, corrected_answer, evidence_map
            )
            execution_steps['hallucination_detection'] = step_meta
            results['hallucination_analysis'] = hallucination_analysis
            self.logger.info("🔬 幻觉检测完成")
            
            # 步骤8: 生成最终报告
            final_report = self._generate_final_report(results, execution_steps)
            results['final_report'] = final_report
            
            total_duration = time.time() - start_time
            self.logger.info(f"🎉 流程完成! 总耗时: {total_duration:.2f}秒")
            
            return {
                'success': True,
                'query': query,
                'results': results,
                'execution_steps': execution_steps,
                'processing_metadata': {
                    'total_duration': total_duration,
                    'timestamp': datetime.now().isoformat(),
                    'steps_completed': list(execution_steps.keys())
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ 处理过程出错: {e}")
            return {
                'success': False,
                'query': query,
                'error': str(e),
                'processing_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'error_step': self._identify_error_step(execution_steps)
                }
            }
    
    def _generate_initial_answer(self, query: str, context: Optional[str]) -> tuple[str, Dict]:
        """生成初始回答"""
        start_time = time.time()
        
        prompt = self.components['templates'].get_initial_answer_prompt(query)
        if context:
            prompt = f"上下文: {context}\n\n{prompt}"
        
        response = self.components['llm_adapter'].call_with_retry(prompt)
        
        duration = time.time() - start_time
        return response['text'], {
            'duration': duration,
            'prompt_length': len(prompt),
            'response_length': len(response['text']),
            'llm_usage': response.get('usage', {})
        }
    
    def _classify_intent(self, query: str) -> tuple[str, Dict]:
        """分类查询意图"""
        start_time = time.time()
        
        intent = self.components['intent_classifier'].classify_intent(query)
        
        duration = time.time() - start_time
        return intent, {
            'duration': duration,
            'detected_intent': intent
        }
    
    def _extract_claims(self, text: str) -> tuple[List[Dict], Dict]:
        """从文本中提取声明"""
        start_time = time.time()
        
        claims = self.components['claim_extractor'].extract_claims(text)
        validation = self.components['claim_extractor'].validate_claims(claims)
        
        duration = time.time() - start_time
        return claims, {
            'duration': duration,
            'claims_count': len(claims),
            'validation_metrics': validation
        }
    
    def _retrieve_evidence(self, query: str, claims: List[Dict], intent: str) -> tuple[Dict, Dict]:
        """检索相关证据"""
        start_time = time.time()
        
        evidence_map = {}
        total_evidence_count = 0
        
        for claim in claims:
            claim_id = claim['id']
            claim_text = claim['text']
            
            # 为每个声明检索证据
            evidence_snippets = self.components['vector_retriever'].search(
                f"{query} {claim_text}", 
                n_results=3
            )
            
            evidence_map[claim_id] = {
                'claim': claim_text,
                'evidence': evidence_snippets,
                'retrieval_query': f"{query} {claim_text}"
            }
            total_evidence_count += len(evidence_snippets)
        
        duration = time.time() - start_time
        return evidence_map, {
            'duration': duration,
            'total_evidence_count': total_evidence_count,
            'claims_with_evidence': len(evidence_map)
        }
    
    def _verify_claims(self, claims: List[Dict], evidence_map: Dict, query: str, intent: str) -> tuple[List[Dict], Dict]:
        """验证声明真实性"""
        start_time = time.time()
        
        verifications = []
        supported_count = 0
        contradicted_count = 0
        
        for claim in claims:
            claim_id = claim['id']
            
            if claim_id in evidence_map:
                evidence_info = evidence_map[claim_id]
                verification_result = self.components['evidence_verifier'].verify_claim(
                    claim['text'], evidence_info['evidence'], query, intent
                )
                
                if verification_result.get('verdict') == 'SUPPORTED':
                    supported_count += 1
                elif verification_result.get('verdict') == 'CONTRADICTED':
                    contradicted_count += 1
                
                verifications.append(verification_result)
            else:
                # 没有检索到证据的声明
                verifications.append({
                    'claim_id': claim_id,
                    'claim': claim['text'],
                    'verdict': 'UNVERIFIED',
                    'confidence': 0.0,
                    'reasoning': '未能检索到相关证据进行验证'
                })
        
        duration = time.time() - start_time
        return verifications, {
            'duration': duration,
            'total_verifications': len(verifications),
            'supported_count': supported_count,
            'contradicted_count': contradicted_count,
            'support_ratio': supported_count / len(verifications) if verifications else 0
        }
    
    def _correct_answer(self, initial_answer: str, verifications: List[Dict], query: str, intent: str) -> tuple[str, Dict]:
        """纠正答案"""
        start_time = time.time()
        
        correction_result = self.components['answer_corrector'].correct_answer(
            initial_answer, verifications, query, intent
        )
        
        duration = time.time() - start_time
        return correction_result['corrected_answer'], {
            'duration': duration,
            'correction_metrics': {
                'supported_claims': correction_result.get('supported_claims', 0),
                'contradicted_claims': correction_result.get('contradicted_claims', 0),
                'length_change': len(correction_result['corrected_answer']) - len(initial_answer)
            }
        }
    
    def _detect_hallucinations(self, query: str, initial_answer: str, corrected_answer: str, evidence_map: Dict) -> tuple[Dict, Dict]:
        """检测幻觉"""
        start_time = time.time()
        
        # 准备证据文本
        evidence_text = self._prepare_evidence_text(evidence_map)
        
        # 使用幻觉检测模板
        prompt = self.components['templates'].get_hallucination_detection_prompt(
            query, initial_answer, corrected_answer, evidence_text
        )
        
        response = self.components['llm_adapter'].call_with_retry(prompt)
        
        try:
            hallucination_analysis = json.loads(response['text'])
        except json.JSONDecodeError:
            hallucination_analysis = {
                'has_hallucination': False,
                'error': '无法解析幻觉检测结果',
                'raw_response': response['text'][:200]
            }
        
        duration = time.time() - start_time
        return hallucination_analysis, {
            'duration': duration,
            'detection_result': hallucination_analysis.get('has_hallucination', False)
        }
    
    def _prepare_evidence_text(self, evidence_map: Dict) -> str:
        """准备证据文本"""
        evidence_parts = []
        for claim_id, info in evidence_map.items():
            evidence_parts.append(f"声明: {info['claim']}")
            for i, evidence in enumerate(info['evidence'], 1):
                evidence_parts.append(f"证据{i}: {evidence['text']} (相似度: {evidence.get('similarity', 0):.3f})")
            evidence_parts.append("")
        
        return "\n".join(evidence_parts)
    
    def _generate_final_report(self, results: Dict, execution_steps: Dict) -> Dict[str, Any]:
        """生成最终报告"""
        return {
            'summary': {
                'query': results.get('query'),
                'intent': results.get('intent'),
                'initial_answer_length': len(results.get('initial_answer', '')),
                'corrected_answer_length': len(results.get('corrected_answer', '')),
                'total_claims': len(results.get('claims', [])),
                'supported_claims': len([v for v in results.get('verifications', []) if v.get('verdict') == 'SUPPORTED']),
                'has_hallucination': results.get('hallucination_analysis', {}).get('has_hallucination', False)
            },
            'quality_metrics': {
                'answer_improvement': self._calculate_improvement_metric(results),
                'evidence_coverage': self._calculate_evidence_coverage(results),
                'verification_confidence': self._calculate_average_confidence(results)
            },
            'recommendations': self._generate_recommendations(results)
        }
    
    def _calculate_improvement_metric(self, results: Dict) -> float:
        """计算答案改进指标"""
        initial_len = len(results.get('initial_answer', ''))
        corrected_len = len(results.get('corrected_answer', ''))
        
        if initial_len == 0:
            return 0.0
        
        # 简单的改进指标：基于长度变化和验证结果
        length_ratio = corrected_len / initial_len
        supported_ratio = len([v for v in results.get('verifications', []) if v.get('verdict') == 'SUPPORTED']) / len(results.get('verifications', [1]))
        
        return (length_ratio + supported_ratio) / 2
    
    def _calculate_evidence_coverage(self, results: Dict) -> float:
        """计算证据覆盖率"""
        claims = results.get('claims', [])
        evidence_map = results.get('evidence_map', {})
        
        if not claims:
            return 0.0
        
        covered_claims = sum(1 for claim in claims if claim['id'] in evidence_map)
        return covered_claims / len(claims)
    
    def _calculate_average_confidence(self, results: Dict) -> float:
        """计算平均置信度"""
        verifications = results.get('verifications', [])
        if not verifications:
            return 0.0
        
        total_confidence = sum(v.get('confidence', 0) for v in verifications)
        return total_confidence / len(verifications)
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于分析结果生成建议
        hallucination_analysis = results.get('hallucination_analysis', {})
        if hallucination_analysis.get('has_hallucination'):
            recommendations.append("检测到潜在幻觉，建议增加证据检索范围")
        
        evidence_coverage = self._calculate_evidence_coverage(results)
        if evidence_coverage < 0.5:
            recommendations.append("证据覆盖率较低，建议扩充知识库")
        
        supported_ratio = len([v for v in results.get('verifications', []) if v.get('verdict') == 'SUPPORTED']) / len(results.get('verifications', [1]))
        if supported_ratio < 0.7:
            recommendations.append("支持声明比例较低，建议优化检索策略")
        
        return recommendations if recommendations else ["回答质量良好，继续保持"]
    
    def _identify_error_step(self, execution_steps: Dict) -> str:
        """识别错误发生的步骤"""
        if not execution_steps:
            return "initialization"
        
        last_step = list(execution_steps.keys())[-1]
        return last_step
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        components_status = {}
        
        for name, component in self.components.items():
            components_status[name] = {
                'initialized': True,
                'type': type(component).__name__
            }
        
        return {
            'status': 'running',
            'components': components_status,
            'timestamp': datetime.now().isoformat(),
            'config_loaded': bool(self.config)
        }
    
    def batch_process(self, queries: List[str], context: Optional[str] = None) -> List[Dict[str, Any]]:
        """批量处理多个查询"""
        results = []
        
        for i, query in enumerate(queries, 1):
            self.logger.info(f"🔄 处理查询 {i}/{len(queries)}: {query[:50]}...")
            
            result = self.process_query(query, context)
            results.append(result)
            
            # 添加进度信息
            result['batch_info'] = {
                'index': i,
                'total': len(queries),
                'progress': f"{i}/{len(queries)}"
            }
        
        return results