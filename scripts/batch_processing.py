#!/usr/bin/env python3
"""
批量处理脚本
用于批量处理查询文件并生成结果报告
"""

import os
import sys
import json
import csv
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.orchestrator import EvidenceEnhancedCorrectionOrchestrator
from src.config.config_loader import load_config

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/batch_processing.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def load_queries_from_file(file_path: str) -> List[str]:
    """从文件加载查询"""
    queries = []
    file_ext = Path(file_path).suffix.lower()
    
    try:
        if file_ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    queries = [item.get('query', '') for item in data if item.get('query')]
                else:
                    queries = [data.get('query', '')] if data.get('query') else []
        elif file_ext == '.csv':
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'query' in row and row['query'].strip():
                        queries.append(row['query'].strip())
        else:  # txt或其他文本文件
            with open(file_path, 'r', encoding='utf-8') as f:
                queries = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        return [q for q in queries if q]
    
    except Exception as e:
        logger.error(f"❌ 读取查询文件失败: {e}")
        return []

def save_results(results: List[Dict[str, Any]], output_format: str, output_path: str = None):
    """保存处理结果"""
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results/batch_results_{timestamp}.{output_format}"
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if output_format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        
        elif output_format == 'csv':
            # 提取所有可能的字段
            all_fields = set()
            for result in results:
                if 'results' in result:
                    all_fields.update(result['results'].keys())
                all_fields.update(result.keys())
            
            fieldnames = list(all_fields)
            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for result in results:
                    writer.writerow(result)
        
        logger.info(f"💾 结果已保存到: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"❌ 保存结果失败: {e}")
        return None

def generate_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """生成批量处理报告"""
    successful = [r for r in results if r.get('success')]
    failed = [r for r in results if not r.get('success')]
    
    # 计算质量指标
    quality_metrics = []
    processing_times = []
    
    for result in successful:
        if 'processing_metadata' in result:
            processing_times.append(result['processing_metadata'].get('total_duration', 0))
        
        if 'results' in result and 'verifications' in result['results']:
            verifications = result['results']['verifications']
            if verifications:
                supported = len([v for v in verifications if v.get('verdict') == 'SUPPORTED'])
                quality_metrics.append(supported / len(verifications))
    
    # 计算统计信息
    avg_quality = sum(quality_metrics) / len(quality_metrics) if quality_metrics else 0
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
    success_rate = len(successful) / len(results) * 100 if results else 0
    
    return {
        'summary': {
            'total_queries': len(results),
            'successful_queries': len(successful),
            'failed_queries': len(failed),
            'success_rate': success_rate,
            'average_processing_time': avg_processing_time,
            'average_quality_score': avg_quality
        },
        'detailed_analysis': {
            'processing_time_stats': {
                'min': min(processing_times) if processing_times else 0,
                'max': max(processing_times) if processing_times else 0,
                'average': avg_processing_time,
                'total': sum(processing_times)
            },
            'quality_distribution': {
                'excellent': len([q for q in quality_metrics if q >= 0.9]),
                'good': len([q for q in quality_metrics if 0.7 <= q < 0.9]),
                'fair': len([q for q in quality_metrics if 0.5 <= q < 0.7]),
                'poor': len([q for q in quality_metrics if q < 0.5])
            }
        },
        'failure_analysis': {
            'total_failures': len(failed),
            'common_errors': _analyze_common_errors(failed),
            'recommendations': _generate_recommendations(failed)
        },
        'timestamp': datetime.now().isoformat()
    }

def _analyze_common_errors(failed_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """分析常见错误"""
    error_counts = {}
    for result in failed_results:
        error = result.get('error', '未知错误')
        error_counts[error] = error_counts.get(error, 0) + 1
    
    return [{'error': error, 'count': count} for error, count in error_counts.items()]

def _generate_recommendations(failed_results: List[Dict[str, Any]]) -> List[str]:
    """生成改进建议"""
    recommendations = []
    
    if failed_results:
        api_errors = len([r for r in failed_results if 'API' in str(r.get('error'))])
        if api_errors > 0:
            recommendations.append("检查API密钥配置和网络连接")
        
        timeout_errors = len([r for r in failed_results if 'timeout' in str(r.get('error')).lower()])
        if timeout_errors > 0:
            recommendations.append("增加请求超时时间或减少并发请求数")
        
        memory_errors = len([r for r in failed_results if 'memory' in str(r.get('error')).lower()])
        if memory_errors > 0:
            recommendations.append("优化内存使用或增加系统内存")
    
    return recommendations if recommendations else ["暂无特定建议，检查日志获取详细信息"]

def process_batch(orchestrator, queries: List[str], context: str = None) -> List[Dict[str, Any]]:
    """批量处理查询"""
    results = []
    total_queries = len(queries)
    
    logger.info(f"🔄 开始批量处理 {total_queries} 个查询")
    start_time = time.time()
    
    for i, query in enumerate(queries, 1):
        logger.info(f"📝 处理进度: {i}/{total_queries} - {query[:50]}...")
        
        try:
            result = orchestrator.process_query(query, context)
            result['batch_index'] = i
            result['total_queries'] = total_queries
            results.append(result)
            
            # 进度报告
            if i % 10 == 0 or i == total_queries:
                elapsed = time.time() - start_time
                eta = (elapsed / i) * (total_queries - i) if i > 0 else 0
                logger.info(f"📊 进度: {i}/{total_queries} ({i/total_queries*100:.1f}%) - ETA: {eta:.1f}s")
                
        except Exception as e:
            error_result = {
                'success': False,
                'query': query,
                'error': str(e),
                'batch_index': i,
                'total_queries': total_queries,
                'timestamp': datetime.now().isoformat()
            }
            results.append(error_result)
            logger.error(f"❌ 查询处理失败: {query[:30]}... - {e}")
    
    total_duration = time.time() - start_time
    logger.info(f"✅ 批量处理完成! 总耗时: {total_duration:.2f}s")
    
    return results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='批量处理查询文件')
    parser.add_argument('--input', '-i', required=True, help='输入文件路径（支持.txt, .json, .csv格式）')
    parser.add_argument('--output', '-o', help='输出文件路径（可选）')
    parser.add_argument('--format', '-f', choices=['json', 'csv'], default='json', help='输出格式')
    parser.add_argument('--context', '-c', help='上下文信息')
    parser.add_argument('--max_queries', '-m', type=int, help='最大处理查询数（用于测试）')
    parser.add_argument('--config', default='config/config.yaml', help='配置文件路径')
    
    args = parser.parse_args()
    
    global logger
    logger = setup_logging()
    
    # 验证输入文件
    if not os.path.exists(args.input):
        logger.error(f"❌ 输入文件不存在: {args.input}")
        return 1
    
    try:
        # 加载配置和初始化系统
        config = load_config(args.config)
        orchestrator = EvidenceEnhancedCorrectionOrchestrator(config)
        logger.info("✅ 系统初始化成功")
        
        # 加载查询
        queries = load_queries_from_file(args.input)
        if not queries:
            logger.error("❌ 没有找到有效的查询")
            return 1
        
        # 限制查询数量（用于测试）
        if args.max_queries and args.max_queries < len(queries):
            queries = queries[:args.max_queries]
            logger.info(f"🔧 限制处理前 {args.max_queries} 个查询")
        
        logger.info(f"📋 加载到 {len(queries)} 个查询")
        
        # 批量处理
        results = process_batch(orchestrator, queries, args.context)
        
        # 生成报告
        report = generate_report(results)
        
        # 保存结果
        output_file = save_results(results, args.format, args.output)
        
        # 输出摘要
        summary = report['summary']
        logger.info(f"""
🎉 批量处理完成!
📊 处理统计:
   • 总查询数: {summary['total_queries']}
   • 成功处理: {summary['successful_queries']}
   • 处理失败: {summary['failed_queries']}
   • 成功率: {summary['success_rate']:.1f}%
   • 平均处理时间: {summary['average_processing_time']:.2f}s
   • 平均质量分: {summary['average_quality_score']:.3f}
💾 结果文件: {output_file}
        """)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ 批量处理失败: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())