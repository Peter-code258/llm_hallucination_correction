#!/usr/bin/env python3
"""
大语言模型幻觉检测与纠正系统 - 主入口文件
集成所有模块，提供命令行界面和API服务入口
"""

import os
import sys
import argparse
import yaml
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.orchestrator import EvidenceEnhancedCorrectionOrchestrator
from src.config.config_loader import load_config, validate_config

class LLMHallucinationCorrectionSystem:
    """大语言模型幻觉检测与纠正系统 - 主控制器"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = None
        self.orchestrator = None
        self.logger = self._setup_logging()
        self._initialize_system()
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/system.log', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger(__name__)
    
    def _initialize_system(self):
        """初始化系统组件"""
        try:
            self.logger.info("🚀 初始化大语言模型幻觉检测与纠正系统")
            
            # 加载配置
            self.config = load_config(self.config_path)
            self.logger.info("✅ 配置文件加载成功")
            
            # 验证配置
            validation_result = validate_config(self.config)
            if not validation_result['valid']:
                self.logger.error(f"❌ 配置验证失败: {validation_result['errors']}")
                raise ValueError("配置文件验证失败")
            
            # 初始化协调器
            self.orchestrator = EvidenceEnhancedCorrectionOrchestrator(self.config)
            self.logger.info("✅ 系统组件初始化完成")
            
            # 检查系统状态
            status = self.orchestrator.get_system_status()
            self.logger.info(f"📊 系统状态: {status['status']}")
            
        except Exception as e:
            self.logger.error(f"❌ 系统初始化失败: {e}")
            raise
    
    def process_single_query(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """处理单个查询"""
        self.logger.info(f"🔍 开始处理查询: {query[:50]}...")
        
        try:
            start_time = datetime.now()
            result = self.orchestrator.process_query(query, context)
            duration = (datetime.now() - start_time).total_seconds()
            
            result['processing_metadata']['total_duration'] = duration
            result['success'] = True
            
            self.logger.info(f"✅ 查询处理完成 (耗时: {duration:.2f}秒)")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 查询处理失败: {e}")
            return {
                'success': False,
                'query': query,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def process_batch_queries(self, queries: List[str], context: Optional[str] = None) -> List[Dict[str, Any]]:
        """批量处理多个查询"""
        self.logger.info(f"📊 开始批量处理 {len(queries)} 个查询")
        
        results = []
        for i, query in enumerate(queries, 1):
            self.logger.info(f"🔄 处理进度: {i}/{len(queries)}")
            
            result = self.process_single_query(query, context)
            result['batch_index'] = i
            result['total_queries'] = len(queries)
            
            results.append(result)
        
        # 生成批量处理报告
        batch_report = self._generate_batch_report(results)
        self.logger.info(f"🎉 批量处理完成: {batch_report['summary']}")
        
        return results
    
    def process_file_queries(self, file_path: str, context: Optional[str] = None) -> List[Dict[str, Any]]:
        """从文件读取并处理查询"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"查询文件不存在: {file_path}")
        
        self.logger.info(f"📁 从文件读取查询: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        self.logger.info(f"📋 读取到 {len(queries)} 个查询")
        return self.process_batch_queries(queries, context)
    
    def interactive_mode(self):
        """交互式模式"""
        self.logger.info("💬 进入交互式模式")
        print("\n" + "="*70)
        print("🧠 大语言模型幻觉检测与纠正系统 - 交互模式")
        print("="*70)
        print("输入您的查询，系统将进行幻觉检测和答案纠正")
        print("输入 'quit', 'exit' 或 'q' 退出")
        print("-"*70)
        
        while True:
            try:
                query = input("\n❓ 请输入查询: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 感谢使用!")
                    break
                
                if not query:
                    continue
                
                # 处理查询
                result = self.process_single_query(query)
                
                # 显示结果
                self._display_result(result)
                
            except KeyboardInterrupt:
                print("\n👋 感谢使用!")
                break
            except Exception as e:
                print(f"❌ 处理错误: {e}")
                continue
    
    def _display_result(self, result: Dict[str, Any]):
        """显示处理结果"""
        if not result['success']:
            print(f"❌ 处理失败: {result.get('error', '未知错误')}")
            return
        
        results_data = result['results']
        metadata = result['processing_metadata']
        
        print(f"\n✅ 处理成功! (耗时: {metadata['total_duration']:.2f}秒)")
        print(f"🎯 意图分类: {results_data.get('intent', '未知')}")
        
        # 显示答案对比
        initial_answer = results_data.get('initial_answer', '')
        corrected_answer = results_data.get('corrected_answer', '')
        
        print(f"\n📝 初始答案 (长度: {len(initial_answer)} 字符):")
        print("-" * 50)
        print(initial_answer[:300] + "..." if len(initial_answer) > 300 else initial_answer)
        print("-" * 50)
        
        print(f"\n✏️ 纠正后答案 (长度: {len(corrected_answer)} 字符):")
        print("-" * 50)
        print(corrected_answer[:300] + "..." if len(corrected_answer) > 300 else corrected_answer)
        print("-" * 50)
        
        # 显示验证统计
        verifications = results_data.get('verifications', [])
        if verifications:
            supported = len([v for v in verifications if v.get('verdict') == 'SUPPORTED'])
            contradicted = len([v for v in verifications if v.get('verdict') == 'CONTRADICTED'])
            total = len(verifications)
            
            print(f"\n📊 声明验证统计:")
            print(f"  • 总声明数: {total}")
            print(f"  • 支持声明: {supported} ({supported/total*100:.1f}%)")
            print(f"  • 矛盾声明: {contradicted} ({contradicted/total*100:.1f}%)")
        
        # 显示幻觉检测结果
        hallucination_analysis = results_data.get('hallucination_analysis', {})
        if hallucination_analysis.get('has_hallucination'):
            print(f"\n⚠️  幻觉检测: 检测到潜在幻觉")
            affected_sections = hallucination_analysis.get('affected_sections', [])
            for section in affected_sections[:3]:  # 显示前3个受影响部分
                print(f"  • {section.get('text', '')[:50]}...")
        else:
            print(f"\n✅ 幻觉检测: 未检测到明显幻觉")
        
        # 显示改进建议
        final_report = results_data.get('final_report', {})
        recommendations = final_report.get('recommendations', [])
        if recommendations:
            print(f"\n💡 改进建议:")
            for rec in recommendations:
                print(f"  • {rec}")
    
    def _generate_batch_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成批量处理报告"""
        successful = [r for r in results if r.get('success')]
        failed = [r for r in results if not r.get('success')]
        
        # 计算质量指标
        quality_metrics = []
        for result in successful:
            if 'results' in result:
                verifications = result['results'].get('verifications', [])
                if verifications:
                    supported = len([v for v in verifications if v.get('verdict') == 'SUPPORTED'])
                    quality_metrics.append(supported / len(verifications))
        
        avg_quality = sum(quality_metrics) / len(quality_metrics) if quality_metrics else 0
        
        return {
            'summary': {
                'total_queries': len(results),
                'successful': len(successful),
                'failed': len(failed),
                'success_rate': len(successful) / len(results) * 100,
                'average_quality': avg_quality
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def export_results(self, results: List[Dict[str, Any]], output_format: str = 'json') -> str:
        """导出处理结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)
        
        if output_format == 'json':
            output_file = output_dir / f"results_{timestamp}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            return str(output_file)
        
        elif output_format == 'csv':
            # 简化的CSV导出
            output_file = output_dir / f"results_{timestamp}.csv"
            # 这里可以添加CSV导出逻辑
            return str(output_file)
        
        else:
            raise ValueError(f"不支持的导出格式: {output_format}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        if not self.orchestrator:
            return {'status': '未初始化'}
        
        status = self.orchestrator.get_system_status()
        return {
            'system': {
                'name': '大语言模型幻觉检测与纠正系统',
                'version': '1.0.0',
                'status': status['status'],
                'initialized': True,
                'timestamp': datetime.now().isoformat()
            },
            'components': status.get('components', {}),
            'config': {
                'llm_provider': self.config['llm']['provider'],
                'vector_db': self.config['vector_db']['collection_name'],
                'retrieval_threshold': self.config['retrieval']['similarity_threshold']
            }
        }

def main():
    """主函数 - 命令行入口点"""
    parser = argparse.ArgumentParser(
        description='大语言模型幻觉检测与纠正系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 交互模式
  python main.py --interactive
  
  # 处理单个查询
  python main.py --query "机器学习的基本概念是什么？"
  
  # 从文件处理查询
  python main.py --file queries.txt
  
  # 批量处理并导出结果
  python main.py --batch queries.txt --export json
  
  # 检查系统状态
  python main.py --status
        """
    )
    
    parser.add_argument('--config', '-c', default='config/config.yaml', 
                       help='配置文件路径 (默认: config/config.yaml)')
    parser.add_argument('--query', '-q', help='直接处理单个查询')
    parser.add_argument('--file', '-f', help='从文件读取查询')
    parser.add_argument('--batch', '-b', help='批量处理文件中的查询')
    parser.add_argument('--interactive', '-i', action='store_true', 
                       help='进入交互模式')
    parser.add_argument('--status', '-s', action='store_true', 
                       help='显示系统状态')
    parser.add_argument('--export', choices=['json', 'csv'], 
                       help='导出结果格式 (json/csv)')
    parser.add_argument('--output', '-o', help='输出文件路径')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='详细输出模式')
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 初始化系统
        system = LLMHallucinationCorrectionSystem(args.config)
        
        # 处理不同模式
        if args.status:
            # 显示系统状态
            info = system.get_system_info()
            print(json.dumps(info, indent=2, ensure_ascii=False))
            return
        
        if args.query:
            # 单次查询模式
            result = system.process_single_query(args.query)
            system._display_result(result)
            
            if args.export:
                output_file = system.export_results([result], args.export)
                print(f"💾 结果已导出到: {output_file}")
        
        elif args.file or args.batch:
            # 文件批量处理模式
            file_path = args.file or args.batch
            results = system.process_file_queries(file_path)
            
            if args.export:
                output_file = system.export_results(results, args.export)
                print(f"💾 批量结果已导出到: {output_file}")
            else:
                # 显示批量处理摘要
                report = system._generate_batch_report(results)
                print(f"\n📊 批量处理报告:")
                print(f"  总查询数: {report['summary']['total_queries']}")
                print(f"  成功处理: {report['summary']['successful']}")
                print(f"  处理失败: {report['summary']['failed']}")
                print(f"  成功率: {report['summary']['success_rate']:.1f}%")
                print(f"  平均质量: {report['summary']['average_quality']:.3f}")
        
        elif args.interactive:
            # 交互模式
            system.interactive_mode()
        
        else:
            # 默认显示帮助
            parser.print_help()
    
    except Exception as e:
        logging.error(f"❌ 系统运行错误: {e}")
        print(f"错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()