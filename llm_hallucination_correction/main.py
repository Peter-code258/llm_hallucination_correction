"""
大语言模型幻觉检测与纠正系统 - 主入口文件
"""

import os
import yaml
import argparse
import json
from datetime import datetime
from src.orchestrator import EvidenceEnhancedCorrectionOrchestrator
"""
from prompt_templates import PromptTemplates

# 快速使用
templates = PromptTemplates()

# 获取意图分类提示词
prompt = templates.get_intent_classification_prompt("你的查询内容")

# 获取答案纠正提示词
correction_prompt = templates.get_correction_prompt(
    intent="事实查询",
    query="原始问题",
    original_answer="需要验证的答案",
    verification_summary="验证结果摘要"
)
"""

def load_config(config_path: str = "config/config.yaml") -> dict:
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 处理环境变量
        if 'llm' in config and 'api_key' in config['llm']:
            api_key = config['llm']['api_key']
            if api_key.startswith('${') and api_key.endswith('}'):
                env_var = api_key[2:-1]
                config['llm']['api_key'] = os.getenv(env_var, '')
        
        return config
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return {}

def initialize_system(config: dict) -> EvidenceEnhancedCorrectionOrchestrator:
    """初始化系统"""
    print("🚀 正在初始化大语言模型幻觉检测与纠正系统...")
    
    try:
        orchestrator = EvidenceEnhancedCorrectionOrchestrator(config)
        
        # 检查系统状态
        status = orchestrator.get_system_status()
        if status['components_initialized']:
            print("✅ 系统初始化成功!")
            print(f"📊 知识库文档数量: {status['vector_db']['count']}")
        else:
            print("⚠️ 系统初始化存在警告")
        
        return orchestrator
        
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        raise

def process_single_query(orchestrator, query: str, original_answer: str):
    """处理单个查询"""
    print(f"\n🔍 开始处理查询: {query}")
    print(f"📝 原始答案: {original_answer[:100]}..." if len(original_answer) > 100 else f"📝 原始答案: {original_answer}")
    
    result = orchestrator.process_correction(query, original_answer)
    
    if result['success']:
        print(f"\n✅ 处理成功!")
        print(f"🎯 检测意图: {result['detected_intent']}")
        print(f"📊 声明统计: {result['analysis_results']['correction_summary']}")
        print(f"⏱️ 总耗时: {result['processing_metadata']['total_duration']:.2f}秒")
        
        print(f"\n📖 纠正后的答案:")
        print("-" * 50)
        print(result['corrected_answer'])
        print("-" * 50)
        
        return result
    else:
        print(f"\n❌ 处理失败: {result['error']['message']}")
        return result

def interactive_mode(orchestrator):
    """交互式模式"""
    print("\n🎮 进入交互式模式 (输入 'quit' 退出)")
    
    while True:
        try:
            # 获取用户输入
            query = input("\n❓ 请输入查询: ").strip()
            if query.lower() in ['quit', 'exit', '退出']:
                break
            if not query:
                continue
            
            original_answer = input("📝 请输入需要验证的原始答案: ").strip()
            if not original_answer:
                print("⚠️ 原始答案不能为空")
                continue
            
            # 处理查询
            result = process_single_query(orchestrator, query, original_answer)
            
            # 询问是否保存结果
            save_choice = input("\n💾 是否保存结果到文件? (y/n): ").strip().lower()
            if save_choice in ['y', 'yes', '是']:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"correction_result_{timestamp}.json"
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"✅ 结果已保存到: {filename}")
                
        except KeyboardInterrupt:
            print("\n👋 感谢使用!")
            break
        except Exception as e:
            print(f"❌ 处理过程中发生错误: {e}")

def batch_mode(orchestrator, input_file: str):
    """批量处理模式"""
    print(f"📂 批量处理模式: {input_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            if input_file.endswith('.json'):
                data = json.load(f)
                queries = data.get('queries', [])
            else:
                # 简单文本格式：每行一个查询和答案，用制表符分隔
                queries = []
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        queries.append({
                            'query': parts[0],
                            'original_answer': parts[1]
                        })
        
        if not queries:
            print("❌ 未找到有效的查询数据")
            return
        
        print(f"📊 找到 {len(queries)} 个查询")
        results = []
        
        for i, item in enumerate(queries, 1):
            print(f"\n🔍 处理第 {i}/{len(queries)} 个查询...")
            result = process_single_query(orchestrator, item['query'], item['original_answer'])
            results.append(result)
        
        # 保存批量结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"batch_results_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "batch_info": {
                    "total_queries": len(queries),
                    "successful_queries": len([r for r in results if r.get('success', False)]),
                    "process_timestamp": timestamp
                },
                "results": results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 批量处理完成! 结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"❌ 批量处理失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='大语言模型幻觉检测与纠正系统')
    parser.add_argument('--config', '-c', default='config/config.yaml', 
                       help='配置文件路径')
    parser.add_argument('--batch', '-b', help='批量处理文件路径')
    parser.add_argument('--query', '-q', help='单个查询')
    parser.add_argument('--answer', '-a', help='原始答案')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    if not config:
        return
    
    # 初始化系统
    try:
        orchestrator = initialize_system(config)
    except Exception as e:
        print(f"❌ 系统启动失败: {e}")
        return
    
    # 运行模式判断
    if args.batch:
        # 批量处理模式
        batch_mode(orchestrator, args.batch)
    
    elif args.query and args.answer:
        # 单个查询处理模式
        process_single_query(orchestrator, args.query, args.answer)
    
    else:
        # 交互式模式
        interactive_mode(orchestrator)

if __name__ == "__main__":

    main()
