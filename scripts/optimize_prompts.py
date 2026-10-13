#!/usr/bin/env python3
"""
自动 prompt 优化入口脚本

用法：
    # 优化单个模板（如事实验证）
    python scripts/optimize_prompts.py --name fact_verification --method mipro

    # 一次性优化全部 6 个模板
    python scripts/optimize_prompts.py --all --method mipro

    # 用 bootstrap 只优化 few-shot demos（更省 API 调用）
    python scripts/optimize_prompts.py --all --method bootstrap

    # 指定训练集 JSON 路径
    python scripts/optimize_prompts.py --name fact_verification \\
        --train data/train/fact_verification.json \\
        --val data/val/fact_verification.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

# 让脚本能在不安装为包的情况下直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dspy

from config.config_loader import load_config
from src.llm.dspy_adapter import configure_dspy_from_config
from src.llm.dspy_signatures import SIGNATURE_REGISTRY
from src.llm.prompt_optimizer import PromptOptimizer


# ==================================================================
# 示例训练数据 - 仅用于演示，真实场景请替换为自己的标注数据
# ==================================================================
def _build_default_trainset(name: str):
    """为每个 signature 返回最小可跑的 demo trainset。"""
    if name == "intent_classification":
        return [
            dspy.Example(query="Python 和 Java 的主要区别是什么？", intent="比较查询").with_inputs("query"),
            dspy.Example(query="如何用 Python 读取 CSV 文件？", intent="方法查询").with_inputs("query"),
            dspy.Example(query="深度学习的定义是什么？", intent="事实查询").with_inputs("query"),
            dspy.Example(query="关于 AI 风险有哪些不同看法？", intent="观点查询").with_inputs("query"),
        ]
    if name == "fact_verification":
        return [
            dspy.Example(
                intent="事实查询", query="Python 由谁创造？",
                claim="Python 由 Guido van Rossum 于 1991 年发布。",
                evidence_text="[证据1] Python 是一门由 Guido van Rossum 在 1991 年首次发布的高级编程语言。",
                verdict="SUPPORTED", confidence=0.95,
                reasoning="证据明确支持该声明。",
                intent_specific_analysis="事实型查询，证据直接命中。",
            ).with_inputs("intent", "query", "claim", "evidence_text"),
            dspy.Example(
                intent="事实查询", query="地球到月球的距离？",
                claim="地月距离约为 384400 公里。",
                evidence_text="[证据1] 月球距地球约 38 万公里。",
                verdict="SUPPORTED", confidence=0.9,
                reasoning="证据数值与声明一致。",
                intent_specific_analysis="数值型事实，证据一致。",
            ).with_inputs("intent", "query", "claim", "evidence_text"),
        ]
    if name == "claim_extraction":
        return [
            dspy.Example(
                text="Python 由 Guido van Rossum 于 1991 年发布。它是解释型语言，支持多种范式。",
                claims=["Python 由 Guido van Rossum 于 1991 年发布。", "Python 是解释型语言。", "Python 支持多种范式。"],
            ).with_inputs("text"),
        ]
    if name == "hallucination_detection":
        return [
            dspy.Example(
                question="Python 创造者是谁？",
                initial_answer="Python 由 Linus Torvalds 创造。",
                verified_answer="Python 由 Guido van Rossum 创造。",
                evidence="Python 由 Guido van Rossum 在 1991 年发布。",
                has_hallucination=True,
                hallucination_type="FACTUAL",
                confidence=0.9,
                reasoning="初始回答把创造者写错为 Linus Torvalds，属事实性幻觉。",
            ).with_inputs("question", "initial_answer", "verified_answer", "evidence"),
        ]
    if name == "answer_correction":
        return [
            dspy.Example(
                intent="事实查询", query="Python 由谁创造？",
                initial_answer="Python 由 Linus Torvalds 创造。",
                verification_summary="声明 CONTRADICTED：证据显示 Python 由 Guido van Rossum 创造。",
                corrected_answer="Python 由 Guido van Rossum 于 1991 年创造。",
            ).with_inputs("intent", "query", "initial_answer", "verification_summary"),
        ]
    if name == "answer_correction_factual":
        return [
            dspy.Example(
                intent="事实查询", query="Python 由谁创造？",
                initial_answer="Python 由 Linus Torvalds 创造。",
                verification_summary="声明 CONTRADICTED：证据显示 Python 由 Guido van Rossum 创造。",
                corrected_answer="Python 由 Guido van Rossum 于 1991 年创造。",
            ).with_inputs("intent", "query", "initial_answer", "verification_summary"),
            dspy.Example(
                intent="事实查询", query="地球到月球距离？",
                initial_answer="约 300000 公里。",
                verification_summary="声明 CONTRADICTED：证据显示约 38 万公里。",
                corrected_answer="地球到月球平均距离约 384400 公里。",
            ).with_inputs("intent", "query", "initial_answer", "verification_summary"),
        ]
    if name == "answer_correction_comparison":
        return [
            dspy.Example(
                intent="比较查询", query="Python 和 Java 的区别？",
                initial_answer="Python 比 Java 快。",
                verification_summary="声明 CONTRADICTED：证据显示 Java 通常比 Python 快。",
                corrected_answer="执行速度上 Java 通常优于 Python；Python 在开发效率上更优。",
            ).with_inputs("intent", "query", "initial_answer", "verification_summary"),
        ]
    if name == "answer_correction_method":
        return [
            dspy.Example(
                intent="方法查询", query="如何用 Python 读 CSV？",
                initial_answer="用 csv 模块，先 open 再 reader。",
                verification_summary="声明 SUPPORTED：csv.reader 是标准做法。",
                corrected_answer="1) import csv; 2) with open(...) as f; 3) csv.reader(f)",
            ).with_inputs("intent", "query", "initial_answer", "verification_summary"),
        ]
    if name == "answer_correction_opinion":
        return [
            dspy.Example(
                intent="观点查询", query="AI 风险有哪些看法？",
                initial_answer="所有人都认为 AI 危险。",
                verification_summary="声明 CONTRADICTED：证据显示存在分歧，部分人认为可控。",
                corrected_answer="对 AI 风险存在两类看法：一派认为存在生存性风险需严管；另一派认为风险可控、技术有益。",
            ).with_inputs("intent", "query", "initial_answer", "verification_summary"),
        ]
    if name == "initial_answer":
        return [
            dspy.Example(
                question="什么是机器学习？",
                answer="机器学习是让计算机从数据中学习规律，从而进行预测或决策的方法。",
            ).with_inputs("question"),
        ]
    return []


def _load_examples_from_json(path: str, input_keys):
    """从 JSON 文件加载 dspy.Example 列表。文件格式：[{...}, {...}]。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    examples = []
    for row in raw:
        ex = dspy.Example(**row)
        if input_keys:
            ex = ex.with_inputs(*input_keys)
        examples.append(ex)
    return examples


def _input_keys_for(name: str):
    """根据 signature 返回输入字段名。"""
    mapping = {
        "initial_answer": ("question",),
        "intent_classification": ("query",),
        "claim_extraction": ("text",),
        "fact_verification": ("intent", "query", "claim", "evidence_text"),
        "hallucination_detection": ("question", "initial_answer", "verified_answer", "evidence"),
        # 通用入口 + 4 个变体共用同一组输入字段
        "answer_correction": ("intent", "query", "initial_answer", "verification_summary"),
        "answer_correction_factual": ("intent", "query", "initial_answer", "verification_summary"),
        "answer_correction_comparison": ("intent", "query", "initial_answer", "verification_summary"),
        "answer_correction_method": ("intent", "query", "initial_answer", "verification_summary"),
        "answer_correction_opinion": ("intent", "query", "initial_answer", "verification_summary"),
    }
    return mapping.get(name, ())


def optimize_one(
    name: str,
    optimizer: PromptOptimizer,
    method: str,
    train_path: str,
    val_path: str,
):
    input_keys = _input_keys_for(name)

    if train_path:
        trainset = _load_examples_from_json(train_path, input_keys)
    else:
        trainset = _build_default_trainset(name)

    valset = None
    if val_path:
        valset = _load_examples_from_json(val_path, input_keys)

    if not trainset:
        print(f"[skip] 模板 '{name}' 无训练数据，跳过。")
        return

    optimizer.optimize(
        name=name,
        trainset=trainset,
        valset=valset,
        optimizer_type=method,
    )


def main():
    parser = argparse.ArgumentParser(description="DSPy 自动 prompt 优化")
    parser.add_argument("--name", type=str, help="单个模板名（见 SIGNATURE_REGISTRY）")
    parser.add_argument("--all", action="store_true", help="优化全部 6 个模板")
    parser.add_argument("--method", type=str, default="mipro", choices=["mipro", "bootstrap"])
    parser.add_argument("--train", type=str, default=None, help="训练集 JSON 路径")
    parser.add_argument("--val", type=str, default=None, help="验证集 JSON 路径")
    parser.add_argument("--output-dir", type=str, default="./data/optimized_prompts")
    args = parser.parse_args()

    # 1) 加载项目配置 + 配置 DSPy 全局 LM
    config = load_config()
    configure_dspy_from_config(config)

    # 2) 创建优化器
    optimizer = PromptOptimizer(output_dir=args.output_dir)

    # 3) 选择要优化的模板列表
    if args.all:
        names = list(SIGNATURE_REGISTRY.keys())
    elif args.name:
        names = [args.name]
    else:
        parser.error("请指定 --name 或 --all")

    for n in names:
        print(f"\n=== 优化 {n} ===")
        optimize_one(n, optimizer, args.method, args.train, args.val)

    print(f"\n✅ 完成。优化后 prompt 已保存至 {args.output_dir}/")


if __name__ == "__main__":
    main()
