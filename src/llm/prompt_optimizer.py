#!/usr/bin/env python3
"""
Prompt 优化器 - 基于 DSPy 自动优化 6 个手工 prompt 模板。

整体流程：
1. 把每个模板映射到一个 DSPy Signature + 一个 DSPy Module（dspy.Predict）。
2. 为每个 Signature 准备 few-shot 训练集（Example 列表）+ 评估 metric。
3. 用 MIPROv2 在数据集上搜索最优 instruction + few-shot demos。
4. 把优化后的 program 序列化到 data/optimized_prompts/<name>.json。
5. 运行时 PromptTemplates 优先加载优化后的 instruction/demos，找不到时回退原模板。

支持的优化器：
- "bootstrap": dspy.BootstrapFewShotWithRandomSearch -- 仅加 few-shot demos，instruction 不变
- "mipro":     dspy.MIPROv2                            -- 同时优化 instruction + demos（推荐）

注：DSPy 3.x 移除了 BootstrapFewShotWithMetric，bootstrap 路径改用
BootstrapFewShotWithRandomSearch（仍接受 metric 参数）。

使用示例：

    from src.llm.dspy_adapter import configure_dspy_from_config
    from src.llm.prompt_optimizer import PromptOptimizer

    configure_dspy_from_config(config)
    optimizer = PromptOptimizer(output_dir="./data/optimized_prompts")
    optimizer.optimize(
        name="fact_verification",
        trainset=train_examples,
        valset=val_examples,
        metric=lambda ex, pred, _=None: 1.0 if ex.intent == pred.intent else 0.0,
        optimizer_type="mipro",
    )
"""

import json
import os
from typing import Any, Callable, Dict, List, Optional

import dspy

from src.llm.dspy_signatures import SIGNATURE_REGISTRY


class PromptOptimizer:
    """基于 DSPy 的 prompt 自动优化器。"""

    def __init__(self, output_dir: str = "./data/optimized_prompts"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 构建 DSPy program（Predict 实例）
    # ------------------------------------------------------------------ #
    def _build_program(self, name: str) -> dspy.Predict:
        """根据 name 取 Signature 并构建 dspy.Predict。"""
        if name not in SIGNATURE_REGISTRY:
            raise ValueError(f"未知 signature: {name}。可选: {list(SIGNATURE_REGISTRY)}")
        signature = SIGNATURE_REGISTRY[name]
        return dspy.Predict(signature)

    # ------------------------------------------------------------------ #
    # 评估 metric 适配
    # ------------------------------------------------------------------ #
    @staticmethod
    def _wrap_metric(metric: Callable) -> Callable:
        """
        DSPy metric 签名：(example: dspy.Example, prediction: Prediction, trace=None) -> float
        兼容用户传入的 (ex, pred) -> float 形式。
        """
        import inspect

        sig = inspect.signature(metric)
        n_params = len(sig.parameters)

        def wrapped(example: dspy.Example, prediction, trace=None) -> float:
            pred = getattr(prediction, "completion", None) or prediction
            if n_params == 2:
                return float(metric(example, pred))
            return float(metric(example, pred, trace))

        return wrapped

    # ------------------------------------------------------------------ #
    # 优化器选择
    # ------------------------------------------------------------------ #
    def _build_optimizer(
        self,
        optimizer_type: str,
        metric: Callable,
        trainset: List[dspy.Example],
        **kwargs,
    ) -> dspy.MIPROv2:
        """构造 DSPy 优化器实例。"""
        metric = self._wrap_metric(metric)

        if optimizer_type == "mipro":
            return dspy.MIPROv2(
                metric=metric,
                num_threads=kwargs.get("num_threads", 4),
                max_labeled_demos=kwargs.get("max_labeled_demos", 4),
                max_bootstrapped_demos=kwargs.get("max_bootstrapped_demos", 4),
                log_dir=kwargs.get("log_dir", "./data/dspy_logs"),
                auto_brief=kwargs.get("auto_brief", True),
            )
        elif optimizer_type == "bootstrap":
            # DSPy 3.x: BootstrapFewShotWithMetric 已移除，改用 RandomSearch 变体
            return dspy.BootstrapFewShotWithRandomSearch(
                metric=metric,
                max_labeled_demos=kwargs.get("max_labeled_demos", 4),
                max_bootstrapped_demos=kwargs.get("max_bootstrapped_demos", 4),
                max_errors=kwargs.get("max_errors", 3),
                num_candidate_programs=kwargs.get("num_candidate_programs", 6),
                num_threads=kwargs.get("num_threads", 4),
            )
        else:
            raise ValueError(f"未知 optimizer_type: {optimizer_type}（支持 mipro / bootstrap）")

    # ------------------------------------------------------------------ #
    # 单模板优化主流程
    # ------------------------------------------------------------------ #
    def optimize(
        self,
        name: str,
        trainset: List[dspy.Example],
        valset: Optional[List[dspy.Example]] = None,
        metric: Optional[Callable] = None,
        optimizer_type: str = "mipro",
        **optimizer_kwargs,
    ) -> Dict[str, Any]:
        """
        对指定模板运行 DSPy 优化并保存结果。

        参数：
            name: SIGNATURE_REGISTRY 键，如 "fact_verification"
            trainset: 训练 example 列表
            valset: 验证 example 列表（MIPROv2 必需）
            metric: 评估函数 (example, prediction, trace=None) -> float
            optimizer_type: "mipro" 或 "bootstrap"
            optimizer_kwargs: 透传给优化器（max_labeled_demos, num_threads 等）

        返回：
            包含 name / instructions / demos / saved_path 的 dict
        """
        if metric is None:
            metric = self._default_metric_for(name)

        program = self._build_program(name)
        optimizer = self._build_optimizer(
            optimizer_type, metric, trainset, **optimizer_kwargs
        )

        # MIPROv2 需要 valset 用于 instruction 搜索；Bootstrap 不需要
        compile_kwargs: Dict[str, Any] = {"trainset": trainset}
        if valset is not None and optimizer_type == "mipro":
            compile_kwargs["valset"] = valset

        print(f"[PromptOptimizer] 开始优化 '{name}'，方法={optimizer_type}")
        optimized = optimizer.compile(program, **compile_kwargs)

        # 序列化
        saved_path = self.save(name, optimized)
        print(f"[PromptOptimizer] 优化完成，已保存至 {saved_path}")

        return {
            "name": name,
            "instructions": self._extract_instructions(optimized),
            "demos": self._extract_demos(optimized),
            "saved_path": saved_path,
        }

    # ------------------------------------------------------------------ #
    # 序列化 / 反序列化
    # ------------------------------------------------------------------ #
    def _extract_instructions(self, program: dspy.Predict) -> str:
        """从优化后的 program 提取 instruction 文本。"""
        try:
            # dspy.Predict 在 program.signature 里有 instructions
            return getattr(program, "instructions", "") or program.signature.instructions
        except Exception:
            return ""

    def _extract_demos(self, program: dspy.Predict) -> List[Dict[str, Any]]:
        """提取 few-shot demos。"""
        demos: List[Dict[str, Any]] = []
        for d in getattr(program, "demos", []):
            try:
                demos.append(dict(d))
            except Exception:
                continue
        return demos

    def save(self, name: str, program: dspy.Predict) -> str:
        """把优化后的 instruction + demos 保存为 JSON。"""
        payload = {
            "name": name,
            "instructions": self._extract_instructions(program),
            "demos": self._extract_demos(program),
        }
        path = os.path.join(self.output_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return path

    def load(self, name: str) -> Optional[Dict[str, Any]]:
        """加载优化后的 prompt 数据。失败返回 None（上层会回退到原模板）。"""
        path = os.path.join(self.output_dir, f"{name}.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # 默认 metric（用于简单的语义匹配任务）
    # ------------------------------------------------------------------ #
    @staticmethod
    def _default_metric_for(name: str) -> Callable:
        """为每种 Signature 提供一个默认 metric。复杂场景用户应自行传入。"""

        if name == "intent_classification":

            def m(ex, pred, trace=None):
                return 1.0 if ex.intent == pred.intent else 0.0

        elif name == "fact_verification":

            def m(ex, pred, trace=None):
                # 命中 verdict 即得 0.7；confidence 越高再补 0.3
                base = 0.7 if ex.verdict == pred.verdict else 0.0
                try:
                    conf = float(pred.confidence)
                except Exception:
                    conf = 0.0
                return base + 0.3 * (conf if ex.verdict == pred.verdict else (1 - conf))

        elif name == "claim_extraction":

            def m(ex, pred, trace=None):
                pred_set = set(getattr(pred, "claims", []) or [])
                ex_set = set(ex.claims)
                if not ex_set:
                    return 0.0
                return len(pred_set & ex_set) / len(ex_set)

        elif name == "hallucination_detection":

            def m(ex, pred, trace=None):
                return 1.0 if ex.has_hallucination == pred.has_hallucination else 0.0

        else:
            # initial_answer / answer_correction / answer_correction_*
            # 升级为 ROUGE-L；当 rouge_score 包不可用时回退到 token-overlap。
            def m(ex, pred, trace=None):
                gold = getattr(ex, "answer", "") or getattr(ex, "corrected_answer", "")
                got = getattr(pred, "answer", "") or getattr(pred, "corrected_answer", "")
                return PromptOptimizer._rouge_l(gold, got)

        return m

    @staticmethod
    def _rouge_l(reference: str, candidate: str) -> float:
        """计算 ROUGE-L F1。rouge_score 包不可用时回退到 token-overlap。"""
        if not reference or not candidate:
            return 0.0
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(
                ["rougeL"], use_stemmer=True
            )
            scores = scorer.score(reference, candidate)
            return float(scores["rougeL"].fmeasure)
        except ImportError:
            # 回退：粗粒度 token overlap，避免硬依赖 rouge_score
            ref_tokens = set(reference.split())
            cand_tokens = set(candidate.split())
            if not ref_tokens:
                return 0.0
            return len(ref_tokens & cand_tokens) / len(ref_tokens)
