from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advrole.prompts import build_actor_messages
from eval.common import chat, load_jsonl, make_client, parallel_map

logger = logging.getLogger(__name__)

DIM_PREFIX = [("SBK", "SBK"), ("SCK", "SCK"), ("RCB", "RCB"),
              ("CM", "CM"), ("TA", "TA")]
DIM_NAMES = {
    "SBK": "Script-Based Knowledge",
    "CM": "Conversation Memory",
    "SCK": "Script-Contradictory Knowledge",
    "RCB": "Role-Cognition Boundary",
    "TA": "Topic Advancement",
}

METRIC_CRITERIA = {
    "SBK": "属性一致性: 评测模型能否根据给定角色介绍中的信息正确回答用户的问题；仅判断对错，不用参考语气",
    "SCK": "幻觉与拒答 - 人设虚假属性: 评测模型能否对用户错误的诱导性提问进行更正",
    "RCB": "幻觉与拒答 - 知识边界: 评测模型能否对角色人设边界外的知识进行合理拒答",
    "TA": "话题推进 - 推动话题: 评测模型是否具有推进话题进行的能力。在用户当前对话表示的信息量比较少导致话题停滞时，回复要主动推动话题进行",
    "CM": "记忆能力 - 问询: 评测模型的思考或回复是否能正确记忆历史对话中的信息。不需要分析对话历史的真实性，不需要考虑角色人设",
}

JUDGE_PROMPT = """请你扮演一个角色扮演对话模型评测人员，严格根据评测标准判断模型回复是否满足{metric}。

## 要扮演的角色{character_name}的介绍:
{character_profile}

## 对话历史内容：
{dialogue_history}

## 待评测模型的回复: {model_response}

## 评测类型和标准：
{criteria}

## !!注意!!：
1. 请专注在给定的评测类型{metric}，不需要评测{metric}以外的维度。
2. 请忽略结果的来源，仅参考评测标准对内容的质量进行评估。
3. 对结果进行简要分析，解释判定的理由。

## 格式如下：
评测结果：正确 / 错误

理由："""

def metric_to_dim(tag: str) -> str:
    t = tag.rstrip("-").upper()
    for pref, dim in DIM_PREFIX:
        if t.startswith(pref):
            return dim
    return "OTHER"

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--model-endpoint", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--model-key", default="EMPTY")
    p.add_argument("--judge-endpoint", required=True)
    p.add_argument("--judge-name", required=True)
    p.add_argument("--judge-key", default="EMPTY")
    p.add_argument("--test", default="data/processed/raiden/test.jsonl")
    p.add_argument("--output", required=True)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.7)
    args = p.parse_args()

    scenarios = load_jsonl(args.test)
    scenarios = [s for s in scenarios
                 if any(metric_to_dim(m) in DIM_NAMES
                        for m in (s.get("meta") or {}).get("metrics", []))]
    if args.max_samples:
        scenarios = scenarios[:args.max_samples]
    logger.info(f"RAIDEN test instances (with reported-dimension tags): {len(scenarios)}")

    client_m = make_client(args.model_endpoint, args.model_key)
    client_j = make_client(args.judge_endpoint, args.judge_key)

    def _gen(s, _i):
        msgs = build_actor_messages(
            character=s["target_character"], profile=s["target_profile"],
            other_profiles="", scenario="", history=s.get("history_text", ""))
        return chat(client_m, args.model_name, msgs,
                    temperature=args.temperature, max_tokens=512)

    responses = parallel_map(_gen, scenarios, max_workers=args.concurrency,
                             desc="generating")

    jobs = []
    for s, r in zip(scenarios, responses):
        seen = set()
        for m in (s.get("meta") or {}).get("metrics", []):
            dim = metric_to_dim(m)
            if dim not in DIM_NAMES or dim in seen:
                continue
            seen.add(dim)
            jobs.append({"scenario": s, "response": r, "dim": dim, "tag": m})

    def _judge(job, _i):
        s = job["scenario"]
        prompt = JUDGE_PROMPT.format(
            metric=DIM_NAMES[job["dim"]],
            character_name=s["target_character"],
            character_profile=s["target_profile"],
            dialogue_history=s.get("history_text", ""),
            model_response=job["response"],
            criteria=METRIC_CRITERIA[job["dim"]],
        )
        raw = chat(client_j, args.judge_name, [{"role": "user", "content": prompt}],
                   temperature=0.0, max_tokens=300)
        correct = 1.0 if re.search(r"评测结果\s*[:：]\s*正确", raw) else 0.0
        return {**job, "correct": correct, "raw": raw[:300]}

    judgments = parallel_map(_judge, jobs, max_workers=args.concurrency, desc="judging")
    judgments = [j for j in judgments if j is not None]

    per_dim: Dict[str, List[float]] = {}
    for j in judgments:
        per_dim.setdefault(j["dim"], []).append(j["correct"])
    dim_scores = {d: round(sum(v) / len(v), 3) for d, v in per_dim.items()}
    ordered = {DIM_NAMES[d]: dim_scores[d] for d in
               ["SBK", "CM", "SCK", "RCB", "TA"] if d in dim_scores}
    avg = round(sum(dim_scores.values()) / len(dim_scores), 3) if dim_scores else None

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"summary": {"dimensions": ordered, "average": avg,
                               "n_judgments": len(judgments)},
                   "judgments": judgments}, f, ensure_ascii=False, indent=2)

    print("\n===== RAIDEN results (zero-shot) =====")
    for k, v in ordered.items():
        print(f"  {k:<32} {v}")
    print(f"  {'Average':<32} {avg}")
    print(f"Saved -> {args.output}")

if __name__ == "__main__":
    main()
