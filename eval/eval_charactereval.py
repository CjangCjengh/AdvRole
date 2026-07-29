from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict, List

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advrole.prompts import build_actor_messages
from eval.common import (chat, load_jsonl, make_client, parallel_map,
                         save_jsonl)

logger = logging.getLogger(__name__)

DIMENSIONS = {
    "Conversational Ability": ["Fluency", "Coherence", "Consistency"],
    "Character Consistency": ["Accuracy", "Hallucination", "Exposure",
                              "Behavior", "Utterance"],
    "Role-playing Attractiveness": ["Humanlikeness", "Communication_skills",
                                    "Diversity", "Empathy"],
}
METRIC_TO_DIM = {m: d for d, ms in DIMENSIONS.items() for m in ms}

MAX_SEQ_LEN = 4096

def generate_for_scenarios(args, scenarios: List[Dict]) -> List[str]:
    if args.responses and os.path.exists(args.responses):
        rows = load_jsonl(args.responses)
        by_id = {r["id"]: r["response"] for r in rows}
        if all(s["id"] in by_id for s in scenarios):
            logger.info(f"Reusing responses from {args.responses}")
            return [by_id[s["id"]] for s in scenarios]

    client = make_client(args.model_endpoint, args.model_key)

    def _one(s, _i):
        msgs = build_actor_messages(
            character=s["target_character"],
            profile=s["target_profile"],
            other_profiles=s.get("other_profiles_text", ""),
            scenario=s.get("scenario", ""),
            history=s.get("history_text", ""),
        )
        r = chat(client, args.model_name, msgs,
                 temperature=args.temperature, max_tokens=args.max_tokens)
        return r.split("\n")[0]

    responses = parallel_map(_one, scenarios, max_workers=args.concurrency,
                             desc="generating")
    if args.responses:
        save_jsonl([{"id": s["id"], "response": r}
                    for s, r in zip(scenarios, responses)], args.responses)
    return responses

def _load_char_rm_classes(model_path: str):

    import importlib.util
    import types

    model_path = os.path.abspath(model_path)
    pkg_name = "_advrole_char_rm"
    for existing in list(sys.modules):
        if existing == pkg_name or existing.startswith(pkg_name + "."):
            del sys.modules[existing]

    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [model_path]
    sys.modules[pkg_name] = pkg
    for sub in ("configuration_baichuan", "modeling_baichuan", "tokenization_baichuan"):
        spec = importlib.util.spec_from_file_location(
            f"{pkg_name}.{sub}", os.path.join(model_path, f"{sub}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"{pkg_name}.{sub}"] = mod
        spec.loader.exec_module(mod)
    modeling = sys.modules[f"{pkg_name}.modeling_baichuan"]
    tokenization = sys.modules[f"{pkg_name}.tokenization_baichuan"]
    return modeling.BaichuanCharRM, tokenization.BaichuanTokenizer

class CharRM:
    def __init__(self, model_path: str, device: str = "cuda"):
        import torch

        self.torch = torch
        model_cls, tokenizer_cls = _load_char_rm_classes(model_path)
        self.tokenizer = tokenizer_cls.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = model_cls.from_pretrained(
            model_path, torch_dtype=torch.bfloat16).to(device).eval()

    def score(self, profile: str, context: str, response: str, metric_zh: str) -> float:
        text = (f"<RoleInfo>\n\n{profile}\n\n<Context>\n\n{context}\n\n"
                f"<Response>\n\n{response}\n\n<Dimension>\n\n{metric_zh}")
        ids = self.tokenizer.encode(text=text, add_special_tokens=False) \
            + [self.tokenizer.eos_token_id]
        if len(ids) > MAX_SEQ_LEN:
            ids = ids[-MAX_SEQ_LEN:]
        input_ids = self.torch.tensor(ids).unsqueeze(0).to(self.model.device)
        with self.torch.no_grad():
            out = self.model(input_ids=input_ids)
            value = out[1].item() if isinstance(out, (tuple, list)) else out.logits.item()
        return value * 4 + 1  # official 1-5 mapping

def score_all(rm: CharRM, scenarios: List[Dict], responses: List[str]) -> List[Dict]:
    records = []
    for s, r in tqdm(zip(scenarios, responses), total=len(scenarios), desc="RM scoring"):
        metrics = (s.get("meta") or {}).get("metrics") or []
        if not metrics:
            continue
        for en, zh in metrics:
            try:
                score = rm.score(s["target_profile"], s.get("history_text", ""), r, zh)
            except Exception as e:
                logger.warning(f"RM error on {s['id']}/{en}: {e}")
                continue
            records.append({"id": s["id"], "metric_en": en, "metric_zh": zh,
                            "score": score})
    return records

def aggregate(records: List[Dict]) -> Dict:
    per_metric: Dict[str, List[float]] = {}
    for rec in records:
        per_metric.setdefault(rec["metric_en"], []).append(rec["score"])

    metric_means = {m: sum(v) / len(v) for m, v in per_metric.items()}
    dim_scores = {}
    for dim, metrics in DIMENSIONS.items():
        present = [metric_means[m] for m in metrics if m in metric_means]
        dim_scores[dim] = round(sum(present) / len(present), 3) if present else None
    valid = [v for v in dim_scores.values() if v is not None]
    return {
        "metric_means": {k: round(v, 3) for k, v in sorted(metric_means.items())},
        "dimensions": dim_scores,
        "overall": round(sum(valid) / len(valid), 3) if valid else None,
        "n_records": len(records),
    }

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--model-endpoint", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--model-key", default="EMPTY")
    p.add_argument("--rm-path", required=True, help="BaichuanCharRM dir")
    p.add_argument("--test", default="data/processed/charactereval/test.jsonl")
    p.add_argument("--output", required=True)
    p.add_argument("--responses", default=None, help="cache path for generations")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    scenarios = load_jsonl(args.test)
    if args.max_samples:
        scenarios = scenarios[:args.max_samples]
    logger.info(f"Test scenarios: {len(scenarios)}")

    responses = generate_for_scenarios(args, scenarios)
    rm = CharRM(args.rm_path, device=args.device)
    records = score_all(rm, scenarios, responses)

    summary = aggregate(records)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f,
                  ensure_ascii=False, indent=2)

    print("\n===== CharacterEval results =====")
    for m, v in summary["metric_means"].items():
        print(f"  {m:<22} {v:.3f} ({METRIC_TO_DIM.get(m, '?')})")
    for d, v in summary["dimensions"].items():
        print(f"  {d:<28} {v}")
    print(f"  {'Overall':<28} {summary['overall']}")
    print(f"Saved -> {args.output}")

if __name__ == "__main__":
    main()
