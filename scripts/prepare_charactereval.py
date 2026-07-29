from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advrole.config import REPO_ROOT

RAW = os.path.join(REPO_ROOT, "data", "raw", "CharacterEval")
OUT = os.path.join(REPO_ROOT, "data", "processed", "charactereval")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=RAW)
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--train-size", type=int, default=2000)
    ap.add_argument("--val-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = os.path.join(args.raw_dir, "data")
    with open(os.path.join(data_dir, "test_data.jsonl"), "r", encoding="utf-8") as f:
        samples = json.load(f)
    with open(os.path.join(data_dir, "character_profiles.json"), "r", encoding="utf-8") as f:
        profiles = json.load(f)
    with open(os.path.join(data_dir, "id2metric.jsonl"), "r", encoding="utf-8") as f:
        id2metric = json.load(f)

    rng = random.Random(args.seed)
    rng.shuffle(samples)

    def to_scenario(s):
        role = s["role"]
        if role not in profiles:
            return None

        profile = profiles[role]
        if not isinstance(profile, str):
            profile = str(profile)
        return {
            "id": f"charactereval_{s['id']}",
            "dataset": "charactereval",
            "target_character": role,
            "target_profile": profile,
            "other_profiles_text": "",
            "scenario": f"作品《{s.get('novel_name', '')}》中的场景。",
            "history_text": s["context"],
            "reference_response": "",
            "lang": "zh",
            "meta": {
                "orig_id": s["id"],
                "novel_name": s.get("novel_name", ""),
                "metrics": id2metric.get(str(s["id"]), []),
            },
        }

    scenarios = [x for x in (to_scenario(s) for s in samples) if x is not None]
    print(f"Total samples: {len(samples)}, usable: {len(scenarios)} "
          f"(skipped {len(samples) - len(scenarios)} without profile)")

    train = scenarios[:args.train_size]
    rest = scenarios[args.train_size:]
    val = rest[:args.val_size]
    test = rest[args.val_size:]

    os.makedirs(args.out_dir, exist_ok=True)
    for name, rows in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(args.out_dir, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(rows)} -> {path}")

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(id2metric, f, ensure_ascii=False)

    from collections import Counter
    cnt = Counter(pair for v in id2metric.values() for pair in (tuple(p) for p in v))
    print("Metric inventory (en/zh -> instances):")
    for (en, zh), n in sorted(cnt.items()):
        print(f"  {en} / {zh}: {n}")

if __name__ == "__main__":
    main()
