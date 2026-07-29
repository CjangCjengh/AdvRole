from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advrole.config import REPO_ROOT

RAW = os.path.join(REPO_ROOT, "data", "raw", "RAIDEN", "release_data")
OUT = os.path.join(REPO_ROOT, "data", "processed", "raiden")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=RAW)
    ap.add_argument("--out-dir", default=OUT)
    args = ap.parse_args()

    with open(os.path.join(args.raw_dir, "npc.json"), "r", encoding="utf-8") as f:
        npcs = json.load(f)
    with open(os.path.join(args.raw_dir, "dialogue.json"), "r", encoding="utf-8") as f:
        dialogues = json.load(f)
    if isinstance(dialogues, dict):
        dialogues = list(dialogues.values())

    scenarios = []
    for i, d in enumerate(dialogues):
        npc = d.get("npc_name", "")
        profile = npcs.get(npc, {})
        profile_text = profile.get("npc_setting", "") if isinstance(profile, dict) else str(profile)
        if not profile_text:
            continue
        history = "\n".join(
            f"{m.get('role', 'user')}: {m.get('text', '')}" for m in d.get("messages", [])
        )
        scenarios.append({
            "id": f"raiden_{i:05d}",
            "dataset": "raiden",
            "target_character": npc,
            "target_profile": profile_text,
            "other_profiles_text": "",
            "scenario": "",
            "history_text": history,
            "reference_response": d.get("reference", ""),
            "lang": "zh",
            "meta": {"metrics": d.get("metrics", []),
                     "npc_type": profile.get("npc_type", "") if isinstance(profile, dict) else ""},
        })

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "test.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in scenarios:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"RAIDEN: {len(scenarios)} scenarios -> {out_path}")

    from collections import Counter
    prefixes = Counter()
    for s in scenarios:
        for m in s["meta"]["metrics"]:
            prefixes[m.rstrip("0123456789")] += 1
    print("Metric prefix distribution:")
    for k, v in sorted(prefixes.items()):
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
