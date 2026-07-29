from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advrole.config import REPO_ROOT

DEFAULT_DEMO_DIR = os.path.join(REPO_ROOT, "data", "raw", "lanobe_demo")
OUT = os.path.join(REPO_ROOT, "data", "processed", "lanobe")
LANGS = ["zh-cn", "zh-tw", "ja", "ko", "th"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-dir", default=DEFAULT_DEMO_DIR)
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--train-size", type=int, default=40)
    args = ap.parse_args()

    if not os.path.isdir(args.demo_dir):
        sys.exit(f"demo dir not found: {args.demo_dir}\n"
                 f"Get the Lanobe demo jsonls from the Lanobe release and "
                 f"pass --demo-dir <dir>.")

    os.makedirs(args.out_dir, exist_ok=True)
    for lang in LANGS:
        src = os.path.join(args.demo_dir, f"{lang}_example.jsonl")
        if not os.path.exists(src):
            print(f"  [skip] {src} not found")
            continue
        samples = []
        with open(src, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    s = json.loads(line)
                    s.setdefault("dataset", "lanobe")
                    s.setdefault("lang", lang)
                    s.setdefault("other_profiles_text", "")
                    s.setdefault("scenario", "")
                    s.setdefault("meta", {})
                    samples.append(s)

        train, test = samples[:args.train_size], samples[args.train_size:]
        for split, rows in [("train", train), ("test", test)]:
            path = os.path.join(args.out_dir, f"{lang}_{split}.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {lang}: train={len(train)} test={len(test)}")

    print(f"Done -> {args.out_dir}")

if __name__ == "__main__":
    main()
