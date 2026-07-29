from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.common import chat, load_jsonl, make_client, save_jsonl
from eval.lanobe_prompts import (LANG_NAME, build_gen_messages,
                                 build_judge_messages)

logger = logging.getLogger(__name__)
ALL_LANGS = ["zh-cn", "zh-tw", "ja", "ko", "th"]
DIMS = ["CC", "SC", "LQ", "IP"]

def cmd_generate(args):
    client = make_client(args.model_endpoint, args.model_key)
    for lang in args.langs:
        path = os.path.join(args.data_dir, f"{lang}_test.jsonl")
        if not os.path.exists(path):
            logger.warning(f"[skip] {path}")
            continue
        samples = load_jsonl(path)
        if args.max_samples:
            samples = samples[:args.max_samples]
        out_dir = os.path.join(args.results_dir, args.model_name, "generations")
        out_path = os.path.join(out_dir, f"{lang}.jsonl")
        done = {r["id"] for r in load_jsonl(out_path)} if os.path.exists(out_path) else set()

        from tqdm import tqdm
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _one(s):
            msgs = build_gen_messages(lang, s)
            r = chat(client, args.model_id, msgs,
                     temperature=args.temperature, max_tokens=args.max_tokens)
            return {"id": s["id"], "response": r}

        remaining = [s for s in samples if s["id"] not in done]
        logger.info(f"[generate:{lang}] {len(remaining)} new / {len(samples)} total")
        rows = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(_one, s) for s in remaining]
            for fut in tqdm(as_completed(futs), total=len(futs), desc=lang):
                rows.append(fut.result())
        if rows:
            os.makedirs(out_dir, exist_ok=True)
            with open(out_path, "a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

def cmd_judge(args):
    client = make_client(args.judge_endpoint, args.judge_key)
    for lang in args.langs:
        gen_path = os.path.join(args.results_dir, args.model_name,
                                "generations", f"{lang}.jsonl")
        if not os.path.exists(gen_path):
            logger.warning(f"[skip] no generations: {gen_path}")
            continue
        gens = {r["id"]: r["response"] for r in load_jsonl(gen_path)}
        samples = {s["id"]: s
                   for s in load_jsonl(os.path.join(args.data_dir, f"{lang}_test.jsonl"))}
        out_dir = os.path.join(args.results_dir, args.model_name, "judgements")
        out_path = os.path.join(out_dir, f"{lang}.jsonl")
        done = {r["id"] for r in load_jsonl(out_path)} if os.path.exists(out_path) else set()
        todo = [sid for sid in gens if sid in samples and sid not in done]
        logger.info(f"[judge:{lang}] {len(todo)} to judge")

        from tqdm import tqdm
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _one(sid):
            msgs = build_judge_messages(lang, samples[sid], gens[sid])
            raw = chat(client, args.judge_id, msgs, temperature=0.0, max_tokens=300)
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            m = re.search(r"\{.*\}", text, re.DOTALL)
            try:
                scores = json.loads(m.group(0) if m else text)
            except (json.JSONDecodeError, AttributeError):
                scores = {"raw": raw[:200], "parse_error": True}
            return {"id": sid, "scores": scores}

        rows = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(_one, sid) for sid in todo]
            for fut in tqdm(as_completed(futs), total=len(futs), desc=f"judge-{lang}"):
                rows.append(fut.result())
        if rows:
            os.makedirs(out_dir, exist_ok=True)
            with open(out_path, "a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

def cmd_summarize(args):
    summary = {}
    all_scores = {d: [] for d in DIMS}
    for lang in args.langs:
        path = os.path.join(args.results_dir, args.model_name,
                            "judgements", f"{lang}.jsonl")
        if not os.path.exists(path):
            continue
        lang_scores = {d: [] for d in DIMS}
        for obj in load_jsonl(path):
            sc = obj.get("scores", {})
            if sc.get("parse_error"):
                continue
            for d in DIMS:
                try:
                    v = float(sc[d])
                    lang_scores[d].append(v)
                    all_scores[d].append(v)
                except (KeyError, TypeError, ValueError):
                    pass
        entry = {d: (round(sum(v) / len(v), 2) if v else None)
                 for d, v in lang_scores.items()}
        vals = [v for v in entry.values() if v is not None]
        entry["avg"] = round(sum(vals) / len(vals), 2) if vals else None
        entry["count"] = len(lang_scores[DIMS[0]])
        summary[lang] = entry

    overall = {d: (round(sum(v) / len(v), 2) if v else None)
               for d, v in all_scores.items()}
    vals = [v for v in overall.values() if v is not None]
    overall["avg"] = round(sum(vals) / len(vals), 2) if vals else None
    summary["overall"] = overall

    out_path = os.path.join(args.results_dir, args.model_name, "summary.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n===== Lanobe results: {args.model_name} =====")
    print(f"{'lang':<8}" + "".join(f"{d:<7}" for d in DIMS) + "avg     N")
    for lang in args.langs:
        if lang in summary:
            s = summary[lang]
            print(f"{lang:<8}" + "".join(f"{str(s.get(d)):<7}" for d in DIMS)
                  + f"{str(s.get('avg')):<8}{s.get('count', 0)}")
    o = summary["overall"]
    print(f"{'overall':<8}" + "".join(f"{str(o.get(d)):<7}" for d in DIMS)
          + str(o.get("avg")))
    print(f"Saved -> {out_path}")

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--model-name", required=True)
    g.add_argument("--model-endpoint", required=True)
    g.add_argument("--model-id", required=True)
    g.add_argument("--model-key", default="EMPTY")
    g.add_argument("--data-dir", default="data/processed/lanobe")
    g.add_argument("--results-dir", default="results/lanobe")
    g.add_argument("--langs", nargs="+", default=ALL_LANGS)
    g.add_argument("--max-samples", type=int, default=0)
    g.add_argument("--temperature", type=float, default=0.7)
    g.add_argument("--max-tokens", type=int, default=512)
    g.add_argument("--concurrency", type=int, default=8)

    j = sub.add_parser("judge")
    j.add_argument("--model-name", required=True)
    j.add_argument("--judge-endpoint", required=True)
    j.add_argument("--judge-id", required=True)
    j.add_argument("--judge-key", default="EMPTY")
    j.add_argument("--data-dir", default="data/processed/lanobe")
    j.add_argument("--results-dir", default="results/lanobe")
    j.add_argument("--langs", nargs="+", default=ALL_LANGS)
    j.add_argument("--concurrency", type=int, default=8)

    s = sub.add_parser("summarize")
    s.add_argument("--model-name", required=True)
    s.add_argument("--results-dir", default="results/lanobe")
    s.add_argument("--langs", nargs="+", default=ALL_LANGS)

    args = p.parse_args()
    {"generate": cmd_generate, "judge": cmd_judge, "summarize": cmd_summarize}[
        args.command](args)

if __name__ == "__main__":
    main()
