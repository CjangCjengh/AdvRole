from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advrole.config import REPO_ROOT

RAW = os.path.join(REPO_ROOT, "data", "raw", "CoSER")
OUT = os.path.join(REPO_ROOT, "data", "processed", "coser")

CHAR_SYSTEM_RE = re.compile(
    r"You are (?:roleplaying|role-playing|playing) as\s+(?P<name>[^.]+?)\.",
    re.IGNORECASE)

def parse_character_system(system: str):

    low = system.lower()
    if "world model" in low or "environmental feedback" in low:
        return None
    m = CHAR_SYSTEM_RE.search(system)
    name = m.group("name").strip() if m else None
    return name, system

def conversations_to_scenarios(conversations, book_hint=""):
    scenarios = []
    system = ""
    turns = []
    for msg in conversations:
        if msg.get("from") == "system":
            system = msg.get("value", "")
        else:
            turns.append(msg)
    parsed = parse_character_system(system)
    if not parsed:
        return scenarios
    name, profile = parsed

    history_lines = []
    for t in turns:
        frm = t.get("from", "")
        val = t.get("value", "")
        if frm in ("gpt", "assistant", "bot"):
            if history_lines:
                scenarios.append({
                    "id": f"coser_{book_hint}_{len(scenarios)}_{abs(hash(val)) % 100000}",
                    "dataset": "coser",
                    "target_character": name or "the character",
                    "target_profile": profile,
                    "other_profiles_text": "",
                    "scenario": "",
                    "history_text": "\n".join(history_lines),
                    "reference_response": val,
                    "lang": "en",
                    "meta": {"book": book_hint},
                })
            history_lines.append(f"{name or 'Character'}: {val}")
        else:
            history_lines.append(f"{frm}: {val}")
    return scenarios

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=RAW)
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--sft-file", default=None,
                    help="CoSER SFT sharegpt json (HF Neph0s/CoSER "
                         "train/sft_conversations_sharegpt.json for full scale)")
    ap.add_argument("--max-train", type=int, default=17762,
                    help="cap train pool size (0 = no cap)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    test_set_path = os.path.join(args.raw_dir, "data", "test", "test_set.json")
    with open(test_set_path, "r", encoding="utf-8") as f:
        test_items = json.load(f)
    shutil.copy(test_set_path, os.path.join(args.out_dir, "test_set_original.json"))

    test_scenarios = []
    for i, item in enumerate(test_items):
        profs = item.get("character_profiles", {})
        majors = item.get("major_characters", [])
        target = majors[0] if majors else (list(profs.keys())[0] if profs else "the character")
        history = "\n".join(
            f"{d.get('character', '?')}: {d.get('message', '')}"
            for d in item.get("dialogues", [])
        )
        test_scenarios.append({
            "id": f"coser_test_{i:03d}",
            "dataset": "coser",
            "target_character": target,
            "target_profile": profs.get(target, ""),
            "other_profiles_text": "\n\n".join(
                f"{n}: {p}" for n, p in profs.items() if n != target),
            "scenario": item.get("scenario", ""),
            "history_text": history,
            "reference_response": "",
            "lang": "en",
            "meta": {"book": item.get("book", ""), "topic": item.get("topic", ""),
                     "i_p": item.get("i_p"), "i_c": item.get("i_c")},
        })

    sft_file = args.sft_file
    if not sft_file:
        for cand in (os.path.join(args.raw_dir, "data", "train",
                                  "example_sft_conversations_sharegpt.json"),
                     os.path.join(args.raw_dir, "data", "train", "sft_sharegpt.json")):
            if os.path.exists(cand):
                sft_file = cand
                break
    train_scenarios = []
    if sft_file and os.path.exists(sft_file):
        with open(sft_file, "r", encoding="utf-8") as f:
            convs = json.load(f)
        print(f"Loaded SFT conversations: {len(convs)} from {sft_file}")
        for c in convs:
            train_scenarios.extend(
                conversations_to_scenarios(c.get("conversations", [])))
    else:
        print("WARNING: no SFT file found; train.jsonl will be empty. "
              "Download train/sft_conversations_sharegpt.json from "
              "https://huggingface.co/datasets/Neph0s/CoSER and pass --sft-file.")

    if args.max_train and len(train_scenarios) > args.max_train:
        train_scenarios = train_scenarios[:args.max_train]

    val = test_scenarios[:16]

    for name, rows in [("train", train_scenarios), ("val", val), ("test", test_scenarios)]:
        path = os.path.join(args.out_dir, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(rows)} -> {path}")

if __name__ == "__main__":
    main()
