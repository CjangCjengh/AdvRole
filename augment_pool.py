from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from tqdm import tqdm

from advrole.data import load_pool, save_jsonl
from advrole.prompts import build_rewriter_messages, rewriter_dialogue_context
from advrole.rewriter import rewrite_from_text

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("advrole.augment")

def make_rewriter_fn(client, model: str, temperature: float, max_tokens: int):
    def _rewrite_one(scenario: Dict) -> Optional[Dict]:
        messages = build_rewriter_messages(
            profile=scenario["target_profile"],
            dialogue_context=rewriter_dialogue_context(scenario),
            character_name=scenario["target_character"],
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            raw = resp.choices[0].message.content or ""
        except Exception as e:
            logger.debug(f"rewrite call failed: {e}")
            return None
        return rewrite_from_text(scenario, raw)

    return _rewrite_one

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rewriter-api-base", required=True)
    p.add_argument("--rewriter-api-key", default="EMPTY")
    p.add_argument("--rewriter-model", required=True)
    p.add_argument("--original-pool", required=True, help="S0 (jsonl/parquet)")
    p.add_argument("--current-pool", default=None,
                   help="this epoch's pool; rewrites are appended to it")
    p.add_argument("--output", required=True, help="output jsonl path")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="only rewrite the first N")
    args = p.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=args.rewriter_api_base, api_key=args.rewriter_api_key,
                    timeout=300.0)

    original = load_pool(args.original_pool)
    if args.limit > 0:
        original = original[:args.limit]
    current = load_pool(args.current_pool) if args.current_pool else list(original)
    logger.info(f"Original pool: {len(original)}; current pool: {len(current)}")

    rewrite_one = make_rewriter_fn(client, args.rewriter_model,
                                   args.temperature, args.max_tokens)

    rewritten: List[Dict] = []
    failed = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(rewrite_one, s): i for i, s in enumerate(original)}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="rewriting"):
            result = fut.result()
            if result is not None:
                rewritten.append(result)
            else:
                failed += 1

    logger.info(f"Rewritten: {len(rewritten)} ok, {failed} failed")
    merged = list(current) + rewritten
    save_jsonl(merged, args.output)
    logger.info(f"Merged pool ({len(merged)}) -> {args.output}")

if __name__ == "__main__":
    main()
