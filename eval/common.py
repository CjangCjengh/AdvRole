from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

def make_client(api_base: str, api_key: Optional[str] = None, timeout: float = 180.0):
    from openai import OpenAI
    return OpenAI(base_url=api_base, api_key=api_key or "EMPTY", timeout=timeout)

def resolve_api_key(cfg_value: str) -> str:

    if not cfg_value or cfg_value == "EMPTY":
        return "EMPTY"
    return os.environ.get(cfg_value, cfg_value)

def chat(client, model: str, messages: List[Dict], *, temperature: float = 0.7,
         max_tokens: int = 512, retries: int = 3) -> str:
    last = ""
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last = str(e)
            time.sleep(min(2 ** attempt, 10))
    logger.warning(f"chat failed after {retries} retries: {last}")
    return ""

def parallel_map(fn: Callable, items: List, max_workers: int = 8,
                 desc: str = "") -> List:

    from tqdm import tqdm
    out = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fn, it, i): i for i, it in enumerate(items)}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=desc):
            i = futs[fut]
            try:
                out[i] = fut.result()
            except Exception as e:
                logger.warning(f"{desc} item {i} failed: {e}")
    return out

def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def save_jsonl(rows: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def generate_responses(client, model: str, scenarios: List[Dict],
                       messages_builder: Callable, *, temperature: float = 0.7,
                       max_tokens: int = 512, max_workers: int = 8,
                       desc: str = "generating") -> List[str]:
    def _one(s, _i):
        return chat(client, model, messages_builder(s),
                    temperature=temperature, max_tokens=max_tokens)

    return parallel_map(_one, scenarios, max_workers=max_workers, desc=desc)
