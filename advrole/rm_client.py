from __future__ import annotations

import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from .prompts import build_reward_messages

logger = logging.getLogger(__name__)

def _extract_json(text: str) -> Optional[dict]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None

class RMClient:
    def __init__(self, api_base: str, api_key: str = "EMPTY", model: str = "",
                 dataset: str = "generic", temperature: float = 0.0,
                 max_tokens: int = 256, timeout: float = 120.0):
        from openai import OpenAI
        self.client = OpenAI(base_url=api_base, api_key=api_key, timeout=timeout)
        self.model = model
        self.dataset = dataset
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._lock = threading.Lock()
        self._fail_count = 0

    def score(self, scenario: Dict, response: str, retries: int = 2) -> float:
        messages = build_reward_messages(self.dataset, scenario, response)
        last_err = None
        for attempt in range(retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                text = resp.choices[0].message.content or ""
                reward = self._parse_reward(text)
                if reward is not None:
                    return reward
                last_err = f"unparseable RM output: {text[:150]!r}"
            except Exception as e:
                last_err = str(e)
        with self._lock:
            self._fail_count += 1
            if self._fail_count % 20 == 1:
                logger.warning(f"RM scoring failures so far: {self._fail_count}; last: {last_err}")
        return 0.0

    def score_batch(self, pairs: List, max_workers: int = 16) -> List[float]:

        scores = [0.0] * len(pairs)

        def _one(i):
            s, r = pairs[i]
            return i, self.score(s, r)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_one, i): i for i in range(len(pairs))}
            for fut in as_completed(futs):
                i, sc = fut.result()
                scores[i] = sc
        return scores

    def _parse_reward(self, text: str) -> Optional[float]:
        data = _extract_json(text)
        if data is None:
            return None
        try:
            if self.dataset == "charactereval":
                if "overall" in data:
                    return (float(data["overall"]) - 1.0) / 4.0
                scores = data.get("scores", {})
                vals = [float(v) for v in scores.values()]
                if vals:
                    return (sum(vals) / len(vals) - 1.0) / 4.0
            elif self.dataset == "coser":
                if "overall" in data:
                    return float(data["overall"]) / 10.0
                keys = ["storyline_consistency", "anthropomorphism",
                        "character_fidelity", "storyline_quality"]
                vals = [float(data[k]) for k in keys if k in data]
                if vals:
                    return sum(vals) / len(vals) / 10.0
            else:
                dims = ["CC", "SC", "LQ", "IP"]
                vals = [float(data[k]) for k in dims if k in data]
                if len(vals) == 4:
                    return sum(vals) / len(vals) / 10.0
        except (TypeError, ValueError, KeyError):
            return None
        return None
