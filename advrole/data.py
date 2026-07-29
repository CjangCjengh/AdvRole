from __future__ import annotations

import copy
import json
import logging
import os
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = ("id", "target_character", "target_profile")

def normalize_scenario(raw: Dict, defaults: Optional[Dict] = None) -> Dict:
    out = copy.deepcopy(raw)
    base = {"dataset": "", "other_profiles_text": "", "scenario": "",
            "history_text": "", "reference_response": "", "lang": "", "meta": {}}
    if defaults:
        base.update(defaults)
    for k, v in base.items():
        out.setdefault(k, v)
    for f in REQUIRED_FIELDS:
        if not out.get(f):
            raise ValueError(f"scenario missing required field '{f}': {str(raw)[:200]}")
    return out

def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def save_jsonl(rows: Iterable[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_pool(path: str) -> List[Dict]:
    if path.endswith(".parquet"):
        import pandas as pd
        return pd.read_parquet(path).to_dict("records")
    return load_jsonl(path)

def save_pool(rows: List[Dict], path: str) -> None:
    if path.endswith(".parquet"):
        import pandas as pd
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        flat = []
        for r in rows:
            r = dict(r)
            if isinstance(r.get("meta"), dict):
                r["meta"] = json.dumps(r["meta"], ensure_ascii=False)
            flat.append(r)
        pd.DataFrame(flat).to_parquet(path, index=False)
    else:
        save_jsonl(rows, path)

def scenarios_to_verl_parquet(scenarios: List[Dict], path: str,
                              data_source: str = "advrole",
                              prompt_builder=None) -> None:

    import pandas as pd
    from .prompts import build_actor_messages

    rows = []
    for s in scenarios:
        messages = (prompt_builder(s) if prompt_builder
                    else build_actor_messages(
                        character=s["target_character"],
                        profile=s["target_profile"],
                        other_profiles=s.get("other_profiles_text", ""),
                        scenario=s.get("scenario", ""),
                        history=s.get("history_text", ""),
                    ))
        scenario_str = json.dumps(_json_safe(s), ensure_ascii=False)
        rows.append({
            "data_source": data_source,
            "prompt": messages,
            "ability": "roleplay",
            "reward_model": {"style": "model", "ground_truth": scenario_str},
            "extra_info": {"index": s.get("id", ""), "scenario": scenario_str},
        })
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    logger.info(f"Wrote verl parquet: {path} ({len(rows)} rows)")

def scenarios_to_rewriter_verl_parquet(scenarios: List[Dict], path: str,
                                       data_source: str = "advrole_rewriter") -> None:

    import pandas as pd
    from .prompts import build_rewriter_messages, rewriter_dialogue_context

    rows = []
    for s in scenarios:
        messages = build_rewriter_messages(
            profile=s["target_profile"],
            dialogue_context=rewriter_dialogue_context(s),
            character_name=s["target_character"],
        )
        scenario_str = json.dumps(_json_safe(s), ensure_ascii=False)
        rows.append({
            "data_source": data_source,
            "prompt": messages,
            "ability": "rewriting",
            "reward_model": {"style": "model", "ground_truth": scenario_str},
            "extra_info": {"index": s.get("id", ""), "scenario": scenario_str},
        })
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    logger.info(f"Wrote rewriter verl parquet: {path} ({len(rows)} rows)")

def _json_safe(obj):

    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)
