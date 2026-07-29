from __future__ import annotations

import json
import logging
import re
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_PROFILE_RE = re.compile(
    r"\[Rewritten Character Profile\]\s*\n?(.*?)(?=\[Rewritten Dialogue Context\]|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_CONTEXT_RE = re.compile(
    r"\[Rewritten Dialogue Context\]\s*\n?(.*?)(?=\Z)",
    re.DOTALL | re.IGNORECASE,
)
_SCENE_RE = re.compile(r"\[Scene\]\s*(.*?)(?=\[Dialogue History\]|\Z)", re.DOTALL)
_HISTORY_RE = re.compile(r"\[Dialogue History\]\s*(.*?)(?=\Z)", re.DOTALL)

def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0]
    return text.strip()

def parse_rewritten_scenario(raw_output: str) -> Optional[Dict]:

    text = strip_code_fence(raw_output)
    if not text:
        return None

    profile_match = _PROFILE_RE.search(text)
    context_match = _CONTEXT_RE.search(text)

    if profile_match:
        return {
            "profile": profile_match.group(1).strip(),
            "dialogue_context": context_match.group(1).strip() if context_match else None,
        }

    try:
        data = json.loads(text)
        if isinstance(data, dict) and "target_profile" in data:
            ctx = None
            if data.get("scenario") or data.get("history_text"):
                ctx = ""
                if data.get("scenario"):
                    ctx += f"[Scene] {data['scenario']}\n\n"
                if data.get("history_text"):
                    ctx += f"[Dialogue History]\n{data['history_text']}"
            return {"profile": data["target_profile"], "dialogue_context": ctx}
    except (json.JSONDecodeError, ValueError):
        pass

    logger.warning(f"Failed to parse rewriter output (len={len(text)})")
    return None

def apply_rewrite(scenario: Dict, parsed: Dict) -> Dict:

    rewritten = dict(scenario)
    if parsed.get("profile"):
        rewritten["target_profile"] = parsed["profile"]

    ctx = parsed.get("dialogue_context")
    if ctx:
        scene_match = _SCENE_RE.search(ctx)
        hist_match = _HISTORY_RE.search(ctx)
        if scene_match:
            rewritten["scenario"] = scene_match.group(1).strip()
        if hist_match:
            rewritten["history_text"] = hist_match.group(1).strip()
        elif not scene_match:
            rewritten["history_text"] = ctx

    rewritten["is_rewritten"] = True
    rewritten["original_id"] = scenario.get("id", "")
    rewritten["id"] = f"{scenario.get('id', 's')}_rw"
    return rewritten

def rewrite_from_text(scenario: Dict, raw_output: str) -> Optional[Dict]:
    parsed = parse_rewritten_scenario(raw_output)
    if parsed is None:
        return None
    return apply_rewrite(scenario, parsed)
