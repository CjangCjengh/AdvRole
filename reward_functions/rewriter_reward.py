import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

_lock = threading.Lock()
_rm_client = None
_actor_client = None

def _get_rm_client():
    global _rm_client
    if _rm_client is None:
        with _lock:
            if _rm_client is None:
                _rm_client = OpenAI(
                    base_url=os.environ.get("RM_API_BASE", "http://localhost:8000/v1"),
                    api_key=os.environ.get("RM_API_KEY", "EMPTY"), timeout=120.0)
    return _rm_client

def _get_actor_client():
    global _actor_client
    if _actor_client is None:
        with _lock:
            if _actor_client is None:
                _actor_client = OpenAI(
                    base_url=os.environ.get("ACTOR_API_BASE", "http://localhost:8001/v1"),
                    api_key=os.environ.get("ACTOR_API_KEY", "EMPTY"), timeout=180.0)
    return _actor_client

ACTOR_SYSTEM = """You are a role-playing actor. Respond strictly in character as {character} based on the provided character profile, scenario, and dialogue history.

Requirements:
1. Stay faithful to the character's personality, speaking style, and background.
2. Be coherent with the scenario and dialogue history; do not introduce unestablished facts.
3. Use brackets [...] for inner thoughts; everything else is spoken dialogue.
4. Output only {character}'s next response. Do not repeat the dialogue history or add explanations."""

ACTOR_USER = """[Character profile]
{profile}

[Other characters' profiles]
{other_profiles}

[Scenario]
{scenario}

[Dialogue history]
{history}

Respond as {character}."""

def _actor_messages(scenario):
    character = scenario.get("target_character", "")
    return [
        {"role": "system", "content": ACTOR_SYSTEM.format(character=character)},
        {"role": "user", "content": ACTOR_USER.format(
            character=character,
            profile=scenario.get("target_profile", "") or "(not provided)",
            other_profiles=scenario.get("other_profiles_text", "") or "(none)",
            scenario=scenario.get("scenario", "") or "(not specified)",
            history=scenario.get("history_text", "") or "(the conversation starts here)",
        )},
    ]

def _generate_actor_response(scenario):
    try:
        resp = _get_actor_client().chat.completions.create(
            model=os.environ.get("ACTOR_MODEL", "actor"),
            messages=_actor_messages(scenario),
            temperature=float(os.environ.get("ACTOR_TEMPERATURE", "0.7")),
            max_tokens=int(os.environ.get("ACTOR_MAX_TOKENS", "512")),
        )
        return resp.choices[0].message.content or ""
    except Exception:
        return ""

def _extract_json(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None

def length_coefficient(text):

    t = (text or "").strip()
    if not t:
        return 1.0
    if any("一" <= c <= "鿿" for c in t):
        length, lo, hi = len(t), 60, 120
    else:
        length, lo, hi = len(t.split()), 50, 100
    if length >= hi:
        return 0.0
    if length > lo:
        return (hi - length) / (hi - lo)
    return 1.0

def _rm_score(scenario, response):
    coeff = length_coefficient(response)
    if coeff == 0.0:
        return 0.0
    dataset = os.environ.get("RM_DATASET", "generic")
    profile = scenario.get("target_profile", "")
    history = scenario.get("history_text", "")
    character = scenario.get("target_character", "")

    if dataset == "charactereval":
        system = "You are a professional roleplay quality evaluation expert."
        user = (f"[Character Profile]\n{profile}\n\n[Dialogue Context]\n{history}\n\n"
                f"[Character Response]\n{response}\n\n"
                "Rate 1-5 on Conversational Ability, Character Consistency and "
                "Roleplay Appeal (11 fine-grained metrics). "
                'Output JSON: {"scores": {...}, "overall": <1-5>, "reason": "..."}')
    elif dataset == "coser":
        system = "You are an expert evaluator for role-playing dialogue quality."
        user = (f"[Character Profile]\n{profile}\n\n[Scenario]\n{scenario.get('scenario','')}\n\n"
                f"[Dialogue Context]\n{history}\n\n[Model Response]\n{response}\n\n"
                "Rate 1-10 on Storyline Consistency, Anthropomorphism, Character "
                "Fidelity, Storyline Quality. "
                'Output JSON: {"storyline_consistency": <1-10>, "anthropomorphism": <1-10>, '
                '"character_fidelity": <1-10>, "storyline_quality": <1-10>, "overall": <1-10>}')
    else:
        system = ("You are a strict but fair role-play evaluator. Score 0-10 on CC, SC, "
                  "LQ, IP. Return strictly JSON: "
                  '{"CC": <score>, "SC": <score>, "LQ": <score>, "IP": <score>}')
        user = (f"[Character to play] {character}\n\n[Character profile]\n{profile}\n\n"
                f"[Other characters' profiles]\n{scenario.get('other_profiles_text','') or '(none)'}\n\n"
                f"[Scenario]\n{scenario.get('scenario','') or '(not specified)'}\n\n"
                f"[Dialogue history]\n{history or '(the conversation starts here)'}\n\n"
                f"[Model response to evaluate]\n{response}\n\nNow produce your JSON judgement.")

    try:
        resp = _get_rm_client().chat.completions.create(
            model=os.environ.get("RM_MODEL", "rm"),
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=float(os.environ.get("RM_TEMPERATURE", "0.0")),
            max_tokens=int(os.environ.get("RM_MAX_TOKENS", "256")),
        )
        data = _extract_json(resp.choices[0].message.content)
        if data is None:
            return None
        raw = None
        if dataset == "charactereval":
            if "overall" in data:
                raw = (float(data["overall"]) - 1.0) / 4.0
            else:
                vals = [float(v) for v in data.get("scores", {}).values()]
                if vals:
                    raw = (sum(vals) / len(vals) - 1.0) / 4.0
        elif dataset == "coser":
            if "overall" in data:
                raw = float(data["overall"]) / 10.0
            else:
                keys = ["storyline_consistency", "anthropomorphism",
                        "character_fidelity", "storyline_quality"]
                vals = [float(data[k]) for k in keys if k in data]
                if vals:
                    raw = sum(vals) / len(vals) / 10.0
        else:
            dims = ["CC", "SC", "LQ", "IP"]
            vals = [float(data[k]) for k in dims if k in data]
            if len(vals) == 4:
                raw = sum(vals) / len(vals) / 10.0
        return raw * coeff if raw is not None else None
    except Exception:
        return None

def _mean_score(scenario, n):

    scores = []

    def _one(_):
        r = _generate_actor_response(scenario)
        return _rm_score(scenario, r)

    with ThreadPoolExecutor(max_workers=max(n, 1)) as ex:
        for s in ex.map(_one, range(n)):
            scores.append(s)
    valid = [s for s in scores if s is not None]
    if not valid:
        return 0.5
    return sum(valid) / len(valid)

# duplicated from advrole/rewriter.py: verl loads this file standalone
_PROFILE_RE = re.compile(
    r"\[Rewritten Character Profile\]\s*\n?(.*?)(?=\[Rewritten Dialogue Context\]|\Z)",
    re.DOTALL | re.IGNORECASE)
_CONTEXT_RE = re.compile(r"\[Rewritten Dialogue Context\]\s*\n?(.*?)\Z",
                         re.DOTALL | re.IGNORECASE)
_SCENE_RE = re.compile(r"\[Scene\]\s*(.*?)(?=\[Dialogue History\]|\Z)", re.DOTALL)
_HIST_RE = re.compile(r"\[Dialogue History\]\s*(.*?)\Z", re.DOTALL)

def _parse_rewrite(original, raw):
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0].strip()
    if not text:
        return None

    pm = _PROFILE_RE.search(text)
    cm = _CONTEXT_RE.search(text)
    profile = pm.group(1).strip() if pm else None
    ctx = cm.group(1).strip() if cm else None

    if profile is None:
        try:
            data = json.loads(text)
            if isinstance(data, dict) and data.get("target_profile"):
                profile = data["target_profile"]
                ctx = None
                if data.get("scenario") or data.get("history_text"):
                    ctx = ""
                    if data.get("scenario"):
                        ctx += f"[Scene] {data['scenario']}\n\n"
                    if data.get("history_text"):
                        ctx += f"[Dialogue History]\n{data['history_text']}"
        except (json.JSONDecodeError, ValueError):
            return None
    if not profile:
        return None

    rewritten = dict(original)
    rewritten["target_profile"] = profile
    if ctx:
        sm, hm = _SCENE_RE.search(ctx), _HIST_RE.search(ctx)
        if sm:
            rewritten["scenario"] = sm.group(1).strip()
        if hm:
            rewritten["history_text"] = hm.group(1).strip()
        elif not sm:
            rewritten["history_text"] = ctx
    rewritten["is_rewritten"] = True
    return rewritten

def _scenario_len(s):
    return (len(str(s.get("target_profile", "")))
            + len(str(s.get("scenario", "")))
            + len(str(s.get("history_text", ""))))

def compute_score(data_source, solution_str, ground_truth, extra_info=None):

    scenario_src = ground_truth
    if extra_info and isinstance(extra_info, dict) and extra_info.get("scenario"):
        scenario_src = extra_info["scenario"]
    try:
        original = json.loads(scenario_src) if isinstance(scenario_src, str) else scenario_src
    except (json.JSONDecodeError, TypeError):
        return -1.0

    rewritten = _parse_rewrite(original, solution_str)
    if rewritten is None:
        return -1.0

    if (rewritten.get("target_profile") == original.get("target_profile")
            and rewritten.get("history_text") == original.get("history_text")
            and rewritten.get("scenario") == original.get("scenario")):
        return 0.0

    if _scenario_len(rewritten) > 1.5 * _scenario_len(original):
        return -5.0

    n = int(os.environ.get("NUM_ACTOR_SAMPLES", "3"))
    original_mean = _mean_score(original, n)
    rewritten_mean = _mean_score(rewritten, n)
    return original_mean - rewritten_mean
