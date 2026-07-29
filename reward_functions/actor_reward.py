import json
import os
import re
import threading

from openai import OpenAI

_lock = threading.Lock()
_client = None

def _get_client():
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = OpenAI(
                    base_url=os.environ.get("RM_API_BASE", "http://localhost:8000/v1"),
                    api_key=os.environ.get("RM_API_KEY", "EMPTY"),
                    timeout=120.0,
                )
    return _client

RM_SYSTEM_GENERIC = """You are a strict but fair role-play evaluator. Score the model response on a 0-10 scale along four dimensions:
- CC (Character Consistency): faithful to the character's personality, style, background?
- SC (Situational Coherence): coherent with scenario and history?
- LQ (Linguistic Quality): fluent, natural language?
- IP (Interaction Proactivity): actively moves the conversation forward?

Return strictly a JSON object: {"CC": <score>, "SC": <score>, "LQ": <score>, "IP": <score>}"""

RM_USER_GENERIC = """[Character to play] {character}

[Character profile]
{profile}

[Other characters' profiles]
{other_profiles}

[Scenario]
{scenario}

[Dialogue history]
{history}

[Model response to evaluate]
{response}

Now produce your JSON judgement."""

RM_SYSTEM_CE = "You are a professional roleplay quality evaluation expert. You will score the model-generated character response on multiple dimensions based on the given character profile and dialogue context."

RM_USER_CE = """Please score the character response based on the following information.

[Character Profile]
{profile}

[Dialogue Context]
{history}

[Character Response]
{response}

Rate on a 1-5 scale along: Conversational Ability (fluency, coherence, consistency), Character Consistency (knowledge accuracy, no hallucination, behavioral plausibility, language style), Roleplay Appeal (human-likeness, communication skills, expression diversity, empathy).

Output JSON: {{"scores": {{"fluency": <1-5>, "coherence": <1-5>, "consistency": <1-5>, "knowledge_accuracy": <1-5>, "no_hallucination": <1-5>, "behavioral_plausibility": <1-5>, "language_style": <1-5>, "human_likeness": <1-5>, "communication_skills": <1-5>, "expression_diversity": <1-5>, "empathy": <1-5>}}, "overall": <1-5>, "reason": "..."}}"""

RM_SYSTEM_COSER = "You are an expert evaluator for role-playing dialogue quality."

RM_USER_COSER = """Evaluate the following role-playing response.

[Character Profile]
{profile}

[Scenario]
{scenario}

[Dialogue Context]
{history}

[Model Response]
{response}

Rate 1-10 on: Storyline Consistency, Anthropomorphism, Character Fidelity, Storyline Quality.
Output JSON: {{"storyline_consistency": <1-10>, "anthropomorphism": <1-10>, "character_fidelity": <1-10>, "storyline_quality": <1-10>, "overall": <1-10>, "comment": "..."}}"""

def _build_messages(dataset, scenario, response):
    profile = scenario.get("target_profile", "")
    history = scenario.get("history_text", "")
    if dataset == "charactereval":
        return [{"role": "system", "content": RM_SYSTEM_CE},
                {"role": "user", "content": RM_USER_CE.format(
                    profile=profile, history=history, response=response)}]
    if dataset == "coser":
        return [{"role": "system", "content": RM_SYSTEM_COSER},
                {"role": "user", "content": RM_USER_COSER.format(
                    profile=profile, scenario=scenario.get("scenario", ""),
                    history=history, response=response)}]
    return [{"role": "system", "content": RM_SYSTEM_GENERIC},
            {"role": "user", "content": RM_USER_GENERIC.format(
                character=scenario.get("target_character", ""),
                profile=profile,
                other_profiles=scenario.get("other_profiles_text", "") or "(none)",
                scenario=scenario.get("scenario", "") or "(not specified)",
                history=history or "(the conversation starts here)",
                response=response)}]

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

def score_response(scenario, response, retries=2):

    coeff = length_coefficient(response)
    if coeff == 0.0:
        return 0.0
    dataset = os.environ.get("RM_DATASET", "generic")
    model = os.environ.get("RM_MODEL", "rm")
    temperature = float(os.environ.get("RM_TEMPERATURE", "0.0"))
    max_tokens = int(os.environ.get("RM_MAX_TOKENS", "256"))
    messages = _build_messages(dataset, scenario, response)

    for _ in range(retries + 1):
        try:
            resp = _get_client().chat.completions.create(
                model=model, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
            data = _extract_json(resp.choices[0].message.content)
            if data is None:
                continue
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
            if raw is not None:
                return raw * coeff
        except Exception:
            continue
    return 0.0

def compute_score(data_source, solution_str, ground_truth, extra_info=None):

    scenario_src = ground_truth
    if extra_info and isinstance(extra_info, dict) and extra_info.get("scenario"):
        scenario_src = extra_info["scenario"]
    try:
        scenario = json.loads(scenario_src) if isinstance(scenario_src, str) else scenario_src
    except (json.JSONDecodeError, TypeError):
        return 0.0
    return score_response(scenario, solution_str or "")
