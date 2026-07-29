from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from typing import Dict, List

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.common import chat, make_client

logger = logging.getLogger(__name__)

TURNS = 30
ENV_EVERY = 5

OUTPUT_FORMAT = ("Your output should include **thought**, **speech**, and **action**. "
                 "Use [your thought] for thoughts, which others can't see. "
                 "Use (your action) for actions, which others can see.")

def character_system_prompt(book: str, character: str, profile: str,
                            plot_summary: str, scenario: str, motivation: str,
                            other_profiles: Dict[str, str]) -> str:
    others = "\n\n".join(f"{n}: {p}" for n, p in other_profiles.items() if n != character)
    return (f"You are {character} from {book}.\n\n"
            f"==={character}'s Profile===\n{profile}\n\n"
            f"===Current Scenario===\nBackground: {plot_summary}\n"
            f"Currently: {scenario}\n\n"
            f"===Information about the other Characters===\n{others}\n\n"
            f"===Your Inner Thoughts===\n{motivation}\n\n"
            f"===Requirements===\n{OUTPUT_FORMAT}\n\n"
            f'Start your response with "{character}: ". Avoid speaking as other '
            f"characters. Speak concisely as humans, limit your response to 60 words.")

def environment_system_prompt(major_characters: List[str], scenario: str) -> str:
    return ("You are a world model for a role-playing game. Your task is to provide "
            "environmental feedback: based on the characters' interactions, dialogues, "
            "and actions, describe the resulting changes in the environment (physical "
            "changes, reactions of background characters, ambient sounds, weather, "
            "atmosphere). Keep descriptions to 1-3 sentences and do not dictate the "
            "actions or dialogue of the main characters "
            f"({', '.join(major_characters)}).\n\nCurrent scenario:\n{scenario}")

def simulate_one(client_m, model_name, client_j, judge_name, item: Dict) -> Dict:
    book = item.get("book", "the story")
    profiles = item.get("character_profiles", {})
    majors = item.get("major_characters", [])
    if not majors or majors[0] not in profiles:
        return {"id": item.get("i_p"), "error": "no major character/profile", "turns": []}
    protagonist = majors[0]
    partner = majors[1] if len(majors) > 1 else None
    scenario = item.get("scenario", "")
    plot_summary = (item.get("plot") or {}).get("summary", "")
    motivations = {kc["name"]: kc.get("thought", "")
                   for kc in item.get("key_characters", [])}

    sys_actor = character_system_prompt(
        book, protagonist, profiles.get(protagonist, ""), plot_summary,
        scenario, motivations.get(protagonist, ""), profiles)
    sys_partner = (character_system_prompt(
        book, partner, profiles.get(partner, ""), plot_summary,
        scenario, motivations.get(partner, ""), profiles) if partner else None)
    sys_env = environment_system_prompt(majors, scenario)

    hist_actor = [{"role": "system", "content": sys_actor}]
    hist_partner = [{"role": "system", "content": sys_partner}] if partner else None
    hist_env = [{"role": "system", "content": sys_env}]
    conversation = []

    def fmt(convo):
        return "\n".join(f"{c['character']}: {c['message']}" for c in convo)

    round_i = 0
    while len([c for c in conversation if c["character"] != "Environment"]) < TURNS:
        round_i += 1
        if partner:
            user_msg = fmt(conversation) or f"(The scene begins. Scenario: {scenario})"
            resp = chat(client_j, judge_name,
                        hist_partner + [{"role": "user", "content": user_msg}],
                        temperature=0.7, max_tokens=200)
            resp = re.sub(rf"^\s*{re.escape(partner)}\s*:\s*", "", resp.strip()).strip()
            conversation.append({"character": partner, "message": resp})
            hist_partner.append({"role": "user", "content": user_msg})
            hist_partner.append({"role": "assistant", "content": resp})

        if round_i % ENV_EVERY == 0:
            resp = chat(client_j, judge_name,
                        hist_env + [{"role": "user", "content": fmt(conversation)}],
                        temperature=0.7, max_tokens=120)
            conversation.append({"character": "Environment", "message": resp.strip()})
            hist_env.append({"role": "user", "content": fmt(conversation)})
            hist_env.append({"role": "assistant", "content": resp.strip()})

        user_msg = fmt(conversation) or f"(The scene begins. Scenario: {scenario})"
        resp = chat(client_m, model_name,
                    hist_actor + [{"role": "user", "content": user_msg}],
                    temperature=0.7, max_tokens=300)
        resp = re.sub(rf"^\s*{re.escape(protagonist)}\s*:\s*", "", resp.strip()).strip()
        conversation.append({"character": protagonist, "message": resp})
        hist_actor.append({"role": "user", "content": user_msg})
        hist_actor.append({"role": "assistant", "content": resp})

    return {"id": f"coser_{item.get('i_p')}_{item.get('i_c')}",
            "book": book, "protagonist": protagonist,
            "turns": conversation, "reference": item.get("dialogues", [])}

DIMENSION_DETAILS = {
    "Storyline Consistency": (
        "Whether the storyline and characters' reactions in the simulated "
        "conversation align well with those in the reference conversation",
        "### Storyline Consistency\n- Type: Storyline Consistency\n"
        "  * Characters' reactions (emotions, attitudes, behaviors) in the "
        "simulated conversation deviate from those in the original conversation"),
    "Anthropomorphism": (
        "How human-like and natural the characters behave",
        "### Anthroporphism\n- Type: Self-identity\n  * Lacks initiative and goals; "
        "behaves like a 'helpful AI assistant'\n- Type: Emotional Depth\n"
        "  * Lacks psychological complexity; speaks out all thoughts directly\n"
        "- Type: Persona Coherence\n  * Inconsistent personality traits\n"
        "- Type: Social Interaction\n  * Rigid reactions; poor social skills"),
    "Character Fidelity": (
        "How well the characters match their established profiles from the book",
        "### Character Fidelity\n- Type: Character Language\n"
        "  * Vocabulary/tone inappropriate for the character\n"
        "- Type: Knowledge & Background\n  * Missing character-specific knowledge; "
        "future information leaks\n- Type: Personality & Behavior\n"
        "  * Emotions/decisions conflict with the profile\n"
        "- Type: Relationship & Social Status\n  * Inappropriate interactions"),
    "Storyline Quality": (
        "How well the conversation maintains logical consistency and narrative quality",
        "### Storyline Quality\n- Type: Flow & Progression\n"
        "  * Unnatural progression; verbose/repetitive dialogue\n"
        "- Type: Logical Consistency\n  * Factual contradictions"),
}

CRITIC_TEMPLATE = """You are a literary critic specializing in character analysis and dialogue evaluation. Given a simulated conversation for a plot in {book}, your task is to evaluate this conversation via the following steps:

1. Read and understand the provided materials about {book}:
   * Story context and scenario.
   * Profiles of the main characters, including {major_characters}.
   * The original conversation from {book} in the same scenario as a reference.

2. Evaluate the simulated conversation in terms of {dimension_name}, i.e., {dimension_brief}.
   Each character message is composed of speech, action (wrapped within parentheses), and inner thoughts (wrapped within square brackets). The inner thoughts are not spoken aloud and are thus invisible to other characters.

## Scenario

### Plot Summary

{plot_summary}

### Current Scenario

{scenario}

## Character Profiles

{character_profiles}

## Original Conversation

{original_conversation}

## Evaluation Criteria

To evaluate the simulated conversation, identify the following types of flaws:

{dimension_criteria}

## Scoring Guidelines

1. Identify all instances of flaws occurred in the simulated conversation.
2. For each flaw identified, determine its level of severity from 1 to 5, where 1 indicates minor, 3 indicates moderate, and 5 indicates severe.

## Output Requirements

Provide your evaluation in JSON format:

{{ "{dimension_name}": {{ "flaws": [ {{ "instance": "<comment>", "type": "<flaw type>", "severity": <1-5> }} ] }} }}

===Dialogue Content===
{simulated_conversation}"""

def _extract_json(text: str):
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0]
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None

def judge_conversation(client_j, judge_name, item: Dict, sim: Dict) -> Dict:
    if sim.get("error"):
        return {"error": sim["error"]}
    book = item.get("book", "the story")
    profiles = item.get("character_profiles", {})
    majors = item.get("major_characters", [])
    convo_str = "\n".join(f"{c['character']}: {c['message']}" for c in sim["turns"])
    ref_str = "\n".join(f"{d.get('character', '?')}: {d.get('message', '')}"
                        for d in item.get("dialogues", []))
    profile_str = "\n\n".join(f"{n}: {p}" for n, p in profiles.items())
    actor_rounds = len([c for c in sim["turns"] if c["character"] in majors])

    scores = {}
    flaws_all = {}
    for dim, (brief, criteria) in DIMENSION_DETAILS.items():
        prompt = CRITIC_TEMPLATE.format(
            book=book, major_characters=", ".join(majors),
            dimension_name=dim, dimension_brief=brief,
            plot_summary=(item.get("plot") or {}).get("summary", ""),
            scenario=item.get("scenario", ""),
            character_profiles=profile_str,
            original_conversation=ref_str,
            dimension_criteria=criteria.replace("{major_characters}", ", ".join(majors)),
            simulated_conversation=convo_str,
        )
        raw = chat(client_j, judge_name, [{"role": "user", "content": prompt}],
                   temperature=0.0, max_tokens=1024)
        data = _extract_json(raw)
        flaws = []
        if data:
            node = data.get(dim, data)
            if isinstance(node, dict):
                flaws = node.get("flaws", []) or []
        severities = [int(f.get("severity", 1)) for f in flaws
                      if isinstance(f, dict) and
                      isinstance(f.get("severity", 1), (int, float))]
        score = max(0.0, min(100.0 - (sum(severities) - 0.3 * actor_rounds) * 5, 100.0))
        scores[dim] = round(score, 2)
        flaws_all[dim] = flaws
    return {"scores": scores, "flaws": flaws_all, "actor_rounds": actor_rounds}

def corpus_ngram_metrics(results: List[Dict]) -> Dict[str, float]:
    import nltk
    from rouge_score import rouge_scorer

    nltk.download("punkt_tab", quiet=True)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    refs, hyps = [], []
    rouge_hits = []
    for r in results:
        if r.get("error"):
            continue
        ref_text = " ".join(d.get("message", "") for d in r.get("reference", []))
        hyp_text = " ".join(c["message"] for c in r.get("turns", [])
                            if c["character"] == r.get("protagonist"))
        if not ref_text.strip() or not hyp_text.strip():
            continue
        refs.append(ref_text)
        hyps.append(hyp_text)
        rouge_hits.append(scorer.score(ref_text, hyp_text)["rougeL"].fmeasure)

    bleu = 0.0
    if refs:
        try:
            ref_toks = [nltk.word_tokenize(r) for r in refs]
            hyp_toks = [nltk.word_tokenize(h) for h in hyps]
            bleu = nltk.translate.bleu_score.corpus_bleu(
                [[rt] for rt in ref_toks], hyp_toks)
        except (ZeroDivisionError, ValueError):
            bleu = 0.0
    return {"bleu": round(bleu, 4),
            "rouge_l": round(sum(rouge_hits) / len(rouge_hits), 4) if rouge_hits else 0.0}

def main():
    global TURNS
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--model-endpoint", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--model-key", default="EMPTY")
    p.add_argument("--judge-endpoint", required=True)
    p.add_argument("--judge-name", required=True)
    p.add_argument("--judge-key", default="EMPTY")
    p.add_argument("--test", default="data/processed/coser/test_set_original.json")
    p.add_argument("--output", required=True)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--turns", type=int, default=TURNS)
    args = p.parse_args()

    TURNS = args.turns

    with open(args.test, "r", encoding="utf-8") as f:
        items = json.load(f)
    if args.max_samples:
        items = items[:args.max_samples]
    logger.info(f"CoSER test items: {len(items)}")

    client_m = make_client(args.model_endpoint, args.model_key)
    client_j = make_client(args.judge_endpoint, args.judge_key)

    sims = []
    for item in tqdm(items, desc="simulating"):
        sims.append(simulate_one(client_m, args.model_name,
                                 client_j, args.judge_name, item))

    judgments = []
    for item, sim in tqdm(zip(items, sims), total=len(items), desc="judging"):
        j = judge_conversation(client_j, args.judge_name, item, sim)
        j["id"] = sim.get("id")
        judgments.append(j)

    dims = list(DIMENSION_DETAILS)
    dim_avgs = {}
    for d in dims:
        vals = [j["scores"][d] for j in judgments if "scores" in j]
        dim_avgs[d] = round(sum(vals) / len(vals), 2) if vals else None
    valid = [v for v in dim_avgs.values() if v is not None]
    ngrams = corpus_ngram_metrics(
        [{"turns": s.get("turns", []), "reference": s.get("reference", []),
          "protagonist": s.get("protagonist"), "error": s.get("error")}
         for s in sims])

    summary = {"dimensions": dim_avgs,
               "average": round(sum(valid) / len(valid), 2) if valid else None,
               **ngrams}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "simulations": sims,
                   "judgments": judgments}, f, ensure_ascii=False, indent=2)

    print("\n===== CoSER results =====")
    for d, v in dim_avgs.items():
        print(f"  {d:<24} {v}")
    print(f"  {'Average':<24} {summary['average']}")
    print(f"  BLEU {ngrams['bleu']}   ROUGE-L {ngrams['rouge_l']}")
    print(f"Saved -> {args.output}")

if __name__ == "__main__":
    main()
