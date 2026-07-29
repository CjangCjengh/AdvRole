from __future__ import annotations

from typing import Dict, List

ACTOR_SYSTEM_PROMPT = """You are a role-playing actor. Respond strictly in character as {character} based on the provided character profile, scenario, and dialogue history.

Requirements:
1. Stay faithful to the character's personality, speaking style, and background.
2. Be coherent with the scenario and dialogue history; do not introduce unestablished facts.
3. Use brackets [...] for inner thoughts; everything else is spoken dialogue.
4. Output only {character}'s next response. Do not repeat the dialogue history or add explanations."""

ACTOR_USER_PROMPT = """[Character profile]
{profile}

[Other characters' profiles]
{other_profiles}

[Scenario]
{scenario}

[Dialogue history]
{history}

Respond as {character}."""

def build_actor_messages(character: str, profile: str, other_profiles: str,
                         scenario: str, history: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": ACTOR_SYSTEM_PROMPT.format(character=character)},
        {"role": "user",
         "content": ACTOR_USER_PROMPT.format(
             character=character,
             profile=profile or "(not provided)",
             other_profiles=other_profiles or "(none)",
             scenario=scenario or "(not specified)",
             history=history or "(the conversation starts here)",
         )},
    ]

REWRITER_SYSTEM_PROMPT = """You are an adversarial scenario editor for role-playing training. Given an original role-playing scenario consisting of a character profile and a dialogue context, your task is to edit either the profile or the context (or both) to create a more challenging version that reveals where the actor model may fail to stay in character.

Your edits should satisfy the following criteria:
- Weakness Exposure: The rewritten scenario should expose where the actor may fail to stay in character. Introduce subtle contradictions, unusual situations, or complex emotional states that require deeper character understanding.
- Diversity: The edit should explore aspects of the character or situation that are underrepresented in the original scenario pool.
- Plausibility: The rewritten scenario must remain internally consistent and realistic within the story's universe.
- Minimal but Effective: Make targeted edits rather than rewriting everything from scratch. Small changes that reveal actor weaknesses are preferred."""

REWRITER_USER_PROMPT = """[Original Character Profile]
{profile}

[Original Dialogue Context]
{dialogue_context}

[Target Character]
{character_name}

Please edit the scenario to better expose the actor model's weaknesses. Output the rewritten profile and dialogue context using exactly these two section headers:

[Rewritten Character Profile]
<the rewritten profile>

[Rewritten Dialogue Context]
<the rewritten scene + dialogue history, in the same [Scene]/[Dialogue History] format>
"""

def build_rewriter_messages(profile: str, dialogue_context: str,
                            character_name: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": REWRITER_SYSTEM_PROMPT},
        {"role": "user",
         "content": REWRITER_USER_PROMPT.format(
             profile=profile,
             dialogue_context=dialogue_context,
             character_name=character_name,
         )},
    ]

def rewriter_dialogue_context(scenario: Dict) -> str:
    ctx = ""
    if scenario.get("scenario"):
        ctx += f"[Scene] {scenario['scenario']}\n\n"
    if scenario.get("history_text"):
        ctx += f"[Dialogue History]\n{scenario['history_text']}"
    return ctx or "(empty)"

RM_SYSTEM_PROMPT_GENERIC = """You are a strict but fair role-play evaluator. You will be given a role-play scenario including a character profile, a dialogue context, and a model-generated response that is supposed to be the next utterance of a designated character. Evaluate the response only; do not rewrite it.

Score the model response on a 0-10 scale (0 = totally fails, 10 = perfect) along the following four dimensions:

- CC (Character Consistency): Does the response stay faithful to the given character's personality, speaking style, background and relationships?
- SC (Situational Coherence): Is the response coherent with the scenario and the dialogue history? Does it avoid introducing facts that contradict or have not been established?
- LQ (Linguistic Quality): Is the language fluent, natural, and appropriate? Free of awkward phrasing, machine-translation artifacts, or wrong-language tokens?
- IP (Interaction Proactivity): Does the response actively move the conversation forward (e.g. by reacting to the partner, expressing the character's intent, or introducing in-character actions/inner thoughts), rather than giving a flat or evasive reply?

Return your judgement strictly as a single JSON object:
{"CC": <score>, "SC": <score>, "LQ": <score>, "IP": <score>}

Do NOT output anything outside of this JSON object."""

RM_USER_PROMPT_GENERIC = """[Character to play] {character}

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

RM_SYSTEM_PROMPT_CHARACTEREVAL = """You are a professional roleplay quality evaluation expert. You will score the model-generated character response on multiple dimensions based on the given character profile and dialogue context."""

RM_USER_PROMPT_CHARACTEREVAL = """Please score the character response based on the following information.

[Character Profile]
{profile}

[Dialogue Context]
{history}

[Character Response]
{response}

Please rate the response on a scale of 1-5 (1=very poor, 5=excellent) along the following dimensions, and provide a brief justification:

1. Conversational Ability:
   - Fluency: Is the response grammatically smooth and naturally expressed?
   - Coherence: Is the response logically coherent with the context?
   - Consistency: Is the response internally consistent and free of contradictions?

2. Character Consistency:
   - Knowledge Accuracy: Does the response correctly reflect information from the character profile?
   - No Hallucination: Does the response avoid fabricating information not present in the character profile?
   - Behavioral Plausibility: Are the character's actions consistent with their identity and personality?
   - Language Style: Does the character's speaking style match their persona?

3. Roleplay Appeal:
   - Human-likeness: Does the response feel like it was written by a human rather than a machine?
   - Communication Skills: Does the character demonstrate good conversational interaction ability?
   - Expression Diversity: Does the response avoid monotonous and repetitive expressions?
   - Empathy: Does the character appropriately respond to the other party's emotions?

Please output in JSON format as follows:
{{"scores": {{"fluency": <1-5>, "coherence": <1-5>, "consistency": <1-5>, "knowledge_accuracy": <1-5>, "no_hallucination": <1-5>, "behavioral_plausibility": <1-5>, "language_style": <1-5>, "human_likeness": <1-5>, "communication_skills": <1-5>, "expression_diversity": <1-5>, "empathy": <1-5>}}, "overall": <1-5>, "reason": "..."}}"""

RM_SYSTEM_PROMPT_COSER = """You are an expert evaluator for role-playing dialogue quality. Given a character profile, a dialogue scenario, and a model-generated response, you will assess the response along multiple role-playing dimensions."""

RM_USER_PROMPT_COSER = """Evaluate the following role-playing response based on the provided context.

[Character Profile]
{profile}

[Scenario]
{scenario}

[Dialogue Context]
{history}

[Model Response]
{response}

Rate the response on a scale of 1-10 (1 = very poor, 10 = excellent) along the following four dimensions:

1. Storyline Consistency: Does the response maintain coherence with the established plot, key events, and narrative direction? Does it avoid introducing contradictions to the storyline?
2. Anthropomorphism: Does the response feel natural and human-like? Does it exhibit emotional depth, self-awareness, and avoid robotic or repetitive patterns?
3. Character Fidelity: Does the response faithfully reflect the character's personality, motivations, speaking style, and behavioral patterns as described in the profile?
4. Storyline Quality: Considered independently, does the response contribute to an engaging and dramatically compelling dialogue? Does it advance the conversation with appropriate pacing and tension?

Output your evaluation strictly as a JSON object:
{{"storyline_consistency": <1-10>, "anthropomorphism": <1-10>, "character_fidelity": <1-10>, "storyline_quality": <1-10>, "overall": <1-10>, "comment": "<brief rationale>"}}"""

def build_reward_messages(dataset: str, scenario: Dict, response: str) -> List[Dict[str, str]]:
    profile = scenario.get("target_profile", "")
    history = scenario.get("history_text", "")
    if dataset == "charactereval":
        return [
            {"role": "system", "content": RM_SYSTEM_PROMPT_CHARACTEREVAL},
            {"role": "user", "content": RM_USER_PROMPT_CHARACTEREVAL.format(
                profile=profile, history=history, response=response)},
        ]
    if dataset == "coser":
        return [
            {"role": "system", "content": RM_SYSTEM_PROMPT_COSER},
            {"role": "user", "content": RM_USER_PROMPT_COSER.format(
                profile=profile,
                scenario=scenario.get("scenario", ""),
                history=history,
                response=response,
            )},
        ]
    return [
        {"role": "system", "content": RM_SYSTEM_PROMPT_GENERIC},
        {"role": "user", "content": RM_USER_PROMPT_GENERIC.format(
            character=scenario.get("target_character", ""),
            profile=profile,
            other_profiles=scenario.get("other_profiles_text", "") or "(none)",
            scenario=scenario.get("scenario", "") or "(not specified)",
            history=history or "(the conversation starts here)",
            response=response,
        )},
    ]
