LANG_NAME = {
    "zh-cn": "Simplified Chinese",
    "zh-tw": "Traditional Chinese",
    "ja":    "Japanese",
    "ko":    "Korean",
    "th":    "Thai",
}

GEN_PROMPTS = {
    "zh-cn": {
        "system": (
            "你是一名角色扮演演员。请严格依据下方提供的角色档案、场景描述与对话历史，"
            "以 {character} 的身份生成下一轮回复。\n"
            "要求：\n"
            "1. 严格保持角色性格、说话风格与背景设定；\n"
            "2. 与场景和对话历史保持一致，不要凭空引入未提及的信息；\n"
            "3. 可以使用方括号 [……] 表示该角色的内心独白，其余为对外说出的台词；\n"
            "4. 仅输出 {character} 的这一轮回复内容，不要复述对话历史，不要附加任何解释。"
        ),
        "user": (
            "【角色档案】\n{profile}\n\n"
            "【其他角色档案】\n{other_profiles}\n\n"
            "【场景】\n{scenario}\n\n"
            "【对话历史】\n{history}\n\n"
            "请以 {character} 的身份生成下一轮回复。"
        ),
    },

    "zh-tw": {
        "system": (
            "你是一名角色扮演演員。請嚴格依據下方提供的角色檔案、場景描述與對話歷史，"
            "以 {character} 的身份生成下一輪回覆。\n"
            "要求：\n"
            "1. 嚴格保持角色性格、說話風格與背景設定；\n"
            "2. 與場景和對話歷史保持一致，不要憑空引入未提及的資訊；\n"
            "3. 可以使用方括號 [……] 表示該角色的內心獨白，其餘為對外說出的台詞；\n"
            "4. 僅輸出 {character} 的這一輪回覆內容，不要複述對話歷史，不要附加任何解釋。"
        ),
        "user": (
            "【角色檔案】\n{profile}\n\n"
            "【其他角色檔案】\n{other_profiles}\n\n"
            "【場景】\n{scenario}\n\n"
            "【對話歷史】\n{history}\n\n"
            "請以 {character} 的身份生成下一輪回覆。"
        ),
    },

    "ja": {
        "system": (
            "あなたはロールプレイの演者です。以下に示すキャラクター設定、場面描写、"
            "会話履歴に厳密に従い、{character} として次の発話を生成してください。\n"
            "要件：\n"
            "1. キャラクターの性格・口調・背景設定を厳密に維持すること；\n"
            "2. 場面と会話履歴に整合し、提示されていない情報を勝手に追加しないこと；\n"
            "3. 角括弧 [……] で囲んだ部分は内心の独白とし、それ以外は発話とする；\n"
            "4. {character} の次の発話のみを出力し、履歴の繰り返しや余計な説明は含めないこと。"
        ),
        "user": (
            "【キャラクター設定】\n{profile}\n\n"
            "【他のキャラクター設定】\n{other_profiles}\n\n"
            "【場面】\n{scenario}\n\n"
            "【会話履歴】\n{history}\n\n"
            "{character} として次の発話を生成してください。"
        ),
    },

    "ko": {
        "system": (
            "당신은 롤플레잉 연기자입니다. 아래 제공된 캐릭터 프로필, 장면 묘사, "
            "대화 기록을 엄격히 따라 {character}의 입장에서 다음 발화를 생성하십시오.\n"
            "요구 사항:\n"
            "1. 캐릭터의 성격, 말투, 배경 설정을 엄격히 유지할 것;\n"
            "2. 장면과 대화 기록과 일치해야 하며, 언급되지 않은 정보를 임의로 도입하지 말 것;\n"
            "3. 대괄호 [……] 안의 내용은 내면의 독백을 의미하며, 그 외는 실제 발화입니다;\n"
            "4. {character}의 다음 발화만 출력하고, 대화 기록의 반복이나 추가 설명은 포함하지 마십시오."
        ),
        "user": (
            "【캐릭터 프로필】\n{profile}\n\n"
            "【다른 캐릭터 프로필】\n{other_profiles}\n\n"
            "【장면】\n{scenario}\n\n"
            "【대화 기록】\n{history}\n\n"
            "{character}의 입장에서 다음 발화를 생성하십시오."
        ),
    },

    "th": {
        "system": (
            "คุณคือนักแสดงบทบาทสมมติ โปรดสร้างบทพูดถัดไปในนามของ {character} "
            "โดยยึดตามโปรไฟล์ตัวละคร คำบรรยายฉาก และประวัติการสนทนาที่ให้ไว้ด้านล่างอย่างเคร่งครัด\n"
            "ข้อกำหนด:\n"
            "1. รักษาบุคลิก น้ำเสียง และฉากหลังของตัวละครอย่างเคร่งครัด;\n"
            "2. สอดคล้องกับฉากและประวัติการสนทนา ห้ามเพิ่มข้อมูลที่ไม่ได้กล่าวถึงโดยพลการ;\n"
            "3. ข้อความในวงเล็บเหลี่ยม [……] ถือเป็นความคิดในใจ ส่วนที่เหลือคือคำพูดที่เปล่งออกมา;\n"
            "4. ให้แสดงผลเฉพาะบทพูดถัดไปของ {character} เท่านั้น "
            "ห้ามทวนประวัติการสนทนาและห้ามเพิ่มคำอธิบายใด ๆ"
        ),
        "user": (
            "【โปรไฟล์ตัวละคร】\n{profile}\n\n"
            "【โปรไฟล์ตัวละครอื่น】\n{other_profiles}\n\n"
            "【ฉาก】\n{scenario}\n\n"
            "【ประวัติการสนทนา】\n{history}\n\n"
            "โปรดสร้างบทพูดถัดไปในนามของ {character}"
        ),
    },
}

_JUDGE_OUTPUT_FORMAT = (
    'Return your judgement strictly as a single JSON object with the following keys '
    'and 0-10 integer (or one-decimal float) scores:\n'
    '{{\n'
    '  "CC": <score 0-10>,   // Character Consistency\n'
    '  "SC": <score 0-10>,   // Situational Coherence\n'
    '  "LQ": <score 0-10>,   // Linguistic Quality\n'
    '  "IP": <score 0-10>,   // Interaction Proactivity\n'
    '  "comment": "<brief 1-2 sentence rationale>"\n'
    '}}\n'
    'Do NOT output anything outside of this JSON object.'
)

_JUDGE_DIMENSIONS_EN = (
    "Score the model response on a 0-10 scale (0 = totally fails, 10 = perfect) along the "
    "following four dimensions:\n"
    "  - CC (Character Consistency): Does the response stay faithful to the given "
    "character's personality, speaking style, background and relationships?\n"
    "  - SC (Situational Coherence): Is the response coherent with the scenario and "
    "the dialogue history? Does it avoid introducing facts that contradict or have not "
    "been established?\n"
    "  - LQ (Linguistic Quality): Is the language fluent, natural, and appropriate to "
    "the target language ({lang_name})? Free of awkward phrasing, machine-translation "
    "artifacts, or wrong-language tokens?\n"
    "  - IP (Interaction Proactivity): Does the response actively move the conversation "
    "forward (e.g. by reacting to the partner, expressing the character's intent, or "
    "introducing in-character actions/inner thoughts), rather than giving a flat or "
    "evasive reply?"
)

JUDGE_PROMPTS = {
    lang: {
        "system": (
            "You are a strict but fair role-play evaluator. You will be given a "
            "role-play scenario in {lang_name}, including a character profile, a scene "
            "description, the dialogue history, and a model-generated response that is "
            "supposed to be the next utterance of a designated character. "
            "Evaluate the response only; do not rewrite it.\n\n"
            + _JUDGE_DIMENSIONS_EN + "\n\n"
            + _JUDGE_OUTPUT_FORMAT
        ).format(lang_name=LANG_NAME[lang]),
        "user": (
            "[Target language] {lang_name}\n"
            "[Character to play] {character}\n\n"
            "[Character profile]\n{profile}\n\n"
            "[Other characters' profiles]\n{other_profiles}\n\n"
            "[Scenario]\n{scenario}\n\n"
            "[Dialogue history]\n{history}\n\n"
            "[Model response to evaluate]\n{response}\n\n"
            "Now produce your JSON judgement."
        ),
    }
    for lang in LANG_NAME
}

def render_dialogue_history(turns):

    lines = []
    for t in turns:
        c = t.get("character", "")
        m = t.get("message", "")
        lines.append(f"{c}: {m}")
    return "\n".join(lines) if lines else "(no prior dialogue)"

def render_other_profiles(profiles_dict, target_character):

    items = []
    for name, prof in profiles_dict.items():
        if name == target_character or not prof:
            continue
        items.append(f"- {name}: {prof}")
    return "\n".join(items) if items else "(none)"

def build_gen_messages(lang, sample):

    tpl = GEN_PROMPTS[lang]
    sys = tpl["system"].format(character=sample["target_character"])
    usr = tpl["user"].format(
        character=sample["target_character"],
        profile=sample["target_profile"],
        other_profiles=sample["other_profiles_text"],
        scenario=sample["scenario"],
        history=sample["history_text"],
    )
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": usr},
    ]

def build_judge_messages(lang, sample, model_response):

    tpl = JUDGE_PROMPTS[lang]
    sys = tpl["system"]
    usr = tpl["user"].format(
        lang_name=LANG_NAME[lang],
        character=sample["target_character"],
        profile=sample["target_profile"],
        other_profiles=sample["other_profiles_text"],
        scenario=sample["scenario"],
        history=sample["history_text"],
        response=model_response,
    )
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": usr},
    ]
