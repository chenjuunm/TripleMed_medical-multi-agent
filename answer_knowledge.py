"""Liked clinical plans: fuzzy cross-patient retrieval, identity filtering, HITL."""
import hashlib
import json
import re
import sqlite3
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from agents import SynthesisOutput, CriticOutput

def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)

TOPICS = {
    "咽痛": ("咽痛", "喉咙痛", "sore throat"),
    "发热": ("发热", "发烧", "fever"),
    "咳嗽": ("咳嗽", "cough"),
    "胸痛": ("胸痛", "chest pain"),
    "气促": ("气促", "呼吸困难", "shortness of breath"),
    "头痛": ("头痛", "headache"),
    "头晕": ("头晕", "dizziness"),
    "腹痛": ("腹痛", "肚子痛", "abdominal pain"),
    "腹泻": ("腹泻", "拉肚子", "diarrhea"),
    "呕吐": ("呕吐", "vomiting"),
    "皮疹": ("皮疹", "rash"),
    "乏力": ("乏力", "疲劳", "fatigue"),
    "心悸": ("心悸", "palpitations"),
    "水肿": ("水肿", "edema"),
    "血压": ("血压", "高血压", "hypertension"),
    "血糖": ("血糖", "糖尿病", "diabetes"),
    "腰痛": ("腰痛", "back pain"),
    "尿痛": ("尿痛", "dysuria"),
}

# This is an identity filter, not a clinical-content summarizer. It deliberately
# preserves diagnoses, examination names, medications, doses and clinical text.
PRIVATE_KEYS = {
    "name", "patientname", "fullname", "姓名", "患者姓名", "患者", "patient",
    "patientid", "caseid", "sourcecaseid", "recordid", "病历号", "患者id",
    "phone", "telephone", "mobile", "电话", "手机号", "email", "邮箱",
    "address", "住址", "地址", "身份证", "身份证号", "idcard", "idnumber",
    "passport", "护照", "nric", "contact", "联系人", "dob", "dateofbirth", "出生日期",
}
LABELLED_ID = re.compile(
    r"(?:患者姓名|姓名|患者名|patient[ _-]?name|name|住址|地址|address|病历号|患者编号|"
    r"patient[ _-]?id|case[ _-]?id|身份证号?|id[ _-]?number|电话|手机(?:号)?|"
    r"phone|mobile|邮箱|email|出生日期|date[ _-]?of[ _-]?birth)"
    r"\s*[:：=]\s*([^\n，,；;。]+)", re.I)
INLINE_NAME = re.compile(r"(?:患者|病人)\s*([\u4e00-\u9fff]{2,4})(?=[，,：:\s])")
CONTACTS = (
    re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
    re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)"),
    re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
    re.compile(r"\b[STFGM]\d{7}[A-Z]\b", re.I),
    re.compile(r"(?<!\d)(?:\+65[- ]?)?[689]\d{3}[- ]?\d{4}(?!\d)"),
)
MASK = "[原患者个人信息已隐藏]"

def _private_key(key):
    return re.sub(r"[\s_-]", "", str(key)).casefold() in PRIVATE_KEYS

def identity_tokens(patient_id, context, payload):
    tokens = {str(patient_id)} if patient_id else set()
    def visit(value, private=False):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, private or _private_key(key))
        elif isinstance(value, list):
            for child in value:
                visit(child, private)
        elif private and isinstance(value, (int, float)):
            tokens.add(str(value))
        elif isinstance(value, str):
            if private and value.strip():
                tokens.add(value.strip())
            tokens.update(match.group(1).strip() for match in LABELLED_ID.finditer(value))
    visit(context)
    visit(payload.get("case_snapshot", {}))
    visit(payload.get("evidence", []))
    for item in payload.get("evidence", []):
        if isinstance(item, dict) and isinstance(item.get("source"), dict):
            for key in ("source_id", "record_id", "source_url"):
                if item["source"].get(key):
                    tokens.add(str(item["source"][key]))
    # Free-text explicitly labelled identities may also occur inside reports.
    visit(payload.get("final_report", ""))
    return sorted(tokens, key=len, reverse=True)

def filter_identity(value, tokens=()):
    if isinstance(value, dict):
        return {key: filter_identity(child, tokens) for key, child in value.items()
                if not _private_key(key)}
    if isinstance(value, list):
        return [filter_identity(child, tokens) for child in value]
    if not isinstance(value, str):
        return value
    text = value
    for token in tokens:
        if token:
            # Do not erase clinical doses merely because an identifier is numeric.
            pattern = (r"(?<![A-Za-z0-9])" + re.escape(token) + r"(?![A-Za-z0-9])"
                       if token.isascii() and token.isalnum() else re.escape(token))
            text = re.sub(pattern, lambda _: MASK, text)
    text = LABELLED_ID.sub(lambda match: match.group(0).replace(match.group(1), MASK), text)
    for match in INLINE_NAME.finditer(text):
        # "患者咽痛" is clinical content, not a name. Do not guess and erase it.
        # Unknown name-like fragments need a structured identity field or manual
        # privacy review before this text is shared with another patient.
        if match.group(1) not in {alias for aliases in TOPICS.values() for alias in aliases}:
            raise ValueError("存在未明确区分的患者姓名/描述，请先核对个人信息或补充结构化姓名字段。")
    for pattern in CONTACTS:
        text = pattern.sub(MASK, text)
    # Clinical source URLs may embed identity in paths/query strings. Do not
    # propagate clickable original URLs as part of a shared plan.
    return re.sub(r"https?://[^\s<>）)]+", "[历史链接已隐藏]", text)

def normalized_complaint(question):
    text = unicodedata.normalize("NFKC", filter_identity(question)).casefold()
    for topic, aliases in TOPICS.items():
        for alias in sorted(aliases, key=len, reverse=True):
            text = text.replace(alias, topic)
    text = re.sub(r"[零一二两三四五六七八九十百\d]+(?:个)?(?:小时|天|日|周|星期|个月|月)(?:左右|余)?", "", text)
    for word in ("请帮我看看", "怎么治疗", "怎么办", "请问", "这几天", "最近", "感觉", "出现", "有点", "有些", "一直"):
        text = text.replace(word, "")
    for word in ("没有", "否认", "不伴", "未见"):
        text = text.replace(word, "无")
    text = re.sub(r"\b(?:no|without)\s*", "无", text)
    clauses = re.split(r"[，,。；;！？!?\n]", text.replace(MASK, ""))
    return "|".join(sorted(part for clause in clauses
                           if (part := re.sub(r"[\W_]+", "", clause))))

def symptom_signature(text):
    # Use the same Chinese negation handling as the existing deterministic gate.
    from graph import _phrase_is_negated
    signature = {}
    for topic, aliases in TOPICS.items():
        signs = set()
        for alias in aliases:
            for match in re.finditer(re.escape(alias), text.casefold()):
                prefix = text[max(0, match.start()-15):match.start()].casefold()
                negative = _phrase_is_negated(text, match.start()) or bool(re.search(r"(?:no|without)\s*$", prefix))
                signs.add("absent" if negative else "mentioned")
        if signs:
            signature[topic] = sorted(signs)
    return signature

def complaint_similarity(left, right):
    left, right = filter_identity(left), filter_identity(right)
    first, second = symptom_signature(left), symptom_signature(right)
    # Looser language is allowed, but new/contradictory recognized symptoms are
    # not an authorization to directly reuse an old clinical plan.
    if first != second:
        return 0.0
    a, b = normalized_complaint(left), normalized_complaint(right)
    if not a or not b or (not first and min(len(a), len(b)) < 4):
        return 0.0
    sequence = SequenceMatcher(None, a, b).ratio()
    grams = lambda text: {text[i:i+2] for i in range(len(text)-1)}
    aa, bb = grams(a), grams(b)
    overlap = len(aa & bb) / max(1, len(aa | bb))
    return max(sequence, overlap)

PAYLOAD_FIELDS = (
    "case_snapshot", "triage_result", "work_plan", "evidence",
    "synthesis_result", "critique_result", "final_report",
)


def reusable_payload(state):
    """Validate the server-held draft before projecting to a safe reference."""
    critique = state.get("critique_result", {})
    synthesis = state.get("synthesis_result", {})
    if not str(state.get("final_report", "")).strip():
        raise ValueError("回答尚未完成，不能加入知识库。")
    if (state.get("triage_result", {}).get("urgency") not in {"low", "medium"}
            or critique.get("verdict") != "pass"
            or critique.get("model_fallback", True)
            or critique.get("severity") == "high"
            or critique.get("issues") or critique.get("safety_flags")
            or synthesis.get("abstain", True)):
        raise ValueError("急症、高风险、拒答或未通过有效审校的回答不能作为共享参考。")
    SynthesisOutput.model_validate(synthesis)
    CriticOutput.model_validate(critique)
    evidence = state.get("evidence", [])
    if not evidence:
        raise ValueError("没有可追溯证据，不能加入知识库。")
    for action in synthesis.get("recommended_actions", []):
        if action.get("type") not in {"question", "test", "referral", "treatment_consideration"}:
            raise ValueError("行动类型不完整，不能加入知识库。")
    return {key: state.get(key, {} if key != "evidence" else []) for key in PAYLOAD_FIELDS}


def shareable_plan(patient_id, question, context, payload):
    admitted = reusable_payload(payload)
    tokens = identity_tokens(patient_id, context, payload)
    clean = filter_identity({
        "schema": "liked-plan-v3",
        "complaint": question,
        "synthesis_result": admitted["synthesis_result"],
        "evidence": admitted["evidence"],
        "report_text": admitted["final_report"],
    }, tokens)
    # Original EHR identifiers/URLs are private. Stable local labels identify
    # historical evidence only, never assert new tests for the current patient.
    for index, item in enumerate(clean["evidence"]):
        item["source"] = {
            "source_id": f"HISTORICAL-EVIDENCE-{index+1}",
            "name": "历史方案证据（非本次患者资料）",
            "is_demo": item.get("source", {}).get("is_demo", False),
        }
        for key in ("query", "retrieved_at"):
            item.pop(key, None)
    return validate_plan(clean)

def validate_plan(value):
    if not isinstance(value, dict) or set(value) != {
            "schema", "complaint", "synthesis_result", "evidence", "report_text"}:
        raise ValueError("方案结构不合法。")
    if value["schema"] != "liked-plan-v3" or not isinstance(value["complaint"], str):
        raise ValueError("方案版本不支持。")
    if not isinstance(value["report_text"], str) or not value["report_text"].strip():
        raise ValueError("历史方案为空。")
    SynthesisOutput.model_validate(value["synthesis_result"])
    if value["synthesis_result"].get("abstain", True):
        raise ValueError("拒答方案不能直接复用。")
    if not isinstance(value["evidence"], list) or not value["evidence"]:
        raise ValueError("历史方案缺少证据。")
    value = filter_identity(value)
    from graph import validate_traceability
    if validate_traceability(value):
        raise ValueError("历史方案引用不完整。")
    # Reapply the identity filter on reads; never expose extra private fields.
    return value

class AnswerKnowledgeBase:
    def __init__(self, path):
        self.path = Path(path).expanduser()

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS liked_plans_v3 (
                id TEXT PRIMARY KEY, payload_json TEXT NOT NULL,
                payload_hash TEXT NOT NULL, saved_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS plan_feedback_v3 (
                case_key TEXT PRIMARY KEY, entry_id TEXT NOT NULL,
                vote TEXT NOT NULL CHECK(vote IN ('up','down')), saved_at TEXT NOT NULL
            );
        """)
        return connection

    def _candidates(self, connection):
        for row in connection.execute("SELECT * FROM liked_plans_v3"):
            try:
                digest = hashlib.sha256(row["payload_json"].encode()).hexdigest()
                if digest != row["payload_hash"] or digest != row["id"]:
                    continue
                plan = validate_plan(json.loads(row["payload_json"]))
                yield row["id"], row["saved_at"], plan
            except (ValueError, TypeError, KeyError, AttributeError):
                continue
        # v1 stored full liked plans: read-only adapter, filter before returning.
        columns = {row[1] for row in connection.execute("PRAGMA table_info(answers)")}
        if {"id", "patient_id", "question", "context_json", "payload_json", "payload_hash", "updated_at", "source_case_id"} <= columns:
            for row in connection.execute("SELECT * FROM answers"):
                try:
                    if hashlib.sha256(row["payload_json"].encode()).hexdigest() != row["payload_hash"]:
                        continue
                    payload = json.loads(row["payload_json"])
                    context = json.loads(row["context_json"])
                    context = {"clinical_context": context, "case_id": row["source_case_id"]}
                    plan = shareable_plan(row["patient_id"], row["question"], context, payload)
                    yield hashlib.sha256(canonical_json(plan).encode()).hexdigest(), row["updated_at"], plan
                except (ValueError, TypeError, KeyError, AttributeError):
                    continue
        # v2 contains themes only and cannot reconstruct a clinical plan.

    def lookup(self, patient_id, question, context):
        if not self.path.exists():
            return []
        query = filter_identity(question, identity_tokens(patient_id, context, {}))
        connection = self._connect()
        try:
            matches = []
            for entry_id, saved_at, plan in self._candidates(connection):
                score = complaint_similarity(query, plan["complaint"])
                if score >= 0.82:
                    try:
                        saved_at = datetime.fromisoformat(saved_at).date().isoformat()
                    except (ValueError, TypeError):
                        continue
                    matches.append({"entry_id": entry_id, "saved_at": saved_at, "score": score, "plan": plan})
            matches.sort(key=lambda item: (-item["score"], item["entry_id"]))
            # Do not combine potentially contradictory old plans.
            return matches[:1]
        finally:
            connection.close()

    def feedback(self, case_id, patient_id, vote, entry_id=""):
        if vote not in {"up", "down"}:
            raise ValueError("无效反馈。")
        case_key = hashlib.sha256(canonical_json(["feedback-v3", patient_id, case_id]).encode()).hexdigest()
        connection = self._connect()
        try:
            with connection:
                connection.execute("""
                    INSERT INTO plan_feedback_v3 VALUES (?, ?, ?, ?)
                    ON CONFLICT(case_key) DO UPDATE SET vote=excluded.vote,
                    entry_id=CASE WHEN excluded.entry_id!='' THEN excluded.entry_id ELSE plan_feedback_v3.entry_id END
                """, (case_key, entry_id, vote, datetime.now(timezone.utc).date().isoformat()))
        finally:
            connection.close()
        return {"vote": vote, "stored": vote == "up", "entry_id": entry_id,
                "message": "方案已收录；高度相似主诉可直接引用，仍需本次医生审批。" if vote == "up"
                else "已记录点踩；不会撤回已收录方案。"}

    def vote(self, case_id, patient_id, question, context, vote, payload=None):
        if vote == "down":
            return self.feedback(case_id, patient_id, vote)
        if vote != "up":
            raise ValueError("无效反馈。")
        context = {"clinical_context": context, "case_id": case_id}
        plan = shareable_plan(patient_id, question, context, payload or {})
        encoded = canonical_json(plan)
        entry_id = hashlib.sha256(encoded.encode()).hexdigest()
        connection = self._connect()
        try:
            with connection:
                connection.execute("INSERT OR IGNORE INTO liked_plans_v3 VALUES (?, ?, ?, ?)",
                    (entry_id, encoded, entry_id, datetime.now(timezone.utc).date().isoformat()))
        finally:
            connection.close()
        return self.feedback(case_id, patient_id, vote, entry_id)
