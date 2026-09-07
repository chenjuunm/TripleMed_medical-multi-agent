import os

# Relax presentation-only Critic findings only for entirely demo-labelled evidence.
CRITIC_DEMO_MODE = os.getenv("CRITIC_DEMO_MODE", "true").lower() in {"1", "true", "yes"}
from pathlib import Path

# 可选远程 OpenAI-compatible API。只要 MODEL_API_KEY 非空，主模型和
# Router 都使用远程 API；留空时保持现有 LM Studio 本地模型行为。
REMOTE_API_KEY = os.getenv("MODEL_API_KEY", "").strip()
USE_REMOTE_API = bool(REMOTE_API_KEY)
REMOTE_API_BASE_URL = os.getenv(
    "MODEL_API_BASE_URL", "https://api.openai.com/v1"
).strip()
REMOTE_AGENT_MODEL = os.getenv("API_AGENT_MODEL", "gpt-5.4-mini").strip()
REMOTE_ROUTER_MODEL = os.getenv("API_ROUTER_MODEL", "gpt-5.4-mini").strip()
REMOTE_VERIFIER_MODEL = os.getenv(
    "API_VERIFIER_MODEL", REMOTE_AGENT_MODEL
).strip()

# LM Studio OpenAI-compatible 本地推理服务。
LM_STUDIO_BASE_URL = os.getenv(
    "LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1"
)
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
LM_STUDIO_TIMEOUT_SECONDS = float(
    os.getenv("LM_STUDIO_TIMEOUT_SECONDS", "300")
)
LM_STUDIO_MAX_RETRIES = int(os.getenv("LM_STUDIO_MAX_RETRIES", "1"))
MODEL_PREFLIGHT_TIMEOUT_SECONDS = float(
    os.getenv("MODEL_PREFLIGHT_TIMEOUT_SECONDS", "5")
)

# 本地默认模型名由 scripts/start_lm_studio_models.sh 创建为稳定 identifier。
LOCAL_AGENT_MODEL = os.getenv("AGENT_MODEL", "medical-main-qwen38-27b")
LOCAL_ROUTER_MODEL = os.getenv("ROUTER_MODEL", "medical-router-qwen35-9b")
LOCAL_VERIFIER_MODEL = os.getenv("VERIFIER_MODEL", LOCAL_AGENT_MODEL)

if USE_REMOTE_API:
    MODEL_BACKEND = "remote_api"
    MODEL_BASE_URL = REMOTE_API_BASE_URL
    MODEL_API_KEY = REMOTE_API_KEY
    AGENT_MODEL = REMOTE_AGENT_MODEL
    ROUTER_MODEL = REMOTE_ROUTER_MODEL
    VERIFIER_MODEL = REMOTE_VERIFIER_MODEL
else:
    MODEL_BACKEND = "lm_studio"
    MODEL_BASE_URL = LM_STUDIO_BASE_URL
    MODEL_API_KEY = LM_STUDIO_API_KEY
    AGENT_MODEL = LOCAL_AGENT_MODEL
    ROUTER_MODEL = LOCAL_ROUTER_MODEL
    VERIFIER_MODEL = LOCAL_VERIFIER_MODEL

# LM Studio 的 Qwen 推理模型默认可能使用很高 reasoning effort，导致结构化
# JSON 正文迟迟不出现。临床推理由显式的多阶段角色分解承担，所有 JSON 角色
# 默认关闭隐藏推理，防止隐藏 token 挤占可验证的结构化正文。
AGENT_REASONING_EFFORT = os.getenv("AGENT_REASONING_EFFORT", "none")
ROUTER_REASONING_EFFORT = os.getenv("ROUTER_REASONING_EFFORT", "none")
VERIFIER_REASONING_EFFORT = os.getenv(
    "VERIFIER_REASONING_EFFORT", AGENT_REASONING_EFFORT
)
AGENT_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", "2048"))
ROUTER_MAX_TOKENS = int(os.getenv("ROUTER_MAX_TOKENS", "1024"))
VERIFIER_MAX_TOKENS = int(os.getenv("VERIFIER_MAX_TOKENS", "1536"))

# 向量数据库配置
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-zh-v1.5")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./chroma_db")

# 资源与安全预算：多智能体按病例复杂度自适应，不无限扩张或反思。
MAX_SPECIALIST_AGENTS = int(os.getenv("MAX_SPECIALIST_AGENTS", "2"))
MAX_RECORD_REQUESTS = int(os.getenv("MAX_RECORD_REQUESTS", "4"))
MAX_GUIDELINE_QUERIES = int(os.getenv("MAX_GUIDELINE_QUERIES", "3"))
MAX_REFLECTION_ROUNDS = int(os.getenv("MAX_REFLECTION_ROUNDS", "1"))

# HITL approval is fail-closed unless a server-side token is explicitly set.
# The token authenticates the local prototype approval channel; production must
# replace it with the institution's identity provider and role-based access.
CLINICIAN_APPROVAL_TOKEN = os.getenv("CLINICIAN_APPROVAL_TOKEN", "")
CLINICIAN_ALLOWED_ROLES = tuple(
    value.strip().lower()
    for value in os.getenv(
        "CLINICIAN_ALLOWED_ROLES", "clinician,doctor,attending"
    ).split(",")
    if value.strip()
)
EXAM_ORDER_APPROVAL_TTL_SECONDS = int(
    os.getenv("EXAM_ORDER_APPROVAL_TTL_SECONDS", "900")
)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Liked answers are stored separately from the guideline vector database.
ANSWER_KNOWLEDGE_DB = os.getenv("ANSWER_KNOWLEDGE_DB", str(Path(__file__).resolve().parent / "answer_knowledge_db" / "answers.sqlite3"))
