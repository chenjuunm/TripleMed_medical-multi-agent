import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from langchain_core.tools import tool
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from huggingface_hub import snapshot_download
import chromadb
from config import EMBEDDING_DEVICE, EMBEDDING_MODEL, VECTOR_DB_PATH

logger = logging.getLogger(__name__)

# All bundled records are fictional retrieval fixtures written for this demo.
# They are not excerpts from clinical guidelines and must not be used as
# diagnosis, treatment, medication, dosage, or emergency-care instructions.
DEMO_NOTICE = "【合成医学资料，仅用于软件检索测试，不构成临床建议】"

mock_guidelines = [
    Document(
        page_content=(
            f"{DEMO_NOTICE}胸痛评估。检索练习要点：起病方式、持续时间、"
            "伴随症状、生命体征、既往病史与已有检查。系统应识别需要及时人工评估的"
            "高风险表现，不得把检索命中直接转换为诊断或治疗方案。"
        ),
        metadata={"source": "合成演示资料：胸痛评估", "source_id": "DEMO-ED-001",
                  "version": "demo-2026-09-15", "is_demo": True,
                  "data_kind": "synthetic_retrieval_fixture"},
    ),
    Document(
        page_content=(
            f"{DEMO_NOTICE}血压异常评估。检索练习要点：重复测量条件、症状、"
            "生命体征趋势、既往诊断、现用药物和已获得的检查结果。不得根据单一数值"
            "自动生成药物、剂量或处置目标。"
        ),
        metadata={"source": "合成演示资料：血压异常评估", "source_id": "DEMO-CV-001",
                  "version": "demo-2026-09-15", "is_demo": True,
                  "data_kind": "synthetic_retrieval_fixture"},
    ),
    Document(
        page_content=(
            f"{DEMO_NOTICE}血糖异常与相关症状。检索练习要点：症状、测量时间与单位、"
            "既往诊断、现用药物、过敏史和已完成的检验。演示记录不提供诊断阈值、"
            "用药选择或剂量建议。"
        ),
        metadata={"source": "合成演示资料：血糖异常评估", "source_id": "DEMO-EN-001",
                  "version": "demo-2026-09-15", "is_demo": True,
                  "data_kind": "synthetic_retrieval_fixture"},
    ),
]

_additional_demo_scenarios = [
    ("DEMO-ENT-001", "咽痛与上呼吸道症状",
     "模拟主诉：咽痛、喉咙痛、鼻塞、流涕。检索练习要点：症状持续时间、是否发热、吞咽情况、接触史、既往用药与过敏史。不要将未提供的信息补写为阴性，也不要由主题命中直接生成抗菌药方案。"),
    ("DEMO-RESP-001", "咳嗽与发热",
     "模拟主诉：咳嗽、咳痰、发热。检索练习要点：起病时间、体温记录、痰的描述、是否气促或胸痛、既往呼吸系统病史及已完成的检查。区分患者自述和客观检查结果。"),
    ("DEMO-RESP-002", "喘息与哮喘病史",
     "模拟主诉：喘息、胸闷，患者自述既往哮喘。检索练习要点：发作频率、活动及夜间症状、可能诱因、既往吸入药物和使用方式、当前生命体征。既往诊断不能代替本次评估，本条不提供急性发作处置方案。"),
    ("DEMO-GI-001", "腹泻与呕吐",
     "模拟主诉：腹泻、呕吐、恶心。检索练习要点：排便及呕吐次数、饮水进食、尿量变化、是否腹痛或便血、饮食旅行史、近期用药。补充资料缺失时明确列为待询问，不生成补液剂量。"),
    ("DEMO-GI-002", "腹痛定位与病程",
     "模拟主诉：腹痛、肚子痛。检索练习要点：疼痛部位、开始时间、持续或间歇、与进食关系、伴随症状、手术史及已完成的查体。不同位置或病程的腹痛不能仅凭相似文字套用同一结论。"),
    ("DEMO-NEURO-001", "头痛病史采集",
     "模拟主诉：头痛、头部疼痛。检索练习要点：首次或反复发作、起病速度、疼痛部位、持续时间、伴随症状、外伤及药物使用。当前患者未提供神经系统查体时，不得写成查体正常。"),
    ("DEMO-NEURO-002", "头晕与眩晕描述",
     "模拟主诉：头晕、眩晕、站立不稳。检索练习要点：旋转感或昏沉感、发作时长、体位关系、伴随听力或神经症状、现用药物和已有测量记录。避免将所有头晕直接归为同一种疾病。"),
    ("DEMO-URO-001", "尿频尿急尿痛",
     "模拟主诉：尿频、尿急、尿痛、排尿不适。检索练习要点：症状持续时间、是否发热或腰痛、妊娠可能性、泌尿系统病史、药物过敏史及已有尿液检查。不能将拟议尿检写成已获得的结果。"),
    ("DEMO-DERM-001", "皮疹与瘙痒",
     "模拟主诉：皮疹、瘙痒、皮肤发红。检索练习要点：发生时间、部位及变化、新接触物或新用药、既往过敏史、伴随全身症状。没有图像或查体时，应标注皮损形态尚未核实。"),
    ("DEMO-MSK-001", "腰痛与活动关系",
     "模拟主诉：腰痛、下背痛。检索练习要点：外伤或活动背景、病程、疼痛放射、肢体感觉与力量变化、排便排尿变化及既往检查。不得将未执行的影像检查描述为阴性。"),
    ("DEMO-EN-002", "乏力与体重变化",
     "模拟主诉：乏力、疲劳、体重变化。检索练习要点：持续时间、睡眠及饮食、活动耐受、既往病史、现用药物及已有检验结果。记录体重与检验值的时间和单位，不能从非特异症状直接确定病因。"),
    ("DEMO-SAFETY-001", "用药核对与过敏信息",
     "模拟场景：药物核对、药物过敏、重复用药。检索练习要点：药物名称、患者实际服用方式、开始或停用时间、自购药和保健品、既往反应描述。区分过敏史不详与明确无过敏；本条不支持自动开药、停药或调整剂量。"),
]
mock_guidelines.extend(
    Document(
        page_content=f"{DEMO_NOTICE}{title}。{content}",
        metadata={"source": f"演示场景资料：{title}", "source_id": source_id,
                  "version": "demo-2026-09-07", "is_demo": True,
                  "topic": title, "data_kind": "synthetic_retrieval_fixture"},
    )
    for source_id, title, content in _additional_demo_scenarios
)

CHROMA_DIR = Path(VECTOR_DB_PATH).expanduser()
if not CHROMA_DIR.is_absolute():
    CHROMA_DIR = Path(__file__).resolve().parent / CHROMA_DIR
COLLECTION_NAME = "medical_guidelines"

_ensemble_retriever = None
_initialization_attempted = False
_initialization_error = ""
_initialization_lock = threading.Lock()


def _tokenize_for_bm25(text: str):
    """Tokenize mixed Chinese/Latin medical text for sparse retrieval."""

    tokens = []
    for segment in re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9.]+", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", segment):
            tokens.extend(segment)
            tokens.extend(segment[index : index + 2] for index in range(len(segment) - 1))
        else:
            tokens.append(segment)
    return tokens


def _add_missing_demo_documents(vectorstore):
    """Add current demos and replace only stale bundled demo source IDs.

    Older collections used generated UUIDs, so compare metadata source_id rather
    than assuming those records already have our deterministic IDs. Documents
    outside the bundled source-ID set are never removed or replaced.
    """
    existing = vectorstore.get(where={"is_demo": True}, include=["metadatas"])
    current = {document.metadata["source_id"]: document for document in mock_guidelines}
    stale_ids = []
    source_ids = set()
    for item_id, metadata in zip(existing.get("ids", []), existing.get("metadatas", [])):
        if not isinstance(metadata, dict):
            continue
        source_id = metadata.get("source_id")
        expected = current.get(source_id)
        if expected and metadata.get("version") != expected.metadata.get("version"):
            stale_ids.append(item_id)
        elif source_id:
            source_ids.add(source_id)
    if stale_ids:
        vectorstore.delete(ids=stale_ids)
    missing = [document for document in mock_guidelines
               if document.metadata["source_id"] not in source_ids]
    if missing:
        vectorstore.add_documents(
            documents=missing,
            ids=[f"bundled-demo:{document.metadata['source_id']}" for document in missing],
        )
    return len(missing)


def _initialize_retriever():
    """Initialize RAG lazily; failure degrades retrieval, not application boot."""

    global _ensemble_retriever, _initialization_attempted, _initialization_error

    if _ensemble_retriever is not None or _initialization_attempted:
        return _ensemble_retriever

    with _initialization_lock:
        if _ensemble_retriever is not None or _initialization_attempted:
            return _ensemble_retriever
        _initialization_attempted = True
        logger.info("正在初始化 RAG 检索器（延迟全局单例）...")

        try:
            model_source = EMBEDDING_MODEL
            configured_path = Path(EMBEDDING_MODEL).expanduser()
            if configured_path.exists():
                model_source = str(configured_path.resolve())
            else:
                try:
                    # Resolve an existing Hub snapshot without any network request.
                    # Only a true cache miss falls back to the repository id below.
                    model_source = snapshot_download(
                        repo_id=EMBEDDING_MODEL, local_files_only=True
                    )
                    logger.info("从本地 Hugging Face 缓存加载嵌入模型。")
                except Exception:
                    logger.info("本地未缓存嵌入模型，将尝试首次联网下载。")

            embeddings = HuggingFaceEmbeddings(
                model_name=model_source,
                model_kwargs={"device": EMBEDDING_DEVICE},
                encode_kwargs={"normalize_embeddings": True},
            )
            chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            existing_collections = [
                collection.name for collection in chroma_client.list_collections()
            ]

            if COLLECTION_NAME in existing_collections:
                logger.info("发现已有向量库集合 '%s'，加载并补齐演示数据。", COLLECTION_NAME)
                vectorstore = Chroma(
                    client=chroma_client,
                    collection_name=COLLECTION_NAME,
                    embedding_function=embeddings,
                )
                added = _add_missing_demo_documents(vectorstore)
                logger.info("已补充 %s 条缺失演示资料；未修改其他文档。", added)
            else:
                logger.info("创建集合 '%s' 并写入演示数据。", COLLECTION_NAME)
                vectorstore = Chroma.from_documents(
                    documents=mock_guidelines,
                    embedding=embeddings,
                    client=chroma_client,
                    collection_name=COLLECTION_NAME,
                    ids=[f"bundled-demo:{document.metadata['source_id']}" for document in mock_guidelines],
                )

            vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            bm25_retriever = BM25Retriever.from_documents(
                mock_guidelines, preprocess_func=_tokenize_for_bm25
            )
            bm25_retriever.k = 3
            _ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever], weights=[0.4, 0.6]
            )
            logger.info("RAG 检索器初始化成功。")
        except Exception as exc:
            _initialization_error = str(exc)
            logger.exception("RAG 初始化失败，知识检索将安全降级: %s", exc)

    return _ensemble_retriever


# ✅ 将 RAG 包装为 LangChain Tool
@tool
def search_medical_guidelines(query: str) -> Dict[str, Any]:
    """检索指南/共识并返回带来源标识的只读证据。"""
    retriever = _initialize_retriever()
    if not retriever:
        return {
            "status": "unavailable",
            "query": query,
            "items": [],
            "message": f"医学知识检索器不可用：{_initialization_error or '初始化失败'}",
        }

    try:
        docs = retriever.invoke(query)
        if not docs:
            return {
                "status": "not_found",
                "query": query,
                "items": [],
                "message": "未找到相关指南。",
            }
        retrieved_at = datetime.now(timezone.utc).isoformat()
        return {
            "status": "found",
            "query": query,
            "items": [
                {
                    "content": document.page_content,
                    "source": document.metadata.get("source", "未知来源"),
                    "source_id": document.metadata.get("source_id"),
                    "source_url": document.metadata.get("source_url"),
                    "version": document.metadata.get("version"),
                    "is_demo": document.metadata.get("is_demo", False),
                    "retrieved_at": retrieved_at,
                }
                for document in docs
            ],
        }
    except Exception as e:
        logging.error(f"RAG 检索出错: {e}")
        return {
            "status": "error",
            "query": query,
            "items": [],
            "message": "医学知识库当前不可用；系统不得把模型记忆伪装成已检索证据。",
        }
