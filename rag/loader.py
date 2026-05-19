import threading
from pathlib import Path
import chromadb
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHROMA_PATH

KNOWLEDGE_BASE = Path(__file__).parent.parent / "knowledge_base"
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

_lock = threading.Lock()
_embeddings: HuggingFaceEmbeddings | None = None
_chroma_client = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        with _lock:
            if _embeddings is None:
                _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def _get_chroma_client():
    """单例 PersistentClient，避免多线程并发创建冲突。"""
    global _chroma_client
    if _chroma_client is None:
        with _lock:
            if _chroma_client is None:
                _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _chroma_client


def get_vectorstore(collection_name: str) -> Chroma:
    return Chroma(
        client=_get_chroma_client(),
        collection_name=collection_name,
        embedding_function=_get_embeddings(),
    )


def _load_collection(subdir: str, collection_name: str) -> int:
    folder = KNOWLEDGE_BASE / subdir
    txt_files = list(folder.glob("*.txt"))
    if not txt_files:
        return 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = []
    for path in txt_files:
        loader = TextLoader(str(path), encoding="utf-8")
        raw = loader.load()
        chunks = splitter.split_documents(raw)
        docs.extend(chunks)

    vectorstore = get_vectorstore(collection_name)
    vectorstore.add_documents(docs)
    return len(docs)


def load_all_documents() -> dict[str, int]:
    counts = {
        "financials": _load_collection("financials", "financials"),
        "macro": _load_collection("macro", "macro"),
    }
    return counts
