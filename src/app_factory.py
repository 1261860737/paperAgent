from typing import Dict, Any

from loguru import logger
import hashlib
from config.settings import settings

from src.indexing.milvus_vdb import MilvusVDB
from src.retrieval.retriever_rerank import Retriever
from src.generation.rag import RAG
from src.workflows.workflow import PaperAgentWorkflow
from src.indexing.memory_store import MemoryStore
from src.indexing.embed_data import embed_file_pipeline, EmbedData


def build_paper_agent_from_pdf(
    pdf_path: str,
    session_id: str = "default",
    max_chars: int = 800,
    overlap_chars: int = 100,
) -> PaperAgentWorkflow:
    """
    给定 PDF 构建完整的 PaperAgentWorkflow（无 metadata 版本）

    步骤：
    1. PDF -> chunks（带 page / chunk_id 等元数据）
    2. 自研 EmbedData 做 embedding + 二进制量化
    3. Milvus Lite 建库 & 写入向量（论文内容向量库）
    4. 构建 Retriever
    5. 构建 RAG
    6. 构建 MemoryStore（长期记忆向量库，驻内存）
    7. 构建 PaperAgentWorkflow    
    """
    logger.info(f"[Factory] Loading and splitting paper: {pdf_path}")

    embed_data = EmbedData(embed_model_name=settings.embedding_model)

    contexts, dense_vecs, binary_vecs, metadatas = embed_file_pipeline(
        pdf_path,
        chunk_size=max_chars,
        overlap=overlap_chars,
        embedder=embed_data,     # 复用同一个 embedder，避免重复加载模型
    )
    if not contexts:
        raise RuntimeError(f"Failed to process document: {pdf_path}")

    safe_hash = hashlib.md5(session_id.encode("utf-8")).hexdigest()
    collection_name = f"legal_{safe_hash}"
    logger.info(f"Generated safe collection name: {collection_name} (from '{session_id}')")

    vdb = MilvusVDB(collection_name=collection_name)
    vdb.initialize_client()
    vdb.create_collection()

    # 根治：metadata 真入库
    vdb.ingest_data_raw(contexts, binary_vecs, metadatas)

    retriever = Retriever(
        vector_db=vdb,          #  用 vdb
        embed_data=embed_data,
        top_k=settings.top_k,
    )

    rag = RAG(retriever=retriever)

    memory_store = MemoryStore(
        embedder=embed_data,
        max_items=getattr(settings, "memory_max_items", 200),
    )

    workflow = PaperAgentWorkflow(
        retriever=retriever,
        rag_system=rag,
        qwen3_api_key=settings.qwen3_api_key,
        qwen3_base_url=settings.qwen3_base_url,
        memory_store=memory_store,
        short_memory_max_turns=getattr(settings, "short_memory_max_turns", 6),
        memory_top_k=getattr(settings, "memory_top_k", 3),
    )
    return workflow

