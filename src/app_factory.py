from typing import Dict, Any

from loguru import logger

from config.settings import settings
from src.ingestion.load_split import load_and_split_paper
from src.indexing.embed_data import EmbedData
from src.indexing.milvus_vdb import MilvusVDB
from src.retrieval.retriever_rerank import Retriever
from src.generation.rag import RAG
from src.workflows.workflow import PaperAgentWorkflow
from src.indexing.memory_store import MemoryStore


def build_paper_agent_from_pdf(
    pdf_path: str,
    session_id: str = "default",
    max_chars: int = 1000,
    overlap_chars: int = 150,
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
    chunks = load_and_split_paper(
        pdf_path,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )
    if not chunks:
        raise RuntimeError(f"从 PDF 中没有切分出 chunk: {pdf_path}")

    logger.info(f"[Factory] Got {len(chunks)} chunks")

    # === 1) 抽出要嵌入的文本（过滤掉纯章节标题） ===
    texts = [c["content"] for c in chunks if not c["is_section_title"]]

    # === 2) 构建 EmbedData（embedding + binary）===
    embed_data = EmbedData(
        embed_model_name=settings.embedding_model,
        batch_size=settings.batch_size,
    )
    embed_data.embed(texts)

    # === 3) 创建 Milvus Lite 库 ===
    collection_name = f"paper_{session_id}"

    vector_db = MilvusVDB(
        collection_name=collection_name,
        vector_dim=settings.vector_dim,
        batch_size=settings.batch_size,
        db_file=settings.milvus_db_path,
    )
    vector_db.initialize_client()
    vector_db.create_collection()
    vector_db.ingest_data(embed_data)

    # === 4) 构建 Retriever（不再传 context_metadata）===
    retriever = Retriever(
        vector_db=vector_db,
        embed_data=embed_data,
        top_k=settings.top_k,
    )

    # === 5) 构建 RAG ===
    rag = RAG(retriever=retriever)

    # 6) 长期记忆 MemoryStore（使用同一个 EmbedData 模型）
    memory_store = MemoryStore(
        embedder=embed_data,
        max_items=getattr(settings, "memory_max_items", 200),
    )

    # === 7) 构建 Workflow ===
    workflow = PaperAgentWorkflow(
        retriever=retriever,
        rag_system=rag,
        qwen3_api_key=settings.qwen3_api_key,
        qwen3_base_url=settings.qwen3_base_url,
        memory_store=memory_store,
        short_memory_max_turns=getattr(settings, "short_memory_max_turns", 6),
        memory_top_k=getattr(settings, "memory_top_k", 3),
    )

    logger.info("[Factory] PaperAgentWorkflow created successfully")
    return workflow
