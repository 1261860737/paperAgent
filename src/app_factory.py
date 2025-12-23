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
    # === 1) 加载 & 切分 & 向量化 (一步到位) ===
    # 使用我们在 embed_data.py 里新加的 pipeline 函数
    # 它会自动调用 load_split.py 的新逻辑
    contexts, dense_vecs, binary_vecs = embed_file_pipeline(
        pdf_path, 
        chunk_size=max_chars, 
        overlap=overlap_chars
    )
    
    if not contexts:
        raise RuntimeError(f"Failed to process document: {pdf_path}")

    # 为了后续构建 MemoryStore，我们需要一个 EmbedData 实例
    # 这里有点 trick，我们重新实例化一个空的，或者你可以修改 pipeline 返回实例
    embed_data = EmbedData(embed_model_name=settings.embedding_model)
    # 手动填充数据，避免重复计算
    embed_data.contexts = contexts
    embed_data.embeddings = dense_vecs
    embed_data.binary_embeddings = binary_vecs


    # === 2) 创建 Milvus 集合并写入 ===
    safe_hash = hashlib.md5(session_id.encode("utf-8")).hexdigest()
    collection_name = f"legal_{safe_hash}"
    logger.info(f"Generated safe collection name: {collection_name} (from '{session_id}')")

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
