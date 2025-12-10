# demo2.py
import asyncio
from pathlib import Path
from loguru import logger

from config.settings import settings
from src.ingestion.load_split import load_and_split_pdf
from src.indexing.embed_data import EmbedData
from src.indexing.milvus_vdb import MilvusVDB
from src.retrieval.retriever_rerank import Retriever
from src.generation.rag import RAG
from src.workflows.workflow import PaperAgentWorkflow  # 你的 workflow.py 路径


async def main():
    pdf_path = Path("data/raft.pdf")
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 文件不存在：{pdf_path}")

    logger.info(f"Loading and splitting PDF: {pdf_path}")
    chunks = load_and_split_pdf(str(pdf_path), chunk_size=512, chunk_overlap=50)
    logger.info(f"Got {len(chunks)} chunks")

    # 1. 嵌入
    embed_data = EmbedData(
        embed_model_name=settings.embedding_model,
        batch_size=settings.batch_size,
    )
    embed_data.embed(chunks)

    # 2. 向量库
    vector_db = MilvusVDB(
        collection_name="paper_demo",
        vector_dim=settings.vector_dim,
        batch_size=settings.batch_size,
        db_file=settings.milvus_db_path,
    )
    vector_db.initialize_client()
    vector_db.create_collection()
    vector_db.ingest_data(embed_data)

    # 3. 检索器 & RAG 实例
    retriever = Retriever(vector_db=vector_db, embed_data=embed_data)
    rag = RAG(
        retriever=retriever,
        llm_model=settings.llm_model,
        qwen3_api_key=settings.qwen3_api_key,
        qwen3_base_url=settings.qwen3_base_url,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )

    # 4. Workflow 实例（注意这里传的是实例）
    workflow = PaperAgentWorkflow(
        retriever=retriever,
        rag_system=rag,
    )

    # 5. 异步跑一次完整 workflow
    question = "这篇论文的主要讲了什么？"
    result = await workflow.run_workflow(question, top_k=3)

    print("====== 问题 ======")
    print(question)
    print("\n====== 最终回答 ======")
    print(result["answer"])
    print("\n====== 是否使用 Web 搜索 ======")
    print(result["web_search_used"])
    print("\n====== 评估结果（GOOD/BAD） ======")
    print(result["evaluation"])


if __name__ == "__main__":
    asyncio.run(main())
