from pathlib import Path
from loguru import logger

from src.indexing.milvus_vdb import MilvusVDB
from src.indexing.embed_data import EmbedData
from src.retrieval.retriever_rerank import Retriever
from src.generation.rag import RAG
from config.settings import settings
from src.ingestion.load_split import load_and_split_pdf  


def main():
    # 1. 指定要读的 PDF 路径
    pdf_path = Path("data/raft.pdf")
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 文件不存在：{pdf_path}")

    # 2. 加载 + 切分 PDF
    logger.info(f"Loading and splitting PDF: {pdf_path}")
    chunks = load_and_split_pdf(str(pdf_path), chunk_size=512, chunk_overlap=50)
    logger.info(f"Got {len(chunks)} text chunks from PDF")

    if not chunks:
        raise RuntimeError("PDF 没有解析出任何文本，检查下 pypdf 或 PDF 内容。")

    # 3. 生成向量（嵌入）
    embed_data = EmbedData(
        embed_model_name=settings.embedding_model,
        batch_size=settings.batch_size,
    )
    embed_data.embed(chunks)  # 真正把文本 -> embedding、binary embedding

    # 4. 初始化 Milvus Lite 向量库，并写入数据
    vector_db = MilvusVDB(
        collection_name="paper_demo",
        vector_dim=settings.vector_dim,
        batch_size=settings.batch_size,
        db_file=settings.milvus_db_path,
    )
    vector_db.initialize_client()
    vector_db.create_collection()
    vector_db.ingest_data(embed_data)  #  此时 embed_data 里已经有 contexts & binary_embeddings 了

    # 5. 构建自研检索器 + RAG
    retriever = Retriever(vector_db=vector_db, embed_data=embed_data)
    rag = RAG(retriever=retriever)

    # 6. 测试问答
    question = "这篇论文的讲了什么？"
    result = rag.get_detailed_response(question)

    print("====== 问题 ======")
    print(question)
    print("\n====== 回答 ======")
    print(result["response"])
    print("\n====== 检索来源（debug 用）======")
    for i, s in enumerate(result.get("sources", []), start=1):
        print(f"[{i}] score={s.get('score')} snippet={s.get('context')[:80]}...")


if __name__ == "__main__":
    main()
