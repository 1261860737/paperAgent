import asyncio
from src.app_factory import build_paper_agent_from_pdf


async def main():
    pdf_path = "data/raft.pdf"
    question = "这篇论文的主要讲了什么？"

    # 1. 从 PDF 构建完整 Agent（索引 + workflow）
    workflow = build_paper_agent_from_pdf(
        pdf_path,
        session_id="raft_demo",
        max_chars=1000,
        overlap_chars=150,
    )

    # 2. 运行 workflow（内部会决定只用论文 or 加上网络搜索）
    result = await workflow.run_workflow(question, top_k=3)

    print("====== 问题 ======")
    print(question)

    print("\n====== 最终回答 ======")
    print(result["answer"])



if __name__ == "__main__":
    asyncio.run(main())
