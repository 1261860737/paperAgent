import asyncio
from loguru import logger

from config.settings import settings
from src.app_factory import build_paper_agent_from_pdf


async def main():
    # 1. 先构建整套 Agent（PDF -> Milvus -> Retriever -> RAG -> Workflow + Memory）
    pdf_path = "data/raft.pdf"
    session_id = "raft_demo"

    logger.info(f"Building PaperAgentWorkflow from PDF: {pdf_path}")
    workflow = build_paper_agent_from_pdf(
        pdf_path=pdf_path,
        session_id=session_id,
        max_chars=1000,
        overlap_chars=150,
    )

    print("\n==== 论文助手已启动 ====")
    print(f"当前论文：{pdf_path}")
    print("输入你的问题，输入 'exit' 或 'q' 退出。\n")

    # 2. 循环问答，测试短期 / 长期记忆
    while True:
        try:
            question = input("问题 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("已退出。")
            break

        # 3. 调用带记忆的工作流
        result = await workflow.run_workflow(question, top_k=settings.top_k)

        answer = result.get("answer", "")
        sources = result.get("sources") or []
        web_used = result.get("web_search_used", False)
        evaluation = result.get("evaluation", "")
        memory_hits = result.get("memory_hits") or []

        # 4. 打印结果：最终回答 + 是否用 Web + RAG 检索来源 + 长期记忆命中
        print("\n====== 问题 ======")
        print(question)

        print("\n====== 最终回答 ======")
        print(answer)

        print("\n====== 评估结果（GOOD/BAD）======")
        print(evaluation)

        print("\n====== 是否使用 Web 搜索 ======")
        print("是" if web_used else "否")

        print("\n====== RAG 检索来源（仅显示前三条）======")
        if not sources:
            print("(无检索结果)")
        else:
            for i, src in enumerate(sources[:3], start=1):
                ctx = (src.get("context") or "").strip()
                if len(ctx) > 200:
                    ctx = ctx[:200] + "..."
                score = src.get("score")
                node_id = src.get("node_id")
                print(f"[{i}] score={score:.4f} node_id={node_id}")
                print(f"    {ctx}\n")

        print("====== 命中的长期记忆（仅显示前三条）======")
        if not memory_hits:
            print("(当前问题尚未命中长期记忆)")
        else:
            for i, m in enumerate(memory_hits[:3], start=1):
                txt = (m.get("text") or "").strip()
                if len(txt) > 200:
                    txt = txt[:200] + "..."
                score = m.get("score")
                print(f"[记忆 {i}] score={score:.3f}")
                print(f"    {txt}\n")

        print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
