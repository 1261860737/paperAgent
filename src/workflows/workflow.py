from __future__ import annotations

from typing import Optional, Any, Dict, List
from collections import deque

from loguru import logger
from openai import OpenAI
from langchain_core.tools import tool 

from config.settings import settings
from src.retrieval.retriever_rerank import Retriever
from src.generation.rag import RAG
from src.tools.firecrawl_search import FirecrawlSearchTool
from src.indexing.memory_store import MemoryStore


# ===== Prompt 模板 =====

ROUTER_EVALUATION_TEMPLATE = """你是一个 RAG 回答质量评估助手，需要判断给定的回答是否足以解答用户的问题。

【用户问题】:
{query}

【RAG 初始回答】:
{rag_response}

评估标准：
- 回答是否直接回应了用户的问题？
- 论述是否连贯、逻辑清晰？
- 是否提供了足够的信息（至少能让用户基本理解）？
- 如果回答中说“我不知道 / 上下文没有信息”，是否确实是因为上下文中确实没有相关内容？

请基于以上标准，给出你的最终评判，只能是以下两种之一：
- "GOOD"：回答已经足够好，可以直接用来回复用户（允许稍微润色）
- "BAD"：回答不完整、不清晰、或明显没答到点，建议使用 Web 搜索补充信息

重要要求：只返回一个大写英文单词：
- GOOD
- BAD

不要输出任何其他文字或标点。

你的评估结果（只能是 GOOD 或 BAD）：
"""

QUERY_OPTIMIZATION_TEMPLATE = """你是一个搜索查询优化助手，需要把原始问题改写成适合 Web 搜索引擎的中文查询。

原始查询:
{query}

指导方针:
- 用中文描述用户的核心信息需求
- 尽量加入领域关键词（如：论文标题、方法名称、缩写、数据集、任务 等）
- 保持简洁，但要保证能够帮助搜索引擎找到权威资料
- 重点是“这篇论文的背景、主要方法、主要贡献、相关工作”等

优化网络搜索查询（中文，一行）:
"""

SYNTHESIS_TEMPLATE = """你是一个论文阅读助手，需要基于「论文内容的 RAG 回答」「Web 搜索结果」「长期记忆」「最近对话历史」综合给出最终答案。

【用户问题】:
{query}

【基于论文内容的 RAG 回答】:
{rag_response}

【来自 Web 搜索的补充信息】:
{web_results}

【与该用户的长期记忆（可能包含他之前问过的相关问题与解释）】:
{memory_context}

【最近几轮对话历史】:
{dialog_history}

回答要求：
1. 优先使用论文本身的内容进行回答；
2. 对于论文中没有讲清楚的地方，可以引用 Web 搜索中的信息进行补充；
3. 如果使用了 Web 搜索的信息，请在对应句子或段落中明确标注“（该部分信息来自网络搜索）”；
4. 如果使用了长期记忆中的内容（例如用户之前的问题或你的解释），可以用“（根据你之前的问题与讨论）”这样的方式自然融入；
5. 回答请使用中文，条理清晰，必要时可以分点说明；
6. 尽量避免重复堆叠相同内容。

请给出综合后的最终回答：
"""

REFINE_TEMPLATE = """你是一个论文阅读助手，请在不改变事实的前提下，对下面这段基于论文内容的回答进行润色和增强：

【原始回答】:
{rag_response}

你还可以参考如下信息（如果有则参考，没有可以忽略）：

【与该用户的长期记忆】:
{memory_context}

【最近几轮对话历史】:
{dialog_history}

要求：
- 用中文回答；
- 结构更清晰，可以适当分点；
- 如果长期记忆中有与当前问题高度相关的内容，可以自然融入；
- 不要添加凭空捏造的内容，如果有不确定的地方，可以保留原回答中的“不确定”表达；
- 可以适当补充逻辑连接词，让读者更容易理解。

请给出润色后的回答：
"""


class PaperAgentWorkflow:
    """
    基于 RAG + OpenAI SDK + LangChain Tools 的论文辅助阅读 Agent Workflow：

    流程：
    1. 使用 RAG（向量库 + 检索 + 生成）得到初始回答；
    2. 用 LLM 评估该回答质量（GOOD / BAD）；
    3. 如果 GOOD：
        - 对 RAG 回答做一次精修，结合“长期记忆 + 最近对话”；
    4. 如果 BAD：
        - 优化用户问题 -> 生成 Web 搜索查询；
        - 调用 FirecrawlSearchTool 进行 Web 搜索；
        - 综合 RAG 回答 + Web 结果 + 长期记忆 + 最近对话，生成最终回答；
    5. 每轮结束：
        - 更新短期记忆（recent turns）
        - 根据策略写入长期记忆 MemoryStore
    """

    def __init__(
        self,
        retriever: Retriever,
        rag_system: RAG,
        qwen3_api_key: Optional[str] = None,
        qwen3_base_url: Optional[str] = None,
        memory_store: Optional[MemoryStore] = None,
        short_memory_max_turns: int = 6,
        memory_top_k: int = 3,
    ) -> None:
        self.retriever = retriever
        self.rag = rag_system

        # ==== OpenAI SDK 客户端（统一在这里配置）====
        api_key = qwen3_api_key or settings.qwen3_api_key
        base_url = qwen3_base_url or settings.qwen3_base_url

        if not api_key:
            raise ValueError("Qwen3 API key 未配置，请在 settings.qwen3_api_key 或环境变量中设置。")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.llm_model = settings.llm_model

        # Web 搜索工具（Firecrawl 或你之后自定义的搜索工具）
        self.web = FirecrawlSearchTool()
    
        # 长期记忆
        self.memory_store = memory_store
        self.memory_top_k = memory_top_k

        # 短期记忆：最近几轮对话（user/assistant）
        self.short_memory = deque(maxlen=short_memory_max_turns) 

        logger.info(
            f"[Workflow] PaperAgentWorkflow initialized | "
            f"short_memory_max_turns={short_memory_max_turns}, "
            f"memory_top_k={memory_top_k}, "
            f"memory_enabled={self.memory_store is not None}"
        )

    # ========= 工具 1：RAG 回答 =========

    # @tool("paper_rag_answer")
    def rag_answer(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        使用 RAG 系统，基于上传的论文内容回答问题。
        返回结构为 RAG.get_detailed_response 的结果。
        """
        logger.info(f"[RAG] Running detailed RAG for query={query!r}, top_k={top_k}")
        result = self.rag.get_detailed_response(query, top_k=top_k)
        return result

    # ========= 工具 2：评估 RAG 回答质量 =========

    # @tool("paper_rag_evaluator")
    def evaluate_rag_answer(self, query: str, rag_response: str) -> str:
        """
        使用 LLM 评估 RAG 回答质量，返回 'GOOD' 或 'BAD'。
        """
        prompt = ROUTER_EVALUATION_TEMPLATE.format(query=query, rag_response=rag_response)

        logger.info("[Eval] Evaluating RAG answer quality...")
        resp = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=4,  # 一个单词足够
        )
        text = (resp.choices[0].message.content or "").strip().upper()
        # 只拿第一个 token，防御一下 LLM 偶尔啰嗦的情况
        label = text.split()[0] if text else "BAD"
        if label not in {"GOOD", "BAD"}:
            label = "BAD"

        logger.info(f"[Eval] Evaluation result: {label}")
        return label

    # ========= 工具 3：优化 Web 搜索查询 =========

    # @tool("paper_web_query_optimizer")
    def optimize_query_for_web(self, query: str) -> str:
        """
        使用 LLM 将用户原始问题改写为更适合 Web 搜索的中文查询。
        """
        prompt = QUERY_OPTIMIZATION_TEMPLATE.format(query=query)
        logger.info("[Web] Optimizing query for web search...")
        resp = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=128,
        )
        optimized = (resp.choices[0].message.content or "").strip()
        logger.info(f"[Web] Optimized query: {optimized!r}")
        return optimized or query

    # ========= 工具 4：执行 Web 搜索 =========

    # @tool("paper_web_search")
    def web_search(self, optimized_query: str, limit: int = 3) -> str:
        """
        使用 FirecrawlSearchTool 进行 Web 搜索，返回整理后的文本摘要。
        """
        logger.info(f"[Web] Running Firecrawl web search, query={optimized_query!r}, limit={limit}")
        try:
            results = self.web.run(query=optimized_query, limit=limit)
        except Exception as e:
            logger.error(f"[Web] Web search failed: {e}")
            results = "由于技术问题，网络搜索失败."
        return results

# ========= 短期记忆相关 =========

    def _update_short_memory(self, query: str, answer: str) -> None:
        """记录最近几轮对话"""
        self.short_memory.append(("user", query))
        self.short_memory.append(("assistant", answer))

    def _build_dialog_history_block(self) -> str:
        """把 short_memory 转成字符串，喂给 LLM 用"""
        if not self.short_memory:
            return "（暂无历史对话）"
        lines: List[str] = []
        for role, content in self.short_memory:
            prefix = "用户：" if role == "user" else "助手："
            lines.append(f"{prefix}{content}")
        return "\n".join(lines)

    # ========= 长期记忆相关 =========

    def _search_long_term_memory(self, query: str) -> List[Dict[str, Any]]:
        if self.memory_store is None:
            return []
        results = self.memory_store.search(
            query=query,
            top_k=self.memory_top_k,
            score_threshold=0.35,  # 可以按需要调
        )
        return results

    def _build_memory_context_block(self, memories: List[Dict[str, Any]]) -> str:
        if not memories:
            return "（当前问题尚无特别相关的长期记忆）"
        lines = []
        for i, m in enumerate(memories, start=1):
            lines.append(f"[记忆 {i}] {m['text']}")
        return "\n".join(lines)

    def _maybe_write_long_term_memory(self, query: str, answer: str) -> None:
        """
        简单策略：如果回答长度比较长，且不是纯闲聊，则写入长期记忆。
        这里先用非常粗暴的策略：长度 > 50 字就写一条。
        """
        if self.memory_store is None:
            return

        text = (answer or "").strip()
        if len(text) < 50:
            return

        memory_text = f"Q: {query}\nA: {text}"
        self.memory_store.add_memory(memory_text)

    # ========= 工具 5：综合 / 精修 最终回答 =========

    # @tool("paper_synthesize_answer")
    def synthesize_answer(
        self,
        query: str,
        rag_response: str,
        web_results: Optional[str] = "",
        use_web_results: bool = False,
        top_k: int = 3,
        dialog_history: Optional[list] = None,
        memory_context: Optional[str] = "",
    ) -> str:
        """
        根据标志决定：
        - 如果 use_web_results=True：综合 RAG 回答 + Web 搜索结果 + 长期记忆 + 对话历史；
        - 否则：对 RAG 回答进行中文润色，同样可以参考长期记忆和对话历史。
        """
        citations = self.retriever.get_citations(query=query, top_k=top_k)
        citation_text = ""
        for c in citations:
            snippet = c["snippet"]
            citation_text += f"[{c['rank']}] {snippet}\n\n"

        # 2) 保证 memory_context / dialog_history 有默认文本，避免变成 None
        memory_block = memory_context or "（当前问题尚无特别相关的长期记忆）"
        dialog_block = dialog_history or "（暂无历史对话）"

        if use_web_results and web_results:
            logger.info("[Synth] Synthesizing answer from RAG + Web results + Memory + Dialog history...")
            prompt = SYNTHESIS_TEMPLATE.format(
                query=query,
                rag_response=rag_response,
                web_results=web_results,
                memory_context=memory_block,
                dialog_history=dialog_block,
            )
        else:
            logger.info("[Synth] Refining RAG-only answer (with Memory + Dialog history)...")
            prompt = REFINE_TEMPLATE.format(
                rag_response=rag_response,
                memory_context=memory_block,
                dialog_history=dialog_block,
                )

        resp = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=settings.max_tokens,
        
        )
        final_answer = resp.choices[0].message.content or ""

        return final_answer.strip()

    # ========= 核心：同步流程调度 =========

    def _run_workflow_sync(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        同步：完整执行一次论文问答工作流。
        返回：
            {
                "answer": 最终回答,
                "rag_response": 初始 RAG 回答,
                "web_search_used": 是否使用了 Web 搜索,
                "web_results": Web 搜索的原始结果（可用于 UI 展示）,
                "sources": RAG 的检索结果元数据,
                "evaluation": "GOOD"/"BAD",
                "query": 原始问题,
                "memory_hits": [...],  # 命中的长期记忆
            }
        """
        logger.info(f"[Workflow] Starting workflow for query={query!r}, top_k={top_k}")

        # Step 0: 先查长期记忆（不会挡住 RAG，只是提供额外 context）
        memory_hits = self._search_long_term_memory(query)
        memory_context = self._build_memory_context_block(memory_hits)
        dialog_history = self._build_dialog_history_block()

        # Step 1: 自研 RAG 初始回答
        rag_result = self.rag_answer(query=query, top_k=top_k)
        rag_response: str = rag_result.get("response", "")
        sources = rag_result.get("sources", [])

        # Step 2: 评估 RAG 回答质量
        evaluation = self.evaluate_rag_answer(query=query, rag_response=rag_response)

        if evaluation == "GOOD":
            # RAG 已经足够好 → 做一个小精修
            final_answer = self.synthesize_answer(
                query=query,
                rag_response=rag_response,
                web_results="",
                use_web_results=False,
                memory_context=memory_context,
                dialog_history=dialog_history,
            )
            web_used = False
            web_results = None
        else:
            # RAG 质量不够 → 走 Web 搜索分支
            optimized_query = self.optimize_query_for_web(query=query)
            web_results = self.web_search(optimized_query=optimized_query, limit=3)
            final_answer = self.synthesize_answer(
                query=query,
                rag_response=rag_response,
                web_results=web_results,
                use_web_results=True,
                memory_context=memory_context,
                dialog_history=dialog_history,
            )
            web_used = True

        # Step 3: 更新 short_memory + 写入长期记忆
        self._update_short_memory(query=query, answer=final_answer)
        self._maybe_write_long_term_memory(query=query, answer=final_answer)

        result = {
            "answer": final_answer,
            "rag_response": rag_response,
            "web_search_used": web_used,
            "web_results": web_results,
            "sources": sources,
            "evaluation": evaluation,
            "query": query,
            "memory_hits": memory_hits,
        }

        logger.info("[Workflow] Finished workflow")
        return result

    # ========= 对外：异步接口 =========

    async def run_workflow(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        异步版本，便于在 FastAPI / Streamlit 等框架中使用。
        当前内部步骤是同步调用，如果你后面想完全异步化，可以用 asyncio.to_thread 包一层。
        """
        # 这里先简单直接同步执行；需要真正非阻塞时，可以改成：
        # return await asyncio.to_thread(self._run_workflow_sync, query, top_k)
        return self._run_workflow_sync(query=query, top_k=top_k)
