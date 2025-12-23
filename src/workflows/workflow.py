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

ROUTER_EVALUATION_TEMPLATE = """你是一个法律 RAG 回答质量评估专家，需要严谨地判断系统检索到的法律依据是否足以回答用户的问题。

【用户问题】:
{query}

【RAG 检索到的法律条文与初始回答】:
{rag_response}

评估标准：
1. **相关性**：检索到的法条是否直接涵盖了用户问题涉及的法律情形？
2. **准确性**：回答是否明确引用了具体的法律法规名称和条款编号？
3. **完整性**：是否遗漏了关键的定罪量刑标准或免责情形？
4. **幻觉检测**：如果回答中出现“根据相关法律”，但RAG上下文中没有具体法条支撑，视为质量不佳。

请给出你的最终评判（二选一）：
- "GOOD"：检索到的法条充分且明确，可以直接基于此生成专业法律意见。
- "BAD"：检索结果缺失、不相关，或仅有笼统描述而无法条支撑，必须联网搜索补充最新的法律法规或案例。

重要要求：只返回一个大写英文单词，不要包含标点：
- GOOD
- BAD

你的评估结果：
"""

QUERY_OPTIMIZATION_TEMPLATE = """你是一个法律检索查询优化专家，需要将用户的自然语言问题转化为高效的法律搜索引擎查询。

原始问题:
{query}

优化原则:
1. **提取法言法语**：将口语转化为法律术语（如“打架”->“故意伤害”、“欠钱不还”->“民间借贷纠纷”）。
2. **明确法律依据**：加入关键词如“中华人民共和国刑法”、“民法典”、“最高法司法解释”、“量刑指导意见”等。
3. **时效性**：如果涉及近期热点或新规，加入“2024”、“最新修订”等关键词。
4. **案例导向**：如果是询问判罚结果，可以加入“典型案例”、“裁判文书”等词。

优化后的搜索查询（中文，一行，精炼）：
"""

SYNTHESIS_TEMPLATE = """你是一个专业的法律智能助手（Paralegal AI），你需要基于「本地法律库检索」「联网法律搜索」「长期记忆」综合生成一份严谨的法律咨询回复。

【用户法律咨询】:
{query}

【本地法律库依据 (RAG)】:
{rag_response}

【联网搜索补充信息 (Web)】:
{web_results}

【历史咨询记忆】:
{memory_context}

【当前对话上下文】:
{dialog_history}

回答撰写要求（严格遵守）：
1. **法条为王**：优先依据本地检索到的确切法律条文回答。引用法条时，必须使用全称（如《中华人民共和国刑法》第二百三十四条）。
2. **区分来源**：
   - 引用本地库内容时，视为“现有法律依据”；
   - 引用联网搜索内容时，需标注“（基于网络检索结果）”，并注意甄别信息的权威性。
3. **结构化输出**：
   - **核心结论**：直接回答合法/违法，或可能的结果。
   - **法律依据**：列出具体的法条原文或概括。
   - **实务建议**：针对用户情况给出起诉、调解或取证建议。
4. **严谨客观**：不要使用“肯定能赢”、“百分之百”等绝对化表述，使用“可能构成”、“存在...风险”等专业表述。
5. **强制免责声明**：回答的最后必须单独一行加上：
   *“注：本回复仅供参考，不构成正式法律意见。具体案件请咨询专业律师或相关司法部门。”*

请生成最终的法律咨询回复：
"""


REFINE_TEMPLATE = """你是一个专业的法律智能助手，请对下面这段基于本地法律库的回答进行润色，使其更具律师风范。

【原始回答】:
{rag_response}

【参考信息】:
{memory_context}
{dialog_history}

润色要求：
1. **去口语化**：使用法言法语（如将“坐牢”改为“承担刑事责任/有期徒刑”）。
2. **逻辑增强**：使用“首先、其次、综上所述”等连接词梳理逻辑。
3. **引用规范**：确保引用的法条格式规范（《法律名称》+ 条款号）。
4. **风险提示**：如果法律规定有模糊地带，应提示诉讼风险。
5. **强制免责声明**：回答最后必须加上：
   *“注：本回复仅供参考，不构成正式法律意见。具体案件请咨询专业律师。”*

请给出润色后的专业法律回答：
"""


class PaperAgentWorkflow:
    """
    基于 RAG + OpenAI SDK + LangChain Tools 的法律辅助助手 (ParalegalAgent) Workflow：

    流程：
    1. 使用 RAG（法律法规向量库）得到初始法条依据；
    2. 用 LLM 评估该回答质量（GOOD / BAD）；
    3. 如果 GOOD：
        - 基于法条生成专业法律意见，结合“长期记忆 + 最近对话”；
    4. 如果 BAD：
        - 优化为法律专业搜索词；
        - 调用 FirecrawlSearchTool 搜索最新法律法规或案例；
        - 综合 RAG + Web + 记忆，生成最终法律建议；
    5. 每轮结束：
        - 更新短期记忆
        - 沉淀有价值的法律咨询记录到长期记忆
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
        使用 RAG 系统，基于本地法律库检索相关法条。
        返回结构为 RAG.get_detailed_response 的结果。
        """
        logger.info(f"[RAG] Running detailed RAG for query={query!r}, top_k={top_k}")
        result = self.rag.get_detailed_response(query, top_k=top_k)
        return result

    # ========= 工具 2：评估 RAG 回答质量 =========

    # @tool("paper_rag_evaluator")
    def evaluate_rag_answer(self, query: str, rag_response: str) -> str:
        """
        使用 LLM 评估RAG检索到的法条是否足以回答问题，返回 'GOOD' 或 'BAD'。
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
        # 获取最近的对话历史
        history = self._build_dialog_history_block()
            
        # 构造更丰富的 Prompt
        prompt = (
                f"你是一个法律搜索优化专家。请根据对话历史和用户问题，生成搜索关键词。\n\n"
                f"【对话历史】:\n{history}\n\n"
                f"【用户当前问题】: {query}\n\n"
                f"请生成一个具体的法律搜索查询（例如包含具体的法律名称）："
        )
        
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
        使用 FirecrawlSearchTool 进行法律网络搜索，返回整理后的文本摘要。
        """
        logger.info(f"[Web] Running Firecrawl web search, query={optimized_query!r}")
        try:
            results = self.web.invoke({"query": optimized_query, "limit": limit})
            # === 🛑 调试打印 ===
            print(f"\n[DEBUG] Web Search Results Length: {len(results)}")
            print(f"[DEBUG] Web Search Content Preview: {results[:200]}...\n")
            # =================
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
        法律咨询通常比较复杂，建议适当放宽写入门槛，或者基于回答的专业度来判断。
        这里保持 > 50 字的粗暴策略，也可以后续改为“是否包含法条引用”来判断价值。
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

    def _run_workflow_sync(self, query: str, top_k: Optional[int] = 3) -> Dict[str, Any]:
        """
        同步：完整执行一次法律咨询工作流。
        """
        logger.info(f"[Workflow] Starting workflow for query={query!r}, top_k={top_k}")

        # Step 0: 先查长期记忆（不会挡住 RAG，只是提供额外 context）
        memory_hits = self._search_long_term_memory(query)
        memory_context = self._build_memory_context_block(memory_hits)
        dialog_history = self._build_dialog_history_block()

        # Step 1: 法律库 RAG 初始检索
        rag_result = self.rag_answer(query=query, top_k=top_k)
        rag_response: str = rag_result.get("response", "")
        sources = rag_result.get("sources", [])

        # Step 2: 评估 RAG 检索到的法条是否充足
        evaluation = self.evaluate_rag_answer(query=query, rag_response=rag_response)

        if evaluation == "GOOD":
            # RAG 法条充足 → 生成专业法律意见
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
            # RAG 法条缺失 → 联网搜索最新法律/案例
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

    async def run_workflow(self, query: str, top_k: Optional[int] = 3) -> Dict[str, Any]:
        """
        异步版本，便于在 FastAPI / Streamlit 等框架中使用。
        当前内部步骤是同步调用，如果你后面想完全异步化，可以用 asyncio.to_thread 包一层。
        """
        # 这里先简单直接同步执行；需要真正非阻塞时，可以改成：
        # return await asyncio.to_thread(self._run_workflow_sync, query, top_k)
        return self._run_workflow_sync(query=query, top_k=top_k)
