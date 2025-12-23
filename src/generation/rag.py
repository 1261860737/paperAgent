from typing import Optional
from loguru import logger
from pydantic import BaseModel

from openai import OpenAI 
from src.retrieval.retriever_rerank import Retriever
from config.settings import settings


class ChatMessage(BaseModel):
    role: str
    content: str


class RAG:
    """
    自研法律 RAG 系统：
    - 召回层：Retriever (Milvus + EmbedData + Rerank)
    - 生成层：OpenAI SDK (Qwen/DeepSeek/GPT)
    """

    def __init__(
        self,
        retriever: Retriever,
        llm_model: str | None = None,
        qwen3_api_key: str | None = None,
        qwen3_base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        base_url: str | None = None,
    ):
        self.retriever = retriever
        self.llm_model = llm_model or settings.llm_model
        self.qwen3_api_key = qwen3_api_key or settings.qwen3_api_key
        self.temperature = temperature if temperature is not None else settings.temperature
        self.max_tokens = max_tokens if max_tokens is not None else settings.max_tokens
        self.qwen3_base_url = qwen3_base_url or settings.qwen3_base_url

        # 初始化 OpenAI 客户端（生成层）
        self.client = self._setup_client()

        # System message
        self.system_message = ChatMessage(
            role="system",
            content=(
                "你是一个专业的法律智能助手（Paralegal AI）。"
                "你的职责是基于检索到的法律法规、司法解释和案例，为用户提供严谨、客观的法律咨询回答。"
                "请务必依据提供的事实和法条作答，严禁提供没有任何法律依据的主观臆测。"
            ),
        )

        # RAG prompt 模板
        self.prompt_template = (
            "【检索到的法律依据】:\n"
            "{context}\n"
            "---------------------\n"
            "请根据上述法律依据，回答下列法律咨询问题。\n\n"
            "回答要求：\n"
            "1. **引用规范**：必须明确引用法律名称和条款编号（例如：“根据《中华人民共和国刑法》第二百三十四条...”）。\n"
            "2. **实事求是**：如果上述【法律依据】中没有包含回答问题所需的信息，请明确说明“当前资料库中未找到相关法律规定”，**严禁编造法条**。\n"
            "3. **结构清晰**：建议按照“法律规定 -> 结合案情 -> 结论建议”的逻辑进行回答。\n"
            "4. **强制免责**：回答的最后一行必须加上：“注：本回复仅供参考，不构成正式法律意见。”\n\n"
            "【用户咨询】: {query}\n"
            "【法律解答】: "
        )

    # ---------- 生成层：OpenAI 官方 SDK ----------

    def _setup_client(self) -> OpenAI:
        if not self.qwen3_api_key:
            raise ValueError(
                "QWEN3 API key is required. Set QWEN3_API_KEY environment variable "
                "or pass qwen3_api_key parameter."
            )

        client = OpenAI(
            api_key=self.qwen3_api_key,
            base_url=self.qwen3_base_url,  # 兼容自建 / 中转
        )
        logger.info(f"Initialized Qwen3 client with model: {self.llm_model}")
        return client

    def _call_llm(self, prompt: str) -> str:
        """
        向 Qwen3 发送 chat completion 请求的地方。
        
        """
        messages = [
            {"role": self.system_message.role, "content": self.system_message.content},
            {"role": "user", "content": prompt},
        ]

        resp = self.client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content

    # ---------- 召回 + 上下文构建 ----------

    def generate_context(self, query: str, top_k: Optional[int] = None) -> str:
        """
        使用自研 Retriever 生成上下文，
        内部调用 retriever.get_combined_context。
        """
        return self.retriever.get_combined_context(query, top_k)

    # ---------- 对外接口：query / get_detailed_response ----------

    def query(self, query: str, top_k: Optional[int] = None) -> str:
        """
        1) 调用检索层生成 context
        2) 用 prompt_template 拼 prompt
        3) 调 OpenAI SDK 得到答案
        """
        # 1. 召回层
        context = self.generate_context(query, top_k)

        # 2. 模板填充
        prompt = self.prompt_template.format(context=context, query=query)

        # 3. 调 OpenAI
        answer = self._call_llm(f"{prompt}")
        return answer

    
    def get_detailed_response(self, query: str, top_k: Optional[int] = None) -> dict:
        """
        返回带检索结果 & 上下文的详尽结构，方便 UI 展示或 debug。
        - response: LLM 最终回答
        - context: 拼好的 context 文本
        - sources: 每条检索结果 + 分数等
        """
        # 检索结果（带 score）
        retrieval_results = self.retriever.search_with_scores(query, top_k)

        # 字符串形式的拼接 context
        context = self.retriever.get_combined_context(query, top_k)

        # LLM 回答
        response = self.query(query, top_k=top_k)

        # 统一封装 sources 信息
        sources: List[Dict[str, Any]] = []
        for r in retrieval_results:
            # 按你 retriever_rerank 的结构来取字段
            sources.append(
                {
                    "id": r.get("id") or r.get("node_id"),
                    "score": r.get("score"),
                    "snippet": r.get("context") or r.get("text") or r.get("snippet") or "",
                    "chunk_index": r.get("metadata", {}).get("chunk_index"),
                }
            )

        return {
            "response": response,
            "context": context,
            "sources": sources,
            "query": query,
            "model": self.llm_model,
        }

    # ---------- 动态可配置部分（你原来就有） ----------

    def set_prompt_template(self, template: str):
        self.prompt_template = template
        logger.info("Updated prompt template")

    def set_system_message(self, content: str):
        self.system_message = ChatMessage(role="system", content=content)
        logger.info("Updated system message")
