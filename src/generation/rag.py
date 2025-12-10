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
    自研 RAG：
    - 召回层：使用自己的 Retriever（Milvus + EmbedData）
    - 生成层：使用 OpenAI 官方 SDK chat.completions
    - 保留原来 generate_context / query / get_detailed_response / 可改系统提示 & 模板
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
                "你是一个能干的论文辅助阅读助手，能根据所提供的背景回答问题。"
                "请务必根据给定的信息作答，并明确指出您不知道的内容。 "
            ),
        )

        # RAG prompt 模板
        self.prompt_template = (
            "CONTEXT:\n"
            "{context}\n"
            "---------------------\n"
            "请根据上述语境信息回答下列问题。 "
            "如果上下文没有包含足够的信息来回答问题，或者即使您知道答案，但它与所提供的上下文无关， "
            "请明确说明您不知道，并解释缺少哪些信息。\n\n "
            "问题: {query}\n"
            "回答: "
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
        实际向 Qwen3 发送 chat completion 请求的地方。
        这里你可以以后扩展为流式、JSON 模式等。
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

        return {
            "response": response,
            "context": context,
            "sources": retrieval_results,
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
