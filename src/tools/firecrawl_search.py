
from typing import Type, Optional
from pydantic import BaseModel, Field
from loguru import logger
from langchain_core.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
from firecrawl import FirecrawlApp
from config.settings import settings
import os
import asyncio


class FirecrawlSearchInput(BaseModel):
    """Firecrawl 网络搜索工具的输入模式."""
    query: str = Field(..., description="The search query to look up on the web.")
    limit: int = Field(3, description="Maximum number of results to fetch.")


class FirecrawlSearchTool(BaseTool):
    """LangChain 风格的 Firecrawl Web 搜索工具。"""

    name: str = "firecrawl_web_search"
    description: str = (
        "Search the web using Firecrawl and return a concise list of results "
        "(title, URL, and short description snippet)."
    )
    args_schema: Type[BaseModel] = FirecrawlSearchInput

    # 在类上声明 api_key 字段，让 pydantic 知道它的存在
    api_key: Optional[str] = Field(
        default=None,
        description="Firecrawl API key",
    )

    def __init__(self, api_key: Optional[str] = None, **data):
        # 统一处理 api_key 的来源，然后交给 BaseTool(pydantic) 去构造
        if api_key is None:
            api_key = settings.firecrawl_api_key or os.getenv("FIRECRAWL_API_KEY")
        super().__init__(api_key=api_key, **data)

    # 同步实现：.run() / .invoke() 会走这个
    def _run(self, query: str, limit: int = 3) -> str:
        api_key = self.api_key
        if not api_key:
            return "Web search unavailable - API not configured."

        try:
            logger.info(f"[Firecrawl] Searching with query={query!r}, limit={limit}")
            app = FirecrawlApp(api_key=api_key)
            response = app.search(query, limit=limit)
            results_list = getattr(response, "data", None)

            if not isinstance(results_list, list) or not results_list:
                return "No relevant web search results found."

            search_contents = []
            for result in results_list:
                if not isinstance(result, dict):
                    continue

                url = result.get("url", "No URL")
                title = result.get("title", "No title")
                description = (result.get("description") or "").strip()
                snippet = description[:1000] if description else "[no description available]"

                search_contents.append(
                    f"Title: {title}\nURL: {url}\nContent: {snippet}"
                )

            return "\n\n---\n\n".join(search_contents) if search_contents else "No relevant web search results found."
        except Exception as e:
            logger.warning(f"Firecrawl failed, switching to DuckDuckGo: {e}")
        
        # 2. 备用方案：DuckDuckGo
        try:
            ddg = DuckDuckGoSearchRun()
            return ddg.run(query)
        except Exception as e:
            return f"Search failed: {e}"

    # 异步实现：以后如果要在真正 async agent 里用，可以 .ainvoke()
    async def _arun(self, query: str, limit: int = 3) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run, query, limit)
