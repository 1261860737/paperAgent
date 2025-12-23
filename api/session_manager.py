from typing import Dict
from datetime import datetime
from loguru import logger
from src.app_factory import build_paper_agent_from_pdf
from src.indexing.memory_store import MemoryStore
from src.indexing.embed_data import EmbedData 
from config.settings import settings

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, any] = {}

    def create_session(self, pdf_path: str) -> str:
        """创建一个新的论文 Session"""
        session_id = self._generate_session_id(pdf_path)

        # 创建 EmbedData 实例作为 embedder
        embedder = EmbedData(embed_model_name=settings.embedding_model, batch_size=settings.batch_size)

        # 创建 memory_store 时传入 embedder
        memory_store = MemoryStore(embedder=embedder)

        workflow = build_paper_agent_from_pdf(
            pdf_path=pdf_path,
            session_id=session_id,
            max_chars=1000,
            overlap_chars=150,
        )

        # 注入 memory_store
        workflow.memory_store = memory_store

        self.sessions[session_id] = {
            "workflow": workflow,
            "memory": memory_store,
            "history": []  # 保存短期历史（用于前端显示）
        }

        logger.info(f"[Session] Created new session: {session_id}")
        return session_id

    def get_workflow(self, session_id: str):
        return self.sessions[session_id]["workflow"]

    def append_history(self, session_id: str, role: str, content: str):
        self.sessions[session_id]["history"].append({"role": role, "content": content})

    def get_history(self, session_id: str):
        return self.sessions[session_id]["history"]

    def _generate_session_id(self, pdf_path: str) -> str:
        base = pdf_path.split("/")[-1].replace(".pdf", "")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base}_{ts}"
