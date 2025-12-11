from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import time

import numpy as np
from loguru import logger

from src.indexing.embed_data import EmbedData


@dataclass
class MemoryItem:
    """一条长期记忆"""
    id: int
    text: str
    embedding: np.ndarray
    created_at: float


class MemoryStore:
    """
    简单版长期记忆向量库（驻内存）：
    - 使用现有 EmbedData 生成向量
    - 用余弦相似度做检索
    - 内存中最多保存 max_items 条，超了就丢最老的
    """

    def __init__(
        self,
        embedder: EmbedData,
        max_items: int = 200,
    ) -> None:
        self.embedder = embedder
        self.max_items = max_items

        self._items: List[MemoryItem] = []
        self._next_id: int = 1

        logger.info(
            f"[MemoryStore] Initialized with max_items={max_items}, "
            f"embed_model={getattr(embedder, 'embed_model_name', 'unknown')}"
        )

    # -------- 写入长期记忆 --------

    def add_memory(self, text: str) -> int:
        """
        写入一条记忆：
        一般建议写入形如：
        Q: xxxx
        A: yyyy（回答的摘要）
        """
        text = (text or "").strip()
        if not text:
            return -1

        # 直接用 EmbedData 的查询接口做单条 embedding
        vec = self.embedder.get_query_embedding(text)
        emb = np.asarray(vec, dtype=np.float32)

        item = MemoryItem(
            id=self._next_id,
            text=text,
            embedding=emb,
            created_at=time.time(),
        )
        self._next_id += 1
        self._items.append(item)

        # 控制上限，超出的话丢最老的
        if len(self._items) > self.max_items:
            self._items.sort(key=lambda m: m.created_at)
            self._items = self._items[-self.max_items :]

        logger.info(f"[MemoryStore] Added memory id={item.id}, len={len(text)} chars")
        return item.id

    # -------- 检索长期记忆 --------

    def search(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        针对当前 query 做长期记忆检索，返回若干条相关记忆：
        [
          {
            "id": int,
            "score": float,
            "text": str,
            "created_at": float,
          },
          ...
        ]
        """
        if not self._items:
            return []

        q_vec = self.embedder.get_query_embedding(query)
        q = np.asarray(q_vec, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-8)

        embs = np.stack([m.embedding for m in self._items], axis=0)
        embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)

        sims = embs @ q  # 余弦相似度
        idxs = np.argsort(-sims)[:top_k]

        results: List[Dict[str, Any]] = []
        for idx in idxs:
            score = float(sims[idx])
            if score < score_threshold:
                continue
            m = self._items[idx]
            results.append(
                {
                    "id": m.id,
                    "score": score,
                    "text": m.text,
                    "created_at": m.created_at,
                }
            )

        logger.info(
            f"[MemoryStore] search(query={query!r}) -> {len(results)} hits, "
            f"top_score={results[0]['score'] if results else 'N/A'}"
        )
        return results
