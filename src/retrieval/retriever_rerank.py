from typing import Optional, List, Dict, Any
from loguru import logger
from pydantic import BaseModel, Field

from src.indexing.milvus_vdb import MilvusVDB
from sentence_transformers import CrossEncoder
from src.indexing.embed_data import EmbedData
from config.settings import settings
from src.ingestion.load_split import load_and_split_document

import bm25s
import re


class TextNode(BaseModel):
    text: str
    id_: str
    metadata: Dict = {}


class NodeWithScore(BaseModel):
    node: TextNode
    score: float

def tokenize_legal(text: str, ngram: int = 2):
    """
    中文用字符 n-gram（默认2），英文/数字用词。
    对法规条文比 jieba 稳定很多（不依赖词典）。
    """
    text = (text or "").strip()
    # 英文/数字词
    words = re.findall(r"[A-Za-z0-9]+", text.lower())

    # 中文字符（去掉空白和常见标点）
    zh = re.sub(r"[^\u4e00-\u9fff]+", "", text)

    grams = []
    if len(zh) >= ngram:
        grams = [zh[i:i+ngram] for i in range(len(zh) - ngram + 1)]

    return words + grams

def zh_num_to_int(s: str) -> int:
    s = str(s).strip()
    if s.isdigit():
        return int(s)
    m = {"零":0,"一":1,"二":2,"两":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9}
    if s in m:
        return m[s]
    if "十" in s:
        left, _, right = s.partition("十")
        left_v = 1 if left == "" else m.get(left, 0)
        right_v = 0 if right == "" else m.get(right, 0)
        return left_v * 10 + right_v
    return 0



class Retriever:
    def __init__(
        self, 
        vector_db: MilvusVDB, 
        embed_data: EmbedData, 
        top_k: int = None,

    ) -> None:
        self.vector_db = vector_db
        self.embed_data = embed_data
        self.top_k = top_k or settings.top_k

        # 推荐模型: BAAI/bge-reranker-v2-m3 (多语言能力强)
        logger.info("Loading Reranker model...")
        self.reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=512)

        # 2. 构建 BM25 索引（只索引 article）
        if self.embed_data.contexts:
            metas = getattr(self.embed_data, "metadatas", None)

            self.bm25_contexts = []
            self.bm25_metadatas = []
            self.bm25_to_orig = []  # bm25_idx -> orig_idx

            for i, ctx in enumerate(self.embed_data.contexts):
                m = metas[i] if metas and i < len(metas) else {}
                if (m or {}).get("node_type", "article") != "article":
                    continue
                self.bm25_to_orig.append(i)
                self.bm25_contexts.append(ctx)
                self.bm25_metadatas.append(m or {})

            logger.info(f"Building BM25 index for {len(self.bm25_contexts)} article documents...")

            corpus_tokens = [tokenize_legal(doc, ngram=2) for doc in self.bm25_contexts]
            self.bm25 = bm25s.BM25()          # Okapi BM25
            self.bm25.index(corpus_tokens)    # 建索引

            logger.info("BM25 index built successfully.")
        else:
            self.bm25 = None
            self.bm25_contexts = []
            self.bm25_metadatas = []
            self.bm25_to_orig = []



    def _bm25_search(self, query: str, top_k: int = 50) -> List[Dict]:
        if not self.bm25:
            return []

        q_tokens = tokenize_legal(query, ngram=2)

        # bm25s 返回 (scores, doc_ids)
        res = self.bm25.retrieve(q_tokens, k=top_k)
        # 兼容不同返回格式
        if isinstance(res, tuple) and len(res) == 2:
            scores, doc_ids = res
        else:
            # bm25s 常见：res.scores / res.documents / res.doc_ids
            scores = getattr(res, "scores", None)
            doc_ids = getattr(res, "doc_ids", None) or getattr(res, "documents", None)

        # 有些版本返回二维
        if isinstance(scores, list) and scores and isinstance(scores[0], list):
            scores = scores[0]
        if isinstance(doc_ids, list) and doc_ids and isinstance(doc_ids[0], list):
            doc_ids = doc_ids[0]


        results = []
        for score, bm25_idx in zip(scores, doc_ids):
            if score <= 0:
                continue
            bm25_idx = int(bm25_idx)

            ctx = self.bm25_contexts[bm25_idx]
            meta = self.bm25_metadatas[bm25_idx]
            orig_idx = self.bm25_to_orig[bm25_idx]

            results.append({
                "context": ctx,
                "id": str(orig_idx),          # 用原 contexts 的索引当 id（至少一致）
                "score": float(score),
                "source": "bm25",
                "metadata": meta,             # 关键：带条号/法名等
            })

        return results



    def search(self, query: str, top_k: Optional[int] = None) -> List[NodeWithScore]:
        """
        混合检索：Milvus(Vector) + BM25(Keyword) -> 去重融合 -> Rerank
        并额外提供“条号意图硬分支”：前N条/第N条 直接按 article_no 精确取回
        """
        if top_k is None:
            top_k = self.top_k

        # ---------- 0) 条号意图硬分支（命中就直接 return，不走 hybrid） ----------
        def _parse_targets(q: str) -> List[int]:
            # 前两条 / 前2条
            m = re.search(r"前\s*([零一二三四五六七八九十百千两0-9]+)\s*条", q)
            if m:
                n = zh_num_to_int(m.group(1))
                return list(range(1, n + 1)) if n > 0 else []

            # 第十条 / 第10条
            m = re.search(r"第\s*([零一二三四五六七八九十百千两0-9]+)\s*条", q)
            if m:
                n = zh_num_to_int(m.group(1))
                return [n] if n > 0 else []

            return []

        targets = _parse_targets(query)
        if targets:
            rows = self.vector_db.fetch_articles(targets)  # 需确保 milvus_vdb.py 有该方法
            if not rows:
                return []

            grouped: Dict[int, List[dict]] = {}
            for r in rows:
                a = int(r.get("article_no") or -1)
                if a <= 0:
                    continue
                grouped.setdefault(a, []).append(r)

            out: List[NodeWithScore] = []
            for a_no in sorted(grouped.keys()):
                parts = grouped[a_no]
                parts.sort(key=lambda x: int(x.get("part_no") or 1))
                text = "\n".join([p.get("context", "") for p in parts if p.get("context")]).strip()

                base = parts[0]
                meta = {
                    "node_type": base.get("node_type"),
                    "law_title": base.get("law_title"),
                    "law_id": base.get("law_id"),
                    "article_no": base.get("article_no"),
                    "article_label": base.get("article_label"),
                    "chapter": base.get("chapter"),
                    "part_no": 1,
                    "part_total": len(parts),
                    "source_file": base.get("source_file"),
                }

                node = TextNode(
                    text=text,
                    id_=str(base.get("id") or f"article-{a_no}"),
                    metadata={"sources": ["article_fetch"], **meta},
                )
                out.append(NodeWithScore(node=node, score=1.0))

            # 这里不截断 top_k：用户问前N条就返回N条
            return out

        # ---------- 1) 原有 hybrid 检索（向量 + BM25） ----------
        initial_top_k = 50

        # 生成查询嵌入并转换为二进制
        query_embedding = self.embed_data.get_query_embedding(query)
        binary_query = self.embed_data.binary_quantize_query(query_embedding)

        # 向量搜索：默认只搜 article，避免目录/页眉页脚干扰
        vector_results = self.vector_db.search(
            binary_query=binary_query,
            top_k=initial_top_k,
            output_fields=[
                "context", "node_type", "law_title", "law_id", "article_no", "article_label",
                "chapter", "part_no", "part_total", "source_file"
            ],
            filter_expr='node_type == "article"',
        )
        for r in vector_results:
            r["source"] = "vector"

        bm25_results = self._bm25_search(query, top_k=initial_top_k)

        unique_candidates: Dict[str, Dict] = {}

        # 向量候选（带 metadata）
        for res in vector_results:
            ctx = res["payload"]["context"]
            unique_candidates[ctx] = {
                "context": ctx,
                "id": str(res["id"]),
                "sources": ["vector"],
                "metadata": {k: v for k, v in res["payload"].items() if k != "context"},
            }


        # BM25 候选
        for res in bm25_results:
            ctx = res["context"]
            if ctx in unique_candidates:
                unique_candidates[ctx]["sources"].append("bm25")
                # 如果向量那边没有 metadata（极少），BM25 有就补上
                if not unique_candidates[ctx].get("metadata") and res.get("metadata"):
                    unique_candidates[ctx]["metadata"] = res["metadata"]
            else:
                unique_candidates[ctx] = {
                    "context": ctx,
                    "id": res["id"],
                    "sources": ["bm25"],
                    "metadata": res.get("metadata") or {},  # BM25 现在有 metadata 了
                }

        candidates = list(unique_candidates.values())
        if not candidates:
            return []


        # ---------- 2) Rerank ----------
        pairs = [[query, doc["context"]] for doc in candidates]
        scores = self.reranker.predict(pairs)

        final_candidates: List[NodeWithScore] = []
        for i, doc in enumerate(candidates):
            node = TextNode(
                text=doc["context"],
                id_=str(doc["id"]),
                # ✅ 关键：把条号/法名等 metadata 合并进去，后续 citations 才能标注第几条
                metadata={"sources": doc["sources"], **(doc.get("metadata") or {})},
            )
            final_candidates.append(NodeWithScore(node=node, score=float(scores[i])))

        final_candidates.sort(key=lambda x: x.score, reverse=True)
        final_results = final_candidates[:top_k]

        if final_results:
            top_doc = final_results[0]
            logger.info(
                f"Top-1: {top_doc.score:.4f} | Source: {top_doc.node.metadata.get('sources')} | Text: {top_doc.node.text[:30]}..."
            )

        return final_results



    def get_contexts(self, query: str, top_k: Optional[int] = None):
        nodes_with_scores = self.search(query, top_k)
        return [node.node.text for node in nodes_with_scores]

    def get_combined_context(self, query: str, top_k: Optional[int] = None):
        contexts = self.get_contexts(query, top_k)
        return "\n\n---\n\n".join(contexts)

    def search_with_scores(self, query: str, top_k: Optional[int] = None):
        nodes_with_scores = self.search(query, top_k)
            
        results = []
        for node_with_score in nodes_with_scores:
            results.append({
                    # [Fix] 补全字段，兼容 rag.py 和 get_citations
                    "context": node_with_score.node.text,
                    "text": node_with_score.node.text,
                    "snippet": node_with_score.node.text,
                    
                    "score": node_with_score.score,
                    
                    # [Fix] 确保 ID 存在
                    "id": node_with_score.node.id_,
                    "node_id": node_with_score.node.id_,
                    
                    # [Fix] 确保 metadata 存在 (BM25 可能会用到)
                    "metadata": getattr(node_with_score.node, "metadata", {})
            })
            
        return results

    def get_citations(self, query: str, top_k: int = 3, snippet_chars: int = 300):
        results = self.search_with_scores(query, top_k)
        citations = []
        for rank, item in enumerate(results, start=1):
            meta = item.get("metadata") or {}
            law = meta.get("law_title") or ""
            a_no = meta.get("article_no")
            p_no = meta.get("part_no")
            p_tt = meta.get("part_total")

            head = ""
            if law and a_no is not None and int(a_no) > 0:
                head = f"《{law}》第{int(a_no)}条"
                if p_tt and int(p_tt) > 1:
                    head += f"（第{int(p_no)}/{int(p_tt)}段）"
                head += "\n"

            context = (item.get("context") or "").strip()
            snippet = context[:snippet_chars] + ("…" if len(context) > snippet_chars else "")
            snippet = head + snippet if head else snippet

            citations.append({
                "rank": rank,
                "node_id": item.get("node_id"),
                "score": item.get("score"),
                "snippet": snippet,
                "metadata": meta,
            })
        return citations
