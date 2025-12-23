from typing import Optional, List, Dict, Any
from loguru import logger
from pydantic import BaseModel, Field
import jieba
from rank_bm25 import BM25Okapi
from src.indexing.milvus_vdb import MilvusVDB
from sentence_transformers import CrossEncoder
from src.indexing.embed_data import EmbedData
from config.settings import settings
from src.ingestion.load_split import load_and_split_document

class TextNode(BaseModel):
    text: str
    id_: str
    metadata: Dict = {}


class NodeWithScore(BaseModel):
    node: TextNode
    score: float

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

        # 2. 构建 BM25 索引 (内存级)
        #    注意：这需要 embed_data.contexts 已经填充了数据
        if self.embed_data.contexts:
            logger.info(f"Building BM25 index for {len(self.embed_data.contexts)} documents...")
            # 对中文文档进行分词
            tokenized_corpus = [list(jieba.cut(doc)) for doc in self.embed_data.contexts]
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info("BM25 index built successfully.")
        else:
            logger.warning("EmbedData contexts are empty! BM25 will not work.")
            self.bm25 = None


    def _bm25_search(self, query: str, top_k: int = 50) -> List[Dict]:
        """
        执行纯关键词检索
        """
        if not self.bm25:
            return []
        
        # 1. 对查询分词
        tokenized_query = list(jieba.cut(query))
        
        # 2. 获取分数
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # 3. 排序并取 Top-K
        #    argsort 返回的是从小到大的索引，所以要反转
        top_n_indexes = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_n_indexes:
            # 过滤掉分数极低的结果 (可选)
            if doc_scores[idx] <= 0:
                continue
                
            results.append({
                "context": self.embed_data.contexts[idx],
                # 假设 contexts 和 embeddings 是对应的，且 Milvus ID 也是顺序对应的 (1, 2, 3...)
                # 如果你的 Milvus ID 是 UUID，这里需要额外的映射机制。
                # 简单起见，这里假设我们能通过内容匹配或索引找回 ID，或者暂时忽略 ID 匹配
                "id": str(idx), 
                "score": float(doc_scores[idx]),
                "source": "bm25"
            })
        return results


    def search(self, query: str, top_k: Optional[int] = None) -> List[NodeWithScore]:
        """
        混合检索：Milvus (Vector) + BM25 (Keyword) -> Rerank
        """
        if top_k is None:
            top_k = self.top_k

        # 1. 粗排：从 Milvus 获取更多候选 (比如 50 个)
        #    二进制搜索非常快，多拿一点没关系
        initial_top_k = 50

        # 生成查询嵌入并转换为二进制
        query_embedding = self.embed_data.get_query_embedding(query)
        binary_query = self.embed_data.binary_quantize_query(query_embedding)

        # 执行矢量搜索
        vector_results = self.vector_db.search(
            binary_query=binary_query,
            top_k=initial_top_k,
            output_fields=["context"]
        )
        # 标记来源
        for r in vector_results:
            r["source"] = "vector"

        # BM25 关键词检索 (字面召回)
        bm25_results = self._bm25_search(query, top_k=initial_top_k)

        # 3. 混合与去重 (Hybrid Fusion)
        # 使用字典按 content 去重 (因为不同路可能召回同一段话)
        # 键是 context (文本内容)，值是结果对象
        unique_candidates = {}

        # 先加向量结果
        for res in vector_results:
            ctx = res["payload"]["context"]
            unique_candidates[ctx] = {
                "context": ctx,
                "id": str(res["id"]),
                "sources": ["vector"]
            }
        # 再加 BM25 结果
        for res in bm25_results:
            ctx = res["context"]
            if ctx in unique_candidates:
                unique_candidates[ctx]["sources"].append("bm25")
            else:
                unique_candidates[ctx] = {
                    "context": ctx,
                    "id": res["id"], # 注意：BM25 的 ID 可能是索引号，需要确保下游不依赖此 ID 查库
                    "sources": ["bm25"]
                }

        candidates = list(unique_candidates.values())
        if not candidates:
            return []

        logger.info(f"Hybrid Retrieval: Vector={len(vector_results)}, BM25={len(bm25_results)}, Merged={len(candidates)}")

        # 4. Rerank 精排
        # -------------------------------------------------------
        pairs = [[query, doc["context"]] for doc in candidates]
        scores = self.reranker.predict(pairs)

        # 5. 组装最终结果
        final_candidates = []
        for i, doc in enumerate(candidates):
            node = TextNode(
                text=doc["context"], 
                id_=str(doc["id"]), 
                metadata={"sources": doc["sources"]} # 记录一下是哪路召回的，方便调试
            )
            final_candidates.append(NodeWithScore(node=node, score=float(scores[i])))

        # 按 Rerank 分数降序
        final_candidates.sort(key=lambda x: x.score, reverse=True)

        # 取 Top-K
        final_results = final_candidates[:top_k]
        
        # [调试日志]
        if final_results:
            top_doc = final_results[0]
            logger.info(f"Top-1: {top_doc.score:.4f} | Source: {top_doc.node.metadata.get('sources')} | Text: {top_doc.node.text[:30]}...")

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
        # 以引文字典格式返回前 k 个检索结果
        results = self.search_with_scores(query, top_k)
        citations = []
        for rank, item in enumerate(results, start=1):
            context = (item.get("context") or "").strip()
            if context:
                snippet = context[:snippet_chars]
                if len(context) > snippet_chars:
                    snippet += "…"
            else:
                snippet = ""

            citations.append({
                "rank": rank,
                "node_id": item.get("node_id"),
                "score": item.get("score"),
                "snippet": snippet,
                "metadata": item.get("metadata") or {},
            })
        return citations