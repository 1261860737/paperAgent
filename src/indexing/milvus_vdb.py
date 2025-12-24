from typing import List
from loguru import logger
from pymilvus import MilvusClient, DataType
from src.indexing.embed_data import EmbedData, batch_iterate
from config.settings import settings

class MilvusVDB:
    """支持二进制量化的 Milvus 向量数据库。"""
    def __init__(
        self,
        collection_name: str = None,
        vector_dim: int = None,
        batch_size: int = None,
        db_file: str = None
    ):
        self.collection_name = collection_name or settings.collection_name
        self.vector_dim = vector_dim or settings.vector_dim
        self.batch_size = batch_size or settings.batch_size
        self.db_file = db_file or settings.milvus_db_path
        self.client = None

    def initialize_client(self):
        try:
            self.client = MilvusClient(self.db_file)
            logger.info(f"Initialized Milvus Lite client with database: {self.db_file}")
        except Exception as e:
            logger.error(f"Failed to initialize Milvus client: {e}")
            raise e

    def create_collection(self):
        """创建支持二进制向量的集合。"""
        if not self.client:
            raise RuntimeError("Milvus client not initialized. Call initialize_client() first.")
        
        # 如果存在，则删除现有集合
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)
            logger.info(f"Dropped existing collection: {self.collection_name}")

        # 为二进制向量创建模式
        schema = self.client.create_schema(
            auto_id=True,
            enable_dynamic_fields=True,
        )

        # 为模式添加字段
        schema.add_field(
            field_name="id", 
            datatype=DataType.INT64, 
            is_primary=True, 
            auto_id=True
        )
        schema.add_field(
            field_name="context", 
            datatype=DataType.VARCHAR, 
            max_length=65535
        )
        schema.add_field(
            field_name="binary_vector", 
            datatype=DataType.BINARY_VECTOR, 
            dim=self.vector_dim
        )
        schema.add_field(field_name="node_type", datatype=DataType.VARCHAR, max_length=32)
        schema.add_field(field_name="law_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="law_title", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="article_no", datatype=DataType.INT64)
        schema.add_field(field_name="article_label", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="chapter", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="part_no", datatype=DataType.INT64)
        schema.add_field(field_name="part_total", datatype=DataType.INT64)
        schema.add_field(field_name="source_file", datatype=DataType.VARCHAR, max_length=256)


        # 为二进制向量创建索引参数
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="binary_vector",
            index_name="binary_vector_index",
            index_type="BIN_FLAT",  # 二进制向量的精确搜索
            metric_type="HAMMING"   # 二进制向量的汉明距离
        )

        # 使用模式和索引创建集合
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )

        logger.info(f"Created collection '{self.collection_name}' with binary vectors (dim={self.vector_dim})")

    def ingest_data_raw(self, contexts: List[str], binary_embeddings: List[bytes], metadatas: List[dict] = None):
        if not self.client:
            raise RuntimeError("Milvus client not initialized. Call initialize_client() first.")

        metadatas = metadatas or [{} for _ in contexts]
        if len(metadatas) != len(contexts):
            raise ValueError(f"metadatas length mismatch: {len(metadatas)} vs contexts {len(contexts)}")

        logger.info(f"Ingesting {len(contexts)} documents...")

        def _s(x, default=""):
            return default if x is None else str(x)

        def _i(x, default=-1):
            try:
                return default if x is None else int(x)
            except Exception:
                return default

        total_inserted = 0
        for batch_ctx, batch_vec, batch_meta in zip(
            batch_iterate(contexts, self.batch_size),
            batch_iterate(binary_embeddings, self.batch_size),
            batch_iterate(metadatas, self.batch_size),
        ):
            data_batch = []
            for context, binary_embedding, meta in zip(batch_ctx, batch_vec, batch_meta):
                data_batch.append({
                    "context": context,
                    "binary_vector": binary_embedding,

                    # ===== 元数据显式入库 =====
                    "node_type": _s(meta.get("node_type"), "article"),
                    "law_id": _s(meta.get("law_id")),
                    "law_title": _s(meta.get("law_title")),
                    "article_no": _i(meta.get("article_no")),
                    "article_label": _s(meta.get("article_label")),
                    "chapter": _s(meta.get("chapter")),
                    "part_no": _i(meta.get("part_no"), 1),
                    "part_total": _i(meta.get("part_total"), 1),
                    "source_file": _s(meta.get("source_file")),
                })

            self.client.insert(collection_name=self.collection_name, data=data_batch)
            total_inserted += len(batch_ctx)
            logger.info(f"Inserted batch: {len(batch_ctx)} documents")

        logger.info(f"Successfully ingested {total_inserted} documents with binary quantization")


    def search(
        self, 
        binary_query: bytes, 
        top_k: int = None,
        output_fields: List[str] = None,
        filter_expr: str = None
    ):
        if not self.client:
            raise RuntimeError("Milvus client not initialized. Call initialize_client() first.")
        
        top_k = top_k or settings.top_k
        output_fields = output_fields or [
        "context", "node_type", "law_title", "law_id", "article_no", "article_label",
        "chapter", "part_no", "part_total", "source_file"
        ]

        # 使用 MilvusClient 进行相似性搜索
        kwargs = dict(
            collection_name=self.collection_name,
            data=[binary_query],
            anns_field="binary_vector",
            search_params={"metric_type": "HAMMING", "params": {}},
            limit=top_k,
            output_fields=output_fields
        )
        if filter_expr:
            kwargs["filter"] = filter_expr

        search_results = self.client.search(**kwargs)

        # 格式结果
        formatted_results = []
        for result in search_results[0]:
            entity = result["entity"]
            payload = {k: entity.get(k) for k in output_fields}

            formatted_results.append({
                "id": result["id"],
                "score": 1.0 / (1.0 + result["distance"]),  # 将汉明距离转换为相似度
                "payload": payload
            })

        return formatted_results

    def query(self, filter_expr: str, output_fields: List[str], limit: int = 1000):
        if not self.client:
            raise RuntimeError("Milvus client not initialized. Call initialize_client() first.")
        # 不同版本参数名可能是 filter 或 expr；MilvusClient 通常用 filter
        return self.client.query(
            collection_name=self.collection_name,
            filter=filter_expr,
            output_fields=output_fields,
            limit=limit,
        )

    def fetch_articles(self, article_nos: List[int]) -> List[dict]:
        nos = sorted({int(x) for x in article_nos if x is not None})
        if not nos:
            return []

        output_fields = [
            "id", "context", "node_type", "law_title", "law_id", "article_no", "article_label",
            "chapter", "part_no", "part_total", "source_file"
        ]
        expr = f'node_type == "article" && article_no in {nos}'
        rows = self.query(expr, output_fields=output_fields, limit=5000)

        # 按条号、分片排序（保证“前两条”拼接顺序正确）
        rows.sort(key=lambda r: (r.get("article_no", -1), r.get("part_no", 1)))
        return rows


    def collection_exists(self):
        if not self.client:
            return False
        return self.client.has_collection(collection_name=self.collection_name)

    def get_collection_info(self):
        if not self.client:
            raise RuntimeError("Milvus client not initialized. Call initialize_client() first.")
        
        if not self.collection_exists():
            return {"exists": False}
        
        # 获取采集统计数据
        stats = self.client.get_collection_stats(collection_name=self.collection_name)
        return {
            "exists": True,
            "row_count": stats["row_count"],
            "collection_name": self.collection_name
        }

    #  关闭数据库连接
    def close(self):
        if self.client:
            self.client.close()
            logger.info("Closed Milvus client connection")