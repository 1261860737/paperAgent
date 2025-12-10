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

    def ingest_data(self, embed_data: EmbedData):
        """将嵌入数据输入矢量数据库。"""
        if not self.client:
            raise RuntimeError("Milvus client not initialized. Call initialize_client() first.")
        
        logger.info(f"Ingesting {len(embed_data.contexts)} documents...")

        total_inserted = 0
        for batch_context, batch_binary_embeddings in zip(
            batch_iterate(embed_data.contexts, self.batch_size),
            batch_iterate(embed_data.binary_embeddings, self.batch_size)
        ):
            # 准备插入数据
            data_batch = []
            for context, binary_embedding in zip(batch_context, batch_binary_embeddings):
                data_batch.append({
                    "context": context,
                    "binary_vector": binary_embedding
                })

            # Insert batch
            self.client.insert(
                collection_name=self.collection_name,
                data=data_batch
            )

            total_inserted += len(batch_context)
            logger.info(f"Inserted batch: {len(batch_context)} documents")

        logger.info(f"Successfully ingested {total_inserted} documents with binary quantization")

    def search(
        self, 
        binary_query: bytes, 
        top_k: int = None,
        output_fields: List[str] = None
    ):
        if not self.client:
            raise RuntimeError("Milvus client not initialized. Call initialize_client() first.")
        
        top_k = top_k or settings.top_k
        output_fields = output_fields or ["context"]

        # 使用 MilvusClient 进行相似性搜索
        search_results = self.client.search(
            collection_name=self.collection_name,
            data=[binary_query],
            anns_field="binary_vector",
            search_params={"metric_type": "HAMMING", "params": {}},
            limit=top_k,
            output_fields=output_fields
        )

        # 格式结果
        formatted_results = []
        for result in search_results[0]:
            formatted_results.append({
                "id": result["id"],
                "score": 1.0 / (1.0 + result["distance"]),  # 将汉明距离转换为相似度
                "payload": {"context": result["entity"]["context"]}
            })

        return formatted_results

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