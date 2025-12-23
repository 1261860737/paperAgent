import numpy as np
from typing import List, Tuple
from loguru import logger
from sentence_transformers import SentenceTransformer
from src.ingestion.load_split import load_and_split_document
from config.settings import settings

def batch_iterate(lst: List, batch_size: int):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i+batch_size]

class EmbedData:
    """处理支持二进制量化的文档嵌入。"""
    def __init__(
        self, 
        embed_model_name: str = None,
        batch_size: int = None,
        cache_folder: str = None
    ):
        self.embed_model_name = embed_model_name or settings.embedding_model
        self.batch_size = batch_size or settings.batch_size
        self.cache_folder = cache_folder or settings.hf_cache_dir
        
        self.embed_model = self._load_embed_model()
        self.embeddings = []
        self.binary_embeddings = []
        self.contexts = []

    def _load_embed_model(self):
        """使用句子变换器加载嵌入模型"""
        logger.info(f"Loading embedding model: {self.embed_model_name}")
        model = SentenceTransformer(
            model_name_or_path=self.embed_model_name,
            cache_folder=self.cache_folder,
            trust_remote_code=True,
        )
        return model

    def _binary_quantize(self, embeddings: List[List[float]]):
        """将 float32 嵌入转换为二进制向量。"""
        embeddings_array = np.array(embeddings)
        binary_embeddings = np.where(embeddings_array > 0, 1, 0).astype(np.uint8)
        
        # 将比特打包成字节（每个字节 8 个维度）
        packed_embeddings = np.packbits(binary_embeddings, axis=1)
        return [vec.tobytes() for vec in packed_embeddings]

    def generate_embedding(self, contexts: List[str]):
        embeddings = self.embed_model.encode(
            sentences=contexts,
            batch_size=min(self.batch_size, max(1, len(contexts))),
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed(self, contexts: List[str]):
        self.contexts = contexts
        logger.info(f"Generating embeddings for {len(contexts)} contexts...")

        for batch_context in batch_iterate(contexts, self.batch_size):
            # 生成 float32 嵌入
            batch_embeddings = self.generate_embedding(batch_context)
            self.embeddings.extend(batch_embeddings)

            # 转换为二进制并存储
            binary_batch = self._binary_quantize(batch_embeddings)
            self.binary_embeddings.extend(binary_batch)

        logger.info(f"Generated {len(self.embeddings)} embeddings with binary quantization")

    def get_query_embedding(self, query: str):
        # 为单个查询生成嵌入
        embedding = self.embed_model.encode(
            sentences=[query],
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return embedding[0].tolist()

    def binary_quantize_query(self, query_embedding: List[float]):
        # 将查询嵌入转换为二进制格式
        embedding_array = np.array([query_embedding])
        binary_embedding = np.where(embedding_array > 0, 1, 0).astype(np.uint8)
        packed_embedding = np.packbits(binary_embedding, axis=1)
        return packed_embedding[0].tobytes()

    def clear(self):
        self.embeddings.clear()
        self.binary_embeddings.clear()
        self.contexts.clear()
        logger.info("Cleared all embeddings and contexts")

    # [新增] 核心流水线：连接 加载器(Loader) 和 嵌入器(Embedder)
# =======================================================
def embed_file_pipeline(file_path: str, chunk_size: int = 800, overlap: int = 100) -> Tuple[List[str], List[List[float]], List[bytes]]:
    """
    处理单个文件：加载 -> 切分 -> 嵌入
    """
    logger.info(f"Starting pipeline for file: {file_path}")
    
    # 1. 调用你已经改好的 load_split.py 中的函数
    chunks = load_and_split_document(file_path, chunk_size=chunk_size, overlap=overlap)
    
    if not chunks:
        logger.warning(f"No chunks generated from {file_path}")
        return [], [], []

    # 2. 初始化嵌入模型
    embedder = EmbedData()
    
    # 3. 生成嵌入 (float32 和 binary 都会生成)
    embedder.embed(chunks)
    
    # 4. 返回结果 (Contexts, Dense Vectors, Binary Vectors)
    return embedder.contexts, embedder.embeddings, embedder.binary_embeddings