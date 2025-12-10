from typing import List
from pathlib import Path
from loguru import logger
from pypdf import PdfReader


def load_and_split_pdf(file_path: str, chunk_size: int = 512, chunk_overlap: int = 50) -> List[str]:
    """
    读取 PDF，并按词数切成重叠的文本块。
    """
    try:
        reader = PdfReader(file_path)
        full_text_parts: List[str] = []

        for page in reader.pages:
            text = page.extract_text() or ""
            if text:
                full_text_parts.append(text)

        full_text = "\n".join(full_text_parts)
        logger.info(f"Extracted {len(full_text)} characters from PDF")

        words = full_text.split()
        chunks: List[str] = []
        i = 0
        step = max(1, chunk_size - chunk_overlap)

        while i < len(words):
            segment = words[i : i + chunk_size]
            chunks.append(" ".join(segment))
            i += step

        chunks = [c for c in chunks if c.strip()]
        logger.info(f"Split into {len(chunks)} chunks")
        return chunks

    except Exception as e:
        logger.error(f"Error loading PDF {file_path}: {e}")
        return []
