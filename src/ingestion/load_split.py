import re
from pypdf import PdfReader
from loguru import logger
from typing import List, Tuple


def split_paper_to_chunks(
    text: str,
    max_chars: int = 1000,
    overlap_chars: int = 200,
) -> List[str]:
    """
    针对学术论文优化的文本分割器
    - 保留段落完整性
    - 识别章节标题
    - 处理公式和引用
    """
    
    # 1. 预处理：识别章节标题（通常全大写或数字开头）
    section_pattern = r'^(?:\d+\.?\s+)?[A-Z][A-Z\s]{3,}$'
    lines = text.split('\n')
    processed_lines = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if re.match(section_pattern, line):
            # 章节标题前后加特殊标记
            processed_lines.append(f'\n\n=== {line} ===\n')
        else:
            processed_lines.append(line)
    
    text = '\n'.join(processed_lines)
    
    # 2. 按段落分割（论文段落通常用双换行分隔）
    paragraphs = re.split(r'\n\s*\n+', text)
    chunks: List[str] = []
    
    for para in paragraphs:
        para = para.strip()
        if not para or len(para) < 10:  # 过滤太短的段落（可能是页码、页眉）
            continue
        
        # 保留章节标题标记
        if para.startswith('==='):
            chunks.append(para)
            continue
        
        # 3. 短段落直接保留
        if len(para) <= max_chars:
            chunks.append(para)
            continue
        
        # 4. 长段落按句子切分（增强的句子边界检测）
        # 考虑论文中的特殊情况：et al., Fig., eq., etc.
        sentences = re.split(
            r'(?<=[.!?])\s+(?=[A-Z])',  # 只在句号后+大写字母处切分
            para
        )
        
        current_chunk = ""
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            
            # 尝试添加当前句子
            test_chunk = (current_chunk + " " + sent).strip()
            
            if len(test_chunk) <= max_chars:
                current_chunk = test_chunk
            else:
                # 当前chunk已满，保存并开始新chunk
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sent
        
        # 保存剩余内容
        if current_chunk:
            chunks.append(current_chunk)
    
    # 5. 添加智能overlap（带上下文感知）
    final_chunks: List[str] = []
    
    for i, chunk in enumerate(chunks):
        # 如果是章节标题，不添加overlap
        if chunk.startswith('==='):
            final_chunks.append(chunk)
            continue
        
        # 正常chunk：添加前一个chunk的结尾作为上下文
        if i > 0 and not chunks[i-1].startswith('==='):
            prev_chunk = chunks[i-1]
            # 取前一个chunk的最后overlap_chars个字符
            overlap_text = prev_chunk[-overlap_chars:].strip()
            
            # 智能截取：尝试从句子边界开始
            sentences_in_overlap = re.split(r'(?<=[.!?])\s+', overlap_text)
            if len(sentences_in_overlap) > 1:
                # 只保留完整句子
                overlap_text = ' '.join(sentences_in_overlap[-2:])
            
            chunk = f"[...{overlap_text}]\n\n{chunk}"
        
        # 对于超长chunk，强制滑窗切分
        if len(chunk) > max_chars * 1.5:  # 允许一定容差
            start = 0
            while start < len(chunk):
                end = start + max_chars
                final_chunks.append(chunk[start:end])
                start = end - overlap_chars
        else:
            final_chunks.append(chunk)
    
    return [c.strip() for c in final_chunks if c.strip()]


def load_and_split_paper(
    file_path: str,
    max_chars: int = 1200,
    overlap_chars: int = 200
) -> List[dict]:
    """
    加载论文PDF并返回带元数据的chunks
    """
    try:
        reader = PdfReader(file_path)
        full_text_parts = []
        page_mapping = []  # 记录每个chunk来自哪一页
        current_len = 0
        
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text:
                full_text_parts.append(text)
                current_len += len(text) + 1  # +1 for the '\n'
                page_mapping.append((current_len, page_num))
        
        full_text = '\n'.join(full_text_parts)
        logger.info(f"Extracted {len(full_text)} characters from {len(reader.pages)} pages")
        
        # 分割文本
        chunks = split_paper_to_chunks(full_text, max_chars, overlap_chars)
        
        # 添加元数据
        chunks_with_metadata = []
        for i, chunk in enumerate(chunks):
            # 简单估算chunk所在页码
            chunk_pos = full_text.find(chunk[:100])
            if chunk_pos == -1:
                page_num = 1 
            else:
                page_num = 1
                for pos, page in page_mapping:
                    if chunk_pos < pos:
                        break
                    page_num = page
            
            chunks_with_metadata.append({
                'content': chunk,
                'chunk_id': i,
                'page': page_num,
                'is_section_title': chunk.startswith('==='),
                'char_count': len(chunk)
            })
        
        logger.info(f"Split into {len(chunks)} chunks")
        logger.info(f"Average chunk size: {sum(len(c) for c in chunks) / len(chunks):.0f} chars")
        
        return chunks_with_metadata
    
    except Exception as e:
        logger.error(f"Error loading PDF: {e}")
        return []


# # 使用示例
# if __name__ == "__main__":
#     chunks = load_and_split_paper("paper.pdf", max_chars=1000, overlap_chars=150)
    
#     for chunk_data in chunks[:3]:  # 显示前3个chunk
#         print(f"\n{'='*60}")
#         print(f"Chunk {chunk_data['chunk_id']} | Page {chunk_data['page']} | {chunk_data['char_count']} chars")
#         print(f"Section Title: {chunk_data['is_section_title']}")
#         print(f"{'='*60}")
#         print(chunk_data['content'][:300] + "...")