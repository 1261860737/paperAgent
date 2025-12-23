import os
import re
import pymupdf4llm
import mammoth
from markdownify import markdownify as md
from typing import List, Dict
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# --------------------------
# 1. 核心转换层：各种格式 -> Markdown
# --------------------------

def convert_pdf_to_markdown(file_path: str) -> str:
    """
    使用 pymupdf4llm 处理 PDF。
    自动处理双栏排版 (Column Layout) 和 表格 (Tables)。
    """
    print(f"[Loader] Processing PDF with Layout Analysis: {file_path}")
    # pymupdf4llm 会自动侦测双栏，按阅读顺序输出
    return pymupdf4llm.to_markdown(file_path)

def convert_docx_to_markdown(file_path: str) -> str:
    """
    使用 mammoth + markdownify 处理 Word。
    Word 的双栏对程序是透明的，无需特殊处理；mammoth 能完美保留 Word 表格结构。
    """
    print(f"[Loader] Processing DOCX: {file_path}")
    with open(file_path, "rb") as docx_file:
        # 1. Word -> HTML (mammoth 擅长保留文档结构)
        result = mammoth.convert_to_html(docx_file)
        html = result.value
        messages = result.messages # 警告信息，可忽略
        
    # 2. HTML -> Markdown
    # strip=['a', 'img'] 表示去掉链接和图片，只留纯文本结构
    markdown_text = md(html, strip=['a', 'img'], heading_style="ATX")
    return markdown_text

# 法律 Markdown 预处理函数
# ==========================================
def preprocess_legal_markdown(md_text: str) -> str:
    """
    pymupdf4llm 可能不会把“第一条”识别为标题。
    我们需要手动用正则给它们加上 Markdown 标题符号 (#)。
    """
    lines = md_text.split('\n')
    processed_lines = []
    
    # 定义正则模式
    # 匹配 "第一章 总则" -> 转换为 "## 第一章 总则" (二级标题)
    chapter_pattern = re.compile(r'^\s*(第[零一二三四五六七八九十百]+章.*)$')
    
    # 匹配 "第一条" -> 转换为 "### 第一条" (三级标题)
    article_pattern = re.compile(r'^\s*(第[零一二三四五六七八九十百]+条.*)$')

    for line in lines:
        # 去除原本可能存在的粗体符号 (**第一章**) 以便处理
        clean_line = line.strip().replace('**', '')
        
        # 如果这一行已经是标题（比如 pymupdf 识别出来的），就跳过
        if clean_line.startswith('#'):
            processed_lines.append(line)
            continue
            
        # 强制升级为 Markdown 标题
        if chapter_pattern.match(clean_line):
            # 给章节加前缀 ##
            processed_lines.append(f"## {clean_line}")
        elif article_pattern.match(clean_line):
            # 给条款加前缀 ###
            processed_lines.append(f"### {clean_line}")
        else:
            processed_lines.append(line)
            
    return '\n'.join(processed_lines)

# --------------------------
# 2. 统一加载入口
# --------------------------

def load_document_to_markdown(file_path: str) -> str:
    """根据后缀分发处理逻辑"""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".pdf":
        return convert_pdf_to_markdown(file_path)
    elif ext in [".docx", ".doc"]:
        return convert_docx_to_markdown(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

# --------------------------
# 3. 切分逻辑 (复用之前的 Markdown 策略)
# --------------------------

def load_and_split_document(file_path: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    对外暴露的主函数：加载 -> 转Markdown -> 结构化切分
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # A. 统一转换为 Markdown
    raw_md_text = load_document_to_markdown(file_path)

    # B. 预处理：强制注入法律结构
    # -------------------------------------------------
    md_text = preprocess_legal_markdown(raw_md_text)

    # B. 定义法律文档的结构层级
    # 适配 Word 和 PDF 转换后的通用 Markdown 标题
    headers_to_split_on = [
        ("#", "header_1"),    # 一级标题 (编/章)
        ("##", "header_2"),   # 二级标题 (节/条)
        ("###", "header_3"),  # 三级标题 (款/项)
    ]

    # C. 按结构切分
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, 
        strip_headers=False 
    )
    md_header_splits = markdown_splitter.split_text(md_text)

    # D. 按长度切分 (防止超长法条)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", "；"]
    )
    final_splits = text_splitter.split_documents(md_header_splits)

    # E. 格式化输出 (带上下文元数据)
    processed_chunks = []
    for doc in final_splits:
        # 将 "Header 1 > Header 2" 这样的路径“烧录”进文本开头
        # 这样即使切分了，模型也知道这段话属于哪一章
        context_parts = [
            doc.metadata.get(h, "") 
            for h in ["header_1", "header_2", "header_3"] 
            if doc.metadata.get(h)
        ]
        
        if context_parts:
            context_str = " > ".join(context_parts)
            final_text = f"【来源：{os.path.basename(file_path)} | 章节：{context_str}】\n{doc.page_content}"
        else:
            final_text = f"【来源：{os.path.basename(file_path)}】\n{doc.page_content}"
            
        processed_chunks.append(final_text)

    print(f"Successfully processed {file_path}: generated {len(processed_chunks)} chunks.")
    return processed_chunks

# # 使用示例
# chunks = load_and_split_document("data/森林草原防灭火条例.pdf", chunk_size=800, overlap=100)

# for i, text in enumerate(chunks[:3], start=1):
#     print("\n" + "="*60)
#     print(f"Chunk {i} | {len(text)} chars")
#     print("="*60)
#     print(text[:300] + "...")