import os
import re
import hashlib
from typing import List, Dict, Any, Tuple, Optional

import pymupdf4llm
import mammoth
from markdownify import markdownify as md
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =========================================================
# 0) 工具：数字解析、文本归一化
# =========================================================

ZH_NUM_MAP = {
    "零": 0, "〇": 0,
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9,
    "十": 10, "百": 100, "千": 1000, "万": 10000
}

def zh_num_to_int(s: str) -> int:
    """支持 1~9999 常见中文数字（简化但够覆盖法律条号）。也支持纯数字。"""
    s = (s or "").strip()
    if not s:
        return 0
    if s.isdigit():
        return int(s)

    # 处理类似“十二”“一百零二”“二千三百四十五”
    total = 0
    num = 0
    unit = 1
    # 从右往左解析
    for ch in reversed(s):
        if ch in ("十", "百", "千", "万"):
            unit = ZH_NUM_MAP[ch]
            if unit == 10000:
                total = max(1, total) * 10000
                unit = 1
            if num == 0:
                num = 1  # “十”=10
            total += num * unit
            num = 0
            unit = 1
        else:
            num = ZH_NUM_MAP.get(ch, 0)
            if unit == 1:
                total += num
                num = 0
    return total

def normalize_text(text: str) -> str:
    """
    把 pymupdf4llm 的噪声 token 和断字问题尽量抹平，保证“第…条”能被稳定识别。
    """
    if not text:
        return ""

    # [年] -> 年, [《] -> 《, [(] -> (, [)] -> )
    text = re.sub(r"\[([^\[\]]{1,3})\]", r"\1", text)
    text = text.replace("[(]", "(").replace("[)]", ")")

    # 合并中文间被空格/换行打断的情况： "中华 人 民" / "中华\n人民" -> "中华人民共和国"
    text = re.sub(r"(?<=[\u4e00-\u9fff])[\s\u3000]+(?=[\u4e00-\u9fff])", "", text)

    # 统一空白
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def stable_law_id(file_path: str, law_title: str) -> str:
    """给每份法律生成稳定 id（文件名+标题）。"""
    base = os.path.basename(file_path)
    raw = f"{base}::{law_title}".encode("utf-8", errors="ignore")
    return "law_" + hashlib.md5(raw).hexdigest()[:16]


# =========================================================
# 1) 各种格式 -> Markdown
# =========================================================

def convert_pdf_to_markdown(file_path: str) -> str:
    """
    PDF -> Markdown（含布局分析）。
    你后续如果要更强的版面能力，可替换/扩展为 pymupdf_layout。
    """
    print(f"[Loader] Processing PDF with Layout Analysis: {file_path}")
    return pymupdf4llm.to_markdown(file_path)

def convert_docx_to_markdown(file_path: str) -> str:
    print(f"[Loader] Processing DOCX: {file_path}")
    with open(file_path, "rb") as docx_file:
        result = mammoth.convert_to_html(docx_file)
        html = result.value
    markdown_text = md(html, strip=["a", "img"], heading_style="ATX")
    return markdown_text

def load_document_to_markdown(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return convert_pdf_to_markdown(file_path)
    if ext in [".docx", ".doc"]:
        return convert_docx_to_markdown(file_path)
    raise ValueError(f"Unsupported file format: {ext}")


# =========================================================
# 2) 目录/页眉页脚识别（根治的关键之一）
# =========================================================

TOC_HINTS = ("目录", "目 录", "contents", "CONTENTS")

def detect_toc_region(text: str) -> Tuple[int, int]:
    """
    粗略识别目录区间：从首次出现“目录”开始，到首次出现“第一章/第一条”等正文锚点之前。
    如果识别不到，返回 (-1, -1)。
    """
    t = text
    # 找目录起点
    toc_start = -1
    for h in TOC_HINTS:
        idx = t.find(h)
        if idx != -1:
            toc_start = idx
            break
    if toc_start == -1:
        return (-1, -1)

    # 找正文锚点：出现“第一章”或“第一条”或“第1条”
    body_anchor = re.search(r"第\s*(?:一|1)\s*(?:章|条)", t[toc_start:])
    if not body_anchor:
        # 目录但没找到正文锚点，就只截取目录后一小段
        return (toc_start, min(len(t), toc_start + 2500))

    toc_end = toc_start + body_anchor.start()
    if toc_end <= toc_start:
        return (-1, -1)
    return (toc_start, toc_end)

def strip_repeated_boilerplate(md_text: str) -> Tuple[str, List[str]]:
    """
    轻量去除页眉页脚/重复行：对“短行”统计频率，频繁出现的视为 boilerplate。
    返回：清理后的全文 + 被剔除的 boilerplate 行（可选择单独存 node）。
    """
    lines = md_text.split("\n")
    freq: Dict[str, int] = {}
    candidates: List[str] = []

    for ln in lines:
        s = ln.strip()
        # 太长不是页眉页脚；太短也可能是空行
        if 5 <= len(s) <= 40:
            freq[s] = freq.get(s, 0) + 1

    # 出现次数阈值：按文档长度自适应
    threshold = 4 if len(lines) > 300 else 3
    boiler = {k for k, v in freq.items() if v >= threshold}

    cleaned = []
    removed = []
    for ln in lines:
        s = ln.strip()
        if s in boiler:
            removed.append(s)
            continue
        cleaned.append(ln)

    cleaned_text = "\n".join(cleaned)
    # 去重 removed
    removed_unique = sorted(set(removed))
    return cleaned_text, removed_unique


# =========================================================
# 3) 全文级“章/条”识别与切片（根治的核心）
# =========================================================

CHAPTER_RE = re.compile(r"(第\s*[零一二三四五六七八九十百千0-9]+\s*章\s*[^\n]{0,30})")
ARTICLE_RE = re.compile(r"(第\s*[零一二三四五六七八九十百千0-9]+\s*条)")

def extract_law_title(md_text: str) -> str:
    """
    粗略提取法律名称：取全文前 2000 字中最像标题的一行（优先 # 标题，其次最长中文行）。
    """
    head = md_text[:2000]
    # 1) Markdown 一级标题
    m = re.search(r"^\s*#\s+(.+)$", head, flags=re.M)
    if m:
        return m.group(1).strip()

    # 2) 找一行较长且中文比例高的
    best = ""
    for ln in head.splitlines():
        s = ln.strip().replace("*", "")
        if len(s) < 6 or len(s) > 40:
            continue
        zh = sum(1 for c in s if "\u4e00" <= c <= "\u9fff")
        if zh / max(1, len(s)) >= 0.6 and len(s) > len(best):
            best = s
    return best or os.path.basename(md_text)  # 兜底

def build_chapter_spans(text: str) -> List[Tuple[int, str]]:
    """返回 [(pos, chapter_title), ...] 按出现顺序"""
    spans = []
    for m in CHAPTER_RE.finditer(text):
        title = normalize_text(m.group(1))
        spans.append((m.start(), title))
    return spans

def find_current_chapter(chapters: List[Tuple[int, str]], pos: int) -> Optional[str]:
    """给定位置，返回最近的章节标题"""
    if not chapters:
        return None
    # 线性倒找（章节数量一般不大）
    for p, t in reversed(chapters):
        if p <= pos:
            return t
    return None

def split_into_article_nodes(
    full_text: str,
    file_path: str,
    law_title: str,
    chunk_size: int,
    overlap: int,
) -> List[Dict[str, Any]]:
    """
    把全文按“条”切成 node；超长条再拆分为多 part，但共享 article_no。
    """
    nodes: List[Dict[str, Any]] = []
    law_id = stable_law_id(file_path, law_title)

    chapters = build_chapter_spans(full_text)

    matches = list(ARTICLE_RE.finditer(full_text))
    if not matches:
        # 兜底：没有识别到条，就把全文当一个 article-like node
        nodes.append({
            "text": full_text.strip(),
            "metadata": {
                "node_type": "article",
                "law_id": law_id,
                "law_title": law_title,
                "article_no": None,
                "article_label": None,
                "chapter": None,
                "part_no": 1,
                "part_total": 1,
                "source_file": os.path.basename(file_path),
            }
        })
        return nodes

    # 定长切分器（只用于超长条）
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", "；", ";", ".", " "],
    )

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)

        article_label_raw = normalize_text(m.group(1))  # “第 一 条”也会被 normalize 合并
        # 提取条号数字
        num_m = re.search(r"第\s*([零一二三四五六七八九十百千0-9]+)\s*条", article_label_raw)
        article_no = zh_num_to_int(num_m.group(1)) if num_m else None

        chapter = find_current_chapter(chapters, start)

        article_text = full_text[start:end].strip()

        # 超长条：拆 part
        if len(article_text) > chunk_size:
            parts = splitter.split_text(article_text)
            part_total = len(parts)

            short_label_match = re.match(r"(第\s*[零一二三四五六七八九十百千0-9]+\s*条)", article_label_raw)
            short_label = short_label_match.group(1) if short_label_match else article_label_raw

            for pi, part in enumerate(parts, start=1):
                current_text = part.strip()
                
                # [关键修改] 如果是第 2 部分及以后，手动在开头注入“（续）第一条”
                if pi > 1:
                    current_text = f"（续）{short_label} {current_text}"

                nodes.append({
                    "text": part.strip(),
                    "metadata": {
                        "node_type": "article",
                        "law_id": law_id,
                        "law_title": law_title,
                        "article_no": article_no,
                        "article_label": article_label_raw,
                        "chapter": chapter,
                        "part_no": pi,
                        "part_total": part_total,
                        "source_file": os.path.basename(file_path),
                    }
                })
        else:
            nodes.append({
                "text": article_text,
                "metadata": {
                    "node_type": "article",
                    "law_id": law_id,
                    "law_title": law_title,
                    "article_no": article_no,
                    "article_label": article_label_raw,
                    "chapter": chapter,
                    "part_no": 1,
                    "part_total": 1,
                    "source_file": os.path.basename(file_path),
                }
            })

    return nodes


# =========================================================
# 4) 对外主函数：输出 nodes（text + metadata）
# =========================================================

def load_and_split_document(
    file_path: str,
    chunk_size: int = 800,
    overlap: int = 100,
    keep_toc: bool = True,
    keep_boilerplate: bool = True,
) -> List[Dict[str, Any]]:
    """
    根治版：返回 List[{"text":..., "metadata": {...}}]
    - 结构化法条 node：node_type="article"，含 article_no
    - 目录 node：node_type="toc"（可选保留）
    - 页眉页脚 node：node_type="boilerplate"（可选保留）
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    raw_md = load_document_to_markdown(file_path)

    # 1) 先清洗：归一化（提升条号识别成功率）
    md_text = normalize_text(raw_md)

    # 2) 轻量剔除重复页眉页脚（但可选择保留为 node）
    md_text2, boiler_lines = strip_repeated_boilerplate(md_text)

    # 3) 目录区间识别（可选保存 toc node，并从正文中剔除 toc 区间）
    toc_start, toc_end = detect_toc_region(md_text2)
    toc_text = ""
    if toc_start != -1 and toc_end != -1 and toc_end > toc_start:
        toc_text = md_text2[toc_start:toc_end].strip()
        body_text = (md_text2[:toc_start] + "\n\n" + md_text2[toc_end:]).strip()
    else:
        body_text = md_text2

    # 4) 识别法律标题 & 按条切片（核心）
    law_title = extract_law_title(body_text)
    law_id = stable_law_id(file_path, law_title)

    nodes: List[Dict[str, Any]] = []

    # 4.1 保存 toc
    if keep_toc and toc_text:
        nodes.append({
            "text": toc_text,
            "metadata": {
                "node_type": "toc",
                "law_id": law_id,
                "law_title": law_title,
                "source_file": os.path.basename(file_path),
            }
        })

    # 4.2 保存 boilerplate
    if keep_boilerplate and boiler_lines:
        nodes.append({
            "text": "\n".join(boiler_lines),
            "metadata": {
                "node_type": "boilerplate",
                "law_id": law_id,
                "law_title": law_title,
                "source_file": os.path.basename(file_path),
            }
        })

    # 4.3 article nodes
    article_nodes = split_into_article_nodes(
        full_text=body_text,
        file_path=file_path,
        law_title=law_title,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    nodes.extend(article_nodes)

    print(f"Successfully processed {file_path}: generated {len(nodes)} nodes "
          f"(articles={sum(1 for n in nodes if n['metadata'].get('node_type')=='article')}).")

    return nodes


# # 使用示例
# chunks = load_and_split_document("data/森林草原防灭火条例.pdf", chunk_size=800, overlap=100)

# for i, text in enumerate(chunks[:3], start=1):
#     print("\n" + "="*60)
#     print(f"Chunk {i} | {len(text)} chars")
#     print("="*60)
#     print(text[:300] + "...")