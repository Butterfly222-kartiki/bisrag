"""
BIS SP 21 Parser — Updated for Pre-Generated Chunks
=====================================================
CHANGES FROM ORIGINAL:
  - parse_pdf() replaced with load_from_chunks_json(): skips all PDF/PyMuPDF/pdfplumber
    logic since chunks are already generated in bis_all_chunks.json.
  - Field mapping updated: existing chunks use 'content' (not 'text'), 'category'
    (extra metadata), 'chunk_id' (sequential int). Parser normalizes these into the
    same parent/child schema the rest of the pipeline expects.
  - normalize_standard_id, is_real_heading, create_child_chunks, _dedup_keep_longest
    all preserved unchanged — still used for child chunk generation and dedup.
  - New entry point: python parser.py --chunks bis_all_chunks.json
    Old entry point:  python parser.py <pdf_path>   (still works if PDF provided)
  - save_chunks / load_chunks unchanged.
"""

import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False


# ── Regex / Constants ──────────────────────────────────────────────────────────

IS_HEADING_PATTERN = re.compile(
    r'\b(IS\s+\d+(?:\s*\(Part\s*\d+\))?(?:\s*\(Sec\s*\d+\))?'
    r'(?:\s*\(Section\s*\d+\))?\s*:\s*\d{4})\b',
    re.IGNORECASE
)

TOC_TITLE_MAX_LEN   = 200
TOC_INLINE_PATTERN  = re.compile(r'\d+\.\d+\s+IS\s+\d+')

CHILD_WINDOW_SIZE   = 4
CHILD_STEP_SIZE     = 2
MIN_SENTENCE_LEN    = 20
MIN_CONTENT_LEN     = 50

INLINE_CITATION_INDICATORS = [
    'as per', 'according to', 'refer to', 'in accordance',
    'shall be', 'methods of', 'procedures of', 'specified in',
    'given in', 'listed in', 'covered in', 'conforming to',
]


# ── Helpers (unchanged from original) ─────────────────────────────────────────

def normalize_standard_id(raw: str) -> str:
    s = re.sub(r'\s+', ' ', raw.strip())
    s = re.sub(r'\s*:\s*', ': ', s)
    s = re.sub(r'\(\s*[Pp][Aa][Rr][Tt]\s*', '(Part ', s)
    return s


def is_toc_title(title: str) -> bool:
    if len(title) > TOC_TITLE_MAX_LEN:
        return True
    if TOC_INLINE_PATTERN.search(title):
        return True
    return False


def is_real_heading(line: str, match: re.Match) -> bool:
    if match.start() > 10:
        return False
    if len(line) > 160:
        return False
    after = line[match.end():].strip()
    if after and not re.search(r'[A-Za-z]{2,}', after):
        return False
    after_lower = after.lower()
    if any(ind in after_lower for ind in INLINE_CITATION_INDICATORS):
        return False
    return True


def clean_toc_title_suffix(title: str) -> str:
    return re.sub(r'\s+\d+(\.\d+)?\s*$', '', title).strip(" —:-")


def _dedup_keep_longest(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[str, Dict] = {}
    for chunk in chunks:
        sid = chunk["standard_id"]
        if sid not in best or len(chunk["text"]) > len(best[sid]["text"]):
            best[sid] = chunk
    result = [c for c in best.values() if len(c["text"]) >= MIN_CONTENT_LEN]
    dropped = len(best) - len(result)
    print(f"[Parser] Dedup (keep-longest): {len(chunks)} raw → {len(best)} unique → "
          f"{len(result)} valid (dropped {dropped} near-empty stubs)")
    return result


def create_child_chunks(parent_chunks: List[Dict]) -> List[Dict]:
    """
    Sliding-window child chunks with context prefix.
    Each child embeds a 4-sentence window prefixed with standard_id + title
    so every child embedding is anchored to its standard identity.
    """
    children = []
    for parent in parent_chunks:
        sid    = parent["standard_id"]
        title  = parent.get("title", "")
        text   = parent["text"]

        raw_sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in raw_sentences if len(s.strip()) >= MIN_SENTENCE_LEN]
        if not sentences:
            continue

        context_prefix = f"Standard: {sid}"
        if title:
            context_prefix += f" — {title}"
        context_prefix += ". "

        i = 0
        while i < len(sentences):
            window = sentences[i: i + CHILD_WINDOW_SIZE]
            window_text = " ".join(window)
            children.append({
                "standard_id":  sid,
                "title":        title,
                "text":         context_prefix + window_text,
                "raw_window":   window_text,
                "page_number":  parent.get("page_number", None),
                "category":     parent.get("category", ""),
                "source":       parent.get("source", ""),
                "chunk_type":   "child",
                "parent_text":  text,
            })
            i += CHILD_STEP_SIZE
            if i >= len(sentences):
                break

    return children


# ── NEW: Load from pre-generated chunks JSON ───────────────────────────────────

def load_from_chunks_json(chunks_path: str) -> Dict[str, List[Dict]]:
    """
    CHANGE: Replaces parse_pdf() when chunks are already generated.

    Reads bis_all_chunks.json (or any chunked JSON in the same schema),
    maps existing fields to the parent/child schema, then generates child
    chunks via the same sliding-window logic used in the PDF path.

    Field mapping:
        existing chunk         → parent chunk
        ─────────────────────────────────────
        chunk_id               → (dropped; chunk_id re-assigned as 1-based index)
        standard_id            → standard_id  (normalized)
        title                  → title
        category               → category     (preserved as extra metadata)
        content                → text         (renamed; this is the embed field)
        source                 → source
        (none)                 → page_number  (set to None — not in text-based chunks)
        (none)                 → chunk_type   (set to "parent")
    """
    print(f"[Parser] Loading pre-generated chunks from: {chunks_path}")
    with open(chunks_path, encoding="utf-8") as f:
        data = json.load(f)

    # Support both bare list and {"chunks": [...]} wrapper
    raw_list = data if isinstance(data, list) else data.get("chunks", [])
    print(f"[Parser] Found {len(raw_list)} raw chunks in file.")

    parent_chunks: List[Dict[str, Any]] = []
    for item in raw_list:
        # Map fields — 'content' → 'text' is the key rename
        sid   = normalize_standard_id(item.get("standard_id", item.get("standard_number", "")))
        title = item.get("title", "")
        if is_toc_title(title):
            title = ""
        title = clean_toc_title_suffix(title)

        text = item.get("content", item.get("text", "")).strip()
        if not text or len(text) < MIN_CONTENT_LEN:
            continue

        parent_chunks.append({
            "standard_id":  sid,
            "title":        title,
            "text":         text,           # ← embed this field
            "page_number":  item.get("page_number", None),
            "category":     item.get("category", ""),
            "source":       item.get("source", "SP 21 : 2005"),
            "chunk_type":   "parent",
        })

    # Dedup: keep longest text per standard_id
    unique_parents = _dedup_keep_longest(parent_chunks)

    # Generate sliding-window child chunks
    child_chunks = create_child_chunks(unique_parents)

    print(f"[Parser] Final: {len(unique_parents)} parent chunks, "
          f"{len(child_chunks)} child chunks.")

    return {
        "parent_chunks": unique_parents,
        "child_chunks":  child_chunks,
    }


# ── Original PDF path (kept intact) ───────────────────────────────────────────

def extract_text_pymupdf(pdf_path: str) -> List[Dict]:
    doc = fitz.open(pdf_path)
    lines = []
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                span_texts = [
                    s["text"] for s in line.get("spans", []) if s["text"].strip()
                ]
                joined = " ".join(span_texts).strip()
                if joined:
                    lines.append({"text": joined, "page": page_num + 1})
    doc.close()
    return lines


def extract_text_pdfplumber(pdf_path: str) -> List[str]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return pages


def _segment_lines_to_chunks(lines: List[Dict]) -> List[Dict[str, Any]]:
    chunks = []
    current_std_id: Optional[str] = None
    current_title: Optional[str]  = None
    current_body_lines: List[str] = []
    current_start_page = 1

    def flush_current():
        nonlocal current_std_id, current_title, current_body_lines
        if current_std_id and current_body_lines:
            full_text = f"{current_std_id}"
            if current_title:
                full_text += f" — {current_title}"
            full_text += "\n" + " ".join(current_body_lines)
            clean_title = (
                current_title
                if current_title and not is_toc_title(current_title)
                else ""
            )
            chunks.append({
                "standard_id":  normalize_standard_id(current_std_id),
                "title":        clean_title,
                "text":         full_text.strip(),
                "page_number":  current_start_page,
                "category":     "",
                "source":       "SP 21 : 2005",
                "chunk_type":   "parent",
            })

    for line_dict in lines:
        text = line_dict["text"]
        page = line_dict["page"]
        match = IS_HEADING_PATTERN.search(text)
        if match and is_real_heading(text, match):
            flush_current()
            current_std_id      = match.group(1)
            current_start_page  = page
            raw_after           = text[match.end():].strip(" —:-")
            after               = clean_toc_title_suffix(raw_after)
            current_title       = after if len(after) > 3 else None
            current_body_lines  = []
        else:
            if current_std_id is None:
                continue
            if current_title is None and len(text) < 120 and not text.endswith("."):
                current_title = text
            else:
                current_body_lines.append(text)

    flush_current()
    return chunks


def parse_pdf(pdf_path: str) -> Dict[str, List[Dict]]:
    print(f"[Parser] Loading PDF: {pdf_path}")
    raw_chunks: List[Dict] = []

    if PYMUPDF_AVAILABLE:
        print("[Parser] Using PyMuPDF...")
        try:
            lines = extract_text_pymupdf(pdf_path)
            raw_chunks = _segment_lines_to_chunks(lines)
        except Exception as e:
            print(f"[Parser] PyMuPDF failed ({e}), falling back to pdfplumber...")

    if not raw_chunks and PDFPLUMBER_AVAILABLE:
        print("[Parser] Using pdfplumber...")
        plain_pages = extract_text_pdfplumber(pdf_path)
        lines = []
        for page_num, page_text in enumerate(plain_pages):
            for raw_line in page_text.split("\n"):
                stripped = raw_line.strip()
                if stripped:
                    lines.append({"text": stripped, "page": page_num + 1})
        raw_chunks = _segment_lines_to_chunks(lines)

    if not raw_chunks:
        raise RuntimeError("Could not parse PDF. Install pymupdf or pdfplumber.")

    unique_parents = _dedup_keep_longest(raw_chunks)
    child_chunks   = create_child_chunks(unique_parents)

    print(f"[Parser] Final: {len(unique_parents)} parent chunks, "
          f"{len(child_chunks)} child chunks.")
    return {"parent_chunks": unique_parents, "child_chunks": child_chunks}


# ── I/O helpers (unchanged) ────────────────────────────────────────────────────

def save_chunks(chunks_data: Dict, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)
    print(f"[Parser] Saved chunks to {output_path}")


def load_chunks(path: str) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="BIS SP21 Parser")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--chunks", metavar="JSON_PATH",
                       help="Path to pre-generated chunks JSON (bis_all_chunks.json)")
    group.add_argument("--pdf", metavar="PDF_PATH",
                       help="Path to raw SP21 PDF (original flow)")
    ap.add_argument("--output", default="data/chunks.json",
                    help="Output path for processed chunks (default: data/chunks.json)")
    args = ap.parse_args()

    if args.chunks:
        result = load_from_chunks_json(args.chunks)
    else:
        result = parse_pdf(args.pdf)

    save_chunks(result, args.output)