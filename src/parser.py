"""
BIS SP 21 Parser
================
Parses BIS standards data from either a pre-generated chunks JSON file or
directly from the raw SP 21 PDF, then produces parent and child chunks ready
for indexing.


Usage (original PDF flow -- requires PyMuPDF or pdfplumber):
    python parser.py --pdf path/to/bis_sp21.pdf

Output (saved to --output, default data/chunks.json):
    parent_chunks  -- one chunk per standard, used for retrieval context
    child_chunks   -- sliding-window sub-chunks, used for dense search
"""

import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False


# -- Regex / Constants ---------------------------------------------------------

IS_HEADING_PATTERN = re.compile(
    r'\b(IS\s+\d+(?:\s*\(Part\s*\d+\))?(?:\s*\(Sec\s*\d+\))?'
    r'(?:\s*\(Section\s*\d+\))?\s*:\s*\d{4})\b',
    re.IGNORECASE
)

TOC_TITLE_MAX_LEN  = 200
TOC_INLINE_PATTERN = re.compile(r'\d+\.\d+\s+IS\s+\d+')

CHILD_WINDOW_SIZE  = 4
CHILD_STEP_SIZE    = 2
MIN_SENTENCE_LEN   = 20
MIN_CONTENT_LEN    = 50

INLINE_CITATION_INDICATORS = [
    'as per', 'according to', 'refer to', 'in accordance',
    'shall be', 'methods of', 'procedures of', 'specified in',
    'given in', 'listed in', 'covered in', 'conforming to',
]

# Covers the range of roman numerals realistically found in BIS part numbers,
# for example (Part II) -> (Part 2), (Part IV) -> (Part 4).
_ROMAN = {
    'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
    'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10,
    'XI': 11, 'XII': 12,
}


# -- Helpers -------------------------------------------------------------------

def normalize_standard_id(raw: str) -> str:
    """
    Converts any raw standard ID string into a consistent canonical form
    so that IDs match across the parser, index, and eval script.

    Examples of what gets normalized:
        'IS269:1989'        -> 'IS 269: 1989'
        'IS 1489(PART1)'    -> 'IS 1489 (Part 1)'
        'IS 456 (PART II)'  -> 'IS 456 (Part 2)'
        'IS 800(PART1/SEC2) -> 'IS 800 (Part 1/Sec 2)'
    """
    s = re.sub(r'\s+', ' ', raw.strip())
    s = re.sub(r'\s*:\s*', ': ', s)

    # Some source IDs omit the space between the number and the opening
    # parenthesis. Insert it so subsequent patterns match consistently.
    s = re.sub(r'(IS\s*\d+)(\()', r'\1 \2', s)

    # Compound part/section variants like (PART 2/SEC 1) must be handled before
    # the simpler (PART N) rule so they are not partially matched.
    def fix_part_sec(m):
        pn = _ROMAN.get(m.group(1).strip().upper(), m.group(1).strip())
        sn = _ROMAN.get(m.group(2).strip().upper(), m.group(2).strip())
        return f'(Part {pn}/Sec {sn})'

    s = re.sub(
        r'\(\s*(?i:part)\s*(\w+)\s*/\s*(?i:sec)\s*(\w+)\s*\)',
        fix_part_sec, s
    )

    # Normalize plain (PART N), converting any roman numeral to an integer.
    def fix_part(m):
        inner = m.group(1).strip()
        num   = _ROMAN.get(inner.upper(), inner)
        return f'(Part {num})'

    s = re.sub(r'\(\s*(?i:part)\s*(\w+\s*)\)', fix_part, s)
    return s


def is_toc_title(title: str) -> bool:
    # Titles that are very long or contain inline IS-number references are
    # almost certainly table-of-contents lines rather than real section headings.
    if len(title) > TOC_TITLE_MAX_LEN:
        return True
    if TOC_INLINE_PATTERN.search(title):
        return True
    return False


def is_real_heading(line: str, match: re.Match) -> bool:
    # Headings appear at or very near the start of a line. A match that begins
    # well into the line is most likely an inline citation, not a heading.
    if match.start() > 10:
        return False
    # Very long lines are almost always body text with an embedded reference.
    if len(line) > 160:
        return False
    after = line[match.end():].strip()
    if after and not re.search(r'[A-Za-z]{2,}', after):
        return False
    # If the text after the ID reads like a citation phrase, treat the whole
    # line as body text and skip it.
    after_lower = after.lower()
    if any(ind in after_lower for ind in INLINE_CITATION_INDICATORS):
        return False
    return True


def clean_toc_title_suffix(title: str) -> str:
    # Strip trailing page numbers that bleed in from table-of-contents lines.
    return re.sub(r'\s+\d+(\.\d+)?\s*$', '', title).strip(" --:-")


def _dedup_keep_longest(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # When multiple chunks share the same standard ID, keep only the one with
    # the most text. Stubs that are too short to be useful are also dropped here.
    best: Dict[str, Dict] = {}
    for chunk in chunks:
        sid = chunk["standard_id"]
        if sid not in best or len(chunk["text"]) > len(best[sid]["text"]):
            best[sid] = chunk
    result = [c for c in best.values() if len(c["text"]) >= MIN_CONTENT_LEN]
    dropped = len(best) - len(result)
    print(f"[Parser] Dedup (keep-longest): {len(chunks)} raw -> {len(best)} unique -> "
          f"{len(result)} valid (dropped {dropped} near-empty stubs)")
    return result


def create_child_chunks(parent_chunks: List[Dict]) -> List[Dict]:
    """
    Produces sliding-window child chunks from each parent chunk.

    Each child window is prefixed with the standard ID, title, and category so
    that a dense search on child chunks still carries enough context to identify
    the parent standard without needing a separate lookup.
    """
    children = []
    for parent in parent_chunks:
        sid      = parent["standard_id"]
        title    = parent.get("title", "")
        text     = parent["text"]
        category = parent.get("category", "")

        raw_sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in raw_sentences if len(s.strip()) >= MIN_SENTENCE_LEN]
        if not sentences:
            continue

        context_prefix = f"Standard: {sid}"
        if title:
            context_prefix += f" -- {title}"
        if category:
            context_prefix += f" [Category: {category}]"
        context_prefix += ". "

        i = 0
        while i < len(sentences):
            window      = sentences[i: i + CHILD_WINDOW_SIZE]
            window_text = " ".join(window)
            children.append({
                "standard_id": sid,
                "title":       title,
                "text":        context_prefix + window_text,
                "raw_window":  window_text,
                "page_number": parent.get("page_number", None),
                "category":    parent.get("category", ""),
                "source":      parent.get("source", ""),
                "chunk_type":  "child",
                "parent_text": text,
            })
            i += CHILD_STEP_SIZE
            if i >= len(sentences):
                break

    return children


# -- Load from pre-generated chunks JSON ---------------------------------------

def load_from_chunks_json(chunks_path: str) -> Dict[str, List[Dict]]:
    """
    Loads chunks from a pre-generated JSON file instead of parsing the PDF.

    This is the recommended path because the JSON has already gone through
    extraction and cleaning, so it is faster and does not require any PDF
    processing dependencies.

    Field mapping applied when reading from the JSON:
        content     -> text         (renamed -- this is the field that gets embedded)
        standard_id -> standard_id  (re-normalized to canonical form)
        category    -> category     (preserved as metadata)
        page_number -> page_number  (None if absent)
        chunk_type  -> "parent"     (all loaded chunks are treated as parents)
    """
    print(f"[Parser] Loading pre-generated chunks from: {chunks_path}")
    with open(chunks_path, encoding="utf-8") as f:
        data = json.load(f)

    raw_list = data if isinstance(data, list) else data.get("chunks", [])
    print(f"[Parser] Found {len(raw_list)} raw chunks in file.")

    parent_chunks: List[Dict[str, Any]] = []
    for item in raw_list:
        sid = normalize_standard_id(
            item.get("standard_id", item.get("standard_number", ""))
        )

        title = item.get("title", "")
        if is_toc_title(title):
            title = ""
        title = clean_toc_title_suffix(title)

        text = item.get("content", item.get("text", "")).strip()
        if not text or len(text) < MIN_CONTENT_LEN:
            continue

        parent_chunks.append({
            "standard_id": sid,
            "title":       title,
            "text":        text,
            "page_number": item.get("page_number", None),
            "category":    item.get("category", ""),
            "source":      item.get("source", "SP 21 : 2005"),
            "chunk_type":  "parent",
        })

    unique_parents = _dedup_keep_longest(parent_chunks)
    child_chunks   = create_child_chunks(unique_parents)

    print(f"[Parser] Final: {len(unique_parents)} parent chunks, "
          f"{len(child_chunks)} child chunks.")

    return {
        "parent_chunks": unique_parents,
        "child_chunks":  child_chunks,
    }


# -- PDF path ------------------------------------------------------------------

def extract_text_pymupdf(pdf_path: str) -> List[Dict]:
    doc   = fitz.open(pdf_path)
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
    # Walk through every line in the document. When we spot a line that looks
    # like an IS standard heading, flush whatever we have been accumulating for
    # the previous standard and start a new chunk.
    chunks = []
    current_std_id: Optional[str] = None
    current_title: Optional[str]  = None
    current_body_lines: List[str] = []
    current_start_page = 1

    def flush_current():
        nonlocal current_std_id, current_title, current_body_lines
        if current_std_id and current_body_lines:
            full_text = current_std_id
            if current_title:
                full_text += f" -- {current_title}"
            full_text += "\n" + " ".join(current_body_lines)
            clean_title = (
                current_title
                if current_title and not is_toc_title(current_title)
                else ""
            )
            chunks.append({
                "standard_id": normalize_standard_id(current_std_id),
                "title":       clean_title,
                "text":        full_text.strip(),
                "page_number": current_start_page,
                "category":    "",
                "source":      "SP 21 : 2005",
                "chunk_type":  "parent",
            })

    for line_dict in lines:
        text  = line_dict["text"]
        page  = line_dict["page"]
        match = IS_HEADING_PATTERN.search(text)
        if match and is_real_heading(text, match):
            flush_current()
            current_std_id     = match.group(1)
            current_start_page = page
            raw_after          = text[match.end():].strip(" --:-")
            after              = clean_toc_title_suffix(raw_after)
            current_title      = after if len(after) > 3 else None
            current_body_lines = []
        else:
            if current_std_id is None:
                continue
            # If we have not captured a title yet and this line looks short
            # enough to be one, treat it as the title rather than body text.
            if current_title is None and len(text) < 120 and not text.endswith("."):
                current_title = text
            else:
                current_body_lines.append(text)

    flush_current()
    return chunks


def parse_pdf(pdf_path: str) -> Dict[str, List[Dict]]:
    """
    Parses the raw SP 21 PDF into parent and child chunks.

    Tries PyMuPDF first for better layout fidelity, then falls back to
    pdfplumber if PyMuPDF is not installed or raises an error.
    """
    print(f"[Parser] Loading PDF: {pdf_path}")
    raw_chunks: List[Dict] = []

    if PYMUPDF_AVAILABLE:
        print("[Parser] Using PyMuPDF...")
        try:
            lines      = extract_text_pymupdf(pdf_path)
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


# -- I/O helpers ---------------------------------------------------------------

def save_chunks(chunks_data: Dict, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)
    print(f"[Parser] Saved chunks to {output_path}")


def load_chunks(path: str) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# -- CLI -----------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="BIS SP21 Parser")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--chunks", metavar="JSON_PATH",
                       help="Path to pre-generated chunks JSON (recommended)")
    group.add_argument("--pdf", metavar="PDF_PATH",
                       help="Path to raw SP21 PDF")
    ap.add_argument("--output", default="data/chunks.json",
                    help="Output path (default: data/chunks.json)")
    args = ap.parse_args()

    if args.chunks:
        result = load_from_chunks_json(args.chunks)
    else:
        result = parse_pdf(args.pdf)

    save_chunks(result, args.output)