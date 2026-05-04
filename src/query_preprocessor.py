"""
query_preprocessor.py
=====================
Handles everything that happens to a user query BEFORE it hits the retrieval
index — without any LLM or external API call.

Pipeline (in order):
  1. Greeting detection   — politely redirect casual greetings.
  2. Spelling correction  — exact-only lookup via the JSON spelling_variants.
  3. Relevance check      — intentionally very lenient; mirrors the old keyword
                            fallback to avoid false rejections.
  4. Dual-query output    — returns BOTH the original (spelling-corrected)
                            query AND an abbreviation-expanded variant.

═══════════════════════════════════════════════════════════════════════════
DUAL-QUERY STRATEGY  (why this exists)
═══════════════════════════════════════════════════════════════════════════
Problem:
  - Eval queries use full words → expanding abbreviations HURTS accuracy
    because extra tokens dilute embedding similarity (99% → 97%).
  - Real users WILL type abbreviations ("OPC", "TMT", "PVC") that may NOT
    appear in the indexed documents.

Solution:
  Return TWO query strings and let the retriever decide how to use them:

    result.queries[0]         → PRIMARY   = spelling-corrected original
                                 (drives the main retrieval — preserves 99%)
    result.expanded_query     → SECONDARY = abbreviations replaced with
                                 full forms (only used as a fallback)

  The retriever has several options (choose one):
    A) Run ONLY the primary query.  Ignore expanded.  (= old behavior, 99%)
    B) Run primary first.  If top-k scores are below a confidence threshold,
       re-run with expanded_query and merge results.
    C) Always run BOTH, retrieve top-k from each, deduplicate & re-rank.
       Primary results get a small score boost so they're preferred when
       both queries return the same document.

  Option A matches the old eval exactly.  Options B/C handle unseen
  abbreviation queries without hurting known performance.

═══════════════════════════════════════════════════════════════════════════

Public surface:
    preprocess(query, abbrev_data) -> PreprocessResult
    IrrelevantQueryError
    load_abbrev_data(json_path) -> dict

PreprocessResult fields:
    is_relevant      : bool
    queries          : list[str]   # [primary_query] — always length 1
    expanded_query   : str         # abbreviation-expanded variant (may == primary)
    has_expansions   : bool        # True when expanded_query differs from primary
    user_message     : str         # non-empty only when is_relevant=False
    was_decomposed   : bool        # always False (decomposition disabled)

Backward-compatibility shim:
    check_relevance_and_expand_query(query, groq_api_key=None) -> (bool, str, str)
    Returns the PRIMARY query as the expanded_query (not the abbreviation-
    expanded one) so existing callers see identical behavior to before.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class IrrelevantQueryError(Exception):
    """
    Raised when the query is a casual greeting or is completely unrelated to
    BIS (Bureau of Indian Standards) standards and product compliance.
    """
    def __init__(self, message: str):
        super().__init__(message)
        self.user_message = message


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PreprocessResult:
    is_relevant: bool
    queries: list[str] = field(default_factory=list)      # [primary_query]
    expanded_query: str = ""       # abbreviation-expanded variant
    has_expansions: bool = False   # True when expanded differs from primary
    user_message: str = ""
    was_decomposed: bool = False


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_DEFAULT_JSON_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "bis_abbrevations.json")
)


def load_abbrev_data(json_path: str = _DEFAULT_JSON_PATH) -> dict:
    """Load and return the BIS abbreviation/spelling JSON as a dict."""
    with open(json_path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_EXPANSION_SECTIONS = {
    "standards_bodies", "cement_types", "cement_admixtures_scm",
    "concrete_types", "aggregates", "lime_types", "steel_types",
    "concrete_products", "asbestos_cement_products", "building_limes",
    "stones", "bricks_masonry", "timber_wood", "gypsum", "bitumen_products",
    "waterproofing_damp_proofing", "floor_wall_roof", "sanitary_plumbing",
    "doors_windows", "hardware_fasteners", "paints_coatings",
    "thermal_insulation", "glass_products", "electrical_products",
    "polymer_materials",
}

_INVERTED_SECTIONS = {
    "common_abbreviations", "cement_extended",
    "measurement_shortcuts", "technical_terms",
}

_SPELLING_SECTIONS = {"spelling_variants"}

_GREETINGS: set[str] = {
    "hi", "hello", "hey", "helo", "hii", "hiii", "howdy", "yo", "sup",
    "good morning", "good afternoon", "good evening", "good night",
    "greetings", "salutations", "namaste", "namaskar", "hai",
    "kya haal", "kya hal", "salam", "assalam",
}

# Domain hints — same lenient set used by the old keyword fallback.
_DOMAIN_HINTS = {
    "bis", "is", "standard", "cement", "steel", "pipe", "water", "concrete",
    "quality", "product", "material", "specification", "manufacture", "rubber",
    "textile", "food", "chemical", "paint", "wire", "cable", "brick", "tile",
    "glass", "plastic", "wood", "timber", "copper", "aluminium", "aluminum",
    "iron", "coal", "oil", "gas", "pressure", "safety", "weight", "dimension",
    "mse", "factory", "industrial", "bureau", "indian", "national", "grade",
    "class", "type", "requirement", "compressive", "tensile", "strength",
}

_NOT_RELEVANT_MSG = (
    "I'm specialized in finding **BIS Standards** for products and manufacturing. 🏭\n\n"
    "Your query doesn't seem related to Indian Standards compliance. "
    "Please describe your product or manufacturing requirement.\n\n"
    "Example: *We manufacture OPC cement — which BIS standard applies?*"
)


def _build_lookup_tables(abbrev_data: dict) -> tuple[dict, dict]:
    """
    Returns (expansion_map, spelling_map).
    expansion_map : {lowercase_key → canonical_full_form}
    spelling_map  : {misspelling → correct_spelling}
    """
    expansion_map: dict[str, str] = {}
    spelling_map: dict[str, str] = {}

    for section_name, section_data in abbrev_data.items():
        if section_name.startswith("_"):
            continue

        if section_name in _EXPANSION_SECTIONS:
            if isinstance(section_data, dict):
                for k, v in section_data.items():
                    if isinstance(v, str):
                        expansion_map[k.lower()] = v

        elif section_name in _INVERTED_SECTIONS:
            if isinstance(section_data, dict):
                for canonical, abbrevs in section_data.items():
                    if isinstance(abbrevs, list):
                        for abbr in abbrevs:
                            if isinstance(abbr, str):
                                expansion_map[abbr.lower()] = canonical.replace("_", " ")

        elif section_name in _SPELLING_SECTIONS:
            if isinstance(section_data, dict):
                for correct, variants in section_data.items():
                    if isinstance(variants, list):
                        for bad in variants:
                            if isinstance(bad, str):
                                spelling_map[bad.lower()] = correct

    return expansion_map, spelling_map


def _normalize(text: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _fix_spellings_exact(tokens: list[str], spelling_map: dict) -> list[str]:
    """Exact-only spelling correction. No fuzzy matching."""
    return [spelling_map.get(tok, tok) for tok in tokens]


def _expand_abbreviations(text: str, expansion_map: dict) -> str:
    """
    REPLACE abbreviations with their full-forms to produce the expanded query.

    "OPC 43 standard" → "Ordinary Portland Cement 43 standard"

    This is ONLY used to build the secondary expanded_query field.
    The primary retrieval query is never modified by this function.

    Uses greedy longest-match (4-gram → 1-gram).
    Special guard: bare "is" only expanded when followed by a digit.
    """
    tokens = text.split()
    expanded: list[str] = []
    i = 0
    while i < len(tokens):
        matched = False
        for n in (4, 3, 2, 1):
            if i + n > len(tokens):
                continue
            candidate = " ".join(tokens[i:i + n]).lower()
            if candidate in expansion_map:
                # Guard: skip expanding bare "is" unless context is standard-like
                if candidate == "is" and n == 1:
                    next_tok = tokens[i + 1].lower() if i + 1 < len(tokens) else ""
                    if not (next_tok.isdigit() or (next_tok.isupper() and len(next_tok) <= 5)):
                        expanded.append(tokens[i])
                        i += 1
                        matched = True
                        break
                expanded.append(expansion_map[candidate])
                i += n
                matched = True
                break
        if not matched:
            expanded.append(tokens[i])
            i += 1
    return " ".join(expanded)


def _is_greeting(query: str) -> bool:
    """Return True if the query is purely a greeting."""
    q = re.sub(r"[!?.,\'\"]+$", "", query).strip().lower()
    return q in _GREETINGS


def _is_mixed_language_query(text: str) -> bool:
    """
    Detect Hinglish / mixed-language queries that are BIS-related.
    Requires at least one Hindi connector + one BIS-intent word.
    """
    hindi_connectors = {"ka", "ke", "ki", "kya", "batao", "kaunsa", "liye", "mein"}
    bis_intent = {"code", "standard", "bis", "number"}
    tokens = set(text.lower().split())
    return bool(tokens & hindi_connectors) and bool(tokens & bis_intent)


def _keyword_relevance_check(query: str) -> bool:
    """Same lenient check as the old preprocessor: one domain-hint word = relevant."""
    words = query.lower().split()
    return any(w in _DOMAIN_HINTS for w in words)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def preprocess(
    raw_query: str,
    abbrev_data: dict,
) -> PreprocessResult:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    raw_query   : The user's raw input string.
    abbrev_data : Loaded BIS abbreviation JSON (call load_abbrev_data once).

    Returns
    -------
    PreprocessResult with:
        queries[0]      — primary query (normalised only — ZERO word changes)
        expanded_query  — spelling-corrected + abbreviation-expanded (fallback)
        has_expansions  — True when expanded differs from primary
    """
    expansion_map, spelling_map = _build_lookup_tables(abbrev_data)

    # ── Step 1: Greeting check ───────────────────────────────────────────
    if _is_greeting(raw_query):
        return PreprocessResult(
            is_relevant=False,
            queries=[],
            user_message=(
                "👋 Hello! I'm the **BIS Standards Finder**.\n\n"
                "Describe your product or compliance requirement and I'll find "
                "the relevant Indian Standards for you.\n\n"
                "Example: *We manufacture OPC cement — which BIS standard applies?*"
            ),
        )

    # ── Step 2: Normalise ────────────────────────────────────────────────
    normalised = _normalize(raw_query)

    # ── Step 3: Build ALL query variants ─────────────────────────────────
    #
    #   primary_query  = normalised ONLY (lowercase + whitespace collapse)
    #                    → ZERO modifications to user's words
    #                    → this is what the retriever uses by default
    #                    → matches old preprocessor exactly → 99% Hit Rate
    #
    #   expanded_query = spelling-corrected + abbreviations replaced
    #                    → only used as fallback when primary retrieval is weak
    #                    → handles typos + unseen abbreviation queries
    #
    primary_query = normalised

    # Spelling correction + abbreviation expansion for the fallback query
    corrected_tokens = _fix_spellings_exact(normalised.split(), spelling_map)
    corrected = " ".join(corrected_tokens)
    expanded_query = _expand_abbreviations(corrected, expansion_map)
    has_expansions = (expanded_query != primary_query)

    # ── Step 4: Relevance check (very lenient) ───────────────────────────

    # Check 1: Mixed-language (Hinglish) queries
    if _is_mixed_language_query(normalised):
        print(f"[QueryPreprocessor] relevant=True (hinglish) | query={primary_query[:80]}")
        return PreprocessResult(
            is_relevant=True,
            queries=[primary_query],
            expanded_query=expanded_query,
            has_expansions=has_expansions,
        )

    # Check 2: IS/BIS number pattern (e.g. "IS 269", "is:269")
    if re.search(r"\bis[\s:–-]?\d{2,5}\b", normalised, re.IGNORECASE):
        print(f"[QueryPreprocessor] relevant=True (IS number) | query={primary_query[:80]}")
        return PreprocessResult(
            is_relevant=True,
            queries=[primary_query],
            expanded_query=expanded_query,
            has_expansions=has_expansions,
        )

    # Check 3: Keyword fallback — one domain-hint word = relevant
    if _keyword_relevance_check(normalised):
        print(f"[QueryPreprocessor] relevant=True (keyword) | query={primary_query[:80]}")
        return PreprocessResult(
            is_relevant=True,
            queries=[primary_query],
            expanded_query=expanded_query,
            has_expansions=has_expansions,
        )

    # Check 4: Check raw query too (belt-and-suspenders)
    if _keyword_relevance_check(raw_query):
        print(f"[QueryPreprocessor] relevant=True (raw keyword) | query={primary_query[:80]}")
        return PreprocessResult(
            is_relevant=True,
            queries=[primary_query],
            expanded_query=expanded_query,
            has_expansions=has_expansions,
        )

    # Check 5: If expansion found abbreviations, the query IS domain-relevant
    # even if no keyword matched (e.g. pure "OPC 43" — no _DOMAIN_HINTS word,
    # but OPC is a known abbreviation → clearly BIS-related)
    if has_expansions:
        print(f"[QueryPreprocessor] relevant=True (abbrev detected) | query={primary_query[:80]}")
        return PreprocessResult(
            is_relevant=True,
            queries=[primary_query],
            expanded_query=expanded_query,
            has_expansions=has_expansions,
        )

    # Truly irrelevant
    print(f"[QueryPreprocessor] relevant=False | query={raw_query!r}")
    return PreprocessResult(
        is_relevant=False,
        queries=[],
        user_message=_NOT_RELEVANT_MSG,
    )


# ---------------------------------------------------------------------------
# Backward-compatibility shim
# ---------------------------------------------------------------------------

def check_relevance_and_expand_query(
    query: str,
    groq_api_key: Optional[str] = None,
) -> tuple[bool, str, str]:
    """
    Drop-in replacement for the original Groq-based function.

    Returns (is_relevant, expanded_query, user_message_if_irrelevant).

    Behavior matches the OLD preprocessor exactly:
      - WITH Groq key  → single LLM call for relevance + expansion
      - WITHOUT Groq key → keyword fallback, query returned untouched

    For the new dual-query strategy (primary + expanded fallback), call
    preprocess() directly instead.
    """
    api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")

    # ── WITH Groq key: use LLM (identical to old code) ───────────────────
    if api_key:
        try:
            from groq import Groq
            client = Groq(api_key=api_key)

            system_prompt = (
                "You are an assistant for a BIS (Bureau of Indian Standards) search engine "
                "that helps Indian businesses find relevant IS standards for their products.\n\n"
                "Given a user query, do TWO things in ONE response:\n"
                "1. Decide if the query is relevant to BIS/IS standards, product compliance, "
                "manufacturing requirements, or material specifications.\n"
                "2. If relevant, expand the query with: relevant IS standard numbers, technical "
                "synonyms, abbreviations (OPC, PPC, PSC, HAC, AAC, TMT, etc.), and related "
                "material/process terms.\n\n"
                "Reply with ONLY valid JSON, no markdown, no extra text:\n"
                "{\n"
                '  "relevant": true,\n'
                '  "expanded_query": "<original query> <space-separated expansion keywords>",\n'
                '  "message": ""\n'
                "}\n"
                "OR if not relevant:\n"
                "{\n"
                '  "relevant": false,\n'
                '  "expanded_query": "",\n'
                '  "message": "<one short friendly sentence explaining you only handle BIS standards>"\n'
                "}"
            )

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": query},
                ],
                max_tokens=160,
                temperature=0.0,
            )

            text = response.choices[0].message.content.strip()
            text = re.sub(r'^```json\s*', '', text)
            text = re.sub(r'\s*```$', '', text)

            import json as _json
            data = _json.loads(text)

            if data.get("relevant", True):
                expanded = data.get("expanded_query", "") or query
                print(f"[QueryPreprocessor] relevant=True, expanded={expanded[:100]}...")
                return True, expanded, ""
            else:
                raw_msg = data.get("message", "")
                user_msg = (
                    f"🏭 {raw_msg}\n\n"
                    "Try something like: *We manufacture steel pipes — which BIS standard applies?*"
                ) if raw_msg else _NOT_RELEVANT_MSG
                print("[QueryPreprocessor] relevant=False")
                return False, "", user_msg

        except Exception as e:
            print(f"[QueryPreprocessor] Groq call failed ({e}). Using keyword fallback.")
            # Fall through to keyword check below

    # ── WITHOUT Groq key (or Groq failed): keyword fallback ──────────────
    q = query.strip().lower()
    q = re.sub(r"[!?.,\'\"]+$", "", q).strip()

    if q in _GREETINGS:
        return False, "", (
            "👋 Hello! I'm the **BIS Standards Finder**.\n\n"
            "Describe your product or compliance requirement and I'll find "
            "the relevant Indian Standards for you.\n\n"
            "Example: *We manufacture OPC cement — which BIS standard applies?*"
        )

    words = q.split()
    if not any(w in _DOMAIN_HINTS for w in words):
        return False, "", _NOT_RELEVANT_MSG

    # Relevant — return query COMPLETELY untouched
    return True, query, ""


def _keyword_relevance_check_with_greeting(query: str) -> tuple[bool, str]:
    """Fallback when the abbreviation JSON is missing."""
    q = re.sub(r"[!?.,\'\"]+$", "", query.strip().lower()).strip()
    if q in _GREETINGS:
        return False, (
            "👋 Hello! I'm the **BIS Standards Finder**.\n\n"
            "Describe your product or compliance requirement and I'll find "
            "the relevant Indian Standards for you.\n\n"
            "Example: *We manufacture OPC cement — which BIS standard applies?*"
        )
    words = q.split()
    if not any(w in _DOMAIN_HINTS for w in words):
        return False, _NOT_RELEVANT_MSG
    return True, ""


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    try:
        data = load_abbrev_data()
    except FileNotFoundError:
        print("ERROR: Could not find bis_abbrevations.json in ./data/", file=sys.stderr)
        sys.exit(1)

    test_cases = [
        "hello",
        "Namaste",
        "OPC 43 standard",
        "concreate standard",
        "opc cement ka is code kya hai",
        "standard for TMT bars and PVC pipe",
        "fly ash brick and AAC block",
        "IS 269",
        "what is the weather today",
        "ppc cement code?",
        "cement ka code",
        "requirements for manufacturing clay roof tiles",
        # Abbreviation-only queries (unseen edge cases)
        "OPC",
        "TMT",
        "PVC pipe",
        "AAC block",
    ]

    print("=" * 70)
    for q in test_cases:
        r = preprocess(q, data)
        print(f"INPUT : {q!r}")
        print(f"  relevant         : {r.is_relevant}")
        print(f"  primary query    : {r.queries}")
        print(f"  expanded query   : {r.expanded_query!r}")
        print(f"  has_expansions   : {r.has_expansions}")
        if r.user_message:
            print(f"  user_message     : {r.user_message[:80]}...")
        print("-" * 70)