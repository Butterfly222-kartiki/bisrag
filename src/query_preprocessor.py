"""
query_preprocessor.py
=====================
Handles everything that happens to a user query BEFORE it hits the retrieval
index:

  1. Relevance check  — is this query about BIS/IS standards at all?
  2. Query expansion  — enrich the query with synonyms, IS numbers, and
                        technical terms so retrieval casts a wider net.

Two code paths exist depending on whether a Groq API key is available:

  WITH key   → single Groq LLM call that does both jobs at once, saving
               one full network round-trip (~300–600 ms) vs the original
               two-call design.

  WITHOUT key → fast keyword-based fallback that catches obvious greetings
                and queries with zero domain vocabulary. No LLM required.

Public surface used by retriever.py:
    check_relevance_and_expand_query(query, groq_api_key) -> (bool, str, str)
    IrrelevantQueryError
"""

import os
import re
import json
from typing import Optional


# ---------------------------------------------------------------------------
# Custom exception — raised when a query is off-topic or a greeting.
# Caught in api.py to return a friendly response instead of a 500 error.
# ---------------------------------------------------------------------------

class IrrelevantQueryError(Exception):
    """
    Raised when the query is a casual greeting or completely unrelated to
    BIS (Bureau of Indian Standards) standards and product compliance.

    The `user_message` attribute holds the human-readable explanation that
    the API layer should return to the frontend.
    """
    def __init__(self, message: str):
        super().__init__(message)
        self.user_message = message


# ---------------------------------------------------------------------------
# Keyword fallback — used when Groq API key is absent or the LLM call fails.
# Catches only the most obvious off-topic cases to avoid false rejections.
# ---------------------------------------------------------------------------

# Common greeting words/phrases we want to redirect rather than search for.
_GREETINGS = {
    "hi", "hello", "hey", "helo", "hii", "hiii", "howdy", "yo", "sup",
    "good morning", "good afternoon", "good evening", "good night",
    "greetings", "salutations", "namaste", "namaskar",
}

# Vocabulary hints that strongly suggest the query is about BIS standards.
# A query with even one of these words is treated as relevant by the fallback.
_DOMAIN_HINTS = {
    "bis", "is", "standard", "cement", "steel", "pipe", "water", "concrete",
    "quality", "product", "material", "specification", "manufacture", "rubber",
    "textile", "food", "chemical", "paint", "wire", "cable", "brick", "tile",
    "glass", "plastic", "wood", "timber", "copper", "aluminium", "aluminum",
    "iron", "coal", "oil", "gas", "pressure", "safety", "weight", "dimension",
    "mse", "factory", "industrial", "bureau", "indian", "national", "grade",
    "class", "type", "requirement", "compressive", "tensile", "strength",
}

# Default message when the query has no recognisable domain vocabulary.
_NOT_RELEVANT_MSG = (
    "I'm specialized in finding **BIS Standards** for products and manufacturing. 🏭\n\n"
    "Your query doesn't seem related to Indian Standards compliance. "
    "Please describe your product or manufacturing requirement.\n\n"
    "Example: *We manufacture OPC cement — which BIS standard applies?*"
)


def _keyword_relevance_check(query: str) -> tuple[bool, str]:
    """
    Fast, zero-dependency relevance check used as a fallback.

    Returns (is_relevant, user_message).
    If is_relevant is True, user_message will be an empty string.
    Only catches two cases:
      - Pure greetings (exact match against _GREETINGS).
      - Queries that share no tokens with _DOMAIN_HINTS.
    """
    # Normalise punctuation so "hello!" and "hello" both match.
    q = query.strip().lower()
    q = re.sub(r"[!?.,\'\"]+$", "", q).strip()

    # Pure greeting — give a welcoming redirect message.
    if q in _GREETINGS:
        return False, (
            "👋 Hello! I'm the **BIS Standards Finder**.\n\n"
            "Describe your product or compliance requirement and I'll find "
            "the relevant Indian Standards for you.\n\n"
            "Example: *We manufacture OPC cement — which BIS standard applies?*"
        )

    # No recognisable domain vocabulary — probably off-topic.
    words = q.split()
    if not any(w in _DOMAIN_HINTS for w in words):
        return False, _NOT_RELEVANT_MSG

    # Looks domain-relevant — let it through.
    return True, ""


# ---------------------------------------------------------------------------
# Primary entry point — single Groq call combining relevance + expansion.
# Falls back to _keyword_relevance_check when Groq is unavailable.
# ---------------------------------------------------------------------------

def check_relevance_and_expand_query(
    query: str,
    groq_api_key: Optional[str] = None,
) -> tuple[bool, str, str]:
    """
    Gate-keeps and enriches a query in one step.

    Returns a 3-tuple:
        (is_relevant, expanded_query, user_message_if_irrelevant)

    - is_relevant        : True if the query is about BIS/IS standards.
    - expanded_query     : Enriched query string ready for vector + BM25 search.
                           Equals the original `query` when expansion is skipped
                           (no key, LLM failure, or irrelevant query).
    - user_message       : Non-empty only when is_relevant is False; contains a
                           friendly explanation for the end user.

    Design note:
        The original code made two sequential Groq calls — one for relevance,
        one for expansion.  Merging them into a single call saves ~300–600 ms
        per request.  The LLM is prompted to return a small JSON object with
        both answers in one shot.
    """
    api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")

    # No API key available — fall back to simple keyword heuristic.
    if not api_key:
        relevant, msg = _keyword_relevance_check(query)
        return relevant, query, msg

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        # System prompt tells the LLM exactly what JSON shape to return so we
        # can parse it reliably without any regex gymnastics.
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
            temperature=0.0,   # Deterministic — we don't want creative expansions.
        )

        text = response.choices[0].message.content.strip()

        # Strip accidental markdown code fences the model sometimes adds.
        text = re.sub(r'^```json\s*', '', text)
        text = re.sub(r'\s*```$', '', text)

        data = json.loads(text)

        if data.get("relevant", True):
            # Use the LLM-expanded query; fall back to original if empty.
            expanded = data.get("expanded_query", "") or query
            print(f"[QueryPreprocessor] relevant=True, expanded={expanded[:100]}...")
            return True, expanded, ""
        else:
            # Build a friendly user-facing message from the LLM's explanation.
            raw_msg = data.get("message", "")
            user_msg = (
                f"🏭 {raw_msg}\n\n"
                "Try something like: *We manufacture steel pipes — which BIS standard applies?*"
            ) if raw_msg else _NOT_RELEVANT_MSG
            print("[QueryPreprocessor] relevant=False")
            return False, "", user_msg

    except Exception as e:
        # Any Groq error (network, quota, parse) → safe keyword fallback.
        print(f"[QueryPreprocessor] Groq call failed ({e}). Using keyword fallback.")
        relevant, msg = _keyword_relevance_check(query)
        return relevant, query, msg
