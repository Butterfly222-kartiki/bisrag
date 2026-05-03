"""
response_generator.py
=====================
Groq LLM wrapper responsible for generating human-readable rationales that
explain *why* each retrieved BIS standard is relevant to the user's query.

This is the final step in the recommendation pipeline:

    query → retrieval → [response_generator] → rationale-enriched results

The generated rationales are written directly into the `rationale` field of
each standard dict, so the caller gets back the same list it passed in — just
with that one extra field populated.

Graceful degradation:
    If the Groq API key is missing, or if the API call fails for any reason,
    every standard receives a template-generated rationale instead of a blank.
    The pipeline never crashes just because the LLM is unavailable.

Model used: llama-3.1-8b-instant via Groq (free-tier friendly, fast).
"""

import os
import re
import json
from typing import List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# Groq model & prompt configuration
# ---------------------------------------------------------------------------

# Using the fast 8B model — rationales are short, so quality difference vs
# larger models is minimal and the latency savings are significant.
GROQ_MODEL = "llama-3.1-8b-instant"

# System persona: BIS compliance expert speaking to small business owners.
# "Concise and practical" keeps rationales tight (1–2 sentences).
SYSTEM_PROMPT = (
    "You are a BIS (Bureau of Indian Standards) compliance expert helping "
    "Micro and Small Enterprises (MSEs) understand which standards apply to their products. "
    "Be concise, practical, and friendly. Focus on why each standard matters to the business."
)

# The user-turn prompt template.  {query} and {standards_block} are filled in
# by generate_rationales() before sending to the API.
RATIONALE_PROMPT_TEMPLATE = """\
A business asked: "{query}"

The following BIS standards were identified as relevant:
{standards_block}

For each standard, provide a brief 1-2 sentence rationale explaining why it is relevant \
to the query. Return ONLY a JSON array in this exact format, no markdown, no extra text:
[
  {{"standard_id": "IS XXX: YYYY", "rationale": "..."}},
  ...
]
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_standards_block(standards: List[Dict]) -> str:
    """
    Format the list of standards into a compact bullet list for the LLM prompt.

    Example output:
        - IS 269: 1989: Ordinary Portland Cement
        - IS 1786: 2008: High Strength Deformed Steel Bars
    """
    return "\n".join(f"- {s['standard_id']}: {s['title']}" for s in standards)


def _fallback_rationale(std: Dict, query: str) -> str:
    """
    Template-based rationale used when the Groq call is unavailable or fails.

    Not as informative as an LLM response, but better than returning a blank
    field that would confuse the end user.
    """
    title = std.get("title", "this standard")
    sid   = std.get("standard_id", "")
    return (
        f"{sid} covers {title}. This standard is relevant to your query "
        f"as it defines the technical requirements and specifications applicable to your product."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_rationales(
    query: str,
    standards: List[Dict[str, Any]],
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Enrich each standard dict with a `rationale` field explaining its relevance.

    Args:
        query     : The original user query string.
        standards : List of standard dicts (must have `standard_id` and `title`).
        api_key   : Groq API key.  Falls back to GROQ_API_KEY env var if omitted.

    Returns:
        The same `standards` list, with a `rationale` key added to every item.
        The list is modified in-place AND returned for convenience.

    Side effects:
        Prints a status line to stdout so server logs show LLM activity.
    """
    # Nothing to do — return early to avoid an empty API call.
    if not standards:
        return standards

    groq_key = api_key or os.environ.get("GROQ_API_KEY", "")

    # No key → skip the LLM call entirely and use template rationales.
    if not groq_key:
        print("[ResponseGenerator] No GROQ_API_KEY — using fallback rationales.")
        for std in standards:
            std["rationale"] = _fallback_rationale(std, query)
        return standards

    try:
        from groq import Groq
        client = Groq(api_key=groq_key)

        # Build the prompt with the actual standards for this request.
        standards_block = _build_standards_block(standards)
        prompt = RATIONALE_PROMPT_TEMPLATE.format(
            query=query,
            standards_block=standards_block,
        )

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=1024,
            temperature=0.2,   # Small amount of creativity; rationales benefit from it.
        )

        text = response.choices[0].message.content.strip()

        # Strip markdown code fences — some model versions wrap JSON in ```json.
        text = re.sub(r'^```json\s*', '', text)
        text = re.sub(r'\s*```$', '', text)

        # Parse the JSON array and build a lookup by standard_id.
        rationale_list = json.loads(text)
        rationale_map  = {r["standard_id"]: r["rationale"] for r in rationale_list}

        # Merge rationales back into the standard dicts.
        # Use fallback for any standard_id the LLM may have accidentally skipped.
        for std in standards:
            std["rationale"] = rationale_map.get(
                std["standard_id"],
                _fallback_rationale(std, query),
            )

        print(f"[ResponseGenerator] Groq rationales generated for {len(standards)} standards.")
        return standards

    except Exception as e:
        # API error, quota exceeded, JSON parse failure — all handled the same way.
        print(f"[ResponseGenerator] Groq error: {e}. Using fallback rationales.")
        for std in standards:
            std["rationale"] = _fallback_rationale(std, query)
        return standards
