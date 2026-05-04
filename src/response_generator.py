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

# System persona: senior BIS compliance consultant, not a generic assistant.
SYSTEM_PROMPT = (
    "You are a senior BIS (Bureau of Indian Standards) compliance consultant with 15+ years of "
    "experience helping Micro and Small Enterprises (MSEs) navigate Indian standards. "
    "Your explanations are clear, specific, and business-focused — you always explain WHAT the "
    "standard requires and WHY it directly applies to the product or activity described in the query. "
    "Never write vague filler like 'this standard applies to your product'. Instead, mention the "
    "specific technical aspect (material, test method, safety requirement, marking, etc.) that makes "
    "the standard relevant."
)

# Improved prompt: asks for structured, specific rationales with concrete details.
RATIONALE_PROMPT_TEMPLATE = """\
A business owner asked: "{query}"

These BIS standards were retrieved as potentially relevant:
{standards_block}

For EACH standard above, write a rationale (5-7 sentences) that:
1. Names the specific product category, process, or material the standard governs.
2. Explains exactly which clause or aspect makes it relevant to the query (e.g., testing method, \
material grade, safety marking, packaging requirement).
3. States the concrete business impact — certification needed, compliance deadline, or quality benefit.

Return ONLY a valid JSON array — no markdown fences, no preamble, no trailing text:
[
  {{"standard_id": "IS XXX: YYYY", "rationale": "...2-3 sentences..."}},
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
    More informative than a blank field; mentions the standard's own title.
    """
    title = std.get("title", "this standard")
    sid   = std.get("standard_id", "")
    text_snippet = std.get("text", "")[:120].strip()
    detail = f" It covers: {text_snippet}..." if text_snippet else ""
    return (
        f"{sid} specifies requirements for {title}."
        f" This standard is directly relevant because it defines the technical "
        f"specifications, testing methods, and quality benchmarks applicable to your product or process."
        f"{detail}"
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
            max_tokens=2048,   # 2–3 sentences per standard; 2048 is safe for up to 10 standards
            temperature=0.15,  # Slightly creative to avoid boilerplate, but still deterministic enough
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