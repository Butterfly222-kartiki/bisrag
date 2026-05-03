"""
Groq LLM wrapper for rationale generation.
Replaces Gemini — uses llama-3.1-8b-instant via Groq API (free tier).
Provides brief, MSE-friendly explanations for why each standard is relevant.
"""

import os
import re
import json
from typing import List, Dict, Any, Optional


GROQ_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = (
    "You are a BIS (Bureau of Indian Standards) compliance expert helping "
    "Micro and Small Enterprises (MSEs) understand which standards apply to their products. "
    "Be concise, practical, and friendly. Focus on why each standard matters to the business."
)

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


def _build_standards_block(standards: List[Dict]) -> str:
    return "\n".join(f"- {s['standard_id']}: {s['title']}" for s in standards)


def _fallback_rationale(std: Dict, query: str) -> str:
    title = std.get("title", "this standard")
    sid = std.get("standard_id", "")
    return (
        f"{sid} covers {title}. This standard is relevant to your query "
        f"as it defines the technical requirements and specifications applicable to your product."
    )


def generate_rationales(
    query: str,
    standards: List[Dict[str, Any]],
    api_key: str = None        # Groq API key
) -> List[Dict[str, Any]]:
    """
    Call Groq (llama-3.1-8b-instant) to generate rationales for retrieved standards.
    Returns the standards list with 'rationale' field added.
    Falls back to a template rationale if Groq is unavailable or key is missing.
    """
    if not standards:
        return standards

    groq_key = api_key or os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        print("[LLM] No GROQ_API_KEY — using fallback rationales.")
        for std in standards:
            std["rationale"] = _fallback_rationale(std, query)
        return standards

    try:
        from groq import Groq
        client = Groq(api_key=groq_key)

        standards_block = _build_standards_block(standards)
        prompt = RATIONALE_PROMPT_TEMPLATE.format(
            query=query,
            standards_block=standards_block
        )

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}
            ],
            max_tokens=512,   # was 1024 — rationales are 1-2 sentences each, 512 is plenty
            temperature=0.0   # was 0.2 — deterministic = slightly faster token sampling
        )

        text = response.choices[0].message.content.strip()
        # Strip markdown code fences if model adds them
        text = re.sub(r'^```json\s*', '', text)
        text = re.sub(r'\s*```$', '', text)

        rationale_list = json.loads(text)
        rationale_map = {r["standard_id"]: r["rationale"] for r in rationale_list}

        for std in standards:
            std["rationale"] = rationale_map.get(
                std["standard_id"],
                _fallback_rationale(std, query)
            )
        print(f"[LLM] Groq rationales generated for {len(standards)} standards.")
        return standards

    except Exception as e:
        print(f"[LLM] Groq error: {e}. Using fallback rationales.")
        for std in standards:
            std["rationale"] = _fallback_rationale(std, query)
        return standards