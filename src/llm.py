"""
Gemini 2.5 Flash LLM wrapper for rationale generation.
Provides brief, MSE-friendly explanations for why each standard is relevant.
"""

import os
from typing import List, Dict, Any

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


GEMINI_MODEL = "gemini-2.5-flash"  # Best free-tier speed/quality ratio

SYSTEM_PROMPT = """You are a BIS (Bureau of Indian Standards) compliance expert helping 
Micro and Small Enterprises (MSEs) understand which standards apply to their products.
Be concise, practical, and friendly. Focus on why each standard matters to the business."""

RATIONALE_PROMPT_TEMPLATE = """
A business asked: "{query}"

The following BIS standards were identified as relevant:
{standards_block}

For each standard, provide a brief 1-2 sentence rationale explaining why it is relevant 
to the query. Return ONLY a JSON array in this exact format:
[
  {{"standard_id": "IS XXX: YYYY", "rationale": "..."}},
  ...
]
"""


def _build_standards_block(standards: List[Dict]) -> str:
    lines = []
    for s in standards:
        lines.append(f"- {s['standard_id']}: {s['title']}")
    return "\n".join(lines)


def generate_rationales(
    query: str,
    standards: List[Dict[str, Any]],
    api_key: str = None
) -> List[Dict[str, Any]]:
    """
    Call Gemini to generate rationales for retrieved standards.
    Returns the standards list with 'rationale' field added.
    Falls back to a template rationale if Gemini is unavailable.
    """
    if not standards:
        return standards

    # Try Gemini
    if GEMINI_AVAILABLE:
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if key:
            try:
                genai.configure(api_key=key)
                model = genai.GenerativeModel(GEMINI_MODEL)

                standards_block = _build_standards_block(standards)
                prompt = RATIONALE_PROMPT_TEMPLATE.format(
                    query=query,
                    standards_block=standards_block
                )

                response = model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.2,
                        max_output_tokens=1024,
                    )
                )

                # Parse JSON response
                import json, re
                text = response.text.strip()
                # Strip markdown code fences if present
                text = re.sub(r'^```json\s*', '', text)
                text = re.sub(r'\s*```$', '', text)

                rationale_list = json.loads(text)
                rationale_map = {r["standard_id"]: r["rationale"] for r in rationale_list}

                # Merge rationales into standards
                for std in standards:
                    std["rationale"] = rationale_map.get(
                        std["standard_id"],
                        _fallback_rationale(std, query)
                    )
                return standards

            except Exception as e:
                print(f"[LLM] Gemini error: {e}. Using fallback rationales.")

    # Fallback: template-based rationale
    for std in standards:
        std["rationale"] = _fallback_rationale(std, query)
    return standards


def _fallback_rationale(std: Dict, query: str) -> str:
    title = std.get("title", "this standard")
    sid = std.get("standard_id", "")
    return (
        f"{sid} covers {title}. This standard is relevant to your query "
        f"as it defines the technical requirements and specifications applicable to your product."
    )
