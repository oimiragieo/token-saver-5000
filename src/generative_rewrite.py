"""
Generative rewrite prompt template generator.

Based on SCOPE (ACL 2025) — provides structured rewrite prompts that
clients can send to their LLM for generative compression. TokenSaver
doesn't make LLM calls itself; instead it prepares optimal prompts.
"""

from typing import Dict, List, Optional


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token)."""
    return max(0, len(text) // 4)


def generate_rewrite_prompt(
    original_text: str,
    target_ratio: float = 0.5,
    preserve_keywords: Optional[List[str]] = None,
) -> dict:
    """Generate a structured rewrite prompt for client-side LLM compression.

    Args:
        original_text: The text to be rewritten/compressed
        target_ratio: Target compression ratio (0.0-1.0)
        preserve_keywords: Keywords that must be preserved in rewrite

    Returns:
        Dict with system_instruction, user_prompt, token counts
    """
    original_tokens = _estimate_tokens(original_text)
    target_tokens = int(original_tokens * target_ratio)

    # Build system instruction
    keyword_instruction = ""
    if preserve_keywords:
        kw_list = ", ".join(preserve_keywords)
        keyword_instruction = f"\n- You MUST preserve these keywords exactly: {kw_list}"

    system_instruction = (
        "You are a precision text compressor. Rewrite the following text to be more concise "
        "while preserving all key information, relationships, and technical accuracy.\n"
        "Rules:\n"
        f"- Target approximately {target_tokens} tokens (currently {original_tokens} tokens)\n"
        "- Preserve all factual claims and technical details\n"
        "- Maintain logical structure and relationships\n"
        "- Remove redundancy, filler words, and unnecessary elaboration\n"
        "- Keep domain-specific terminology intact"
        f"{keyword_instruction}"
    )

    user_prompt = (
        f"Compress the following text to approximately {target_ratio:.0%} of its original length:\n\n"
        f"{original_text}"
    )

    return {
        "system_instruction": system_instruction,
        "user_prompt": user_prompt,
        "original_token_count": original_tokens,
        "target_token_count": target_tokens,
        "target_ratio": target_ratio,
    }
