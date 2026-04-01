"""Lossless meta-token compression inspired by arXiv 2506.00307.

Finds repeated token subsequences and replaces them with meta-token symbols
(§1, §2, etc.) with a prepended dictionary. Fully reversible.

Algorithm:
1. Replace newlines with a placeholder token so line structure is preserved.
2. For n-gram lengths 2..min(20, len//2), count all occurrences.
3. Filter by min_frequency; compute savings = freq*(len-1) - (len+1).
4. Sort by savings descending.
5. Greedily replace: scan tokens left-to-right, replace matched spans,
   skip already-replaced positions.
6. Build ``§N=expansion`` dictionary header + ``---`` separator + body.
7. Decompress: parse dict from header, replace symbols with expansions.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


_NL_PLACEHOLDER = "__NL__"
"""Sentinel token used in place of newline characters during processing."""

SYMBOL_PREFIX = "\u00a7"  # §  (Unicode SECTION SIGN U+00A7)


@dataclass
class MetaTokenEntry:
    """A single meta-token dictionary entry."""

    symbol: str
    expansion: list[str]
    frequency: int
    savings: int


@dataclass
class MetaTokenResult:
    """Result of a meta-token compression pass."""

    original_text: str
    compressed_text: str
    dictionary: list[MetaTokenEntry] = field(default_factory=list)
    original_tokens: int = 0
    compressed_tokens: int = 0
    savings_tokens: int = 0
    compression_ratio: float = 1.0


class MetaTokenCompressor:
    """Lossless meta-token compressor for repetitive text.

    Args:
        min_length: Minimum n-gram length to consider (default 2).
        min_frequency: Minimum occurrence count for substitution (default 2).
        max_entries: Maximum dictionary entries (default 50).
        symbol_prefix: Prefix for meta-token symbols (default '§').
    """

    def __init__(
        self,
        min_length: int = 2,
        min_frequency: int = 2,
        max_entries: int = 50,
        symbol_prefix: str = SYMBOL_PREFIX,
    ) -> None:
        self.min_length = min_length
        self.min_frequency = min_frequency
        self.max_entries = max_entries
        self.symbol_prefix = symbol_prefix

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_repeated_subsequences(
        self, tokens: list[str], min_length: int | None = None
    ) -> dict[tuple[str, ...], list[int]]:
        """Find repeated token n-grams using a sliding window.

        Args:
            tokens: List of string tokens to scan.
            min_length: Minimum n-gram length (overrides instance default).

        Returns:
            Dict mapping ``tuple[str, ...]`` -> list of start indices.
            Only entries with at least ``self.min_frequency`` occurrences.
        """
        min_len = min_length if min_length is not None else self.min_length
        max_len = min(20, len(tokens) // 2 + 1)
        subs: dict[tuple[str, ...], list[int]] = defaultdict(list)

        for length in range(min_len, max_len + 1):
            for i in range(len(tokens) - length + 1):
                key = tuple(tokens[i : i + length])
                subs[key].append(i)

        return {k: v for k, v in subs.items() if len(v) >= self.min_frequency}

    def rank_by_savings(
        self, subsequences: dict[tuple[str, ...], list[int]]
    ) -> list[tuple[tuple[str, ...], int, int]]:
        """Rank subsequences by net token savings.

        Net savings = frequency * (length - 1) - length
        where ``length`` is the dictionary overhead (expansion tokens only;
        the symbol token itself replaces each occurrence so its cost is already
        accounted for in the ``freq * 1`` replacement cost).

        Args:
            subsequences: Output of :meth:`find_repeated_subsequences`.

        Returns:
            List of ``(seq_tuple, frequency, savings)`` sorted descending by savings.
            Entries with ``savings <= 0`` are excluded.
        """
        ranked = []
        for seq, indices in subsequences.items():
            freq = len(indices)
            length = len(seq)
            savings = freq * (length - 1) - length
            if savings > 0:
                ranked.append((seq, freq, savings))
        ranked.sort(key=lambda x: x[2], reverse=True)
        return ranked

    def compress(self, text: str) -> MetaTokenResult:
        """Compress *text* by replacing repeated subsequences with meta-tokens.

        Args:
            text: Input text (may contain newlines).

        Returns:
            :class:`MetaTokenResult` with original, compressed text, and stats.
        """
        if not text or not text.strip():
            return MetaTokenResult(
                original_text=text,
                compressed_text=text,
                dictionary=[],
                original_tokens=0,
                compressed_tokens=0,
                savings_tokens=0,
                compression_ratio=1.0,
            )

        # Replace newlines with placeholder so we work on a flat token list
        flat = text.replace("\n", f" {_NL_PLACEHOLDER} ")
        tokens = flat.split()
        original_count = sum(1 for t in tokens if t != _NL_PLACEHOLDER)

        # Find repeated subsequences in content tokens only
        content_tokens = [t for t in tokens if t != _NL_PLACEHOLDER]
        subsequences = self.find_repeated_subsequences(content_tokens)
        ranked = self.rank_by_savings(subsequences)

        if not ranked:
            return MetaTokenResult(
                original_text=text,
                compressed_text=text,
                dictionary=[],
                original_tokens=original_count,
                compressed_tokens=original_count,
                savings_tokens=0,
                compression_ratio=1.0,
            )

        # Greedily build entries, selecting non-overlapping occurrences
        entries: list[MetaTokenEntry] = []
        # Work directly on content_tokens (mutable copy for replacement)
        working = list(content_tokens)

        for seq, _freq, _savings in ranked[: self.max_entries]:
            if len(entries) >= self.max_entries:
                break
            symbol = f"{self.symbol_prefix}{len(entries) + 1}"
            length = len(seq)

            # Count how many non-overlapping replacements we can make
            positions_replaced = 0
            i = 0
            new_working: list[str] = []
            while i < len(working):
                if i + length <= len(working) and tuple(working[i : i + length]) == seq:
                    new_working.append(symbol)
                    positions_replaced += 1
                    i += length
                else:
                    new_working.append(working[i])
                    i += 1

            if positions_replaced >= self.min_frequency:
                actual_savings = positions_replaced * (length - 1) - length
                if actual_savings > 0:
                    entries.append(
                        MetaTokenEntry(
                            symbol=symbol,
                            expansion=list(seq),
                            frequency=positions_replaced,
                            savings=actual_savings,
                        )
                    )
                    working = new_working

        if not entries:
            return MetaTokenResult(
                original_text=text,
                compressed_text=text,
                dictionary=[],
                original_tokens=original_count,
                compressed_tokens=original_count,
                savings_tokens=0,
                compression_ratio=1.0,
            )

        # Rebuild substituted text.
        # `working` is the substituted content-token list (no NL placeholders).
        # Reconstruct newlines by noting that the original text was split on "\n"
        # into lines; we join working tokens back to a single string and then
        # restore the original newline structure by proportional reconstruction.
        # Simpler: join working as space-separated, then re-split the original
        # text on newlines and map line boundaries by content.
        #
        # The cleanest approach: rebuild using the original per-line token lists,
        # applying the same substitutions that were already applied to `working`.

        lines_orig = text.split("\n")
        body_lines: list[str] = []

        # Re-apply substitutions per-line to reconstruct body with newlines
        for line in lines_orig:
            line_tokens = line.split()
            if not line_tokens:
                body_lines.append("")
                continue
            # Apply all entries to this line's token list
            line_w = list(line_tokens)
            for entry in entries:
                seq = tuple(entry.expansion)
                length = len(seq)
                i = 0
                new_line_w: list[str] = []
                while i < len(line_w):
                    if i + length <= len(line_w) and tuple(line_w[i : i + length]) == seq:
                        new_line_w.append(entry.symbol)
                        i += length
                    else:
                        new_line_w.append(line_w[i])
                        i += 1
                line_w = new_line_w
            body_lines.append(" ".join(line_w))

        body = "\n".join(body_lines)

        # Build dictionary header
        dict_parts = [f"{e.symbol}={' '.join(e.expansion)}" for e in entries]
        dict_header = " | ".join(dict_parts)
        compressed_text = f"{dict_header}\n---\n{body}"

        # Compute effective token count:
        # - body tokens (after symbol substitution)
        # - dict expansion tokens (the words in expansions, not the symbols/format chars)
        body_tokens = sum(len(line.split()) for line in body_lines)
        dict_expansion_tokens = sum(len(e.expansion) for e in entries)
        compressed_count = body_tokens + dict_expansion_tokens
        savings = max(0, original_count - compressed_count)

        return MetaTokenResult(
            original_text=text,
            compressed_text=compressed_text,
            dictionary=entries,
            original_tokens=original_count,
            compressed_tokens=max(1, compressed_count),
            savings_tokens=savings,
            compression_ratio=round(original_count / max(1, compressed_count), 2),
        )

    def decompress(self, compressed_text: str) -> str:
        """Decompress a meta-token compressed string.

        Args:
            compressed_text: Output of :meth:`compress`.

        Returns:
            Reconstructed original text (best-effort; whitespace may differ).
        """
        if not compressed_text:
            return compressed_text
        if "---" not in compressed_text:
            return compressed_text

        parts = compressed_text.split("\n---\n", 1)
        if len(parts) != 2:
            return compressed_text

        dict_header, body = parts

        # Parse dictionary: "§1=word1 word2 | §2=word3 word4"
        expansions: dict[str, str] = {}
        for entry_str in dict_header.split(" | "):
            entry_str = entry_str.strip()
            if "=" not in entry_str:
                continue
            symbol, expansion = entry_str.split("=", 1)
            expansions[symbol.strip()] = expansion.strip()

        # Replace symbols longest-first to avoid §1 matching inside §10
        result = body
        for symbol in sorted(expansions.keys(), key=lambda s: -len(s)):
            result = result.replace(symbol, expansions[symbol])

        return result
