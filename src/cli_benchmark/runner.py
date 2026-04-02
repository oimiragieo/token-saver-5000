"""Main benchmark orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .compressor import compress_text
from .corpus import CORPUS_DIR, CorpusEntry, build_prompt, load_all_corpus, load_corpus
from .project_scaffold import cleanup, create_vanilla, create_with_mcp
from .providers import is_available, run_prompt
from .results import BenchmarkReport, CLIResult, ComparisonResult

if TYPE_CHECKING:
    from ..search_compress_pipeline import SearchCompressResult


def run_benchmark(
    mode: str = "skill",
    sizes: list[str] | None = None,
    providers: list[str] | None = None,
    corpus_dir: Path = CORPUS_DIR,
    dry_run: bool = False,
    repeats: int = 1,
    model_claude: str | None = None,
    model_gemini: str | None = None,
    model_codex: str | None = None,
    model_opencode: str | None = None,
    verbose: bool = False,
) -> BenchmarkReport:
    """Run the full benchmark suite.

    Args:
        mode: "skill", "mcp", or "both"
        sizes: List of corpus sizes to test (default: all)
        providers: List of providers to test (default: all available)
        corpus_dir: Path to corpus directory
        dry_run: If True, skip actual CLI calls
        repeats: Number of times to repeat each configuration
        model_claude: Override Claude model
        model_gemini: Override Gemini model
        model_codex: Override Codex model
        model_opencode: Override OpenCode model
        verbose: Print progress
    """
    report = BenchmarkReport(
        metadata={
            "mode": mode,
            "dry_run": dry_run,
            "repeats": repeats,
            "model_claude": model_claude,
            "model_gemini": model_gemini,
            "model_codex": model_codex,
            "model_opencode": model_opencode,
        }
    )

    if sizes is None or sizes == ["all"]:
        entries = load_all_corpus(corpus_dir)
    else:
        entries = [load_corpus(s, corpus_dir) for s in sizes]

    if providers is None or providers == ["all"]:
        providers = ["claude", "gemini", "codex", "opencode"]

    available = {p for p in providers if is_available(p) or dry_run}
    skipped = set(providers) - available
    if skipped and verbose:
        print(f"Skipping unavailable providers: {', '.join(skipped)}")

    for entry in entries:
        for provider in sorted(available):
            if provider == "claude":
                model = model_claude
            elif provider == "gemini":
                model = model_gemini
            elif provider == "opencode":
                model = model_opencode
            else:
                model = model_codex

            if mode in ("skill", "both"):
                for _ in range(repeats):
                    try:
                        result = _run_skill_comparison(
                            entry, provider, model, corpus_dir, dry_run, verbose
                        )
                        report.add(result)
                    except Exception as e:
                        if verbose:
                            print(f"    ERROR: {e}")
                        # Record a failed comparison with zeroed compressed result
                        report.add(
                            ComparisonResult(
                                corpus_name=entry.name,
                                provider=provider,
                                mode="skill",
                                baseline=CLIResult(provider=provider),
                                compressed=CLIResult(provider=provider),
                            )
                        )

            if mode in ("mcp", "both"):
                for _ in range(repeats):
                    try:
                        result = _run_mcp_comparison(
                            entry, provider, model, corpus_dir, dry_run, verbose
                        )
                        report.add(result)
                    except Exception as e:
                        if verbose:
                            print(f"    ERROR: {e}")
                        report.add(
                            ComparisonResult(
                                corpus_name=entry.name,
                                provider=provider,
                                mode="mcp",
                                baseline=CLIResult(provider=provider),
                                compressed=CLIResult(provider=provider),
                            )
                        )

    return report


def run_search_compress_benchmark(
    corpus_dir: Path,
    queries: list[str] | None = None,
    verbose: bool = False,
) -> list[SearchCompressResult]:
    """Run search-then-compress benchmark on code corpus.

    Searches the code corpus for each query, compresses only the matched files,
    and compares against naive (all-files) and compress-all baselines.

    Args:
        corpus_dir: Path to the benchmarks directory (parent of corpus/).
        queries: List of search queries; uses 3 built-in defaults if None.
        verbose: Print per-query progress to stdout.

    Returns:
        List of SearchCompressResult, one per query.
    """
    from ..search_compress_pipeline import search_then_compress

    code_dir = corpus_dir / "corpus" / "code"
    if not code_dir.exists():
        if verbose:
            print(f"Code corpus not found at {code_dir}")
        return []

    if queries is None:
        queries = [
            "authentication token JWT",
            "database connection pool",
            "rate limiting middleware API",
        ]

    results = []
    for query in queries:
        if verbose:
            print(f"  Query: {query!r}")
        result = search_then_compress(str(code_dir), query)
        results.append(result)
        if verbose:
            print(f"    Files: {result.files_scanned} scanned, {result.files_matched} matched")
            print(f"    Naive: {result.naive_all_tokens:,} tokens")
            print(f"    Compress-all: {result.naive_compressed_tokens:,} tokens")
            print(f"    Search+compress: {result.search_compress_tokens:,} tokens")
            print(f"    vs naive: {result.search_vs_naive_savings_pct}% savings")
            print(f"    vs compress-all: {result.search_vs_compress_all_savings_pct}% savings")

    return results


def _run_skill_comparison(
    entry: CorpusEntry,
    provider: str,
    model: str | None,
    corpus_dir: Path,
    dry_run: bool,
    verbose: bool,
) -> ComparisonResult:
    """Run a skill-mode comparison: raw vs pre-compressed context."""
    if verbose:
        print(f"  [{provider}] Skill mode: {entry.name} ({entry.line_count} lines)")

    # Compress the corpus (skipped in dry-run to avoid subprocess overhead)
    doc_original_tokens = 0
    doc_compressed_tokens = 0
    if dry_run:
        compressed_text_str = entry.content
        compression_ratio = 0.0
    else:
        if verbose:
            print(f"    Compressing {entry.name}...")
        compressed = compress_text(entry.content)
        compressed_text_str = compressed.compressed_text
        compression_ratio = compressed.compression_ratio
        doc_original_tokens = compressed.original_tokens
        doc_compressed_tokens = compressed.compressed_tokens
        if verbose:
            print(
                f"    Compression: {compressed.original_tokens} -> {compressed.compressed_tokens} tokens "
                f"({compression_ratio:.1f}x)"
            )

    # Build prompts
    raw_prompt = build_prompt(entry.content, corpus_dir)
    compressed_prompt = build_prompt(compressed_text_str, corpus_dir)

    # Run baseline (raw context)
    if verbose:
        print(f"    Running baseline ({len(raw_prompt)} chars)...")
    baseline = run_prompt(provider, raw_prompt, model=model, dry_run=dry_run)

    # Run compressed
    if verbose:
        print(f"    Running compressed ({len(compressed_prompt)} chars)...")
    compressed_result = run_prompt(provider, compressed_prompt, model=model, dry_run=dry_run)

    return ComparisonResult(
        corpus_name=entry.name,
        provider=provider,
        mode="skill",
        baseline=baseline,
        compressed=compressed_result,
        compression_ratio=compression_ratio,
        document_original_tokens=doc_original_tokens,
        document_compressed_tokens=doc_compressed_tokens,
    )


def _run_mcp_comparison(
    entry: CorpusEntry,
    provider: str,
    model: str | None,
    corpus_dir: Path,
    dry_run: bool,
    verbose: bool,
) -> ComparisonResult:
    """Run an MCP-mode comparison: vanilla project vs Token Saver project."""
    if verbose:
        print(f"  [{provider}] MCP mode: {entry.name} ({entry.line_count} lines)")

    prompt = build_prompt("Read corpus.txt and use its content.", corpus_dir)

    # Create scaffold directories
    vanilla_dir = create_vanilla(entry.file_path, provider)
    mcp_dir = create_with_mcp(entry.file_path, provider)

    try:
        if verbose:
            print(f"    Vanilla dir: {vanilla_dir}")
            print(f"    MCP dir: {mcp_dir}")

        # Run baseline in vanilla project
        if verbose:
            print("    Running baseline (vanilla project)...")
        baseline = run_prompt(provider, prompt, model=model, cwd=vanilla_dir, dry_run=dry_run)

        # Run with Token Saver MCP
        if verbose:
            print("    Running with Token Saver MCP...")
        compressed_result = run_prompt(provider, prompt, model=model, cwd=mcp_dir, dry_run=dry_run)

        return ComparisonResult(
            corpus_name=entry.name,
            provider=provider,
            mode="mcp",
            baseline=baseline,
            compressed=compressed_result,
        )
    finally:
        cleanup(vanilla_dir)
        cleanup(mcp_dir)
