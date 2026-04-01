"""CLI provider wrappers for Claude Code and Gemini CLI.

Calls the native CLIs directly (not via omega wrappers) to preserve
full JSON output including usage/token data.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .pricing import compute_cost
from .results import CLIResult


def _find_cli(name: str) -> str | None:
    """Find a CLI executable on PATH."""
    return shutil.which(name)


def is_available(provider: str) -> bool:
    """Check if a CLI provider is installed."""
    if provider == "claude":
        return _find_cli("claude") is not None
    elif provider == "gemini":
        return _find_cli("gemini") is not None
    elif provider == "codex":
        return _find_cli("codex") is not None
    return False


def run_prompt(
    provider: str,
    prompt: str,
    model: str | None = None,
    cwd: str | Path | None = None,
    dry_run: bool = False,
    timeout: int = 600,
) -> CLIResult:
    """Run a prompt through a CLI provider and return structured results.

    Calls the native CLIs directly to get full JSON output with usage data.
    The omega wrappers strip usage fields, so we bypass them for benchmarking.
    """
    cmd = _build_command(provider, model)

    if dry_run:
        print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        if cwd:
            print(f"[DRY RUN] Working directory: {cwd}")
        print(f"[DRY RUN] Prompt length: {len(prompt)} chars (via stdin)")
        return CLIResult(provider=provider, model=model or "", is_dry_run=True)

    result = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        cwd=str(cwd) if cwd else None,
    )

    if result.returncode != 0 and not result.stdout.strip():
        raise RuntimeError(
            f"{provider} CLI failed (exit {result.returncode}): {result.stderr[:500]}"
        )

    if provider == "codex":
        return _parse_codex_result(result.stdout)

    raw_json = _parse_json_output(result.stdout)
    if provider == "claude":
        return _parse_claude_result(raw_json, result.stdout)
    else:
        return _parse_gemini_result(raw_json, result.stdout, model)


def _build_command(provider: str, model: str | None) -> list[str]:
    """Build native CLI command. Prompt is always sent via stdin.

    Both CLIs read from stdin when no positional prompt is given,
    which avoids Windows command-line length limits (~8KB).
    """
    if provider == "claude":
        cli = _find_cli("claude")
        if cli is None:
            raise RuntimeError("claude CLI not found on PATH")
        # -p: print mode, --output-format json: full structured JSON with usage
        # --no-session-persistence: prevents cache reuse between benchmark runs
        cmd = [
            cli,
            "-p",
            "--output-format",
            "json",
            "--dangerously-skip-permissions",
            "--no-session-persistence",
        ]
        if model:
            cmd.extend(["--model", model])
        return cmd
    elif provider == "gemini":
        cli = _find_cli("gemini")
        if cli is None:
            raise RuntimeError("gemini CLI not found on PATH")
        cmd = [cli, "--output-format", "json", "--yolo"]
        if model:
            cmd.extend(["-m", model])
        return cmd
    elif provider == "codex":
        cli = _find_cli("codex")
        if cli is None:
            raise RuntimeError("codex CLI not found on PATH")
        # exec --json: outputs JSONL events including turn.completed with usage
        # --dangerously-bypass-approvals-and-sandbox: non-interactive headless mode
        # Prompt is sent via stdin
        cmd = [cli, "exec", "--json", "--dangerously-bypass-approvals-and-sandbox"]
        if model:
            cmd.extend(["--model", model])
        return cmd
    raise ValueError(f"Unknown provider: {provider}")


def _parse_json_output(stdout: str) -> dict:
    """Extract JSON from CLI stdout, handling potential non-JSON prefix/suffix."""
    stdout = stdout.strip()
    # Try direct parse first
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        pass
    # Try to find JSON object in output
    start = stdout.find("{")
    end = stdout.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(stdout[start:end])
        except json.JSONDecodeError:
            pass
    return {}


def _parse_claude_result(data: dict, raw: str) -> CLIResult:
    """Parse Claude Code JSON output into CLIResult.

    Claude puts detailed per-model token counts in modelUsage (camelCase)
    and aggregate (often zero) in usage (snake_case). We prefer modelUsage
    for accurate counts.
    """
    # Try modelUsage first (has actual per-model counts)
    model_usage = data.get("modelUsage", {})
    model_name = ""
    input_tokens = 0
    output_tokens = 0
    cache_read = 0
    cache_creation = 0

    if model_usage:
        # Get the first (usually only) model entry
        model_name = next(iter(model_usage), "")
        mu = model_usage.get(model_name, {})
        input_tokens = mu.get("inputTokens", 0)
        output_tokens = mu.get("outputTokens", 0)
        cache_read = mu.get("cacheReadInputTokens", 0)
        cache_creation = mu.get("cacheCreationInputTokens", 0)

    # Fall back to usage block if modelUsage was empty
    if input_tokens == 0 and output_tokens == 0:
        usage = data.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cache_read = usage.get("cache_read_input_tokens", 0)

    # Total input includes cache creation (first-request cost)
    total_input = input_tokens + cache_creation + cache_read

    return CLIResult(
        provider="claude",
        model=model_name or data.get("model", ""),
        input_tokens=total_input,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cache_creation_tokens=cache_creation,
        total_cost_usd=data.get("total_cost_usd", 0.0),
        wall_time_ms=data.get("duration_ms", 0.0),
        tool_calls=0,
        num_turns=data.get("num_turns", 0),
        raw_response=data.get("result", ""),
        raw_json=data,
    )


def _parse_gemini_result(data: dict, raw: str, model: str | None) -> CLIResult:
    """Parse Gemini CLI JSON output into CLIResult.

    Gemini's stats are nested per-model: stats.models.{name}.tokens.{field}.
    We aggregate across all models.

    IMPORTANT: Gemini's ``input`` field is NET of cache (billed tokens only),
    while ``prompt`` is the TOTAL content tokens sent. For compression savings
    comparison we use ``prompt`` (stable measure), and for cost we use ``input``
    (billed). Cache read tokens are tracked separately.
    """
    stats = data.get("stats", {})

    # Aggregate from per-model breakdown (current Gemini CLI format)
    prompt_tokens = 0  # total content tokens (stable for savings comparison)
    billed_input = 0  # net of cache (for cost calculation)
    output_tokens = 0
    cached = 0
    wall_time_ms = 0.0
    detected_model = model or ""
    models = stats.get("models", {})
    if models:
        for model_name, model_stats in models.items():
            if not detected_model:
                detected_model = model_name
            tokens = model_stats.get("tokens", {})
            prompt_tokens += tokens.get("prompt", tokens.get("input", 0))
            billed_input += tokens.get("input", 0)
            output_tokens += tokens.get("candidates", 0)
            cached += tokens.get("cached", 0)
            api = model_stats.get("api", {})
            wall_time_ms += api.get("totalLatencyMs", 0)
    else:
        # Fallback: flat stats format (stream-json or older versions)
        prompt_tokens = stats.get("input_tokens", 0)
        billed_input = prompt_tokens
        output_tokens = stats.get("output_tokens", 0)
        cached = stats.get("cached", 0)
        wall_time_ms = stats.get("duration_ms", 0.0)

    cost = compute_cost(detected_model or "gemini-2.5-flash", billed_input, output_tokens, cached)

    # Tool call count from stats.tools if present
    tools_stats = stats.get("tools", {})
    tool_calls = tools_stats.get("totalCalls", 0) if isinstance(tools_stats, dict) else 0

    return CLIResult(
        provider="gemini",
        model=detected_model,
        input_tokens=prompt_tokens,  # total content tokens (cache-independent)
        output_tokens=output_tokens,
        cache_read_tokens=cached,
        total_cost_usd=cost,
        wall_time_ms=wall_time_ms,
        tool_calls=tool_calls,
        num_turns=0,
        raw_response=data.get("response", ""),
        raw_json=data,
    )


def _parse_codex_result(stdout: str) -> CLIResult:
    """Parse Codex CLI JSONL output into CLIResult.

    Codex exec --json streams one JSON object per line (not a single JSON blob).
    We scan all lines to find:
      - The ``turn.completed`` event, which contains ``usage`` with token counts.
      - The last ``item.completed`` event with an ``agent_message`` for response text.
    """
    events: list[dict] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    # Extract usage from turn.completed
    input_tokens = 0
    output_tokens = 0
    cache_read_tokens = 0
    usage_event: dict = {}
    for event in events:
        if event.get("type") == "turn.completed":
            usage_event = event.get("usage", {})
            input_tokens = usage_event.get("input_tokens", 0)
            output_tokens = usage_event.get("output_tokens", 0)
            cache_read_tokens = usage_event.get("cached_input_tokens", 0)
            break

    # Extract response text from the last item.completed agent_message
    raw_response = ""
    for event in reversed(events):
        if event.get("type") == "item.completed":
            details = event.get("details", {})
            if details.get("type") == "agent_message":
                content_list = details.get("content", [])
                texts = [
                    c.get("text", "")
                    for c in content_list
                    if isinstance(c, dict) and c.get("type") == "output_text"
                ]
                raw_response = "\n".join(texts)
                break

    # Net input (total minus cache reads, which are already included in input_tokens
    # by Codex; use input_tokens directly as the billed amount)
    cost = compute_cost("gpt-5.1-codex", input_tokens, output_tokens, cache_read_tokens)

    return CLIResult(
        provider="codex",
        model="gpt-5.1-codex",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        total_cost_usd=cost,
        raw_response=raw_response,
        raw_json={"events": events, "usage": usage_event},
    )
