"""The MCP handler chain must import without torch.

WHAT THIS COST. The runtime image dropped torch (it is a build-time dependency:
the ONNX exports need optimum, serving does not). Boot passed, embeddings
passed, a real compression passed -- and the MCP gateway returned **500 on
every request**, caught only by the post-deploy smoke test, after staging had
already taken the image:

    src/handlers/mcp_core.py       -> from . import compression_handlers
    src/handlers/compression_handlers.py -> from ..types import HandlerContext
    src/types.py                   -> from src.adaptive_rate_allocator import ...
    src/adaptive_rate_allocator.py -> import torch      <-- module scope
    ModuleNotFoundError: No module named 'torch'

`types.py` imports only `ContextWindowAdapter` and `MultiLevelSemanticEncoder`,
for TypedDict annotations. Neither has a single torch reference. The module they
live in also defines `AdaptiveRateAllocator` (an `nn.Module`), and one
module-scope `import torch` on its behalf made torch a hard requirement of the
ENTIRE MCP surface -- for a class no production code imports.

WHY THE BUILD GATE MISSED IT. The in-image gate ran with
`MCP_GATEWAY_ENABLED=0`. It disabled the one subsystem that broke. A gate that
switches off a subsystem cannot testify about it, and "the app booted" reads as
covering far more than it does.

WHY A SUBPROCESS. In-process this can only pass: pytest has already imported
torch through some other module, so `sys.modules` is poisoned before the first
assertion. The claim is about a fresh interpreter in an image where torch is
ABSENT, which is simulated here by blocking the import.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Block torch at the meta-path, then import the chain. This reproduces the
# torch-free image inside a normal dev environment -- uninstalling torch to run
# the suite is not a thing anyone will do, so the guard has to create the
# condition itself.
_BLOCK_TORCH = """
import sys

class _BlockTorch:
    def find_module(self, name, path=None):
        return self if name == "torch" or name.startswith("torch.") else None
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError(f"No module named {name!r} (blocked by the test)")
        return None

sys.meta_path.insert(0, _BlockTorch())
assert "torch" not in sys.modules, "torch was already imported - the block is too late"
"""


def _run(body: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _BLOCK_TORCH + body],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=600,
    )


def test_the_block_itself_works() -> None:
    """NON-VACUITY: if the block does not block, every test below is theatre."""
    proc = _run("\nimport torch\nprint('IMPORTED')\n")
    assert "IMPORTED" not in proc.stdout, (
        "torch imported despite the block - the remaining tests in this file "
        "prove nothing about a torch-free image"
    )


def test_adaptive_rate_allocator_imports_without_torch() -> None:
    """The two torch-free classes must be importable with torch absent."""
    proc = _run(
        "\nfrom src.adaptive_rate_allocator import ContextWindowAdapter, "
        "MultiLevelSemanticEncoder\nprint('OK', bool(ContextWindowAdapter), "
        "bool(MultiLevelSemanticEncoder))\n"
    )
    assert (
        "OK True True" in proc.stdout
    ), f"import failed without torch.\nstderr tail: {proc.stderr.strip()[-600:]}"


def test_the_mcp_handler_chain_imports_without_torch() -> None:
    """THE HEADLINE -- this is the exact chain that 500'd on staging."""
    proc = _run(
        "\nfrom src.handlers.mcp_core import setup_mcp_tools\n"
        "tools = setup_mcp_tools(profile='full')\n"
        "print('TOOLS', len(tools))\n"
    )
    line = next((x for x in proc.stdout.splitlines() if x.startswith("TOOLS")), None)
    assert line, (
        "the MCP handler chain could not import without torch -- this is the "
        f"defect that returned 500 on every gateway request.\nstderr tail: "
        f"{proc.stderr.strip()[-800:]}"
    )
    count = int(line.split()[1])
    assert count > 15, f"tool catalogue collapsed to {count} tools without torch"


def test_the_server_context_builds_without_torch() -> None:
    """IMPORTING the chain is not enough -- the SERVER CONTEXT must build too.

    The first fix deferred torch to construction time and the catalogue test
    went green, so it looked done. But `server_factory_service.build`
    constructs `ContextWindowAdapter` on every `SemanticModulatorServer()`, and
    that constructor built the torch-backed allocator eagerly. The gateway
    caught the ImportError and logged:

        MCP gateway: could not initialize server context - tools will receive
        an empty context and may fail at runtime

    A green catalogue over a broken context is the worst available outcome: the
    tools LIST fine and fail when called. A deferral that only moves a failure
    from import to construction has deferred nothing for a caller that always
    constructs.
    """
    proc = _run(
        "\nfrom src.adaptive_rate_allocator import ContextWindowAdapter\n"
        "a = ContextWindowAdapter(None)\n"
        "print('BUILT', type(a).__name__)\n"
    )
    assert "BUILT ContextWindowAdapter" in proc.stdout, (
        "ContextWindowAdapter cannot be constructed without torch -- "
        "SemanticModulatorServer() will fail and the MCP gateway will serve an "
        f"empty context.\nstderr tail: {proc.stderr.strip()[-600:]}"
    )


def test_the_allocator_DEGRADES_without_torch_it_does_not_raise() -> None:
    """SUPERSEDES an earlier assertion that this raised ImportError.

    Raising was the wrong contract and production proved it: `adapt_to_context_window`
    is a LIVE MCP TOOL, and in the torch-free image it returned "Internal error in
    tool" against production - caught by the 156-tool sweep AFTER every deploy job
    had gone green. Degrading the engine's guts is one thing; 500ing a tool a
    customer can call is another.

    The replacement contract: a torch-free image gets a deterministic numpy
    allocator over the SAME [0.10, 0.30] levels. Worth stating why that is not a
    downgrade - the torch path runs an UNTRAINED network plus Gumbel sampling, so
    it was a near-random pick in that band, different on every call for identical
    inputs.
    """
    proc = _run(
        "\nimport networkx as nx\n"
        "from src.adaptive_rate_allocator import _build_adaptive_rate_allocator_cls\n"
        "alloc = _build_adaptive_rate_allocator_cls()()\n"
        "g = nx.Graph(); g.add_edges_from([(1,2),(2,3),(3,1),(3,4)])\n"
        "kw = dict(graph=g, available_context_tokens=50000, "
        "max_context_tokens=100000, query_priority=0.5)\n"
        "r1, d = alloc(**kw)\n"
        "r2, _ = alloc(**kw)\n"
        "print('ALLOC', r1, 0.10 <= r1 <= 0.30, r1 == r2, d['allocator'])\n"
    )
    line = next((x for x in proc.stdout.splitlines() if x.startswith("ALLOC")), None)
    assert line, (
        "the allocator raised instead of degrading - a torch-free image would "
        f"500 adapt_to_context_window.\nstderr tail: {proc.stderr.strip()[-600:]}"
    )

    _, ratio, in_band, deterministic, which = line.split()
    assert in_band == "True", f"ratio {ratio} is outside the documented [0.10, 0.30] band"
    assert deterministic == "True", "identical inputs gave different ratios"
    assert which == "numpy_deterministic", (
        f"diagnostics must name the path that produced the number, got {which!r} - "
        f"a caller cannot otherwise tell a deterministic ratio from a sampled one"
    )


def test_the_symbol_still_resolves_without_torch_and_says_which_path() -> None:
    """The module attribute must resolve, and the caller must be able to tell.

    Degrading silently is the failure mode worth guarding here: if the torch-free
    allocator were indistinguishable from the torch one, nobody could explain why
    a ratio stopped varying between identical calls. The class NAME carries it.
    """
    proc = _run(
        "\nimport src.adaptive_rate_allocator as m\n"
        "cls = m.AdaptiveRateAllocator\n"
        "print('CLASS', cls.__name__)\n"
    )
    line = next((x for x in proc.stdout.splitlines() if x.startswith("CLASS")), None)
    assert line, (
        "the AdaptiveRateAllocator attribute did not resolve without torch.\n"
        f"stderr tail: {proc.stderr.strip()[-500:]}"
    )
    name = line.split()[1]
    assert name == "_NumpyRateAllocator", (
        f"expected the torch-free stand-in, got {name!r} - either torch leaked "
        f"into this subprocess or the fallback silently returned the torch class"
    )
