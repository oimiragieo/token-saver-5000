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


def test_the_deferred_allocator_still_raises_when_actually_used() -> None:
    """The deferral must not turn a hard requirement into a silent no-op.

    `adapt_to_context_window` genuinely needs torch. Making construction lazy is
    correct only if USING it still fails loudly and names the reason.
    """
    proc = _run(
        "\nfrom src.adaptive_rate_allocator import ContextWindowAdapter\n"
        "a = ContextWindowAdapter(None)\n"
        "try:\n"
        "    a.rate_allocator\n"
        "    print('NO_RAISE')\n"
        "except ImportError as exc:\n"
        "    print('RAISED', 'torch' in str(exc))\n"
    )
    assert "RAISED True" in proc.stdout, (
        f"using the allocator without torch must raise an ImportError naming "
        f"torch; got {proc.stdout.strip()[-200:]}"
    )


def test_the_torch_class_still_raises_a_useful_error_without_torch() -> None:
    """Degrading is not the same as pretending.

    `AdaptiveRateAllocator` genuinely needs torch. Without it the failure must
    name the reason, not surface as an AttributeError on a module that quietly
    stopped exposing the symbol.
    """
    proc = _run(
        "\nimport src.adaptive_rate_allocator as m\n"
        "try:\n"
        "    m.AdaptiveRateAllocator\n"
        "    print('NO_RAISE')\n"
        "except ImportError as exc:\n"
        "    print('RAISED', 'torch' in str(exc))\n"
    )
    assert "RAISED True" in proc.stdout, (
        f"expected an ImportError naming torch; got: {proc.stdout.strip()[-300:]} "
        f"/ {proc.stderr.strip()[-300:]}"
    )
