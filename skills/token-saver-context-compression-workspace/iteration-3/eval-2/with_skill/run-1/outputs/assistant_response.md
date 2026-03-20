I'm unable to read files in this session â€” the standard Read/Bash/Glob tools aren't available and there's no Python LSP server configured. I can't inspect the skill scripts or the fixture file to give you the **exact** command.

Here's what I need to answer precisely:

1. The `skills/token-saver-context-compression/` directory contains 12 scripts, but I can't list them or read the SKILL.md to find the workflow entry point and the fidelity-check / "fail loudly" mechanism.

2. Based on the project's architecture (semantic compression + fidelity scoring via `src/fidelity_advisor.py` and `src/semantic_fidelity.py`), the "loud failure" almost certainly means the **semantic fidelity score** of the compressed output falls below a safety threshold â€” meaning the compression lost too much meaning to reliably answer questions from the compressed version.

**To get the exact answer**, I need file-reading capability. Could you either:
- Grant Read/Bash tool permissions, or
- Paste the contents of `skills/token-saver-context-compression/SKILL.md` so I can identify the exact command and failure semantics?