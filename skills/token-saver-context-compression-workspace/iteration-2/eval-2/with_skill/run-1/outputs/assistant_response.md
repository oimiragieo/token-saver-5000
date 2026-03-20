I don't have file-reading tools (Read, Bash, Glob, Grep) available in this session â€” only Hugging Face and LSP tools. I need those local tools to inspect the skill scripts and give you the exact command.

However, based on the CLAUDE.md context, I can point you in the right direction:

The skill lives at `skills/token-saver-context-compression/` which contains **12 scripts**. To find the exact full workflow command and the "fail loudly" safety gate, you'd need to check:

1. **`skills/token-saver-context-compression/SKILL.md`** â€” the skill definition with usage instructions
2. The 12 scripts in that directory â€” one will be a pipeline/workflow runner, and one will be a fidelity/safety check

**What I can tell you from CLAUDE.md context:**

The project has a **`recommend_fidelity`** MCP tool and a **`fidelity_advisor`** module (`src/fidelity_advisor.py`, 90% coverage) plus a **`compression_advisor`** (`src/compression_advisor.py`, 91% coverage). The "loud failure" almost certainly comes from the fidelity check â€” it means the **compressed output lost too much semantic information to safely answer questions from**. The compression was too aggressive (fidelity too low), so the system refuses to serve the result rather than risk hallucination or missing critical context.

**To get the exact command**, I need you to grant me access to the `Read` or `Bash` tools so I can inspect the skill scripts. Could you re-run with those tools enabled, or check `skills/token-saver-context-compression/SKILL.md` yourself for the workflow entry point?