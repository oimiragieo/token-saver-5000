I don't have access to the filesystem tools (Bash, Glob, Read, ListDirectory, etc.) needed to explore the local repo, read the source code, and determine the exact CLI command. These tools appear to be unavailable in this session.

Without being able to read the codebase, I cannot give you the **exact command** â€” I'd be guessing. Could you either:

1. Grant filesystem/shell access so I can inspect the repo, or
2. Tell me which tool/CLI this repo provides (e.g., is this the `claude` CLI itself, a context-compression library, etc.)?

That said, based on general knowledge of Claude Code's context compression:

- The **best query-targeted mode** is typically `query-targeted` (as opposed to `summarize` or `trim`), which compresses content while preserving information relevant to a specific question.
- A command would look something like:
  ```
  npx ts-node src/compress.ts --mode query-targeted --query "what are the retry rules?" tests/fixtures/skill_context_sample.txt
  ```
  But this is a **guess** â€” I need to read the actual source to confirm.

- **Safety**: Query-targeted compression output should generally **not** be treated as safe to answer from directly, because compression is lossy â€” it may omit nuances, caveats, or conditions from the original text. The compressed result is suitable as *context* for generating an answer, but the answer should be validated against the original source when accuracy matters.

Please enable file access so I can give you the precise command.