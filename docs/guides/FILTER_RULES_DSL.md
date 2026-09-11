# Filter Rules DSL (`.gotcontext.toml`)

User-defined CLI output filtering applied after built-in command strategies.

| Field | Value |
|-------|-------|
| **Module** | `src/filter_rules.py` |
| **Version** | 0.11.0 |

## Config locations

1. **Project** — `.gotcontext.toml` in the working directory
2. **User global** — `~/.config/gotcontext/filters.toml`

Rules are loaded by `FilterRuleEngine` and matched against a **command hint** string (derived from the invoked command or auto-detected output type).

## Pipeline stages (order fixed)

Applied in sequence inside `FilterRule.apply()`:

| Stage | Field | Effect |
|-------|-------|--------|
| 1 | `strip_ansi` | Remove ANSI escape codes |
| 2 | `strip_lines_matching` | Drop lines matching regex patterns |
| 3 | `keep_lines_matching` | Keep only lines matching patterns |
| 4 | `truncate_lines_at` | Cap each line length |
| 5 | `head_lines` | Keep first N lines |
| 6 | `tail_lines` | Keep last N lines |
| 7 | `max_lines` | Cap total lines (head + tail with gap) |
| 8 | `on_empty` | Replacement text when output becomes empty |

Invalid regex in a stage skips that stage (fail-soft).

## Example rule

```toml
[[filters]]
name = "pytest-quiet"
description = "Drop passing dots from pytest output"
match_command = "pytest"

strip_ansi = true
strip_lines_matching = ["^\\.+$", "^=+ FAILURES =+$"]
max_lines = 200
on_empty = "(no output after filter)"
```

## Integration

`src/cli_output_optimizer.py` runs 11 built-in strategies (git, test, lint, docker, …) then applies matching user rules as a **post-filter** stage.

MCP tool: `filter_cli_output` (`token_optimization_handlers`).

## Safety

- Regex compilation uses safe patterns; compilation errors skip the stage
- Rules never execute shell commands — text transformation only

## Related

- [PROXY.md](../reference/PROXY.md) — compresses MCP tool results (different layer)
- [MCP Tools Guide](./MCP_TOOLS_GUIDE.md) — `filter_cli_output` parameters
