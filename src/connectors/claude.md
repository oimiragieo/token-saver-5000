# Folder guide: `src/connectors/`

Breadcrumb for AI navigation. Master index: [`CLAUDE.md`](../../CLAUDE.md).

## Contents

### Python modules

#### `base.py`

Base types for managed connector feeds.

| Kind | Name |
|------|------|
| `def` | `sanitize_segment` |
| `class` | `ConnectorDocument` |
| `class` | `BaseConnector` |

#### `github_connector.py`

GitHub export connector normalization.

| Kind | Name |
|------|------|
| `class` | `GitHubConnector` |

#### `s3_connector.py`

S3 export connector normalization.

| Kind | Name |
|------|------|
| `class` | `S3Connector` |

#### `slack_export_connector.py`

Slack export connector normalization.

| Kind | Name |
|------|------|
| `class` | `SlackExportConnector` |

#### `web_connector.py`

Web URL and sitemap style connector normalization.

| Kind | Name |
|------|------|
| `class` | `WebConnector` |

---

Symbols are **top-level only** (nested methods and inner functions are not listed). Regenerate: `python scripts/generate_claude_folder_guides.py`.
