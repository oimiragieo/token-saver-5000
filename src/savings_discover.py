"""Discover missed token savings opportunities.

Scans files or text blobs to estimate what COULD be compressed,
using the same heuristics as the compression pipeline. Reports
ranked opportunities with estimated savings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .token_estimation import TokenEstimator

# Files below this token count are not worth compressing
MIN_COMPRESSIBLE_TOKENS = 100

# Files at or above this token count benefit most from compression
RECOMMENDED_THRESHOLD = 500

# Estimated compression ratios by file type (conservative)
COMPRESSION_ESTIMATES: Dict[str, float] = {
    ".py": 0.85,
    ".js": 0.83,
    ".ts": 0.83,
    ".tsx": 0.82,
    ".jsx": 0.82,
    ".java": 0.84,
    ".go": 0.84,
    ".rs": 0.83,
    ".c": 0.82,
    ".cpp": 0.82,
    ".h": 0.80,
    ".md": 0.88,
    ".txt": 0.87,
    ".json": 0.90,
    ".yaml": 0.85,
    ".yml": 0.85,
    ".toml": 0.83,
    ".xml": 0.88,
    ".html": 0.86,
    ".css": 0.80,
    ".sql": 0.85,
    ".sh": 0.82,
    ".log": 0.92,
}

DEFAULT_COMPRESSION_RATIO = 0.85

# Skip these directories when scanning
SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    ".env",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    ".semantic_modulator_data",
    "htmlcov",
    ".eggs",
    "*.egg-info",
}

# Skip binary/large file extensions
SKIP_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".so",
    ".dll",
    ".exe",
    ".bin",
    ".dat",
    ".db",
    ".sqlite",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".svg",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".pdf",
    ".mp3",
    ".mp4",
    ".avi",
    ".mov",
    ".wav",
}


@dataclass
class SavingsOpportunity:
    """A single file or text blob that could benefit from compression."""

    path: str
    token_count: int
    estimated_compressed: int
    estimated_savings_pct: float
    recommendation: str  # "compress", "direct_read", "skip"

    @property
    def tokens_saved(self) -> int:
        return self.token_count - self.estimated_compressed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "token_count": self.token_count,
            "estimated_compressed": self.estimated_compressed,
            "tokens_saved": self.tokens_saved,
            "estimated_savings_pct": round(self.estimated_savings_pct, 1),
            "recommendation": self.recommendation,
        }


@dataclass
class DiscoveryReport:
    """Aggregated report of all savings opportunities."""

    opportunities: List[SavingsOpportunity] = field(default_factory=list)
    total_tokens_scanned: int = 0
    total_potential_savings: int = 0
    files_scanned: int = 0
    files_compressible: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "files_scanned": self.files_scanned,
                "files_compressible": self.files_compressible,
                "total_tokens_scanned": self.total_tokens_scanned,
                "total_potential_savings": self.total_potential_savings,
                "potential_savings_pct": (
                    round(self.total_potential_savings / self.total_tokens_scanned * 100, 1)
                    if self.total_tokens_scanned > 0
                    else 0.0
                ),
            },
            "opportunities": [o.to_dict() for o in self.opportunities],
        }


class ContextAnalyzer:
    """Analyzes files and text to discover compression opportunities."""

    def __init__(self, estimator: Optional[TokenEstimator] = None):
        self._estimator = estimator or TokenEstimator()

    def analyze_text(
        self,
        text: str,
        label: str = "<text>",
        file_ext: str = "",
    ) -> SavingsOpportunity:
        """Analyze a single text blob for compression potential."""
        token_count = self._estimator.count_tokens(text)
        ratio = COMPRESSION_ESTIMATES.get(file_ext, DEFAULT_COMPRESSION_RATIO)

        if token_count < MIN_COMPRESSIBLE_TOKENS:
            return SavingsOpportunity(
                path=label,
                token_count=token_count,
                estimated_compressed=token_count,
                estimated_savings_pct=0.0,
                recommendation="skip",
            )

        if token_count < RECOMMENDED_THRESHOLD:
            return SavingsOpportunity(
                path=label,
                token_count=token_count,
                estimated_compressed=token_count,
                estimated_savings_pct=0.0,
                recommendation="direct_read",
            )

        estimated_compressed = max(1, int(token_count * (1 - ratio)))
        savings_pct = (1 - estimated_compressed / token_count) * 100

        return SavingsOpportunity(
            path=label,
            token_count=token_count,
            estimated_compressed=estimated_compressed,
            estimated_savings_pct=savings_pct,
            recommendation="compress",
        )

    def analyze_file(self, file_path: str) -> Optional[SavingsOpportunity]:
        """Analyze a single file for compression potential."""
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            return None

        ext = path.suffix.lower()
        if ext in SKIP_EXTENSIONS:
            return None

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except (OSError, PermissionError):
            return None

        return self.analyze_text(text, label=str(path), file_ext=ext)

    def scan_directory(
        self,
        directory: str,
        max_files: int = 500,
    ) -> DiscoveryReport:
        """Scan a directory recursively for compression opportunities.

        Returns a DiscoveryReport with ranked opportunities (highest savings first).
        """
        report = DiscoveryReport()
        root = Path(directory)

        if not root.exists() or not root.is_dir():
            return report

        file_count = 0
        for path in root.rglob("*"):
            if file_count >= max_files:
                break

            # Skip directories in SKIP_DIRS
            if any(skip in path.parts for skip in SKIP_DIRS):
                continue

            if not path.is_file():
                continue

            ext = path.suffix.lower()
            if ext in SKIP_EXTENSIONS:
                continue

            file_count += 1
            opportunity = self.analyze_file(str(path))
            if opportunity is None:
                continue

            report.files_scanned += 1
            report.total_tokens_scanned += opportunity.token_count

            if opportunity.recommendation == "compress":
                report.opportunities.append(opportunity)
                report.files_compressible += 1
                report.total_potential_savings += opportunity.tokens_saved

        # Sort by tokens_saved descending (biggest wins first)
        report.opportunities.sort(key=lambda o: o.tokens_saved, reverse=True)
        return report

    def analyze_items(
        self,
        items: List[Dict[str, str]],
    ) -> DiscoveryReport:
        """Analyze a list of text items for compression potential.

        Each item should have 'text' and optionally 'label' and 'file_ext' keys.
        """
        report = DiscoveryReport()

        for item in items:
            text = item.get("text", "")
            label = item.get("label", "<text>")
            file_ext = item.get("file_ext", "")

            opportunity = self.analyze_text(text, label=label, file_ext=file_ext)
            report.files_scanned += 1
            report.total_tokens_scanned += opportunity.token_count

            if opportunity.recommendation == "compress":
                report.opportunities.append(opportunity)
                report.files_compressible += 1
                report.total_potential_savings += opportunity.tokens_saved

        report.opportunities.sort(key=lambda o: o.tokens_saved, reverse=True)
        return report


def format_report(report: DiscoveryReport, top_n: int = 20) -> str:
    """Format a DiscoveryReport as human-readable text."""
    lines = []
    lines.append("=== Token Savings Discovery Report ===")
    lines.append(f"Files scanned: {report.files_scanned}")
    lines.append(f"Files worth compressing: {report.files_compressible}")
    lines.append(f"Total tokens scanned: {report.total_tokens_scanned:,}")
    lines.append(f"Potential savings: {report.total_potential_savings:,} tokens")

    if report.total_tokens_scanned > 0:
        pct = report.total_potential_savings / report.total_tokens_scanned * 100
        lines.append(f"Potential reduction: {pct:.1f}%")

    if report.opportunities:
        lines.append("")
        lines.append(f"Top {min(top_n, len(report.opportunities))} opportunities:")
        for opp in report.opportunities[:top_n]:
            lines.append(
                f"  {opp.path}: ~{opp.token_count:,} tokens "
                f"→ ~{opp.estimated_compressed:,} compressed "
                f"({opp.estimated_savings_pct:.0f}% savings)"
            )

    return "\n".join(lines)
