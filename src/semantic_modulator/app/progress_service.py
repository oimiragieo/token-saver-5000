"""App-layer progress bar rendering service."""

from __future__ import annotations


class ProgressRenderService:
    """Renders compact terminal-friendly progress bars for server diagnostics."""

    @staticmethod
    def create_progress_bar(percentage: float, width: int = 40) -> str:
        filled = int((percentage / 100) * width)
        empty = width - filled

        if percentage >= 100:
            bar = "█" * width
            return f"[{bar}] [CRIT] FULL"
        if percentage >= 80:
            bar = "█" * filled + "░" * empty
            return f"[{bar}] [WARN] {percentage:.0f}%"

        bar = "█" * filled + "░" * empty
        return f"[{bar}] [OK] {percentage:.0f}%"
