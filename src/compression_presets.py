"""
Compression profile presets for common use cases.

Named presets map to fidelity/ratio combinations optimized
for specific workflows like code review, chat, or research.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class CompressionPreset:
    """A named compression configuration."""

    name: str
    description: str
    skeleton_ratio: float
    fidelity: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "skeleton_ratio": self.skeleton_ratio,
            "fidelity": self.fidelity,
        }

    def to_prompt_seed(self) -> dict:
        prompt_name = f"compression-{self.name}"
        return {
            "name": prompt_name,
            "description": (
                f"Managed prompt template seeded from compression preset '{self.name}'. "
                f"{self.description}."
            ),
            "system_prompt": (
                "You are a token-saving prompt optimizer. Preserve factual accuracy, "
                "structural anchors, and retrieval-critical context. Prefer stable "
                "instructions and output deterministic sections."
            ),
            "user_prompt_template": (
                "Use the '{preset_name}' compression strategy for the following content.\n"
                "Target skeleton ratio: {skeleton_ratio}\n"
                "Target fidelity: {fidelity}\n"
                "Use case: {use_case}\n\n"
                "Content to transform:\n{content}"
            ),
            "variables": ["preset_name", "skeleton_ratio", "fidelity", "use_case", "content"],
            "metadata": {
                "preset_name": self.name,
                "default_skeleton_ratio": self.skeleton_ratio,
                "default_fidelity": self.fidelity,
            },
            "source_preset": self.name,
            "deployment_label": "production",
        }


_PRESETS = {
    "code-review": CompressionPreset(
        name="code-review",
        description="High fidelity, preserves structure and detail for thorough code review",
        skeleton_ratio=0.5,
        fidelity="DETAILED",
    ),
    "chat": CompressionPreset(
        name="chat",
        description="Balanced compression for conversational context, keeps outlines",
        skeleton_ratio=0.25,
        fidelity="OUTLINE",
    ),
    "research": CompressionPreset(
        name="research",
        description="Moderate compression preserving structure for research analysis",
        skeleton_ratio=0.35,
        fidelity="STRUCTURE",
    ),
    "aggressive": CompressionPreset(
        name="aggressive",
        description="Maximum compression for large codebases, abstract summaries only",
        skeleton_ratio=0.1,
        fidelity="ABSTRACT",
    ),
    "balanced": CompressionPreset(
        name="balanced",
        description="Default balance between compression and detail preservation",
        skeleton_ratio=0.2,
        fidelity="STRUCTURE",
    ),
}


def get_preset(name: str) -> CompressionPreset:
    """Get a compression preset by name.

    Args:
        name: Preset name (e.g., 'code-review', 'chat', 'research', 'aggressive')

    Returns:
        CompressionPreset with configured values

    Raises:
        ValueError: If preset name is unknown
    """
    preset = _PRESETS.get(name)
    if preset is None:
        available = ", ".join(sorted(_PRESETS.keys()))
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return preset


def list_presets() -> List[CompressionPreset]:
    """List all available compression presets."""
    return list(_PRESETS.values())
