"""Shared helpers for patching F11_RANKER_PATH across constants + facade modules."""

from __future__ import annotations


def f11_ranker_path_orig() -> tuple[str, str, str]:
    import src.constants as _constants
    import src.semantic_compressor as _sc
    import src.handlers.compression_handlers as _ch

    return (_constants.F11_RANKER_PATH, _sc.F11_RANKER_PATH, _ch.F11_RANKER_PATH)


def set_f11_ranker_path(path: str) -> tuple[str, str, str]:
    import src.constants as _constants
    import src.semantic_compressor as _sc
    import src.handlers.compression_handlers as _ch

    orig = f11_ranker_path_orig()
    normalized = path.lower().strip()
    _constants.F11_RANKER_PATH = normalized
    _sc.F11_RANKER_PATH = normalized
    _ch.F11_RANKER_PATH = normalized
    return orig


def restore_f11_ranker_path(orig: tuple[str, str, str]) -> None:
    import src.constants as _constants
    import src.semantic_compressor as _sc
    import src.handlers.compression_handlers as _ch

    _constants.F11_RANKER_PATH, _sc.F11_RANKER_PATH, _ch.F11_RANKER_PATH = orig
