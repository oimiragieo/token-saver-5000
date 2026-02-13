"""Unit tests for shared node identity parsing helpers."""

from src.node_identity import collect_file_ids, extract_file_id_from_node


def test_extract_file_id_from_text_node_suffix():
    assert extract_file_id_from_node("paper_n12") == "paper"
    assert extract_file_id_from_node("design_notes_n0") == "design_notes"


def test_extract_file_id_from_code_node():
    assert extract_file_id_from_node("main.py::parse_args") == "main.py"


def test_extract_file_id_ignores_non_index_n_segments():
    assert extract_file_id_from_node("design_notes") == "design_notes"
    assert extract_file_id_from_node("plan_neto") == "plan_neto"
    assert extract_file_id_from_node("chapter_nX") == "chapter_nX"


def test_collect_file_ids_supports_mixed_node_formats():
    node_ids = ["doc_a_n0", "doc_a_n1", "main.py::run", "main.py::parse_args", "notes"]
    assert collect_file_ids(node_ids) == {"doc_a", "main.py", "notes"}
