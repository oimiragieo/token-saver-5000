"""Tests for GitHub connector normalization."""

from src.connectors.github_connector import GitHubConnector


def test_collects_repo_files_into_documents():
    connector = GitHubConnector()
    documents = connector.collect_documents(
        {
            "repo": "octo/example",
            "files": [{"path": "src/app.py", "content": "print('hello')", "ref": "main"}],
        }
    )

    assert len(documents) == 1
    assert documents[0].metadata["repo"] == "octo/example"
    assert documents[0].metadata["path"] == "src/app.py"
