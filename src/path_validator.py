"""
Path Validator for File Sync Operations

Prevents path traversal attacks (CWE-22) by validating file paths against
an allowed directory whitelist. This is critical for security when accepting
file paths from MCP tool calls.

Security Features:
- Resolves path traversal sequences (.., symlinks)
- Validates paths are within allowed directories
- Returns absolute paths for consistent storage
- Detailed error messages for debugging

Usage:
    validator = PathValidator(allowed_base_dirs=["/home/user/documents"])
    safe_path = validator.validate("/home/user/documents/file.txt")  # OK
    safe_path = validator.validate("../../etc/passwd")  # ValueError
"""

import logging
import os
import re
from urllib.parse import urlparse
from typing import List, Optional

logger = logging.getLogger("path_validator")


class PathValidator:
    """
    Validates file paths to prevent path traversal attacks (CWE-22).

    Ensures all file paths are within allowed base directories by:
    1. Resolving relative paths, .., and symlinks to absolute paths
    2. Checking if the resolved path starts with an allowed base directory
    3. Rejecting any path outside the whitelist

    This prevents attacks like:
    - Path traversal: ../../etc/passwd
    - Symlink traversal: ln -s /etc/passwd safe.txt
    - Absolute path injection: /etc/passwd

    Args:
        allowed_base_dirs: List of allowed base directories (default: current working directory)

    Example:
        >>> validator = PathValidator(allowed_base_dirs=[os.getcwd()])
        >>> validator.validate("docs/file.txt")  # OK - within cwd
        '/home/user/project/docs/file.txt'
        >>> validator.validate("../../etc/passwd")  # ValueError - outside cwd
    """

    def __init__(self, allowed_base_dirs: Optional[List[str]] = None):
        """
        Initialize path validator with allowed base directories.

        Args:
            allowed_base_dirs: List of allowed base directories
                              None = current working directory only
        """
        if allowed_base_dirs is None:
            # Default: only allow files in current working directory
            self.allowed_base_dirs = [os.getcwd()]
        else:
            # Resolve all base dirs to absolute paths and normalize
            self.allowed_base_dirs = [
                os.path.normpath(os.path.abspath(d)) for d in allowed_base_dirs
            ]

        # Sort by length (longest first) to match most specific directories first
        self.allowed_base_dirs.sort(key=len, reverse=True)

        logger.info(
            f"PathValidator initialized with {len(self.allowed_base_dirs)} allowed directories"
        )
        for base_dir in self.allowed_base_dirs:
            logger.debug(f"  Allowed: {base_dir}")

    def validate(self, file_path: str) -> str:
        """
        Validate file path and return absolute path if safe.

        Security checks:
        1. Path cannot be empty
        2. Resolve to absolute path (resolves .., symlinks)
        3. Path must be within allowed directories

        Args:
            file_path: Path to validate (relative or absolute)

        Returns:
            Absolute normalized path if valid

        Raises:
            ValueError: If path is invalid or outside allowed directories

        Example:
            >>> validator.validate("docs/file.txt")
            '/home/user/project/docs/file.txt'
            >>> validator.validate("../../etc/passwd")
            ValueError: Access denied - path outside allowed directories
        """
        # Check 1: Path cannot be empty
        if not file_path or not isinstance(file_path, str):
            raise ValueError("file_path cannot be empty or non-string")

        # Check 2: Resolve to absolute path
        # This resolves:
        # - Relative paths (./file.txt, ../file.txt)
        # - Path traversal (.., ../..)
        # - Symlinks (ln -s /etc/passwd safe.txt)
        # - Redundant separators (foo//bar)
        try:
            # os.path.realpath resolves symlinks AND converts to absolute
            # os.path.normpath normalizes path separators and removes redundant parts
            abs_path = os.path.normpath(os.path.realpath(file_path))
        except (OSError, ValueError) as e:
            raise ValueError(
                f"Invalid file path: {e}\n" f"Tip: Ensure the path is valid and accessible"
            ) from e

        # Check 3: Path must be within allowed directories
        # Use startswith to check if path is within allowed directory tree
        # Note: We normalized both allowed_base_dirs and abs_path, so this is safe
        is_allowed = any(
            abs_path.startswith(base_dir + os.sep) or abs_path == base_dir
            for base_dir in self.allowed_base_dirs
        )

        if not is_allowed:
            # Provide detailed error message for debugging
            allowed_str = "\n   ".join(self.allowed_base_dirs)
            raise ValueError(
                f"Access denied: file_path must be within allowed directories\n"
                f"   Allowed directories:\n   {allowed_str}\n"
                f"   Attempted path: {abs_path}\n"
                f"Tip: Use relative paths or paths within the current directory"
            )

        logger.debug(f"Path validated: {file_path} → {abs_path}")
        return abs_path

    def is_allowed(self, file_path: str) -> bool:
        """
        Check if a path is allowed without raising an exception.

        This is useful for conditional logic where you want to check
        if a path is valid without handling exceptions.

        Args:
            file_path: Path to check

        Returns:
            True if path is allowed, False otherwise

        Example:
            >>> if validator.is_allowed("docs/file.txt"):
            ...     safe_path = validator.validate("docs/file.txt")
        """
        try:
            self.validate(file_path)
            return True
        except ValueError:
            return False

    def get_allowed_directories(self) -> List[str]:
        """
        Get list of allowed base directories.

        Returns:
            List of absolute paths for allowed directories
        """
        return self.allowed_base_dirs.copy()

    def add_allowed_directory(self, directory: str) -> None:
        """
        Add a new allowed base directory.

        This can be used to dynamically expand the allowed directories
        during runtime (e.g., when user selects a new project folder).

        Args:
            directory: Directory path to add

        Raises:
            ValueError: If directory doesn't exist or is invalid

        Example:
            >>> validator.add_allowed_directory("/home/user/new_project")
        """
        if not os.path.isdir(directory):
            raise ValueError(f"Directory does not exist: {directory}")

        abs_dir = os.path.normpath(os.path.abspath(directory))

        if abs_dir not in self.allowed_base_dirs:
            self.allowed_base_dirs.append(abs_dir)
            # Re-sort by length (longest first)
            self.allowed_base_dirs.sort(key=len, reverse=True)
            logger.info(f"Added allowed directory: {abs_dir}")

    def validate_web_url(self, url: str) -> str:
        """Validate an HTTP(S) URL for connector ingestion."""
        if not url or not isinstance(url, str):
            raise ValueError("url cannot be empty or non-string")
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Web connector only supports absolute http/https URLs")
        if parsed.username or parsed.password:
            raise ValueError("Credentials in URLs are not allowed")
        return url

    def validate_github_repo(self, repo: str) -> str:
        """Validate GitHub repo identifier format."""
        if not repo or not isinstance(repo, str):
            raise ValueError("repo cannot be empty or non-string")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo.strip()):
            raise ValueError("GitHub repo must use 'owner/repo' format")
        return repo.strip()

    def validate_s3_bucket(self, bucket: str) -> str:
        """Validate S3 bucket naming used by connector feeds."""
        if not bucket or not isinstance(bucket, str):
            raise ValueError("bucket cannot be empty or non-string")
        bucket = bucket.strip()
        if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket) or ".." in bucket:
            raise ValueError("Invalid S3 bucket name")
        return bucket.strip()

    def remove_allowed_directory(self, directory: str) -> None:
        """
        Remove an allowed base directory.

        Args:
            directory: Directory path to remove

        Raises:
            ValueError: If directory is not in allowed list

        Example:
            >>> validator.remove_allowed_directory("/home/user/old_project")
        """
        abs_dir = os.path.normpath(os.path.abspath(directory))

        if abs_dir not in self.allowed_base_dirs:
            raise ValueError(f"Directory not in allowed list: {abs_dir}")

        self.allowed_base_dirs.remove(abs_dir)
        logger.info(f"Removed allowed directory: {abs_dir}")
