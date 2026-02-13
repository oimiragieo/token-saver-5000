#!/usr/bin/env python3
"""
Migrate Legacy Pickle Files to Safe JSON + NumPy Format

Security migration script for Token Saver 5000 v0.8.0.
Converts legacy pickle files to the secure JSON + NumPy format
to prevent CWE-502 (Deserialization of Untrusted Data) vulnerabilities.

Usage:
    python scripts/migrate_pickle.py [--dry-run] [--backup] [--storage-dir PATH]

Options:
    --dry-run       Show what would be migrated without making changes
    --backup        Create .pkl.bak backup files (default: True)
    --no-backup     Delete pickle files after migration (no backup)
    --storage-dir   Path to storage directory (default: .semantic_modulator_data)

Examples:
    # Preview migration
    python scripts/migrate_pickle.py --dry-run

    # Run migration with backups
    python scripts/migrate_pickle.py --backup

    # Run migration without backups (deletes pickle files)
    python scripts/migrate_pickle.py --no-backup

Security Notes:
    - This script intentionally uses pickle.load() during migration
    - Only run this on TRUSTED pickle files that YOU created
    - After migration, legacy pickle files are rejected by the server
    - The new format uses JSON for structure and numpy.save for embeddings

Version: 0.8.0
"""

import argparse
import json
import logging
import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constants
PERSISTENCE_FORMAT_VERSION = 2  # v1 = pickle, v2 = JSON + numpy
DEFAULT_STORAGE_DIR = ".semantic_modulator_data"


def find_legacy_pickle_files(storage_dir: Path) -> List[Path]:
    """
    Find all legacy pickle files in the storage directory.

    Args:
        storage_dir: Path to the storage directory

    Returns:
        List of pickle file paths
    """
    pickle_files = []

    documents_dir = storage_dir / "documents"
    if documents_dir.exists():
        pickle_files.extend(documents_dir.glob("*_graph.pkl"))
        pickle_files.extend(documents_dir.glob("*_chunks.pkl"))

    afm_dir = storage_dir / "afm_history"
    if afm_dir.exists():
        pickle_files.extend(afm_dir.glob("*.pkl"))

    # Also check root storage dir
    pickle_files.extend(storage_dir.glob("*.pkl"))

    return sorted(pickle_files)


def load_pickle_safely(pickle_path: Path) -> Any:
    """
    Load a pickle file.

    WARNING: Only use this on TRUSTED pickle files.
    Pickle can execute arbitrary code during deserialization.

    Args:
        pickle_path: Path to pickle file

    Returns:
        Deserialized data
    """
    logger.warning(f"Loading pickle file (TRUSTED): {pickle_path}")
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def save_as_safe_format(
    data: Any,
    output_base: Path,
    data_type: str,
) -> bool:
    """
    Save data in safe JSON + numpy format.

    Args:
        data: The data to save
        output_base: Base path for output files (without extension)
        data_type: Type of data ("graph", "chunks", "afm")

    Returns:
        True if successful
    """
    try:
        if data_type == "graph":
            return _save_graph_data(data, output_base)
        elif data_type == "chunks":
            return _save_chunks_data(data, output_base)
        elif data_type == "afm":
            return _save_afm_data(data, output_base)
        else:
            logger.error(f"Unknown data type: {data_type}")
            return False
    except Exception as e:
        logger.error(f"Failed to save {data_type} data: {e}")
        return False


def _save_graph_data(graph_data: Dict[str, Any], output_base: Path) -> bool:
    """Save graph data in safe format."""
    json_file = output_base.with_suffix(".json")
    embeddings_file = output_base.parent / (output_base.stem.replace("_graph", "_embeddings.npz"))

    # Separate embeddings from graph structure
    embeddings_dict = {}
    safe_graph_data = {
        "nodes": [],
        "edges": graph_data.get("edges", []),
        "metadata": graph_data.get("metadata", {}),
        "_format_version": PERSISTENCE_FORMAT_VERSION,
    }

    # Process nodes - extract embeddings separately
    for node_data in graph_data.get("nodes", []):
        node_copy = dict(node_data) if isinstance(node_data, dict) else {"data": node_data}
        node_id = node_copy.get("id", node_copy.get("node_id", str(len(safe_graph_data["nodes"]))))

        # Extract embedding if present
        if "embedding" in node_copy:
            emb = node_copy.pop("embedding")
            if isinstance(emb, np.ndarray):
                embeddings_dict[str(node_id)] = emb
            elif isinstance(emb, list):
                embeddings_dict[str(node_id)] = np.array(emb)

        safe_graph_data["nodes"].append(node_copy)

    # Save JSON structure
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(safe_graph_data, f, indent=2, default=str)
    logger.info(f"  Saved JSON: {json_file}")

    # Save embeddings as numpy arrays
    if embeddings_dict:
        ids = list(embeddings_dict.keys())
        embeddings = np.array([embeddings_dict[id_] for id_ in ids])
        np.savez(embeddings_file, embeddings=embeddings)

        # Save IDs as separate JSON (secure format)
        ids_file = embeddings_file.with_suffix(".ids.json")
        with open(ids_file, "w", encoding="utf-8") as f:
            json.dump(ids, f)

        logger.info(f"  Saved embeddings: {embeddings_file}")
        logger.info(f"  Saved IDs: {ids_file}")

    return True


def _save_chunks_data(chunks_data: Dict[str, Any], output_base: Path) -> bool:
    """Save chunks data in safe format."""
    json_file = output_base.with_suffix(".json")
    embeddings_file = output_base.parent / (
        output_base.stem.replace("_chunks", "_chunk_embeddings.npz")
    )

    # Separate embeddings from chunk data
    embeddings_dict = {}
    safe_chunks = {}

    for chunk_id, chunk in chunks_data.items():
        chunk_copy = dict(chunk) if isinstance(chunk, dict) else {"data": str(chunk)}

        # Handle object with attributes
        if hasattr(chunk, "__dict__"):
            chunk_copy = dict(chunk.__dict__)

        # Extract embedding if present
        if "embedding" in chunk_copy:
            emb = chunk_copy.pop("embedding")
            if isinstance(emb, np.ndarray):
                embeddings_dict[str(chunk_id)] = emb
            elif isinstance(emb, list):
                embeddings_dict[str(chunk_id)] = np.array(emb)

        safe_chunks[chunk_id] = chunk_copy

    # Add version marker
    safe_data = {
        "chunks": safe_chunks,
        "_format_version": PERSISTENCE_FORMAT_VERSION,
    }

    # Save JSON structure
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(safe_data, f, indent=2, default=str)
    logger.info(f"  Saved JSON: {json_file}")

    # Save embeddings
    if embeddings_dict:
        ids = list(embeddings_dict.keys())
        embeddings = np.array([embeddings_dict[id_] for id_ in ids])
        np.savez(embeddings_file, embeddings=embeddings)

        ids_file = embeddings_file.with_suffix(".ids.json")
        with open(ids_file, "w", encoding="utf-8") as f:
            json.dump(ids, f)

        logger.info(f"  Saved embeddings: {embeddings_file}")
        logger.info(f"  Saved IDs: {ids_file}")

    return True


def _save_afm_data(afm_data: Dict[str, Any], output_base: Path) -> bool:
    """Save AFM (dialogue history) data in safe format."""
    json_file = output_base.with_suffix(".json")

    # Convert to JSON-serializable format
    safe_data = {
        "_format_version": PERSISTENCE_FORMAT_VERSION,
    }

    for key, value in afm_data.items():
        if isinstance(value, list):
            # Convert message objects to dicts if needed
            safe_list = []
            for item in value:
                if hasattr(item, "__dict__"):
                    safe_list.append(dict(item.__dict__))
                elif isinstance(item, dict):
                    safe_list.append(item)
                else:
                    safe_list.append(str(item))
            safe_data[key] = safe_list
        elif hasattr(value, "__dict__"):
            safe_data[key] = dict(value.__dict__)
        else:
            safe_data[key] = value

    # Save JSON
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(safe_data, f, indent=2, default=str)
    logger.info(f"  Saved JSON: {json_file}")

    return True


def determine_data_type(pickle_path: Path) -> str:
    """Determine the type of data in a pickle file based on filename."""
    name = pickle_path.stem.lower()
    if "_graph" in name:
        return "graph"
    elif "_chunks" in name or "chunk" in name:
        return "chunks"
    elif "afm" in name or "history" in name or "dialogue" in name:
        return "afm"
    else:
        return "afm"  # Default to AFM format for unknown types


def migrate_file(
    pickle_path: Path,
    backup: bool = True,
    dry_run: bool = False,
) -> Tuple[bool, str]:
    """
    Migrate a single pickle file to safe format.

    Args:
        pickle_path: Path to pickle file
        backup: Whether to create backup
        dry_run: If True, don't actually migrate

    Returns:
        Tuple of (success, message)
    """
    if dry_run:
        return True, f"[DRY RUN] Would migrate: {pickle_path}"

    try:
        # Determine data type and output path
        data_type = determine_data_type(pickle_path)
        output_base = pickle_path.with_suffix("")  # Remove .pkl extension

        logger.info(f"Migrating: {pickle_path} ({data_type})")

        # Load pickle data
        data = load_pickle_safely(pickle_path)

        # Save in safe format
        success = save_as_safe_format(data, output_base, data_type)

        if not success:
            return False, f"Failed to save migrated data for {pickle_path}"

        # Handle old pickle file
        if backup:
            backup_path = pickle_path.with_suffix(".pkl.bak")
            shutil.move(str(pickle_path), str(backup_path))
            logger.info(f"  Backed up to: {backup_path}")
        else:
            os.remove(pickle_path)
            logger.info(f"  Deleted: {pickle_path}")

        return True, f"Successfully migrated: {pickle_path}"

    except Exception as e:
        return False, f"Error migrating {pickle_path}: {e}"


def main():
    """Main entry point for migration script."""
    parser = argparse.ArgumentParser(
        description="Migrate legacy pickle files to safe JSON + numpy format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without making changes",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        default=True,
        help="Create .pkl.bak backup files (default)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Delete pickle files after migration (no backup)",
    )
    parser.add_argument(
        "--storage-dir",
        type=Path,
        default=Path(DEFAULT_STORAGE_DIR),
        help=f"Path to storage directory (default: {DEFAULT_STORAGE_DIR})",
    )

    args = parser.parse_args()

    # Determine backup setting
    backup = not args.no_backup

    print("=" * 70)
    print("Token Saver 5000 - Pickle to JSON Migration Tool")
    print("=" * 70)
    print(f"Storage directory: {args.storage_dir.absolute()}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE MIGRATION'}")
    print(f"Backup: {'Yes (creating .pkl.bak files)' if backup else 'No (deleting pickle files)'}")
    print()

    # Check storage directory exists
    if not args.storage_dir.exists():
        print(f"[WARN] Storage directory does not exist: {args.storage_dir}")
        print("       No files to migrate.")
        return 0

    # Find pickle files
    pickle_files = find_legacy_pickle_files(args.storage_dir)

    if not pickle_files:
        print("[OK] No legacy pickle files found. Already migrated or fresh install.")
        return 0

    print(f"Found {len(pickle_files)} legacy pickle file(s):")
    for pf in pickle_files:
        print(f"  - {pf}")
    print()

    # Confirm if not dry run
    if not args.dry_run:
        print("[WARN] This script will load pickle files using pickle.load().")
        print("       Only proceed if you TRUST these files (e.g., you created them).")
        print()
        response = input("Continue with migration? [y/N]: ").strip().lower()
        if response != "y":
            print("Aborted.")
            return 1
        print()

    # Migrate files
    success_count = 0
    failure_count = 0
    results = []

    for pickle_path in pickle_files:
        success, message = migrate_file(pickle_path, backup=backup, dry_run=args.dry_run)
        results.append((success, message))

        if success:
            success_count += 1
            print(f"[OK] {message}")
        else:
            failure_count += 1
            print(f"[FAIL] {message}")

    # Summary
    print()
    print("=" * 70)
    print("Migration Summary")
    print("=" * 70)
    print(f"Total files:    {len(pickle_files)}")
    print(f"Successful:     {success_count}")
    print(f"Failed:         {failure_count}")

    if failure_count > 0:
        print()
        print("[WARN] Some migrations failed. Check the errors above.")
        print("       Failed files will still block server startup until migrated.")
        return 1

    if not args.dry_run and success_count > 0:
        print()
        print("[OK] Migration complete!")
        print("     The server will now use the secure JSON + numpy format.")
        if backup:
            print("     Backup files (.pkl.bak) can be deleted after verifying migration.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
