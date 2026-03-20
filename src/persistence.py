"""
Persistent Storage Layer for Semantic Modulator

Provides persistent storage for semantic graphs using ChromaDB.
Automatically saves and loads document state across server restarts.

Features:
- Persistent vector storage with ChromaDB
- Automatic save on document ingestion
- Automatic load on server start
- Metadata preservation
- Node and edge persistence
- Graceful degradation if ChromaDB unavailable

Security Notes (v0.8.0):
- Migrated from pickle to JSON + numpy for security (CWE-502 fix)
- numpy.load(allow_pickle=False) used explicitly to prevent code execution
- Legacy pickle files are loaded with warning but new saves use safe format
- File format version tracking for backward compatibility
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import tempfile
import shutil

from .identity_scope import compose_scoped_file_id, display_file_id, has_scope, scope_matches

# File format version for migration tracking
PERSISTENCE_FORMAT_VERSION = 2  # v1 = pickle, v2 = JSON + numpy

try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logging.warning("ChromaDB not available, using fallback JSON persistence")


logger = logging.getLogger("persistence")


class PersistenceManager:
    """
    Manages persistent storage for semantic graphs and dialogue history.

    Storage backends:
    1. ChromaDB (preferred) - Vector database for embeddings
    2. JSON/Pickle fallback - File-based storage if ChromaDB unavailable
    """

    def __init__(self, storage_dir: str = ".semantic_modulator_data"):
        """
        Initialize persistence manager.

        Args:
            storage_dir: Directory for persistent storage
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

        # Initialize ChromaDB if available
        self.use_chromadb = CHROMADB_AVAILABLE
        self.chroma_client = None

        if self.use_chromadb:
            try:
                self.chroma_client = chromadb.PersistentClient(
                    path=str(self.storage_dir / "chromadb"),
                    settings=Settings(anonymized_telemetry=False, allow_reset=True),
                )
                logger.info(f"ChromaDB initialized at {self.storage_dir}/chromadb")
            except Exception as e:
                logger.warning(f"Failed to initialize ChromaDB: {e}, using JSON fallback")
                self.use_chromadb = False
                self.chroma_client = None

        # Fallback storage paths
        self.documents_dir = self.storage_dir / "documents"
        self.documents_dir.mkdir(exist_ok=True)

        self.afm_dir = self.storage_dir / "afm_history"
        self.afm_dir.mkdir(exist_ok=True)

        logger.info(f"Persistence manager initialized (ChromaDB: {self.use_chromadb})")

    @staticmethod
    def _atomic_write_json(filepath: Path, data: dict) -> None:
        """Write JSON atomically via temp file + rename to prevent corruption."""
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=filepath.parent, suffix=".tmp", prefix=filepath.stem
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            shutil.move(tmp_path, filepath)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    @staticmethod
    def _atomic_write_npz(filepath: Path, **arrays) -> None:
        """Write numpy arrays atomically via temp file + rename."""
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=filepath.parent, suffix=".tmp", prefix=filepath.stem
        )
        os.close(tmp_fd)
        try:
            np.savez(tmp_path, **arrays)
            shutil.move(
                tmp_path + ".npz" if os.path.exists(tmp_path + ".npz") else tmp_path, filepath
            )
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except BaseException:
            for p in (tmp_path, tmp_path + ".npz"):
                if os.path.exists(p):
                    os.remove(p)
            raise

    # =========================================================================
    # Safe Serialization Helpers (v0.8.0 - CWE-502 fix)
    # =========================================================================

    def _save_graph_data_safe(self, file_id: str, graph_data: Dict[str, Any]) -> bool:
        """
        Save graph data using safe JSON + numpy format (v0.8.0 security fix).

        Uses JSON for structure and numpy.save for embeddings.
        This prevents CWE-502 (Deserialization of Untrusted Data) vulnerabilities.

        Args:
            file_id: Document identifier
            graph_data: NetworkX graph data (nodes, edges, metadata)

        Returns:
            True if saved successfully
        """
        graph_file = self.documents_dir / f"{file_id}_graph.json"
        embeddings_file = self.documents_dir / f"{file_id}_embeddings.npy"

        try:
            # Separate embeddings from graph structure
            embeddings_dict = {}
            safe_graph_data = {"nodes": [], "edges": [], "metadata": graph_data.get("metadata", {})}

            # Process nodes - extract embeddings separately
            for node_data in graph_data.get("nodes", []):
                node_copy = dict(node_data)
                node_id = node_copy.get("id", node_copy.get("node_id", "unknown"))

                # Extract embedding if present
                if "embedding" in node_copy:
                    emb = node_copy.pop("embedding")
                    if isinstance(emb, np.ndarray):
                        embeddings_dict[node_id] = emb
                    elif isinstance(emb, list):
                        embeddings_dict[node_id] = np.array(emb)

                safe_graph_data["nodes"].append(node_copy)

            # Copy edges
            safe_graph_data["edges"] = graph_data.get("edges", [])

            # Add format version marker
            safe_graph_data["_format_version"] = PERSISTENCE_FORMAT_VERSION

            # Save JSON structure
            with open(graph_file, "w", encoding="utf-8") as f:
                json.dump(safe_graph_data, f, indent=2, default=str)

            # Save embeddings as numpy array (allow_pickle=False by default for .npy)
            if embeddings_dict:
                # Stack embeddings into single array with ID mapping
                ids = list(embeddings_dict.keys())
                embeddings = np.array([embeddings_dict[id_] for id_ in ids])
                # Save both the embeddings array and ID mapping
                np.savez(
                    embeddings_file.with_suffix(".npz"),
                    embeddings=embeddings,
                    ids=np.array(ids, dtype=object),
                )

            logger.debug(f"Saved graph data safely for {file_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save graph data safely for {file_id}: {e}")
            return False

    def _load_graph_data_safe(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Load graph data from safe JSON + numpy format (v0.8.0 security fix).

        Falls back to legacy pickle with warning if new format not found.

        Args:
            file_id: Document identifier

        Returns:
            Graph data dictionary or None if not found
        """
        graph_file = self.documents_dir / f"{file_id}_graph.json"
        embeddings_file = self.documents_dir / f"{file_id}_embeddings.npz"
        legacy_pickle = self.documents_dir / f"{file_id}_graph.pkl"

        try:
            # Try new safe format first
            if graph_file.exists():
                with open(graph_file, "r", encoding="utf-8") as f:
                    graph_data = json.load(f)

                # Load embeddings if available
                if embeddings_file.exists():
                    # SECURITY: allow_pickle=False prevents code execution (CWE-502)
                    npz_data = np.load(embeddings_file, allow_pickle=False)
                    embeddings = npz_data["embeddings"]

                    # Load IDs from JSON file (v0.8.0 audit fix - Issue 2)
                    # SECURITY: IDs stored as JSON to avoid pickle vulnerability
                    ids_file = embeddings_file.with_suffix(".ids.json")
                    if ids_file.exists():
                        with open(ids_file, "r", encoding="utf-8") as f:
                            ids = json.load(f)
                    elif "ids" in npz_data.files:
                        # Legacy fallback: IDs in numpy (deprecated)
                        logger.warning(
                            "Loading legacy numpy IDs for graph - re-save to migrate to secure format"
                        )
                        # Still use allow_pickle=False, will fail on object arrays (intentional)
                        ids = npz_data["ids"].tolist()
                    else:
                        # Fallback: reconstruct IDs from graph_data
                        ids = [n.get("id", n.get("node_id")) for n in graph_data.get("nodes", [])]

                    # Reconstruct embedding mapping
                    emb_map = {str(ids[i]): embeddings[i] for i in range(len(ids))}

                    # Attach embeddings back to nodes
                    for node_data in graph_data.get("nodes", []):
                        node_id = node_data.get("id", node_data.get("node_id"))
                        if node_id and str(node_id) in emb_map:
                            node_data["embedding"] = emb_map[str(node_id)]

                logger.debug(f"Loaded graph data (safe format) for {file_id}")
                return graph_data

            # v0.8.0 SECURITY FIX: Fail hard on legacy pickle files (CWE-502)
            # pickle.load() can execute arbitrary code, so we refuse to load these
            elif legacy_pickle.exists():
                raise ValueError(
                    f"SECURITY: Legacy pickle file detected for '{file_id}'.\n"
                    f"  File: {legacy_pickle}\n"
                    f"\n"
                    f"Pickle files are a security risk (CWE-502: arbitrary code execution).\n"
                    f"Please run the migration script to convert to safe format:\n"
                    f"\n"
                    f"  python scripts/migrate_pickle.py\n"
                    f"\n"
                    f"Or delete the legacy file and re-ingest the document."
                )

            return None

        except ValueError:
            # Re-raise security-related ValueError (legacy pickle rejection)
            raise
        except Exception as e:
            logger.error(f"Failed to load graph data for {file_id}: {e}")
            return None

    def _serialize_chunks_safe(self, chunks: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize SemanticNode chunks to JSON-safe format.

        Converts numpy arrays to lists for JSON serialization.

        Args:
            chunks: Dictionary of node_id -> SemanticNode

        Returns:
            JSON-serializable dictionary
        """
        safe_chunks = {}

        for node_id, node in chunks.items():
            safe_node = {
                "node_id": node.node_id,
                "text": node.text,
                "importance": float(node.importance),
                "metadata": node.metadata,
            }
            # Convert embedding to list for JSON
            if hasattr(node, "embedding") and node.embedding is not None:
                if isinstance(node.embedding, np.ndarray):
                    safe_node["embedding"] = node.embedding.tolist()
                else:
                    safe_node["embedding"] = list(node.embedding)

            safe_chunks[node_id] = safe_node

        return safe_chunks

    def _deserialize_chunks_safe(self, safe_chunks: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deserialize JSON chunks back to SemanticNode format.

        Converts embedding lists back to numpy arrays.

        Args:
            safe_chunks: JSON-loaded chunk dictionary

        Returns:
            Dictionary of node_id -> SemanticNode-like dict (or actual SemanticNode)
        """
        from .semantic_compressor import SemanticNode

        chunks = {}

        for node_id, node_data in safe_chunks.items():
            embedding = None
            if "embedding" in node_data:
                embedding = np.array(node_data["embedding"])

            chunks[node_id] = SemanticNode(
                node_id=node_data["node_id"],
                text=node_data["text"],
                embedding=embedding,
                importance=node_data["importance"],
                metadata=node_data.get("metadata", {}),
            )

        return chunks

    # =========================================================================
    # Document Persistence
    # =========================================================================

    def save_document(
        self,
        file_id: str,
        chunks: Dict[str, Any],
        graph_data: Dict[str, Any],
        metadata: Dict[str, Any],
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> bool:
        """
        Save document to persistent storage.

        Args:
            file_id: Document identifier
            chunks: Dictionary of SemanticNode objects
            graph_data: NetworkX graph data
            metadata: Document metadata

        Returns:
            True if saved successfully
        """
        internal_file_id = compose_scoped_file_id(
            file_id,
            workspace_id=workspace_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )
        try:
            if self.use_chromadb:
                return self._save_document_chromadb(internal_file_id, chunks, graph_data, metadata)
            else:
                return self._save_document_json(internal_file_id, chunks, graph_data, metadata)
        except Exception as e:
            logger.error(f"Failed to save document {internal_file_id}: {e}", exc_info=True)
            return False

    def _save_document_chromadb(
        self,
        file_id: str,
        chunks: Dict[str, Any],
        graph_data: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> bool:
        """Save document using ChromaDB."""
        # Create or get collection for this document
        collection_name = f"doc_{file_id}".replace("-", "_").replace(".", "_")

        try:
            # Delete existing collection if present
            try:
                self.chroma_client.delete_collection(name=collection_name)
            except Exception:
                pass

            collection = self.chroma_client.create_collection(
                name=collection_name, metadata={"file_id": file_id, **metadata}
            )

            # Prepare data for ChromaDB
            node_ids = []
            embeddings = []
            documents = []
            metadatas = []

            for node_id, node in chunks.items():
                node_ids.append(node_id)
                embeddings.append(node.embedding.tolist())
                documents.append(node.text)
                metadatas.append(
                    {
                        "importance": float(node.importance),
                        "position": node.metadata.get("position", 0),
                        "tokens": node.metadata.get("tokens", 0),
                        "entities": json.dumps(node.metadata.get("entities", [])),
                    }
                )

            # Add to collection
            collection.add(
                ids=node_ids, embeddings=embeddings, documents=documents, metadatas=metadatas
            )

            # Save graph structure using safe format (v0.8.0 - CWE-502 fix)
            # Uses JSON + numpy instead of pickle
            self._save_graph_data_safe(file_id, graph_data)

            logger.info(f"[OK] Saved document {file_id} to ChromaDB ({len(chunks)} nodes)")
            return True

        except Exception as e:
            logger.error(f"ChromaDB save failed for {file_id}: {e}")
            return False

    def _save_document_json(
        self,
        file_id: str,
        chunks: Dict[str, Any],
        graph_data: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> bool:
        """Save document using safe JSON format (v0.8.0 - CWE-502 fix)."""
        doc_file = self.documents_dir / f"{file_id}.json"
        embeddings_file = self.documents_dir / f"{file_id}_chunks.npz"

        try:
            # Serialize chunks to JSON-safe format
            safe_chunks = self._serialize_chunks_safe(chunks)

            # Extract embeddings for numpy storage
            embeddings_dict = {}
            for node_id, chunk_data in safe_chunks.items():
                if "embedding" in chunk_data:
                    embeddings_dict[node_id] = np.array(chunk_data.pop("embedding"))

            data = {
                "_format_version": PERSISTENCE_FORMAT_VERSION,
                "file_id": file_id,
                "chunks": safe_chunks,
                "graph_data": graph_data,
                "metadata": metadata,
            }

            # Atomic save: JSON structure
            self._atomic_write_json(doc_file, data)

            # Atomic save: embeddings + IDs
            if embeddings_dict:
                ids = list(embeddings_dict.keys())
                embeddings = np.array([embeddings_dict[id_] for id_ in ids])
                self._atomic_write_npz(embeddings_file, embeddings=embeddings)
                ids_file = embeddings_file.with_suffix(".ids.json")
                self._atomic_write_json(ids_file, ids)

            logger.info(f"[OK] Saved document {file_id} to JSON ({len(chunks)} nodes)")
            return True

        except Exception as e:
            logger.error(f"JSON save failed for {file_id}: {e}")
            return False

    def load_document(
        self,
        file_id: str,
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Load document from persistent storage.

        Args:
            file_id: Document identifier

        Returns:
            Dictionary with chunks, graph_data, metadata, or None if not found
        """
        internal_file_id = compose_scoped_file_id(
            file_id,
            workspace_id=workspace_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )
        try:
            if self.use_chromadb:
                return self._load_document_chromadb(internal_file_id)
            else:
                return self._load_document_json(internal_file_id)
        except Exception as e:
            logger.error(f"Failed to load document {internal_file_id}: {e}", exc_info=True)
            return None

    def _load_document_chromadb(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Load document from ChromaDB."""
        collection_name = f"doc_{file_id}".replace("-", "_").replace(".", "_")

        try:
            collection = self.chroma_client.get_collection(name=collection_name)

            # Get all items
            results = collection.get(include=["embeddings", "documents", "metadatas"])

            # Reconstruct chunks
            from .semantic_compressor import SemanticNode

            chunks = {}

            for i, node_id in enumerate(results["ids"]):
                chunks[node_id] = SemanticNode(
                    node_id=node_id,
                    text=results["documents"][i],
                    embedding=np.array(results["embeddings"][i]),
                    importance=results["metadatas"][i]["importance"],
                    metadata={
                        "position": results["metadatas"][i]["position"],
                        "tokens": results["metadatas"][i]["tokens"],
                        "entities": json.loads(results["metadatas"][i]["entities"]),
                    },
                )

            # Load graph structure using safe method (v0.8.0 - CWE-502 fix)
            # Falls back to legacy pickle with warning if safe format not found
            graph_data = self._load_graph_data_safe(file_id)
            if graph_data is None:
                logger.warning(f"No graph data found for {file_id}")
                graph_data = {}

            logger.info(f"[OK] Loaded document {file_id} from ChromaDB ({len(chunks)} nodes)")

            return {
                "chunks": chunks,
                "graph_data": graph_data,
                "metadata": collection.metadata,
            }

        except Exception as e:
            logger.debug(f"ChromaDB load failed for {file_id}: {e}")
            return None

    def _load_document_json(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Load document from safe JSON format (v0.8.0 - CWE-502 fix).

        Tries new JSON format first, falls back to legacy pickle with warning.
        """
        json_file = self.documents_dir / f"{file_id}.json"
        embeddings_file = self.documents_dir / f"{file_id}_chunks.npz"
        legacy_pickle = self.documents_dir / f"{file_id}.pkl"

        try:
            # Try new safe JSON format first
            if json_file.exists():
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Load embeddings if available
                if embeddings_file.exists():
                    # SECURITY: allow_pickle=False prevents code execution (CWE-502)
                    npz_data = np.load(embeddings_file, allow_pickle=False)
                    embeddings = npz_data["embeddings"]

                    # Load IDs from JSON file (v0.8.0 audit fix - Issue 2)
                    # SECURITY: IDs stored as JSON to avoid pickle vulnerability
                    ids_file = embeddings_file.with_suffix(".ids.json")
                    if ids_file.exists():
                        with open(ids_file, "r", encoding="utf-8") as f:
                            ids = json.load(f)
                    elif "ids" in npz_data.files:
                        # Legacy fallback: IDs in numpy (deprecated)
                        logger.warning(
                            f"Loading legacy numpy IDs for {file_id} - re-save to migrate to secure format"
                        )
                        # Still use allow_pickle=False, will fail on object arrays (intentional)
                        ids = npz_data["ids"].tolist()
                    else:
                        # Fallback: use chunk keys from data
                        ids = list(data.get("chunks", {}).keys())

                    # Attach embeddings back to chunks
                    emb_map = {str(ids[i]): embeddings[i] for i in range(len(ids))}
                    for node_id, chunk_data in data.get("chunks", {}).items():
                        if str(node_id) in emb_map:
                            chunk_data["embedding"] = emb_map[str(node_id)]

                # Deserialize chunks to SemanticNode objects
                if "chunks" in data:
                    data["chunks"] = self._deserialize_chunks_safe(data["chunks"])

                logger.info(
                    f"[OK] Loaded document {file_id} from JSON ({len(data.get('chunks', {}))} nodes)"
                )
                return data

            # v0.8.0 SECURITY FIX: Fail hard on legacy pickle files (CWE-502)
            # pickle.load() can execute arbitrary code, so we refuse to load these
            elif legacy_pickle.exists():
                raise ValueError(
                    f"SECURITY: Legacy pickle file detected for document '{file_id}'.\n"
                    f"  File: {legacy_pickle}\n"
                    f"\n"
                    f"Pickle files are a security risk (CWE-502: arbitrary code execution).\n"
                    f"Please run the migration script to convert to safe format:\n"
                    f"\n"
                    f"  python scripts/migrate_pickle.py\n"
                    f"\n"
                    f"Or delete the legacy file and re-ingest the document."
                )

            return None

        except ValueError:
            # Re-raise security-related ValueError (legacy pickle rejection)
            raise
        except Exception as e:
            logger.error(f"JSON load failed for {file_id}: {e}")
            return None

    def list_documents(
        self,
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[str]:
        """
        List all persisted document IDs.

        Returns:
            List of file_ids
        """
        if self.use_chromadb:
            file_ids = self._list_documents_chromadb()
        else:
            file_ids = self._list_documents_json()

        if not has_scope(workspace_id, user_id, agent_id, session_id):
            return file_ids

        return [
            display_file_id(file_id)
            for file_id in file_ids
            if scope_matches(
                file_id,
                workspace_id=workspace_id,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
            )
        ]

    def _list_documents_chromadb(self) -> List[str]:
        """List documents from ChromaDB."""
        try:
            collections = self.chroma_client.list_collections()
            file_ids = []
            for collection in collections:
                if collection.name.startswith("doc_"):
                    # Extract file_id from collection metadata
                    file_id = collection.metadata.get("file_id")
                    if file_id:
                        file_ids.append(file_id)
            return file_ids
        except Exception as e:
            logger.error(f"Failed to list ChromaDB documents: {e}")
            return []

    def _list_documents_json(self) -> List[str]:
        """List documents from JSON storage (v0.8.0 - supports both new and legacy formats)."""
        try:
            file_ids = set()

            # Find new JSON format files
            for f in self.documents_dir.glob("*.json"):
                # Exclude graph, chunk, and ID files (v0.8.0 audit fix - Issue 2)
                # ID files have format: {doc_id}_chunks.ids.json (stem = {doc_id}_chunks.ids)
                if (
                    not f.stem.endswith("_graph")
                    and not f.stem.endswith("_chunks")
                    and not f.stem.endswith("_chunks.ids")
                ):
                    file_ids.add(f.stem)

            # Find legacy pickle files (for backward compatibility)
            for f in self.documents_dir.glob("*.pkl"):
                # Exclude graph files
                if not f.stem.endswith("_graph"):
                    file_ids.add(f.stem)

            return list(file_ids)
        except Exception as e:
            logger.error(f"Failed to list JSON documents: {e}")
            return []

    def delete_document(
        self,
        file_id: str,
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> bool:
        """
        Delete document from persistent storage.

        Args:
            file_id: Document identifier

        Returns:
            True if deleted successfully
        """
        internal_file_id = compose_scoped_file_id(
            file_id,
            workspace_id=workspace_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )
        try:
            if self.use_chromadb:
                return self._delete_document_chromadb(internal_file_id)
            else:
                return self._delete_document_json(internal_file_id)
        except Exception as e:
            logger.error(f"Failed to delete document {internal_file_id}: {e}")
            return False

    def _delete_document_chromadb(self, file_id: str) -> bool:
        """Delete document from ChromaDB (v0.8.0 - cleans up all file formats)."""
        collection_name = f"doc_{file_id}".replace("-", "_").replace(".", "_")

        try:
            self.chroma_client.delete_collection(name=collection_name)

            # Delete all graph file formats (new JSON + numpy and legacy pickle)
            files_to_delete = [
                self.documents_dir / f"{file_id}_graph.json",
                self.documents_dir / f"{file_id}_graph.pkl",
                self.documents_dir / f"{file_id}_embeddings.npz",
            ]
            for f in files_to_delete:
                if f.exists():
                    f.unlink()

            logger.info(f"[OK] Deleted document {file_id} from ChromaDB")
            return True

        except Exception as e:
            logger.error(f"ChromaDB delete failed for {file_id}: {e}")
            return False

    def _delete_document_json(self, file_id: str) -> bool:
        """Delete document from JSON storage (v0.8.0 - cleans up all file formats)."""
        deleted = False

        try:
            # Delete all possible file formats (new and legacy)
            # v0.8.0 audit fix - Issue 2: Also delete IDs JSON file
            files_to_delete = [
                self.documents_dir / f"{file_id}.json",
                self.documents_dir / f"{file_id}.pkl",
                self.documents_dir / f"{file_id}_chunks.npz",
                self.documents_dir / f"{file_id}_chunks.ids.json",  # v0.8.0: IDs stored as JSON
                self.documents_dir / f"{file_id}_graph.json",
                self.documents_dir / f"{file_id}_graph.pkl",
                self.documents_dir / f"{file_id}_embeddings.npz",
            ]

            for f in files_to_delete:
                if f.exists():
                    f.unlink()
                    deleted = True

            if deleted:
                logger.info(f"[OK] Deleted document {file_id} from JSON storage")

            return deleted

        except Exception as e:
            logger.error(f"JSON delete failed for {file_id}: {e}")
            return False

    # =========================================================================
    # AFM Dialogue History Persistence
    # =========================================================================

    def _serialize_message_safe(self, msg: Any) -> Dict[str, Any]:
        """Serialize a Message object to JSON-safe format (v0.8.0 - CWE-502 fix)."""
        # Handle ImportanceLevel enum - store as string value
        importance_val = msg.importance
        if hasattr(importance_val, "value"):
            importance_val = importance_val.value  # Enum.value -> "critical", "relevant", etc.

        # Handle FidelityLevel enum - store as string value
        fidelity_val = getattr(msg, "intended_fidelity", None)
        if fidelity_val is not None and hasattr(fidelity_val, "value"):
            fidelity_val = fidelity_val.value

        safe_msg = {
            "role": msg.role,
            "content": msg.content,
            "importance": importance_val,
            "turn_index": msg.turn_index,
            "timestamp": getattr(msg, "timestamp", 0.0),
            "message_id": getattr(msg, "message_id", None),
            "relevance_score": getattr(msg, "relevance_score", 0.0),
            "intended_fidelity": fidelity_val,
            "compressed_summary": getattr(msg, "compressed_summary", None),
            "placeholder_stub": getattr(msg, "placeholder_stub", None),
        }
        # Handle embedding
        if hasattr(msg, "embedding") and msg.embedding is not None:
            if isinstance(msg.embedding, np.ndarray):
                safe_msg["embedding"] = msg.embedding.tolist()
            else:
                safe_msg["embedding"] = list(msg.embedding)
        return safe_msg

    def _deserialize_message_safe(self, msg_data: Dict[str, Any]) -> Any:
        """Deserialize JSON message back to Message object (v0.8.0 - CWE-502 fix)."""
        from .afm import Message, ImportanceLevel, FidelityLevel

        embedding = None
        if "embedding" in msg_data and msg_data["embedding"] is not None:
            embedding = np.array(msg_data["embedding"])

        # Reconstruct ImportanceLevel enum from string value
        importance_val = msg_data.get("importance", "trivial")
        if isinstance(importance_val, str):
            # Map string values to ImportanceLevel enum
            importance_map = {
                "critical": ImportanceLevel.CRITICAL,
                "relevant": ImportanceLevel.RELEVANT,
                "trivial": ImportanceLevel.TRIVIAL,
            }
            importance_val = importance_map.get(importance_val, ImportanceLevel.TRIVIAL)

        # Reconstruct FidelityLevel enum from string value
        fidelity_val = msg_data.get("intended_fidelity", "placeholder")
        if isinstance(fidelity_val, str):
            fidelity_map = {
                "full": FidelityLevel.FULL,
                "compressed": FidelityLevel.COMPRESSED,
                "placeholder": FidelityLevel.PLACEHOLDER,
            }
            fidelity_val = fidelity_map.get(fidelity_val, FidelityLevel.PLACEHOLDER)

        return Message(
            role=msg_data["role"],
            content=msg_data["content"],
            importance=importance_val,
            turn_index=msg_data["turn_index"],
            timestamp=msg_data.get("timestamp", 0.0),
            message_id=msg_data.get("message_id"),
            embedding=embedding,
            relevance_score=msg_data.get("relevance_score", 0.0),
            intended_fidelity=fidelity_val,
            compressed_summary=msg_data.get("compressed_summary"),
            placeholder_stub=msg_data.get("placeholder_stub"),
        )

    def save_afm_history(
        self,
        session_id: str,
        messages: List[Any],
        turn_counter: int,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """
        Save AFM dialogue history using safe JSON format (v0.8.0 - CWE-502 fix).

        Args:
            session_id: Session identifier
            messages: List of Message objects
            turn_counter: Current turn counter
            metadata: Optional session metadata

        Returns:
            True if saved successfully
        """
        history_file = self.afm_dir / f"{session_id}.json"
        embeddings_file = self.afm_dir / f"{session_id}_embeddings.npz"

        try:
            # Serialize messages to JSON-safe format
            safe_messages = [self._serialize_message_safe(msg) for msg in messages]

            # Extract embeddings for numpy storage
            embeddings_list = []
            for i, msg_data in enumerate(safe_messages):
                if "embedding" in msg_data and msg_data["embedding"]:
                    embeddings_list.append((i, msg_data.pop("embedding")))

            data = {
                "_format_version": PERSISTENCE_FORMAT_VERSION,
                "session_id": session_id,
                "messages": safe_messages,
                "turn_counter": turn_counter,
                "metadata": metadata or {},
            }

            # Atomic save: JSON structure
            self._atomic_write_json(history_file, data)

            # Atomic save: embeddings
            if embeddings_list:
                indices = np.array([e[0] for e in embeddings_list])
                embeddings = np.array([e[1] for e in embeddings_list])
                self._atomic_write_npz(embeddings_file, indices=indices, embeddings=embeddings)

            logger.info(f"[OK] Saved AFM history {session_id} ({len(messages)} messages)")
            return True

        except Exception as e:
            logger.error(f"Failed to save AFM history {session_id}: {e}")
            return False

    def load_afm_history(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load AFM dialogue history (v0.8.0 - supports safe JSON and legacy pickle).

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with messages, turn_counter, metadata, or None
        """
        json_file = self.afm_dir / f"{session_id}.json"
        embeddings_file = self.afm_dir / f"{session_id}_embeddings.npz"
        legacy_pickle = self.afm_dir / f"{session_id}.pkl"

        try:
            # Try new safe JSON format first
            if json_file.exists():
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Load embeddings if available
                if embeddings_file.exists():
                    npz_data = np.load(embeddings_file, allow_pickle=False)
                    indices = npz_data["indices"]
                    embeddings = npz_data["embeddings"]

                    # Attach embeddings back to messages
                    for idx, emb in zip(indices, embeddings):
                        if idx < len(data["messages"]):
                            data["messages"][idx]["embedding"] = emb.tolist()

                # Deserialize messages to Message objects
                data["messages"] = [self._deserialize_message_safe(msg) for msg in data["messages"]]

                logger.info(
                    f"[OK] Loaded AFM history {session_id} ({len(data['messages'])} messages)"
                )
                return data

            # v0.8.0 SECURITY FIX: Fail hard on legacy pickle files (CWE-502)
            # pickle.load() can execute arbitrary code, so we refuse to load these
            elif legacy_pickle.exists():
                raise ValueError(
                    f"SECURITY: Legacy pickle file detected for AFM session '{session_id}'.\n"
                    f"  File: {legacy_pickle}\n"
                    f"\n"
                    f"Pickle files are a security risk (CWE-502: arbitrary code execution).\n"
                    f"Please run the migration script to convert to safe format:\n"
                    f"\n"
                    f"  python scripts/migrate_pickle.py\n"
                    f"\n"
                    f"Or delete the legacy file and restart the session."
                )

            return None

        except json.JSONDecodeError as e:
            # Handle corrupted JSON files gracefully (not a security issue)
            logger.warning(f"Corrupted JSON in AFM history {session_id}: {e}")
            return None
        except ValueError:
            # Re-raise security-related ValueError (legacy pickle rejection)
            raise
        except Exception as e:
            logger.error(f"Failed to load AFM history {session_id}: {e}")
            return None

    def list_afm_sessions(self) -> List[str]:
        """
        List all AFM session IDs (v0.8.0 - supports both new and legacy formats).

        Returns:
            List of session_ids
        """
        try:
            session_ids = set()

            # Find new JSON format files
            for f in self.afm_dir.glob("*.json"):
                session_ids.add(f.stem)

            # Find legacy pickle files
            for f in self.afm_dir.glob("*.pkl"):
                session_ids.add(f.stem)

            return list(session_ids)
        except Exception as e:
            logger.error(f"Failed to list AFM sessions: {e}")
            return []

    def delete_afm_history(self, session_id: str) -> bool:
        """
        Delete AFM dialogue history (v0.8.0 - cleans up all file formats).

        Args:
            session_id: Session identifier

        Returns:
            True if deleted successfully
        """
        deleted = False

        try:
            files_to_delete = [
                self.afm_dir / f"{session_id}.json",
                self.afm_dir / f"{session_id}.pkl",
                self.afm_dir / f"{session_id}_embeddings.npz",
            ]

            for f in files_to_delete:
                if f.exists():
                    f.unlink()
                    deleted = True

            if deleted:
                logger.info(f"[OK] Deleted AFM history {session_id}")

            return deleted

        except Exception as e:
            logger.error(f"Failed to delete AFM history {session_id}: {e}")
            return False

    # =========================================================================
    # File Sync Metadata Persistence (NEW in v0.4.0)
    # =========================================================================

    def save_file_sync_metadata(self, metadata_dict: Dict[str, Dict]) -> bool:
        """
        Save file sync metadata.

        Args:
            metadata_dict: Dictionary of file_id -> metadata

        Returns:
            True if saved successfully
        """
        sync_file = self.storage_dir / "file_sync_metadata.json"

        try:
            with open(sync_file, "w", encoding="utf-8") as f:
                json.dump(metadata_dict, f, indent=2)

            logger.info(f"[OK] Saved file sync metadata ({len(metadata_dict)} files)")
            return True

        except Exception as e:
            logger.error(f"Failed to save file sync metadata: {e}")
            return False

    def load_file_sync_metadata(self) -> Optional[Dict[str, Dict]]:
        """
        Load file sync metadata.

        Returns:
            Dictionary of file_id -> metadata, or None if not found
        """
        sync_file = self.storage_dir / "file_sync_metadata.json"

        if not sync_file.exists():
            return None

        try:
            with open(sync_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            logger.info(f"[OK] Loaded file sync metadata ({len(data)} files)")
            return data

        except Exception as e:
            logger.error(f"Failed to load file sync metadata: {e}")
            return None

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dictionary with storage stats
        """
        stats = {
            "storage_dir": str(self.storage_dir),
            "backend": "ChromaDB" if self.use_chromadb else "JSON/Pickle",
            "chromadb_available": CHROMADB_AVAILABLE,
            "documents_count": len(self.list_documents()),
            "afm_sessions_count": len(self.list_afm_sessions()),
        }

        # Calculate disk usage
        try:
            total_size = sum(f.stat().st_size for f in self.storage_dir.rglob("*") if f.is_file())
            stats["disk_usage_mb"] = total_size / (1024 * 1024)
        except Exception as e:
            logger.error(f"Failed to calculate disk usage: {e}")
            stats["disk_usage_mb"] = 0

        return stats

    def clear_all(self) -> bool:
        """
        Clear all persistent storage (DANGEROUS!).

        v0.8.0: Updated to clean up all file formats (JSON, pickle, numpy).

        Returns:
            True if cleared successfully
        """
        try:
            if self.use_chromadb and self.chroma_client:
                self.chroma_client.reset()

            # Clear all document files (new and legacy formats)
            for pattern in ["*.pkl", "*.json", "*.npz"]:
                for f in self.documents_dir.glob(pattern):
                    f.unlink()
                for f in self.afm_dir.glob(pattern):
                    f.unlink()

            logger.warning("[WARN]  Cleared all persistent storage")
            return True

        except Exception as e:
            logger.error(f"Failed to clear storage: {e}")
            return False
