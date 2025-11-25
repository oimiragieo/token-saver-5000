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
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

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
                logger.info(f"✅ ChromaDB initialized at {self.storage_dir}/chromadb")
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

    # =========================================================================
    # Document Persistence
    # =========================================================================

    def save_document(
        self,
        file_id: str,
        chunks: Dict[str, Any],
        graph_data: Dict[str, Any],
        metadata: Dict[str, Any],
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
        try:
            if self.use_chromadb:
                return self._save_document_chromadb(file_id, chunks, graph_data, metadata)
            else:
                return self._save_document_json(file_id, chunks, graph_data, metadata)
        except Exception as e:
            logger.error(f"Failed to save document {file_id}: {e}", exc_info=True)
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

            # Save graph structure separately (ChromaDB doesn't store graphs)
            graph_file = self.documents_dir / f"{file_id}_graph.pkl"
            with open(graph_file, "wb") as f:
                pickle.dump(graph_data, f)

            logger.info(f"✅ Saved document {file_id} to ChromaDB ({len(chunks)} nodes)")
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
        """Save document using JSON/pickle fallback."""
        doc_file = self.documents_dir / f"{file_id}.pkl"

        try:
            data = {
                "file_id": file_id,
                "chunks": chunks,
                "graph_data": graph_data,
                "metadata": metadata,
            }

            with open(doc_file, "wb") as f:
                pickle.dump(data, f)

            logger.info(f"✅ Saved document {file_id} to JSON ({len(chunks)} nodes)")
            return True

        except Exception as e:
            logger.error(f"JSON save failed for {file_id}: {e}")
            return False

    def load_document(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Load document from persistent storage.

        Args:
            file_id: Document identifier

        Returns:
            Dictionary with chunks, graph_data, metadata, or None if not found
        """
        try:
            if self.use_chromadb:
                return self._load_document_chromadb(file_id)
            else:
                return self._load_document_json(file_id)
        except Exception as e:
            logger.error(f"Failed to load document {file_id}: {e}", exc_info=True)
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

            # Load graph structure
            graph_file = self.documents_dir / f"{file_id}_graph.pkl"
            with open(graph_file, "rb") as f:
                graph_data = pickle.load(f)

            logger.info(f"✅ Loaded document {file_id} from ChromaDB ({len(chunks)} nodes)")

            return {
                "chunks": chunks,
                "graph_data": graph_data,
                "metadata": collection.metadata,
            }

        except Exception as e:
            logger.debug(f"ChromaDB load failed for {file_id}: {e}")
            return None

    def _load_document_json(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Load document from JSON/pickle fallback."""
        doc_file = self.documents_dir / f"{file_id}.pkl"

        if not doc_file.exists():
            return None

        try:
            with open(doc_file, "rb") as f:
                data = pickle.load(f)

            logger.info(f"✅ Loaded document {file_id} from JSON ({len(data['chunks'])} nodes)")
            return data

        except Exception as e:
            logger.error(f"JSON load failed for {file_id}: {e}")
            return None

    def list_documents(self) -> List[str]:
        """
        List all persisted document IDs.

        Returns:
            List of file_ids
        """
        if self.use_chromadb:
            return self._list_documents_chromadb()
        else:
            return self._list_documents_json()

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
        """List documents from JSON fallback."""
        try:
            files = list(self.documents_dir.glob("*.pkl"))
            # Exclude graph files
            doc_files = [f for f in files if not f.stem.endswith("_graph")]
            return [f.stem for f in doc_files]
        except Exception as e:
            logger.error(f"Failed to list JSON documents: {e}")
            return []

    def delete_document(self, file_id: str) -> bool:
        """
        Delete document from persistent storage.

        Args:
            file_id: Document identifier

        Returns:
            True if deleted successfully
        """
        try:
            if self.use_chromadb:
                return self._delete_document_chromadb(file_id)
            else:
                return self._delete_document_json(file_id)
        except Exception as e:
            logger.error(f"Failed to delete document {file_id}: {e}")
            return False

    def _delete_document_chromadb(self, file_id: str) -> bool:
        """Delete document from ChromaDB."""
        collection_name = f"doc_{file_id}".replace("-", "_").replace(".", "_")

        try:
            self.chroma_client.delete_collection(name=collection_name)

            # Delete graph file
            graph_file = self.documents_dir / f"{file_id}_graph.pkl"
            if graph_file.exists():
                graph_file.unlink()

            logger.info(f"✅ Deleted document {file_id} from ChromaDB")
            return True

        except Exception as e:
            logger.error(f"ChromaDB delete failed for {file_id}: {e}")
            return False

    def _delete_document_json(self, file_id: str) -> bool:
        """Delete document from JSON fallback."""
        doc_file = self.documents_dir / f"{file_id}.pkl"

        try:
            if doc_file.exists():
                doc_file.unlink()
                logger.info(f"✅ Deleted document {file_id} from JSON")
                return True
            return False

        except Exception as e:
            logger.error(f"JSON delete failed for {file_id}: {e}")
            return False

    # =========================================================================
    # AFM Dialogue History Persistence
    # =========================================================================

    def save_afm_history(
        self,
        session_id: str,
        messages: List[Any],
        turn_counter: int,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """
        Save AFM dialogue history.

        Args:
            session_id: Session identifier
            messages: List of Message objects
            turn_counter: Current turn counter
            metadata: Optional session metadata

        Returns:
            True if saved successfully
        """
        history_file = self.afm_dir / f"{session_id}.pkl"

        try:
            data = {
                "session_id": session_id,
                "messages": messages,
                "turn_counter": turn_counter,
                "metadata": metadata or {},
            }

            with open(history_file, "wb") as f:
                pickle.dump(data, f)

            logger.info(f"✅ Saved AFM history {session_id} ({len(messages)} messages)")
            return True

        except Exception as e:
            logger.error(f"Failed to save AFM history {session_id}: {e}")
            return False

    def load_afm_history(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load AFM dialogue history.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with messages, turn_counter, metadata, or None
        """
        history_file = self.afm_dir / f"{session_id}.pkl"

        if not history_file.exists():
            return None

        try:
            with open(history_file, "rb") as f:
                data = pickle.load(f)

            logger.info(f"✅ Loaded AFM history {session_id} ({len(data['messages'])} messages)")
            return data

        except Exception as e:
            logger.error(f"Failed to load AFM history {session_id}: {e}")
            return None

    def list_afm_sessions(self) -> List[str]:
        """
        List all AFM session IDs.

        Returns:
            List of session_ids
        """
        try:
            files = list(self.afm_dir.glob("*.pkl"))
            return [f.stem for f in files]
        except Exception as e:
            logger.error(f"Failed to list AFM sessions: {e}")
            return []

    def delete_afm_history(self, session_id: str) -> bool:
        """
        Delete AFM dialogue history.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted successfully
        """
        history_file = self.afm_dir / f"{session_id}.pkl"

        try:
            if history_file.exists():
                history_file.unlink()
                logger.info(f"✅ Deleted AFM history {session_id}")
                return True
            return False

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

            logger.info(f"✅ Saved file sync metadata ({len(metadata_dict)} files)")
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

            logger.info(f"✅ Loaded file sync metadata ({len(data)} files)")
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

        Returns:
            True if cleared successfully
        """
        try:
            if self.use_chromadb and self.chroma_client:
                self.chroma_client.reset()

            # Clear JSON files
            for f in self.documents_dir.glob("*.pkl"):
                f.unlink()
            for f in self.afm_dir.glob("*.pkl"):
                f.unlink()

            logger.warning("⚠️  Cleared all persistent storage")
            return True

        except Exception as e:
            logger.error(f"Failed to clear storage: {e}")
            return False
