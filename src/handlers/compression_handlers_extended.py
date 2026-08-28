"""Compression handlers: batch, directory, codebase, and extended ops."""

from .compression_handlers_common import (
    chunks_for_file,
    validate_file_id,
    _has_scope_args,
    _resolve_awaitable,
    _scoped_file_id,
    _scope_filtered_duplicates,
)
from .compression_handlers_common import *  # noqa: F403, F401


async def handle_batch_ingest(context: HandlerContext, args: Dict[str, Any]) -> str:
    """
    Handle batch_ingest_documents MCP tool (v0.6.0, rate limiting v0.7.0).

    Ingests multiple documents concurrently with bounded parallelism,
    progress tracking, and error isolation.

    Args:
        context: Server context dict
        args: Tool arguments:
            - documents: List of {file_id, text, metadata} objects
            - max_concurrent: Optional max concurrent ingestions (default 4)

    Returns:
        JSON string with batch results:
        {
            "total": 10,
            "successful": 9,
            "failed": 1,
            "results": [
                {"file_id": "doc1", "success": true, "processing_time": 1.2},
                {"file_id": "doc2", "success": false, "error": "ValueError: text too short"}
            ],
            "summary": "Batch complete: 9/10 succeeded in 12.5s"
        }

    Raises:
        ValueError: If validation fails
        RateLimitExceededError: If rate limit exceeded (v0.7.0)
    """
    # Rate limiting for batch operations (v0.7.0 security hardening)
    try:
        await RATE_LIMITERS["batch_ingest"].acquire(blocking=True)
    except RateLimitExceededError:
        raise ValueError(
            "Rate limit exceeded for batch ingestion. Please retry in a moment.\n"
            "Tip: The server allows ~2 batch operations/second to prevent resource exhaustion."
        )

    from ..batch_manager import BatchCompressionManager, BatchDocument

    # Extract arguments
    documents_list = args.get("documents", [])
    max_concurrent = args.get("max_concurrent", 4)

    # Validate arguments
    if not documents_list:
        raise SmartError.missing_required_field("documents", "batch_ingest_documents")

    if not isinstance(documents_list, list):
        raise ValueError(
            f"documents must be a list, got {type(documents_list).__name__}\n"
            "Tip: Provide a list of objects with 'file_id' and 'text' fields"
        )

    if not (1 <= max_concurrent <= 8):
        raise ValueError(
            f"max_concurrent must be between 1 and 8, got {max_concurrent}\n"
            "Tip: Use 4 for balanced performance, 1 for sequential, 8 for maximum throughput"
        )

    # Validate each document
    batch_documents = []
    for i, doc in enumerate(documents_list):
        if not isinstance(doc, dict):
            raise ValueError(
                f"Document {i} must be an object, got {type(doc).__name__}\n"
                "Tip: Each document should have 'file_id' and 'text' fields"
            )

        file_id = doc.get("file_id")
        text = doc.get("text")
        metadata = doc.get("metadata", {})

        # Validate file_id
        if not file_id:
            raise SmartError.missing_required_field(
                f"documents[{i}].file_id", "batch_ingest_documents"
            )

        if not isinstance(file_id, str):
            raise ValueError(
                f"documents[{i}].file_id must be a string, got {type(file_id).__name__}"
            )

        # Validate text exists (allow empty strings for batch error isolation)
        if text is None:
            raise SmartError.missing_required_field(
                f"documents[{i}].text", "batch_ingest_documents"
            )

        if not isinstance(text, str):
            raise ValueError(f"documents[{i}].text must be a string, got {type(text).__name__}")

        # Scope the file_id (mirrors handle_ingest's contract, and the
        # sibling handle_ingest_directory) so two tenants batch-ingesting the
        # same plain file_id (e.g. "notes") don't collide on the process-wide
        # compressor store keyed by internal scoped id. must_exist=False
        # mirrors handle_ingest's ingest-time validation.
        scoped_file_id = _scoped_file_id(file_id, args)
        validate_file_id(scoped_file_id, context, must_exist=False)

        # Create BatchDocument using the internal scoped id; the caller-visible
        # display id is restored from result.file_id below via display_file_id().
        batch_documents.append(BatchDocument(file_id=scoped_file_id, text=text, metadata=metadata))

    # Log batch operation
    logger.info(
        f"Starting batch ingestion: {len(batch_documents)} documents, "
        f"max_concurrent={max_concurrent}"
    )

    # Create batch manager and execute
    manager = BatchCompressionManager(
        compressor=context["compressor"], max_concurrent=max_concurrent
    )

    import time

    start_time = time.time()
    results = await manager.compress_batch(batch_documents)
    total_time = time.time() - start_time

    # Format results
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful

    result_list = []
    for result in results:
        entry = {
            # result.file_id is the internal scoped id (see BatchDocument
            # construction above) -- unscope it so the caller sees the raw
            # file_id it passed in, never the internal tenant-scoped key.
            "file_id": display_file_id(result.file_id),
            "success": result.success,
            "processing_time": round(result.processing_time, 2),
        }

        if result.success:
            # Include skeleton summary
            if result.result:
                entry["skeleton_preview"] = result.result.skeleton_text[:200] + "..."
                entry["compression_ratio"] = result.result.compression_ratio
        else:
            entry["error"] = result.error

        result_list.append(entry)

    # Create response
    response = {
        "total": len(results),
        "successful": successful,
        "failed": failed,
        "max_concurrent": max_concurrent,
        "total_time": round(total_time, 2),
        "avg_time_per_doc": round(total_time / len(results), 2) if results else 0.0,
        "results": result_list,
        "summary": (
            f"Batch complete: {successful}/{len(results)} succeeded in {total_time:.1f}s "
            f"(avg {total_time/len(results):.2f}s/doc)"
        ),
    }

    if failed > 0:
        failed_ids = [display_file_id(r.file_id) for r in results if not r.success]
        response["failed_file_ids"] = failed_ids
        response["tip"] = (
            f"[WARN] {failed} documents failed. Check error messages for each failed document."
        )

    # v1.34.28 (F12 class-completion): batch_ingest bypasses handle_ingest by
    # calling BatchCompressionManager.compress_batch() directly, so the F12
    # tracker wiring on handle_ingest does NOT cover this path. Record one
    # tracker event per successful result so per-session savings reflect bulk
    # ingest activity. Same lazy-import + swallow pattern.
    try:
        from .token_optimization_handlers import _get_tracker

        _sid = args.get("session_id") or "default"
        _model = args.get("model") or "claude-sonnet-4-6"
        _tracker = _get_tracker(_sid, _model)
        for _r in results:
            if _r.success and _r.result is not None:
                _tracker.record(
                    tool_name="batch_ingest_documents",
                    original_tokens=getattr(_r.result, "total_tokens", 0),
                    compressed_tokens=getattr(_r.result, "skeleton_tokens", 0),
                    model=_model,
                )
    except Exception as exc:
        logger.warning(f"SavingsTracker.record failed for batch_ingest: {exc}")

    logger.info(response["summary"])

    return json.dumps(response, indent=2)


async def handle_ingest_directory(context: HandlerContext, args: Dict[str, Any]) -> str:
    """
    Handle ingest_directory MCP tool (v0.9.0).

    Bulk ingest code files from a directory using glob patterns.
    Uses PathValidator for security (prevents path traversal attacks).
    Leverages BatchCompressionManager for concurrent processing.

    Args:
        context: Server context dict with path_validator and compressor
        args: Tool arguments:
            - directory: Directory path to scan
            - patterns: Glob patterns for files to include (default: ['*.py', '*.js', '*.ts'])
            - exclude_patterns: Patterns to exclude (default: node_modules, __pycache__, venv)
            - max_files: Maximum files to ingest (default: 50, max: 100)
            - max_concurrent: Maximum concurrent ingestions (default: 4, max: 8)

    Returns:
        JSON string with directory ingestion results

    Raises:
        ValueError: If directory is invalid or path traversal detected
    """
    import os
    import time
    from pathlib import Path
    from ..batch_manager import BatchCompressionManager, BatchDocument

    # Extract arguments with defaults
    directory = args.get("directory", "")
    patterns = args.get("patterns", ["*.py", "*.js", "*.ts"])
    exclude_patterns = args.get(
        "exclude_patterns",
        [
            "**/node_modules/**",
            "**/__pycache__/**",
            "**/venv/**",
            "**/.venv/**",
            "**/.git/**",
        ],
    )
    max_files = args.get("max_files", 50)
    max_concurrent = args.get("max_concurrent", 4)

    # Validate directory is provided
    if not directory:
        raise SmartError.missing_required_field("directory", "ingest_directory")

    # Validate and normalize directory path using PathValidator (CWE-22 protection)
    path_validator = context["path_validator"]
    try:
        validated_dir = path_validator.validate(directory)
    except ValueError as e:
        raise ValueError(
            f"Invalid directory path: {e}\n"
            "Tip: Directory must be within the current working directory or user home directory."
        )

    # Check directory exists
    if not os.path.isdir(validated_dir):
        raise ValueError(
            f"Directory not found: {validated_dir}\n" "Tip: Provide an existing directory path."
        )

    # Validate parameters
    if not (1 <= max_files <= 100):
        raise ValueError(
            f"max_files must be between 1 and 100, got {max_files}\n"
            "Tip: Use smaller values for faster ingestion."
        )

    if not (1 <= max_concurrent <= 8):
        raise ValueError(
            f"max_concurrent must be between 1 and 8, got {max_concurrent}\n"
            "Tip: Use 4 for balanced performance."
        )

    # Scan directory for matching files
    logger.info(f"Scanning directory: {validated_dir}")
    logger.info(f"Patterns: {patterns}, Excludes: {exclude_patterns}")

    matched_files = []
    dir_path = Path(validated_dir)

    for pattern in patterns:
        # Use recursive glob for patterns with ** and non-recursive otherwise
        if "**" in pattern:
            for file_path in dir_path.rglob(pattern.replace("**/", "")):
                if file_path.is_file():
                    matched_files.append(file_path)
        else:
            # Check in immediate directory and subdirectories
            for file_path in dir_path.rglob(pattern):
                if file_path.is_file():
                    matched_files.append(file_path)

    # Remove duplicates
    matched_files = list(set(matched_files))

    # Apply exclusions
    # P2-1 fix: Use PurePath.match() for proper glob pattern support
    from pathlib import PurePath

    def is_excluded(path: Path) -> bool:
        """Check if path matches any exclusion pattern using proper glob matching.

        PurePath.match() handles both simple patterns (*.pyc) and ** patterns natively.
        Simple patterns like '*.pyc' match from the right, so 'src/file.pyc' matches '*.pyc'.
        """
        path_obj = PurePath(str(path))

        for exclude in exclude_patterns:
            try:
                # PurePath.match() supports ** patterns and simple patterns natively
                # e.g., '*.pyc' matches 'src/file.pyc', '**/node_modules/**' matches nested dirs
                if path_obj.match(exclude):
                    return True
            except ValueError:
                # Invalid pattern - log and skip
                logger.warning(f"Invalid exclude pattern: {exclude}")
                continue
        return False

    filtered_files = [f for f in matched_files if not is_excluded(f)]

    # Validate each file path
    validated_files = []
    for file_path in filtered_files:
        try:
            validated_path = path_validator.validate(str(file_path))
            validated_files.append(Path(validated_path))
        except ValueError:
            logger.warning(f"Skipping file outside allowed directories: {file_path}")
            continue

    # Limit files
    if len(validated_files) > max_files:
        logger.info(f"Limiting from {len(validated_files)} to {max_files} files")
        validated_files = validated_files[:max_files]

    if not validated_files:
        return json.dumps(
            {
                "status": "no_files",
                "directory": validated_dir,
                "patterns": patterns,
                "message": "No matching files found in directory.",
                "tip": "Try different patterns or check the directory path.",
            },
            indent=2,
        )

    # Read files and prepare batch documents
    batch_documents = []
    skipped_files = []

    for file_path in validated_files:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()

            # Generate file_id from relative path
            try:
                rel_path = file_path.relative_to(dir_path)
                file_id = str(rel_path).replace(os.sep, "/")
            except ValueError:
                file_id = file_path.name

            batch_documents.append(
                BatchDocument(
                    file_id=_scoped_file_id(file_id, args),
                    text=text,
                    metadata={
                        "source_path": str(file_path),
                        "file_path": str(file_path),
                        "file_size": len(text),
                    },
                )
            )
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            skipped_files.append({"path": str(file_path), "error": str(e)})

    if not batch_documents:
        return json.dumps(
            {
                "status": "read_failed",
                "directory": validated_dir,
                "skipped_files": skipped_files,
                "message": "Could not read any files.",
            },
            indent=2,
        )

    # Log operation
    logger.info(
        f"Starting directory ingestion: {len(batch_documents)} files, "
        f"max_concurrent={max_concurrent}"
    )

    # Create batch manager and execute
    manager = BatchCompressionManager(
        compressor=context["compressor"], max_concurrent=max_concurrent
    )

    start_time = time.time()
    results = await manager.compress_batch(batch_documents)
    total_time = time.time() - start_time

    # Format results
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful

    result_list = []
    for result in results:
        display_id = display_file_id(result.file_id)
        entry = {
            "file_id": display_id,
            "success": result.success,
            "processing_time": round(result.processing_time, 2),
        }

        if result.success:
            if result.result:
                entry["compression_ratio"] = result.result.compression_ratio
                entry["node_count"] = result.result.total_nodes
        else:
            entry["error"] = result.error

        result_list.append(entry)

    # Create response
    response = {
        "status": "complete",
        "directory": validated_dir,
        "patterns": patterns,
        "total_files_found": len(matched_files),
        "files_after_exclusion": len(filtered_files),
        "files_ingested": len(batch_documents),
        "successful": successful,
        "failed": failed,
        "skipped_read_errors": len(skipped_files),
        "total_time": round(total_time, 2),
        "avg_time_per_file": round(total_time / len(results), 2) if results else 0.0,
        "results": result_list,
        "summary": (
            f"Directory ingestion complete: {successful}/{len(batch_documents)} files "
            f"succeeded in {total_time:.1f}s"
        ),
    }

    if skipped_files:
        response["skipped_files"] = skipped_files

    if failed > 0:
        failed_ids = [r.file_id for r in results if not r.success]
        response["failed_file_ids"] = failed_ids

    # v1.34.28 (F12 class-completion): ingest_directory bypasses handle_ingest
    # (BatchCompressionManager.compress_batch() → compressor.ingest_file_async()
    # directly), so the F12 tracker wiring on handle_ingest does NOT cover this
    # path. Record one event per successful result so per-session savings reflect
    # directory-scale ingest activity.
    try:
        from .token_optimization_handlers import _get_tracker

        _sid = args.get("session_id") or "default"
        _model = args.get("model") or "claude-sonnet-4-6"
        _tracker = _get_tracker(_sid, _model)
        for _r in results:
            if _r.success and _r.result is not None:
                _tracker.record(
                    tool_name="ingest_directory",
                    original_tokens=getattr(_r.result, "total_tokens", 0),
                    compressed_tokens=getattr(_r.result, "skeleton_tokens", 0),
                    model=_model,
                )
    except Exception as exc:
        logger.warning(f"SavingsTracker.record failed for ingest_directory: {exc}")

    logger.info(response["summary"])

    return json.dumps(response, indent=2)


# =========================================================================
# Diff Re-ingestion, Cross-doc Dedup, Preset Tools
# =========================================================================


async def handle_diff_reingest(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Re-ingest a document preserving unchanged chunk embeddings."""
    file_id = args.get("file_id")
    scoped_file_id = _scoped_file_id(file_id, args) if file_id else None
    text = args.get("text")

    if not file_id or not text:
        return json.dumps({"error": "Both 'file_id' and 'text' are required"}, indent=2)

    try:
        compressor = context["compressor"]
        result = await compressor.diff_reingest_async(scoped_file_id, text)

        # Persist updated document to disk (same as handle_ingest)
        try:
            import networkx as nx

            graph_data = nx.node_link_data(compressor.graphs[scoped_file_id], edges="links")
            success = context["persistence"].save_document(
                file_id=scoped_file_id,
                chunks=dict(chunks_for_file(compressor.chunks, scoped_file_id)),
                graph_data=graph_data,
                metadata=compressor.file_metadata.get(scoped_file_id, {}),
            )
            success = await _resolve_awaitable(success)
            if success:
                logger.info(f"[OK] Persisted diff-reingested document {file_id}")
            else:
                logger.warning(f"[WARN] Failed to persist diff-reingested {file_id}")
        except Exception as e:
            logger.error(f"Failed to persist diff-reingested {file_id}: {e}")

        # Save version history
        try:
            checksum = hashlib.md5(text.encode()).hexdigest()
            await context["version_manager"].add_version_async(
                doc_id=scoped_file_id,
                content=text,
                checksum=checksum,
                metadata={},
                compression_stats={
                    "chunks_unchanged": result.chunks_unchanged,
                    "chunks_updated": result.chunks_updated,
                    "chunks_added": result.chunks_added,
                    "chunks_removed": result.chunks_removed,
                },
            )
            logger.info(f"[OK] Registered version for diff-reingested {file_id}")
        except Exception as e:
            logger.warning(f"[WARN] Failed to save version for diff-reingested {file_id}: {e}")

        return json.dumps(
            {
                "status": "success",
                "file_id": file_id,
                "chunks_unchanged": result.chunks_unchanged,
                "chunks_updated": result.chunks_updated,
                "chunks_added": result.chunks_added,
                "chunks_removed": result.chunks_removed,
                "message": (
                    f"Diff re-ingestion complete: {result.chunks_unchanged} unchanged, "
                    f"{result.chunks_updated} updated, {result.chunks_added} added, "
                    f"{result.chunks_removed} removed"
                ),
            },
            indent=2,
        )
    except ValueError as e:
        return json.dumps({"error": str(e)}, indent=2)
    except Exception as e:
        logger.error(f"Diff re-ingestion failed: {e}")
        return json.dumps({"error": f"Diff re-ingestion failed: {e}"}, indent=2)


async def handle_find_duplicates(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Find near-duplicate chunks across different documents."""
    threshold = args.get("threshold", 0.9)
    timeout_seconds = args.get("timeout_seconds", 30.0)

    try:
        compressor = context["compressor"]
        duplicates = compressor.find_duplicates(
            threshold=threshold, timeout_seconds=timeout_seconds
        )
        if _has_scope_args(args):
            duplicates = _scope_filtered_duplicates(duplicates, args)
        return json.dumps(
            {
                "status": "success",
                "duplicate_count": len(duplicates),
                "threshold": threshold,
                "duplicates": duplicates[:50],
                "message": f"Found {len(duplicates)} duplicate pairs above {threshold} similarity",
            },
            indent=2,
        )
    except Exception as e:
        logger.error(f"Duplicate detection failed: {e}")
        return json.dumps({"error": f"Duplicate detection failed: {e}"}, indent=2)


async def handle_get_presets(context: HandlerContext, args: Dict[str, Any]) -> str:
    """List available compression presets."""
    from ..compression_presets import list_presets

    presets = list_presets()
    return json.dumps(
        {
            "status": "success",
            "presets": [p.to_dict() for p in presets],
            "message": f"{len(presets)} compression presets available",
        },
        indent=2,
    )


async def handle_check_context_budget(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Check context budget usage and recommend compression action."""
    current_tokens = args.get("current_tokens")
    if current_tokens is None:
        return json.dumps({"error": "'current_tokens' is required"}, indent=2)
    context_limit = args.get("context_limit", 200_000)

    from ..token_threshold import check_context_budget

    result = check_context_budget(current_tokens, context_limit)
    return json.dumps(result.to_dict(), indent=2)


# =============================================================================
# Phase 5 Handlers — Research-based features (2025 papers)
# =============================================================================


async def handle_prune_by_relevance(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Prune document nodes by query relevance (AttentionRAG)."""
    from ..attention_pruning import prune_by_relevance

    doc_id = args.get("doc_id", "")
    query = args.get("query", "")
    keep_ratio = args.get("keep_ratio", 0.5)

    compressor = context["compressor"]
    if doc_id not in compressor.graphs:
        return json.dumps({"error": f"Document '{doc_id}' not found"})

    # Collect node embeddings for this doc. Boundary-safe match (extract_file_id_from_node)
    # not bare startswith — else file_ids sharing a prefix ('report' vs 'report_archive')
    # cross-contaminate: 'report_archive_n3'.startswith('report') is True. (audit P1-5)
    node_embeddings = {}
    for nid, node in compressor.chunks.items():
        if extract_file_id_from_node(nid) == doc_id:
            node_embeddings[nid] = node.embedding

    if not node_embeddings:
        return json.dumps({"error": "No nodes found for document"})

    # Encode query
    from ..embeddings import EmbeddingManager

    emb_mgr = EmbeddingManager()
    query_emb = emb_mgr.encode([query])[0]

    kept_ids = prune_by_relevance(node_embeddings, query_emb, keep_ratio)

    return json.dumps(
        {
            "doc_id": doc_id,
            "total_nodes": len(node_embeddings),
            "kept_nodes": len(kept_ids),
            "kept_node_ids": kept_ids,
            "compression_ratio": round(len(kept_ids) / len(node_embeddings), 3),
        },
        indent=2,
    )


async def handle_multi_level_skeleton(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Generate 3-tier skeleton: headline, summary, full (Squeezed Attention)."""
    from ..multi_level_skeleton import generate_multi_level_skeleton

    doc_id = args.get("doc_id", "")
    compressor = context["compressor"]

    if doc_id not in compressor.graphs:
        return json.dumps({"error": f"Document '{doc_id}' not found"})

    # Build node list with importance scores. Boundary-safe match (see prune_by_relevance;
    # bare startswith cross-contaminates prefix-sharing file_ids — audit P1-5).
    nodes = []
    for nid, node in compressor.chunks.items():
        if extract_file_id_from_node(nid) == doc_id:
            nodes.append(
                {
                    "node_id": nid,
                    "text": node.text,
                    "importance": node.importance,
                }
            )

    result = generate_multi_level_skeleton(nodes)
    return json.dumps(
        {
            "doc_id": doc_id,
            "levels": result,
        },
        indent=2,
    )


async def handle_evict_stale(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Evict stale documents based on access recency (DynamicKV/ACON)."""
    max_age_hours = args.get("max_age_hours", 1.0)
    max_age_seconds = max_age_hours * 3600

    compressor = context["compressor"]
    tracker = getattr(compressor, "_access_tracker", None)

    if tracker is None:
        return json.dumps(
            {
                "evicted": [],
                "message": "Access tracking not enabled. Access documents first.",
            }
        )

    stale_ids = tracker.find_stale(max_age_seconds=max_age_seconds)

    evicted = []
    for doc_id in stale_ids:
        if doc_id in compressor.graphs:
            evicted.append(doc_id)

    return json.dumps(
        {
            "stale_documents": stale_ids,
            "evictable": evicted,
            "max_age_hours": max_age_hours,
        },
        indent=2,
    )


async def handle_advise_context(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Analyze context and recommend optimal strategy (MCP Best Practices)."""
    from ..context_advisor import advise_context

    compressor = context["compressor"]

    doc_stats = []
    for file_id, graph in compressor.graphs.items():
        total_tokens = sum(
            compressor.chunks[nid].metadata.get("tokens", 0)
            for nid in graph.nodes
            if nid in compressor.chunks
        )
        avg_importance = 0.0
        if graph.nodes:
            avg_importance = sum(
                compressor.chunks[nid].importance for nid in graph.nodes if nid in compressor.chunks
            ) / len(graph.nodes)

        doc_stats.append(
            {
                "doc_id": file_id,
                "tokens": total_tokens,
                "importance": round(avg_importance, 3),
            }
        )

    advice = advise_context(doc_stats)
    return json.dumps(advice, indent=2)


async def handle_get_compression_insights(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Get compression replay insights (ACON)."""
    compressor = context["compressor"]
    replay_log = getattr(compressor, "_compression_replay", None)

    if replay_log is None:
        return json.dumps({"insights": {}, "message": "No compression history recorded yet."})

    insights = replay_log.get_insights()
    return json.dumps({"insights": insights}, indent=2)


async def handle_generate_rewrite_prompt(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Generate rewrite prompt for client-side LLM compression (SCOPE)."""
    from ..generative_rewrite import generate_rewrite_prompt

    text = args.get("text", "")
    target_ratio = args.get("target_ratio", 0.5)
    preserve_keywords = args.get("preserve_keywords", [])

    if not text:
        doc_id = args.get("doc_id", "")
        compressor = context["compressor"]
        if doc_id in compressor.graphs:
            text = " ".join(
                compressor.chunks[nid].text
                for nid in compressor.graphs[doc_id].nodes
                if nid in compressor.chunks
            )

    prompt = generate_rewrite_prompt(text, target_ratio, preserve_keywords or None)
    return json.dumps(prompt, indent=2)


async def handle_compress_codebase(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Handle compress_codebase tool call.

    Uses tensor-grep AST analysis when available; falls back to a plain
    directory glob when tensor-grep is not installed.
    """
    import glob
    from pathlib import Path as _Path

    from ..tensor_grep_integration import code_search, get_repo_map, is_available

    directory = args.get("directory", ".")
    query = args.get("query")
    max_files = int(args.get("max_files", 50))

    # CWE-22 (2026-08-26 audit HIGH): confine `directory` to cwd+home before it
    # reaches the tg subprocess or glob — mirrors handle_ingest_directory. Without
    # this, an MCP client could enumerate any server-readable dir (e.g. "/etc").
    path_validator = context.get("path_validator")
    if path_validator is not None:
        try:
            directory = path_validator.validate(directory)
        except ValueError as e:
            return json.dumps(
                {
                    "status": "error",
                    "error": f"Invalid directory path: {e}",
                    "message": "directory must be within the current working "
                    "directory or user home directory.",
                }
            )
    else:
        # Fail-open is observable, not silent: the hosted server always injects
        # path_validator (contract-enforced key), so a None here means a hand-rolled
        # context — warn so a future lightweight-context refactor can't silently
        # regress CWE-22 confinement (matches handle_should_compress precedent).
        logger.warning(
            "path_validator absent from context; directory not confined (CWE-22 guard skipped)"
        )

    tg_available = is_available()
    result: Dict[str, Any] = {
        "status": "success",
        "tensor_grep_available": tg_available,
        "directory": directory,
    }

    if tg_available:
        repo_map = get_repo_map(directory)
        result["files_found"] = len(repo_map.files)
        result["symbols_found"] = len(repo_map.symbols)
        files = repo_map.files[:max_files]
        if query:
            search = code_search(query, directory)
            matched_files = list({m.get("file", "") for m in search.matches})
            result["query_matched_files"] = len(matched_files)
            files = matched_files[:max_files] if matched_files else files
        result["selected_files"] = files
    else:
        # Fallback: list directory using glob patterns
        py_files = glob.glob(str(_Path(directory) / "**/*.py"), recursive=True)
        js_files = glob.glob(str(_Path(directory) / "**/*.js"), recursive=True)
        ts_files = glob.glob(str(_Path(directory) / "**/*.ts"), recursive=True)
        files = sorted(set(py_files + js_files + ts_files))[:max_files]
        result["files_found"] = len(files)
        result["selected_files"] = files

    result["message"] = (
        f"Found {len(result.get('selected_files', []))} files. "
        "Use ingest_context on individual files to compress them."
    )
    return json.dumps(result, default=str)


async def handle_search_code(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Handle search_code tool call.

    Fast regex or literal code search using tensor-grep trigram index.
    Falls back gracefully if tensor-grep is not installed.
    """
    from ..tensor_grep_integration import code_search, is_available

    pattern = args.get("pattern", "")
    directory = args.get("directory", ".")

    # CWE-22 (2026-08-26 audit HIGH): confine `directory` before it reaches the
    # tg subprocess — same guard as handle_compress_codebase / ingest_directory.
    path_validator = context.get("path_validator")
    if path_validator is not None:
        try:
            directory = path_validator.validate(directory)
        except ValueError as e:
            return json.dumps(
                {
                    "status": "error",
                    "error": f"Invalid directory path: {e}",
                    "message": "directory must be within the current working "
                    "directory or user home directory.",
                }
            )
    else:
        # Fail-open is observable, not silent: the hosted server always injects
        # path_validator (contract-enforced key), so a None here means a hand-rolled
        # context — warn so a future lightweight-context refactor can't silently
        # regress CWE-22 confinement (matches handle_should_compress precedent).
        logger.warning(
            "path_validator absent from context; directory not confined (CWE-22 guard skipped)"
        )

    if not is_available():
        return json.dumps(
            {
                "status": "fallback",
                "message": "tensor-grep not installed. Install with: pip install tensor-grep",
                "tensor_grep_available": False,
            }
        )

    result = code_search(pattern, directory)
    return json.dumps(
        {
            "status": "success",
            "pattern": result.pattern,
            "total_matches": result.total_matches,
            "matches": result.matches[:50],
            "tensor_grep_available": result.available,
        }
    )
