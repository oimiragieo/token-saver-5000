"""Compression handlers: ingest, read, search, manage documents."""

from .. import constants
from .compression_handlers_common import (
    chunks_for_file,
    classify_estimate_accuracy,
    resolve_anchored_node_ids,
    run_read_skeleton_pipeline,
    _call_explicit_optional_method,
    _compressor_temporal_graph,
    _generate_skeleton_with_optional_filters,
    _has_scope_args,
    _is_structured_markdown,
    _resolve_awaitable,
    _resolve_chunking_strategy,
    _scoped_file_id,
    _scoped_global_stats,
    _scope_filtered_file_ids,
    _scope_filtered_results,
    _scope_kwargs,
    _scope_label,
    _SMALL_INPUT_TOKEN_THRESHOLD,
    _temporal_excluded_node_ids,
    _temporal_filter_search_results,
    _temporal_graph,
)
from .compression_handlers_common import *  # noqa: F403, F401


def _f11_ranker_path() -> str:
    from ..constants import F11_RANKER_PATH

    return F11_RANKER_PATH


async def handle_ingest(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Handle ingest_context tool call.

    Args:
        context: Server context dict with compressor, persistence, resource_manager, etc.
        args: Tool arguments containing text, file_id, optional file_path and metadata

    Returns:
        JSON string with ingestion results including compression stats

    Raises:
        ValueError: If validation fails
        RuntimeError: If ingestion fails
        RateLimitExceededError: If rate limit exceeded (v0.7.0)
    """
    # Rate limiting (v0.7.0 security hardening)
    try:
        await RATE_LIMITERS["ingest"].acquire(blocking=True)
    except RateLimitExceededError:
        raise ValueError(
            "Rate limit exceeded for document ingestion. Please retry in a moment.\n"
            "Tip: The server allows ~10 ingestions/second to prevent resource exhaustion."
        )

    # --- Resolve text from inline 'text' or remote 'file_url' ---
    text_arg: str | None = args.get("text")
    file_url: str | None = args.get("file_url")
    source_url: str | None = None  # stamped on response when file_url is used

    if text_arg is not None and file_url is not None:
        raise ValueError(
            "'text' and 'file_url' are mutually exclusive — provide only one.\n"
            "Tip: Use 'text' for inline content or 'file_url' to fetch from a remote HTTPS URL."
        )

    if file_url is not None:
        try:
            text_arg = await fetch_url(file_url)
            source_url = file_url
            logger.info(f"Fetched {len(text_arg):,} chars from {file_url}")
        except URLFetchError as exc:
            raise ValueError(
                f"Failed to fetch file_url '{file_url}': {exc}\n" f"Error code: {exc.code}"
            ) from exc

    if text_arg is None:
        raise ValueError(
            "Either 'text' or 'file_url' is required.\n"
            "Tip: Provide inline document content via 'text', "
            "or a remote HTTPS URL via 'file_url'."
        )

    text: str = text_arg

    file_id = args["file_id"]
    file_path = args.get("file_path")  # Optional file path for sync tracking
    metadata = args.get("metadata")
    scoped_file_id = _scoped_file_id(file_id, args)

    # 2026-07-06 knob-honesty fix (architecture plan Move 5): the schema has
    # long advertised `skeleton_ratio` but nothing ever read it — an agent
    # asking for more compression got a silent no-op (same bug class as the
    # activation-honesty fix). Validate it here; `set_file_skeleton_ratio`
    # below is what actually makes it take effect.
    skeleton_ratio_arg = args.get("skeleton_ratio")
    if skeleton_ratio_arg is not None and skeleton_ratio_arg != "auto":
        _is_number = isinstance(skeleton_ratio_arg, (int, float)) and not isinstance(
            skeleton_ratio_arg, bool
        )
        if not _is_number or not (0.0 < skeleton_ratio_arg <= 1.0):
            raise ValueError(
                f"Invalid skeleton_ratio: {skeleton_ratio_arg!r}\n"
                "Tip: skeleton_ratio must be a number in (0.0, 1.0], or the string 'auto'."
            )

    # Text content length validation (v0.7.0 security hardening)
    text_bytes = len(text.encode("utf-8"))
    if text_bytes > MAX_TEXT_LENGTH_BYTES:
        raise ValueError(
            f"Text content too large: {text_bytes:,} bytes (max: {MAX_TEXT_LENGTH_BYTES:,})\n"
            "Tip: Split large documents into smaller chunks before ingestion."
        )

    # SECURITY: Validate file_path to prevent path traversal (CWE-22)
    if file_path:
        try:
            # PathValidator resolves .., symlinks, and validates against allowed directories
            file_path = context["path_validator"].validate(file_path)
            logger.info(f"File path validated: {file_path}")
        except ValueError as e:
            raise ValueError(
                f"Invalid file_path: {str(e)}\n"
                "[TIP] Security: File paths must be within allowed directories to prevent path traversal attacks"
            ) from e

    # Validation
    if not text or len(text.strip()) == 0:
        raise ValueError(
            "text cannot be empty\n"
            "Tip: Provide document content to ingest (minimum ~20 characters recommended)"
        )

    if len(text) < 20:
        raise ValueError(
            f"text is too short ({len(text)} chars)\n"
            "Tip: Provide at least 20 characters for meaningful semantic analysis"
        )

    validate_file_id(scoped_file_id, context, must_exist=False)

    # Check resource limits BEFORE ingestion
    # v0.8.0 audit fix: use async wrapper to avoid blocking event loop
    text_size = len(text.encode("utf-8"))
    allowed, error_msg = await context["resource_manager"].check_document_size_async(
        scoped_file_id, text_size
    )
    if not allowed:
        raise ValueError(error_msg)

    logger.info(
        f"Ingesting document: {scoped_file_id} ({len(text)} chars, {text_size / 1024:.1f}KB)"
    )

    # NEW v0.4.1: Provide compression estimate before actual compression
    advisor = CompressionAdvisor()
    # 2026-07-06: mirror the real ratio the engine will use instead of always
    # previewing at the hardcoded 0.2 — a caller-supplied skeleton_ratio (or
    # the "auto"/unset adaptive default) now shapes the estimate too.
    _estimate_ratio = skeleton_ratio_arg
    if _estimate_ratio is None or _estimate_ratio == "auto":
        _estimate_ratio = compute_adaptive_ratio(len(text.split()))
    estimate = advisor.estimate_compression(text, skeleton_ratio=_estimate_ratio)
    logger.info(
        f"Compression estimate: {estimate.compression_ratio:.1f}× "
        f"({estimate.original_tokens} → ~{estimate.estimated_compressed} tokens)"
    )

    try:
        chunking_strategy, chunking_strategy_used = _resolve_chunking_strategy(args, text)
        # Thread the (validated) caller-requested ratio through as a
        # per-document override BEFORE ingest — the baseline skeleton cached
        # for read_skeleton's cache_stable_prefix is generated INSIDE
        # ingest_file_async, so the override must already be in place.
        context["compressor"].set_file_skeleton_ratio(scoped_file_id, skeleton_ratio_arg)
        skeleton = await context["compressor"].ingest_file_async(
            text, scoped_file_id, metadata, chunking_strategy=chunking_strategy
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to ingest document: {str(e)}\n"
            "Tip: Check that text is valid and file_id contains only alphanumeric and underscores"
        ) from e

    # Register with resource manager
    # v0.8.0 audit fix: use async wrapper to avoid blocking event loop
    await context["resource_manager"].register_document_async(scoped_file_id, text_size)

    # Persist to storage
    try:
        import networkx as nx

        graph_data = nx.node_link_data(context["compressor"].graphs[scoped_file_id], edges="links")
        success = context["persistence"].save_document(
            file_id=scoped_file_id,
            chunks=dict(chunks_for_file(context["compressor"].chunks, scoped_file_id)),
            graph_data=graph_data,
            metadata=context["compressor"].file_metadata.get(scoped_file_id, {}),
        )
        success = await _resolve_awaitable(success)
        if success:
            logger.info(f"[OK] Persisted document {file_id}")
        else:
            logger.warning(f"[WARN]  Failed to persist {file_id}, will be lost on restart")
    except Exception as e:
        logger.error(f"Failed to persist {file_id}: {e}")

    # NEW: Register with file sync manager and version manager
    checksum = hashlib.md5(text.encode()).hexdigest()
    context["sync_manager"].register_file(scoped_file_id, file_path, text)
    try:
        # v0.8.0 audit fix: use async wrapper to avoid blocking event loop
        await context["version_manager"].add_version_async(
            doc_id=scoped_file_id,
            content=text,
            checksum=checksum,
            file_path=file_path,
            metadata=metadata or {},
            compression_stats={
                "total_tokens": skeleton.total_tokens,
                "skeleton_tokens": skeleton.skeleton_tokens,
                "compression_ratio": skeleton.compression_ratio,
            },
        )
        logger.info(f"[OK] Registered version history for {file_id}")
    except Exception as e:
        logger.warning(f"[WARN]  Failed to save version history for {file_id}: {e}")

    # Save file sync metadata to persistence
    try:
        metadata_export = context["sync_manager"].export_metadata()
        success = context["persistence"].save_file_sync_metadata(metadata_export)
        success = await _resolve_awaitable(success)
        if success:
            logger.info(f"[OK] Saved file sync metadata for {len(metadata_export)} documents")
        else:
            logger.warning("[WARN]  Failed to save file sync metadata")
    except Exception as e:
        logger.error(f"Failed to save file sync metadata: {e}")

    # Initialize retrieval history
    context["retrieval_history"][scoped_file_id] = []

    # Compare estimate vs actual (v0.4.1+; relative-error grading 2026-07-25)
    actual_ratio = skeleton.compression_ratio
    estimate_accuracy = classify_estimate_accuracy(
        actual=actual_ratio, estimated=estimate.compression_ratio
    )

    token_savings_percent = (
        round((1 - skeleton.skeleton_tokens / skeleton.total_tokens) * 100, 1)
        if skeleton.total_tokens > 0
        else 0.0
    )

    # Surface the honest estimate the advisor already computed. Pre-fix reasoning/
    # confidence were logged then dropped, so a connected MCP agent got no "is this
    # worth it?" signal (the activation gap). isinstance guards keep the response
    # JSON-safe when a test injects a bare Mock estimate — a real CompressionEstimate
    # always yields str reasoning/confidence + int estimated_compressed.
    estimate_out: Dict[str, Any] = {
        "estimated_ratio": estimate.compression_ratio,
        "accuracy": estimate_accuracy,
    }
    if isinstance(getattr(estimate, "estimated_compressed", None), int):
        estimate_out["estimated_compressed"] = estimate.estimated_compressed
    if isinstance(getattr(estimate, "confidence", None), str):
        estimate_out["confidence"] = estimate.confidence
    if isinstance(getattr(estimate, "reasoning", None), str):
        estimate_out["reasoning"] = estimate.reasoning

    # Small-doc honesty note (mirrors the REST path). Below ~200 tokens the skeleton
    # overhead exceeds the savings, so a first-time evaluator sees "it got bigger" and
    # bounces. Only set when the input is genuinely small AND no real saving landed.
    note = None
    if token_savings_percent <= 0.0 and 0 < skeleton.total_tokens < _SMALL_INPUT_TOKEN_THRESHOLD:
        note = (
            f"Input too small to compress: at {skeleton.total_tokens} tokens the "
            "semantic-skeleton overhead exceeds the savings. Compression pays off on "
            "documents of ~1,000+ tokens — try a larger file to see real savings."
        )

    # Build JSON response
    response = {
        "status": "success",
        "file_id": file_id,
        "total_nodes": skeleton.total_nodes,
        "total_tokens": skeleton.total_tokens,
        "skeleton_tokens": skeleton.skeleton_tokens,
        "compression_ratio": skeleton.compression_ratio,
        "token_savings": skeleton.total_tokens - skeleton.skeleton_tokens,
        "token_savings_percent": token_savings_percent,
        "estimate": estimate_out,
        "note": note,
        "chunking_strategy_used": chunking_strategy_used,
        "message": f"Document ingested successfully with {skeleton.total_nodes} semantic nodes",
    }

    try:
        response["cost_savings"] = compute_cost_savings(
            original_tokens=skeleton.total_tokens,
            compressed_tokens=skeleton.skeleton_tokens,
        ).to_dict()
    except Exception as exc:
        logger.warning(f"Cost savings calculation failed for '{file_id}': {exc}")
        response["cost_savings"] = None

    # F12 (2026-05-23 dogfood Sentry-MCP discovery): wire the SavingsTracker
    # from token_optimization_handlers so `get_savings_report` /
    # `get_savings_inline` actually see ingest activity. Pre-fix the tracker
    # was dead infrastructure — never recorded events, so every customer's
    # session report returned $0 even after real compression. Lazy import to
    # avoid module-load circularity (token_optimization_handlers imports
    # from here transitively).
    try:
        from .token_optimization_handlers import _get_tracker

        _sid = args.get("session_id") or "default"
        _model_for_savings = args.get("model") or "claude-sonnet-4-6"
        _get_tracker(_sid, _model_for_savings).record(
            tool_name="ingest_context",
            original_tokens=skeleton.total_tokens,
            compressed_tokens=skeleton.skeleton_tokens,
            model=_model_for_savings,
        )
    except Exception as exc:
        logger.warning(f"SavingsTracker.record failed for '{file_id}': {exc}")

    # Record Prometheus metrics for observability
    try:
        metrics = get_metrics()
        fidelity_label = args.get("fidelity_level", "BALANCED")
        metrics.record_compression_ratio(skeleton.compression_ratio, fidelity_label)
        metrics.increment_documents_processed("ingest", fidelity_label, "success")
        metrics.set_active_documents(len(context["compressor"].graphs))
    except Exception as exc:
        logger.warning(f"Metrics recording failed for '{file_id}': {exc}")

    # Phase 5: Record access and compression replay for optimization
    try:
        compressor = context["compressor"]
        content_type = args.get("content_type", "general")
        await _call_explicit_optional_method(
            compressor, "_access_tracker", "record_access", scoped_file_id
        )
        replay_hook = inspect.getattr_static(compressor, "_compression_replay", None)
        if replay_hook is not None:
            from ..fidelity_scoring import compute_fidelity_score

            fidelity = 0.0
            try:
                original_text = text  # use resolved text (may have come from file_url)
                if original_text and skeleton.skeleton_text:
                    emb_mgr = compressor.model
                    fidelity = compute_fidelity_score(
                        original_text, skeleton.skeleton_text, lambda texts: emb_mgr.encode(texts)
                    )
            except Exception as exc:
                logger.warning(f"Fidelity scoring failed for '{file_id}': {exc}")
            await _call_explicit_optional_method(
                compressor,
                "_compression_replay",
                "record",
                doc_id=scoped_file_id,
                content_type=content_type,
                input_tokens=skeleton.total_tokens,
                output_tokens=skeleton.skeleton_tokens,
                ratio=skeleton.compression_ratio,
                fidelity_score=fidelity,
            )
            response["fidelity_score"] = round(fidelity, 4)
        temporal_graph = _compressor_temporal_graph(compressor)
        if temporal_graph is not None:
            graph = compressor.graphs.get(scoped_file_id)
            facts = []
            if graph is not None:
                for node_id in graph.nodes():
                    node = compressor.chunks.get(node_id)
                    if node is None or not node_id.startswith(scoped_file_id):
                        continue
                    facts.append(
                        {
                            "fact_id": node_id,
                            "content": node.text,
                            "metadata": {
                                "tokens": node.metadata.get("tokens", 0),
                                "importance": round(node.importance, 4),
                                "entities": node.metadata.get("entities", [])[:5],
                            },
                        }
                    )
            temporal_graph.record_document_state(
                scoped_file_id,
                facts,
                metadata={
                    "content_type": content_type,
                    "compression_ratio": round(skeleton.compression_ratio, 4),
                },
            )
            temporal_graph.record_event(
                "document_ingested",
                doc_id=scoped_file_id,
                summary=f"Ingested {len(facts)} facts",
                metadata={"content_type": content_type},
            )
    except Exception as exc:
        logger.warning(f"Phase 5 replay/tracking failed for '{file_id}': {exc}")

    if file_path:
        response["file_sync_enabled"] = True
        response["file_path"] = file_path
        response["version"] = 1

    if source_url is not None:
        response["source_url"] = source_url

    # F6: optional ingest+query in one call
    inline_query = args.get("query")
    if inline_query and skeleton.total_nodes >= 3:
        try:
            query_skeleton_payload = run_read_skeleton_pipeline(
                compressor=context["compressor"],
                file_id=scoped_file_id,
                selection_mode="query_guided",
                query=inline_query,
                top_k=args.get("top_k", 5),
                min_similarity=args.get("min_similarity", 0.35),
            )
            # F7 (2026-05-23 dogfood Sentry GOTCONTEXT-API-H): the pipeline
            # dict contains a raw SkeletonResponse dataclass under
            # "final_skeleton" — embedding it directly here causes json.dumps
            # below to raise "TypeError: Object of type SkeletonResponse is
            # not JSON serializable". Project to scalar fields (matching the
            # handle_read_skeleton response shape) so the inline-query result
            # is JSON-safe end-to-end.
            inline_skeleton = query_skeleton_payload["final_skeleton"]
            response["query_skeleton"] = {
                "total_nodes": inline_skeleton.total_nodes,
                "total_tokens": inline_skeleton.total_tokens,
                "skeleton_tokens": inline_skeleton.skeleton_tokens,
                "compression_ratio": inline_skeleton.compression_ratio,
                "skeleton_text": inline_skeleton.skeleton_text,
                "node_map": inline_skeleton.node_map,
                "selection_mode_resolved": query_skeleton_payload.get(
                    "selection_mode_resolved", query_skeleton_payload["final_stage"]
                ),
                "evidence": query_skeleton_payload.get("evidence"),
                "pipeline": {
                    "final_stage": query_skeleton_payload["final_stage"],
                    "stage_count": query_skeleton_payload["stage_count"],
                    "stages": query_skeleton_payload["stages"],
                },
            }
        except Exception as exc:
            logger.warning(f"Inline query failed for '{file_id}': {exc}")
            response["query_skeleton"] = None

    return json.dumps(response, indent=2)


async def handle_read_skeleton(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Handle read_skeleton tool call.

    Args:
        context: Server context dict
        args: Tool arguments containing file_id

    Returns:
        JSON string with skeleton data and optional staleness warning

    Raises:
        RuntimeError: If reading skeleton fails
    """
    file_id = args["file_id"]
    scoped_file_id = _scoped_file_id(file_id, args)
    selection_mode = args.get("selection_mode", "auto")
    query = args.get("query")
    top_k = args.get("top_k", 5)
    min_similarity = args.get("min_similarity", 0.35)
    excluded_node_ids = _temporal_excluded_node_ids(context, args, scoped_file_id)

    valid_modes = {"baseline", "query_guided", "evidence_aware", "auto"}
    if selection_mode not in valid_modes:
        raise ValueError(
            f"Invalid selection_mode: '{selection_mode}'\n"
            f"[TIP] Valid modes: {sorted(valid_modes)}"
        )
    # query_guided and evidence_aware require an explicit query;
    # auto resolves the query internally from doc structure when not supplied.
    if selection_mode in {"query_guided", "evidence_aware"} and not query:
        raise ValueError(
            f"query is required when selection_mode='{selection_mode}'\n"
            "[TIP] Provide a natural-language query to guide anchor selection."
        )

    validate_file_id(scoped_file_id, context, must_exist=True)

    logger.info(f"Reading skeleton: {scoped_file_id}")

    # NEW: Check file sync status before reading
    # v1.34.20 (dogfood F2): suppress the warning when the source file is
    # absent on this server. That branch fires for every hosted-MCP ingest
    # that passes file_path — the customer's client-side path naturally
    # doesn't exist on the Fly container. The staleness warning is meant
    # for "your file changed locally, refresh" not "we never saw your file
    # in the first place." Only emit when has_source_file=True (legitimate
    # local-disk drift, the only case where refresh_document/diff_cached_file
    # is the right next step).
    staleness_warning = None
    if scoped_file_id in context["sync_manager"].file_metadata:
        status = context["sync_manager"].check_file_sync(scoped_file_id)
        if not status["in_sync"] and status.get("has_source_file", False):
            staleness_warning = {
                "is_stale": True,
                "reason": status["reason"],
                "cached_time": status.get("cached_mtime"),
                "current_time": status.get("current_mtime"),
                "recommendation": f"Use refresh_document('{file_id}') to update or diff_cached_file('{file_id}') to see changes",
            }

    try:
        compressor = context["compressor"]
        anchored_keywords = args.get("anchored_keywords", [])
        anchored_node_ids = (
            resolve_anchored_node_ids(compressor.chunks, scoped_file_id, anchored_keywords)
            if anchored_keywords
            else set()
        )

        # F3: Reconstruct raw_text for auto-mode heuristic from stored chunks.
        # Chunks are sorted by node_id (which encodes insertion order) so the
        # concatenated text preserves document structure well enough for heading
        # and finding-count heuristics.
        # Guard: compressor.chunks may be a Mock or absent in unit-test contexts;
        # fall back to None (auto resolves to baseline) rather than raising.
        raw_text_for_auto: str | None = None
        if selection_mode == "auto":
            try:
                file_chunks = chunks_for_file(compressor.chunks, scoped_file_id)
                raw_text_for_auto = "\n\n".join(node.text for _, node in file_chunks)
            except (AttributeError, TypeError):
                raw_text_for_auto = None

        # A3: offload the synchronous CPU-bound pipeline (query encode + skeleton
        # generation + MMR selection) onto a worker thread so the event loop is
        # not blocked while one request compresses. Output is unchanged — this is
        # the same call, just run via asyncio.to_thread (mirrors the _encode_async
        # pattern ingest already uses).
        pipeline = await asyncio.to_thread(
            run_read_skeleton_pipeline,
            compressor=compressor,
            file_id=scoped_file_id,
            selection_mode=selection_mode,
            query=query,
            top_k=top_k,
            min_similarity=min_similarity,
            anchor_node_ids=anchored_node_ids,
            excluded_node_ids=excluded_node_ids,
            raw_text=raw_text_for_auto,
        )
        skeleton_response = pipeline["final_skeleton"]

        # Phase 5: Record access for decay tracking
        try:
            await _call_explicit_optional_method(
                compressor,
                "_access_tracker",
                "record_access",
                scoped_file_id,
                access_type="read_skeleton",
            )
            temporal_graph = _compressor_temporal_graph(compressor)
            if temporal_graph is not None:
                temporal_graph.record_access(
                    scoped_file_id,
                    access_type="read_skeleton",
                    metadata={"query": query, "selection_mode": selection_mode},
                )
        except Exception as exc:
            logger.warning(f"Access tracking failed for '{file_id}': {exc}")

        # Build JSON response
        #
        # N5 contract: `cache_stable_prefix`, whenever present, MUST be query-
        # independent. It exists so a caller can build a stable KV-cache prefix
        # across calls with different `query` values (arXiv 2607.15516) — a
        # value that silently varies per query defeats that prefix cache and,
        # below ~6x compression, can cost MORE than not compressing at all.
        #
        # `_baseline_skeleton_cache` is populated at ingest time and is purely
        # in-process (semantic_compressor.py:1451) — a worker restart, an
        # eviction, or an ingest that happened in a different worker are all
        # routine ways for it to miss. On a miss we RECOMPUTE the baseline
        # (query-free) skeleton directly from the already-ingested graph via
        # `_generate_skeleton(file_id)` with no `query` argument (chosen over
        # silently omitting the field — see semantic_compressor.py:1693 — this
        # is a pure node-selection/render pass over already-embedded chunks, no
        # network or model call, so the recompute is cheap even on a cold
        # cache). If even that fails (e.g. the graph itself is gone), we surface
        # `cache_stable_prefix: None` rather than hand back a plausible-looking
        # but query-conditioned value — a consumer can `is None`-check rather
        # than being silently handed the wrong thing.
        cache_stable_prefix = skeleton_response.skeleton_text
        if selection_mode != "baseline":
            baseline_cache = getattr(compressor, "_baseline_skeleton_cache", None)
            if isinstance(baseline_cache, dict) and scoped_file_id in baseline_cache:
                cache_stable_prefix = baseline_cache[scoped_file_id]
            elif isinstance(baseline_cache, dict):
                # Real cache dict present but missing this file_id — the
                # routine miss case this fix targets. A compressor without a
                # dict-shaped `_baseline_skeleton_cache` at all (e.g. a bare
                # test double) doesn't support this mechanism; leave it as-is
                # rather than forcing an extra call it never asked for.
                try:
                    # A3-style offload: this recompute is a synchronous,
                    # CPU-bound pass (node selection + render over already-
                    # embedded chunks). The main pipeline above never calls a
                    # sync compressor method directly on the event loop for
                    # exactly this reason (see the `asyncio.to_thread` offload
                    # a few lines up) -- this recompute must not either, or a
                    # routine cache miss stalls every other in-flight request.
                    baseline_response = await asyncio.to_thread(
                        compressor._generate_skeleton, scoped_file_id
                    )
                    cache_stable_prefix = baseline_response.skeleton_text
                    # Write-through: the next miss for this file_id is now a hit.
                    if isinstance(baseline_cache, dict):
                        baseline_cache[scoped_file_id] = cache_stable_prefix
                except Exception as exc:
                    logger.warning(
                        f"cache_stable_prefix baseline recompute failed for '{file_id}': {exc}",
                        exc_info=True,
                    )
                    cache_stable_prefix = None

        response = {
            "file_id": file_id,
            "total_nodes": skeleton_response.total_nodes,
            "total_tokens": skeleton_response.total_tokens,
            "skeleton_tokens": skeleton_response.skeleton_tokens,
            "compression_ratio": skeleton_response.compression_ratio,
            "skeleton_text": skeleton_response.skeleton_text,
            "cache_stable_prefix": cache_stable_prefix,
            "node_map": skeleton_response.node_map,
            "selection_mode": args.get("selection_mode", "auto"),
            "selection_mode_resolved": pipeline.get(
                "selection_mode_resolved", pipeline["final_stage"]
            ),
            "temporal_filters": {
                "as_of": (
                    format_timestamp(coerce_timestamp(args["as_of"])) if args.get("as_of") else None
                ),
                "include_invalidated": args.get("include_invalidated", False),
            },
            "pipeline": {
                "final_stage": pipeline["final_stage"],
                "stage_count": pipeline["stage_count"],
                "stages": pipeline["stages"],
            },
        }

        if anchored_keywords:
            response["anchored_nodes"] = sorted(anchored_node_ids)
        if query:
            response["query"] = query
        evidence_info = pipeline["evidence"]
        if evidence_info:
            evidence_info["node_ids"] = [
                node_id for node_id in evidence_info["node_ids"] if node_id not in excluded_node_ids
            ]
            response["evidence"] = evidence_info

        if staleness_warning:
            response["staleness_warning"] = staleness_warning

        # v1.34.28 (F12 class-completion): re-compression on every read_skeleton
        # call produces real token savings (total → skeleton). Wire the tracker
        # so agents querying get_savings_report mid-session see this activity.
        # Same lazy-import + swallow-failure pattern as handle_ingest (v1.34.27).
        try:
            from .token_optimization_handlers import _get_tracker

            _sid = args.get("session_id") or "default"
            _model = args.get("model") or "claude-sonnet-4-6"
            _get_tracker(_sid, _model).record(
                tool_name="read_skeleton",
                original_tokens=skeleton_response.total_tokens,
                compressed_tokens=skeleton_response.skeleton_tokens,
                model=_model,
            )
        except Exception as exc:
            logger.warning(f"SavingsTracker.record failed for read_skeleton '{file_id}': {exc}")

        return json.dumps(response, indent=2)
    except Exception as e:
        raise RuntimeError(
            f"Failed to read skeleton for '{file_id}': {str(e)}\n"
            f"Tip: Verify the document was ingested successfully with get_stats()"
        ) from e


async def handle_modulate_region(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Handle modulate_region tool call.

    Args:
        context: Server context dict
        args: Tool arguments containing node_ids and optional fidelity_level

    Returns:
        Modulated content with optional staleness warning

    Raises:
        ValueError: If validation fails
        RuntimeError: If modulation fails
    """
    # v1.34.30 (F10): accept singular `node_id` as a convenience for the
    # one-region case — wraps to [node_id]. Customers who try the
    # singular intuitive call (modulate_region(node_id="x")) now succeed
    # instead of getting "Input validation error: 'node_ids' is a
    # required property". Canonical name remains `node_ids`.
    if "node_ids" in args:
        node_ids = args["node_ids"]
    elif "node_id" in args:
        node_ids = [args["node_id"]]
    else:
        raise ValueError(
            "modulate_region requires `node_ids` (list) or `node_id` (single string)\n"
            '[TIP] For one region: node_id="<id>". For many: node_ids=["<id1>", "<id2>"]'
        )
    fidelity_str = args.get("fidelity_level", "RAW")

    # Validation
    validate_node_ids(node_ids, context)

    # NEW: Check file sync status before modulating
    file_id = extract_file_id_from_node(node_ids[0]) if node_ids else None
    warning = ""
    if file_id and file_id in context["sync_manager"].file_metadata:
        status = context["sync_manager"].check_file_sync(file_id)
        if not status["in_sync"]:
            warning = f"""
[WARN]  WARNING: Cache may be stale for '{file_id}'!

{status['reason']}

[TIP] Use refresh_document('{file_id}') to update

Proceeding with cached version...
---

"""

    # Convert string to enum with validation
    try:
        fidelity = FidelityLevel[fidelity_str]
    except KeyError:
        valid_levels = [level.name for level in FidelityLevel]
        raise ValueError(
            f"Invalid fidelity_level: '{fidelity_str}'\n"
            f"[TIP] Valid levels: {valid_levels}\n"
            f"   ABSTRACT: ~10 tokens (summary only)\n"
            f"   OUTLINE: ~30 tokens (summary + section markers)\n"
            f"   STRUCTURE: ~50 tokens (headers + entities)\n"
            f"   DETAILED: ~100 tokens (summary + excerpts)\n"
            f"   RAW: Full original text"
        )

    logger.info(f"Modulating {len(node_ids)} nodes at {fidelity_str} fidelity")

    # Track retrieval for blind spot detection
    for node_id in node_ids:
        file_id = extract_file_id_from_node(node_id)
        if file_id not in context["retrieval_history"]:
            context["retrieval_history"][file_id] = []
        if node_id not in context["retrieval_history"][file_id]:
            context["retrieval_history"][file_id].append(node_id)

    try:
        result = context["compressor"].modulate_region(node_ids, fidelity)
        return warning + result
    except Exception as e:
        raise RuntimeError(
            f"Failed to modulate region: {str(e)}\n"
            f"Tip: Verify node IDs are valid with read_skeleton()"
        ) from e


async def handle_search_semantic(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Handle search_semantic tool call.

    v0.9.0: Now returns similarity scores alongside importance (PageRank) scores.
    - similarity: How well the node matches the search query (cosine similarity)
    - importance: How central the node is in the document graph (PageRank)

    Args:
        context: Server context dict
        args: Tool arguments containing query, optional file_id and top_k

    Returns:
        JSON string with search results including similarity scores
    """
    query = args["query"]
    file_id = args.get("file_id")
    scoped_file_id = _scoped_file_id(file_id, args) if file_id else None
    top_k = args.get("top_k", 5)
    evidence_aware = args.get("evidence_aware", False)
    min_similarity = args.get("min_similarity", 0.35)
    search_top_k = max(top_k * 5, top_k) if _has_scope_args(args) and not file_id else top_k

    logger.info(f"Semantic search: '{query}' in {scoped_file_id or 'scoped files'}")

    # A3: offload the synchronous CPU-bound retrieval (query encode + cosine /
    # RRF ranking) onto a worker thread so the event loop is not blocked while
    # one request searches. Output is unchanged — same calls via asyncio.to_thread
    # (mirrors the _encode_async pattern ingest already uses).
    compressor = context["compressor"]
    # Fallback label when the ranker doesn't surface a per-call score_type
    # (older/stub compressor or a test Mock without the typed API): mirror the
    # pre-Path-G behavior — Path C is RRF, everything else cosine.
    _fallback_score_type = "rrf" if _f11_ranker_path() == "c" else "cosine"

    def _valid_score_type(candidate: object) -> str:
        # Value-validate rather than presence-detect: a unittest.mock.Mock
        # auto-vivifies any attribute, so `getattr(mock, "score_type")` is a
        # Mock, not a str — accept only the known labels, else fall back.
        return candidate if candidate in ("cosine", "rrf") else _fallback_score_type

    if evidence_aware:
        evidence = await asyncio.to_thread(
            compressor.retrieve_evidence,
            query=query,
            file_id=scoped_file_id,
            top_k=search_top_k,
            min_similarity=min_similarity,
        )
        search_results = _scope_filtered_results(evidence.scores, args)
        # retrieve_evidence surfaces the per-call ranker label (Path G gate may
        # have fused or fallen back). Read + value-validate it.
        _ranker_score_type = _valid_score_type(getattr(evidence, "score_type", None))
    else:
        evidence = None
        # Prefer the typed ranker so Path G reports "rrf" only when the gate
        # actually fused for THIS query, "cosine" when it fell back to dense
        # (blocker-3 fix). The typed method returns (results, score_type); only
        # trust it when it returns exactly that shape with a known label —
        # otherwise (stub / Mock) use the plain call + fallback label.
        typed_fn = getattr(compressor, "search_semantic_with_scores_typed", None)
        raw_results = None
        _ranker_score_type = _fallback_score_type
        if callable(typed_fn):
            typed_out = await asyncio.to_thread(typed_fn, query, scoped_file_id, search_top_k)
            if (
                isinstance(typed_out, tuple)
                and len(typed_out) == 2
                and typed_out[1] in ("cosine", "rrf")
            ):
                raw_results, _ranker_score_type = typed_out
        if raw_results is None:
            raw_results = await asyncio.to_thread(
                compressor.search_semantic_with_scores,
                query,
                scoped_file_id,
                search_top_k,
            )
            _ranker_score_type = _fallback_score_type
        search_results = _scope_filtered_results(raw_results, args)

    search_results = _temporal_filter_search_results(context, search_results, args)[:top_k]

    # Council patch P2: score_type field distinguishes RRF from cosine scores.
    # Callers must NOT treat RRF scores as cosine similarity values.
    #
    # Audit P2-3: the label is a function of the ACTIVE RANKER PATH only, NOT of
    # evidence_aware. retrieve_evidence() (the evidence_aware path) goes through
    # the typed ranker too, so under Path C its scores are RRF.
    #
    # Blocker-3 fix (2026-07-08 codex review): under Path G the label is
    # PER-CALL — "rrf" iff the gate fused for THIS query, else "cosine". We now
    # take the label from what the ranker ACTUALLY ran (typed API /
    # EvidenceResult.score_type), never a hardcoded `== "c"` check. Path "a" and
    # "c" labels are unchanged (typed ranker returns "cosine"/"rrf" for them).
    _score_type = _ranker_score_type

    # Build structured results with both similarity and importance
    results = []
    for node_id, similarity_score in search_results:
        node = context["compressor"].chunks[node_id]
        summary = context["compressor"]._generate_summary(node.text, max_length=100)
        results.append(
            {
                "node_id": node_id,
                "similarity": round(similarity_score, 3),  # Query match score
                "score_type": _score_type,  # "cosine" (Path A) or "rrf" (Path C)
                "importance": round(node.importance, 3),  # PageRank centrality
                "summary": summary,
                "tokens": node.metadata.get("tokens", 0),
            }
        )

    # Build JSON response
    response = {
        "query": query,
        "file_id": file_id,
        "evidence_aware": evidence_aware,
        "score_type": _score_type,  # Top-level: callers can detect Path A→C fallback
        "total_results": len(results),
        "results": results,
        "tip": "Use modulate_region() with node_ids to retrieve full content",
        "score_explanation": {
            "similarity": (
                "RRF score (rank-fusion, not cosine; higher = better match)"
                if _score_type == "rrf"
                else "Semantic match to query (higher = better match)"
            ),
            "importance": "PageRank centrality in document graph (higher = more central)",
        },
        "temporal_filters": {
            "as_of": (
                format_timestamp(coerce_timestamp(args["as_of"])) if args.get("as_of") else None
            ),
            "include_invalidated": args.get("include_invalidated", False),
        },
    }
    if evidence is not None:
        response["evidence"] = {
            "sufficient": evidence.sufficient,
            "best_score": round(evidence.best_score, 3),
            "threshold": evidence.threshold,
            "used_expanded_search": evidence.used_expanded_search,
            "message": evidence.message,
        }

    # Phase 5: Record access for decay tracking
    try:
        compressor = context["compressor"]
        if hasattr(compressor, "_access_tracker"):
            accessed_files = set()
            for r in results:
                fid = extract_file_id_from_node(r["node_id"])
                if fid:
                    accessed_files.add(fid)
            for fid in accessed_files:
                await _call_explicit_optional_method(
                    compressor,
                    "_access_tracker",
                    "record_access",
                    fid,
                    access_type="search_semantic",
                )
                temporal_graph = _compressor_temporal_graph(compressor)
                if temporal_graph is not None:
                    temporal_graph.record_access(
                        fid,
                        access_type="search_semantic",
                        metadata={"query": query},
                    )
    except Exception as exc:
        logger.warning(f"Access tracking failed during semantic search: {exc}")

    return json.dumps(response, indent=2)


async def handle_get_stats(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Handle get_stats tool call.

    Args:
        context: Server context dict
        args: Tool arguments containing optional file_id

    Returns:
        Formatted statistics
    """
    file_id = args.get("file_id")
    scoped_file_id = _scoped_file_id(file_id, args) if file_id else None

    stats = (
        context["compressor"].get_stats(scoped_file_id)
        if scoped_file_id
        else (
            _scoped_global_stats(context, args)
            if _has_scope_args(args)
            else context["compressor"].get_stats()
        )
    )

    if file_id:
        display_stats_file_id = file_id or display_file_id(scoped_file_id)
        scope_label = _scope_label(scoped_file_id or file_id)
        scope_line = f"Scope: {scope_label}\n\n" if scope_label else ""
        result = f"""
[STATS] Document Statistics: {display_stats_file_id}

{scope_line}Total nodes: {stats['total_nodes']}
Total edges: {stats['total_edges']}
Original tokens: {stats['total_tokens']:,}
Skeleton tokens: {stats['skeleton_tokens']:,}
Compression ratio: {stats['compression_ratio']:.1f}x

Token savings: {stats['total_tokens'] - stats['skeleton_tokens']:,} ({(1 - stats['skeleton_tokens'] / stats['total_tokens']) * 100:.1f}%)

Metadata: {json.dumps(stats['metadata'], indent=2)}
"""
    else:
        scope_lines = (
            [
                (
                    f"  - {display_file_id(fid)} ({label})"
                    if (label := _scope_label(fid))
                    else f"  - {display_file_id(fid)}"
                )
                for fid in _scope_filtered_file_ids(list(context["compressor"].graphs.keys()), args)
            ]
            if _has_scope_args(args)
            else None
        )
        files_output = (
            chr(10).join(scope_lines) if scope_lines else ", ".join(stats.get("files", []))
        )
        result = f"""
[STATS] Global Statistics

Total files ingested: {stats.get('total_files', stats.get('total_documents', 0))}
Total nodes: {stats.get('total_nodes', 0)}

Files: {files_output}
"""

    return result


async def handle_list_documents(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Handle list_documents tool call.

    Args:
        context: Server context dict
        args: Tool arguments (none required)

    Returns:
        Formatted document inventory
    """
    logger.info("Listing all ingested documents")

    # Get all unique file_ids from chunks (supports both text and code node formats)
    file_ids = sorted(
        _scope_filtered_file_ids(list(collect_file_ids(context["compressor"].chunks.keys())), args)
    )

    if not file_ids:
        return """
[DOC] Document Inventory

No documents ingested yet.

[TIP] Use ingest_context(text, file_id) to add documents.
"""

    # Build structured inventory
    documents = []
    for internal_file_id in sorted(file_ids):
        stats = context["compressor"].get_stats(internal_file_id)
        metadata = stats.get("metadata", {})
        visible_file_id = display_file_id(internal_file_id)

        doc_info = {
            "file_id": visible_file_id,
            "title": metadata.get("title", visible_file_id),
            "scope_label": _scope_label(internal_file_id),
            "total_nodes": stats["total_nodes"],
            "total_tokens": stats["total_tokens"],
            "skeleton_tokens": stats["skeleton_tokens"],
            "compression_ratio": stats["compression_ratio"],
            "metadata": metadata,
        }
        documents.append(doc_info)

    # Format output
    result_lines = ["[DOC] Document Inventory\n"]
    result_lines.append(f"Total documents: {len(documents)}\n")

    for i, doc in enumerate(documents, 1):
        result_lines.append(f"{i}. [{doc['file_id']}]")
        if doc["title"] != doc["file_id"]:
            result_lines.append(f"   Title: {doc['title']}")
        if doc["scope_label"]:
            result_lines.append(f"   Scope: {doc['scope_label']}")
        result_lines.append(f"   Nodes: {doc['total_nodes']}")
        result_lines.append(
            f"   Tokens: {doc['total_tokens']:,} → {doc['skeleton_tokens']:,} ({doc['compression_ratio']:.1f}x compression)"
        )

        # Include relevant metadata
        if "author" in doc["metadata"]:
            result_lines.append(f"   Author: {doc['metadata']['author']}")
        if "date" in doc["metadata"]:
            result_lines.append(f"   Date: {doc['metadata']['date']}")
        if "tags" in doc["metadata"]:
            tags = ", ".join(doc["metadata"]["tags"][:3])
            result_lines.append(f"   Tags: {tags}")

        result_lines.append("")  # Blank line between documents

    result_lines.append("[TIP] Next steps:")
    result_lines.append("  - read_skeleton(file_id) - View compressed structure")
    result_lines.append("  - search_semantic(query) - Find relevant content")
    result_lines.append("  - get_stats(file_id) - Detailed statistics")

    return "\n".join(result_lines)


async def handle_delete_document(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Handle delete_document tool call.

    Args:
        context: Server context dict
        args: Tool arguments containing file_id and confirm flag

    Returns:
        Confirmation prompt or deletion success message

    Raises:
        RuntimeError: If deletion fails
    """
    file_id = args["file_id"]
    scoped_file_id = _scoped_file_id(file_id, args)
    confirm = args.get("confirm", False)

    # Validation
    validate_file_id(scoped_file_id, context, must_exist=True)

    if not confirm:
        return f"""
[WARN]  DELETE CONFIRMATION REQUIRED

You are about to delete document: {file_id}

This will:
  -Remove all {len(chunks_for_file(context['compressor'].chunks, scoped_file_id))} semantic nodes from memory
  -Delete persistent storage (cannot be undone)
  -Clear retrieval history for this document

To proceed, call again with confirm=true:
  delete_document(file_id="{file_id}", confirm=true)

Tip: Use list_documents() to see all available documents first
"""

    logger.info(f"Deleting document: {scoped_file_id}")

    # Get stats before deletion
    stats = context["compressor"].get_stats(scoped_file_id)
    node_count = stats["total_nodes"]

    # Delete from memory
    # IMPORTANT: compressor.chunks, compressor.graphs, and
    # compressor.file_metadata are properties on CodeCompressionAdapter
    # that return *copies* of the underlying dicts.  Direct del on those
    # copies is a no-op against the real storage.  Use
    # delete_document_from_memory when available; fall back to direct
    # mutation only when the compressor exposes real dicts (plain
    # SemanticCompressor).
    try:
        compressor = context["compressor"]
        if hasattr(compressor, "delete_document_from_memory"):
            # CodeCompressionAdapter path — mutates real underlying dicts
            compressor.delete_document_from_memory(scoped_file_id)
        else:
            # SemanticCompressor path — chunks/graphs/file_metadata are real dicts
            chunks_to_delete = [k for k, _ in chunks_for_file(compressor.chunks, scoped_file_id)]
            for chunk_id in chunks_to_delete:
                del compressor.chunks[chunk_id]
            if scoped_file_id in compressor.graphs:
                del compressor.graphs[scoped_file_id]
            if scoped_file_id in compressor.file_metadata:
                del compressor.file_metadata[scoped_file_id]

        # Remove retrieval history (plain dict on context — safe to mutate directly)
        if scoped_file_id in context["retrieval_history"]:
            del context["retrieval_history"][scoped_file_id]

        logger.info(f"[OK] Removed {file_id} from memory ({node_count} nodes)")

    except Exception as e:
        logger.error(f"Failed to delete {file_id} from memory: {e}")
        raise RuntimeError(f"Failed to delete from memory: {e}")

    # Delete from persistent storage
    try:
        success = context["persistence"].delete_document(scoped_file_id)
        if success:
            logger.info(f"[OK] Deleted {file_id} from persistent storage")
        else:
            logger.warning(f"[WARN]  Failed to delete {file_id} from persistent storage")
    except Exception as e:
        logger.error(f"Failed to delete {file_id} from storage: {e}")

    # Unregister from resource manager
    # v0.8.0 audit fix: use async wrapper to avoid blocking event loop
    try:
        await context["resource_manager"].unregister_document_async(scoped_file_id)
    except Exception as e:
        logger.warning(f"Failed to unregister {file_id} from resource manager: {e}")

    # NEW: Clean up file sync metadata and version history
    try:
        context["sync_manager"].remove_metadata(scoped_file_id)
        # v0.8.0 audit fix: use async wrapper to avoid blocking event loop
        await context["version_manager"].delete_versions_async(scoped_file_id)
        # Save metadata after removal
        metadata_export = context["sync_manager"].export_metadata()
        context["persistence"].save_file_sync_metadata(metadata_export)
        logger.info(f"[OK] Cleaned up file sync metadata and version history for {file_id}")
    except Exception as e:
        logger.warning(f"Failed to clean up file sync data for {file_id}: {e}")

    return f"""
[DELETE] Document Deleted Successfully

File ID: {file_id}
Nodes removed: {node_count}
Memory freed: ~{node_count * 2}KB (estimated)

[OK] Document has been permanently deleted from:
   -Memory (semantic graph, chunks, metadata)
   -Persistent storage (ChromaDB/JSON)
   -Resource tracking

[TIP] Remaining documents: {len(collect_file_ids(context["compressor"].chunks.keys()))}
   Use list_documents() to see what's left.
"""


async def handle_adapt_to_context_window(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Handle adapt_to_context_window tool call (JSCCM-inspired).

    Args:
        context: Server context dict
        args: Tool arguments containing file_id, available_tokens, optional max_tokens and query_priority

    Returns:
        Adapted skeleton text

    Raises:
        ValueError: If validation fails
        RuntimeError: If adaptation fails
    """
    file_id = args["file_id"]
    available_tokens = args["available_tokens"]
    max_tokens = args.get("max_tokens", 100000)
    query_priority = args.get("query_priority", 0.5)

    # Validation
    validate_file_id(file_id, context, must_exist=True)
    validate_token_count(available_tokens, max_tokens)

    if not 0.0 <= query_priority <= 1.0:
        raise ValueError(
            f"query_priority must be between 0.0 and 1.0, got {query_priority}\n"
            "Tip: 0.0 = low priority, 0.5 = medium, 1.0 = high priority"
        )

    logger.info(
        f"Adapting skeleton for {file_id}: {available_tokens}/{max_tokens} tokens available"
    )

    try:
        result = context["context_window_adapter"].adapt_to_context_window(
            file_id=file_id,
            available_tokens=available_tokens,
            max_tokens=max_tokens,
            query_priority=query_priority,
        )
        return result
    except Exception as e:
        raise RuntimeError(
            f"Failed to adapt to context window: {str(e)}\n"
            "Tip: This is a JSCCM-inspired feature. Check that the document exists and token counts are valid."
        ) from e


async def handle_multilevel_encode(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Handle multilevel_encode tool call (JSCCM-inspired).

    Args:
        context: Server context dict
        args: Tool arguments containing file_id and available_tokens

    Returns:
        Multi-level encoded skeleton

    Raises:
        ValueError: If validation fails
        RuntimeError: If encoding fails
    """
    file_id = args["file_id"]
    available_tokens = args["available_tokens"]

    # Validation
    validate_file_id(file_id, context, must_exist=True)
    validate_token_count(available_tokens)

    logger.info(
        f"Generating multi-level encoding for {file_id}: {available_tokens} tokens available"
    )

    try:
        result = context["multilevel_encoder"].generate_adaptive_skeleton(file_id, available_tokens)
        return result
    except Exception as e:
        raise RuntimeError(
            f"Failed to generate multi-level encoding: {str(e)}\n"
            "Tip: This JSCCM-inspired feature requires Main + Auxiliary + Detail branches.\n"
            "   Try with at least 1000 tokens available for meaningful output."
        ) from e


async def handle_recommend_fidelity(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Handle recommend_fidelity tool call (NEW in v0.4.1).

    Args:
        context: Server context dict (not used but required for handler signature)
        args: Tool arguments containing use_case, num_nodes, optional token_budget and query_complexity

    Returns:
        JSON string with fidelity recommendation, reasoning, token estimate, and alternatives

    Raises:
        ValueError: If validation fails

    Example:
        >>> handle_recommend_fidelity(
        ...     {"use_case": "question_answering", "num_nodes": 3, "token_budget": 200},
        ...     context
        ... )
        '{"recommended_level": "STRUCTURE", "confidence": 0.9, ...}'
    """
    use_case_str = args["use_case"]
    num_nodes = args["num_nodes"]
    token_budget = args.get("token_budget")
    query_complexity = args.get("query_complexity", "medium")

    # Validation
    if num_nodes < 1:
        raise ValueError(
            f"num_nodes must be at least 1, got {num_nodes}\n"
            "Tip: Specify how many nodes you plan to retrieve"
        )

    if num_nodes > 1000:
        raise ValueError(
            f"num_nodes is very high ({num_nodes})\n"
            "Tip: Consider retrieving fewer nodes for better token efficiency.\n"
            "   Most queries work well with 3-10 nodes."
        )

    if token_budget is not None:
        if token_budget < 10:
            raise ValueError(
                f"token_budget is too low ({token_budget})\n"
                "Tip: Even ABSTRACT fidelity needs ~10 tokens per node.\n"
                f"   For {num_nodes} nodes, minimum budget: {num_nodes * 10} tokens"
            )

        if token_budget > 1_000_000:
            raise ValueError(
                f"token_budget is very high ({token_budget:,})\n"
                "Tip: Most use cases work well with 100-10,000 token budgets."
            )

    if query_complexity not in ["simple", "medium", "complex"]:
        raise ValueError(
            f"query_complexity must be 'simple', 'medium', or 'complex', got '{query_complexity}'\n"
            "Tip: Use 'medium' if unsure"
        )

    # Convert string to enum
    try:
        use_case_enum = UseCase(use_case_str)
    except ValueError:
        valid_cases = [case.value for case in UseCase]
        raise ValueError(
            f"Unknown use_case: '{use_case_str}'\n" f"[TIP] Valid options: {', '.join(valid_cases)}"
        )

    # Get recommendation
    logger.info(
        f"Recommending fidelity for use_case={use_case_str}, "
        f"num_nodes={num_nodes}, budget={token_budget}, complexity={query_complexity}"
    )

    advisor = FidelityAdvisor()
    rec = advisor.recommend(use_case_enum, num_nodes, token_budget, query_complexity)

    # Format response. recommended_level is the enum NAME (e.g. "DETAILED") —
    # modulate_region's fidelity_level takes the label, not the integer. The
    # pre-fix code emitted .value, so the usage_tip told agents to pass
    # fidelity_level='5', which modulate_region rejects (#92; codex production
    # dogfood, 2026-06-12). The numeric level stays available alongside.
    response = {
        "recommended_level": rec.recommended_level.name,
        "recommended_level_value": rec.recommended_level.value,
        "confidence": rec.confidence,
        "reasoning": rec.reasoning,
        "token_estimate": rec.token_estimate,
        "alternatives": rec.alternatives,
        "usage_tip": (
            f"Use modulate_region with fidelity_level='{rec.recommended_level.name}' "
            f"to retrieve {num_nodes} nodes (~{rec.token_estimate} tokens)"
        ),
    }

    return json.dumps(response, indent=2)


# ===========================
# Batch Processing Handler
# ===========================
