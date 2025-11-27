#!/usr/bin/env python
"""
Observability Demo - OpenTelemetry Distributed Tracing

Demonstrates all features of the observability module including:
- Basic trace creation
- Async context propagation
- Error handling and exception recording
- Integration with structured logging
- Nested spans
- Span attributes and events
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.observability import configure_observability, get_observability
from src.structured_logging import configure_structlog, get_logger


def demo_basic_tracing():
    """Demo 1: Basic tracing with attributes."""
    print("\n=== Demo 1: Basic Tracing ===")

    observe = get_observability()

    with observe.trace("compress", doc_id="abc123", fidelity_level="BALANCED"):
        # Simulate compression work
        print("  Compressing document...")

        # Set attributes during operation
        observe.set_attribute("compression_ratio", 7.5)
        observe.set_attribute("token_count", 500)
        observe.set_attribute("status", "success")

        print("  Compression completed")
        print(f"  Trace context: {observe.get_current_trace_context()}")


def demo_error_handling():
    """Demo 2: Error handling and exception recording."""
    print("\n=== Demo 2: Error Handling ===")

    observe = get_observability()

    try:
        with observe.trace("compress_with_error", doc_id="error123"):
            print("  Starting compression...")
            observe.set_attribute("status", "processing")

            # Simulate error
            raise ValueError("Document exceeds size limit")
    except ValueError as e:
        print(f"  Error caught and recorded: {e}")


def demo_nested_spans():
    """Demo 3: Nested spans."""
    print("\n=== Demo 3: Nested Spans ===")

    observe = get_observability()

    with observe.trace("batch_ingest", batch_id="batch-456", batch_size=3):
        print("  Batch ingestion started")

        for i in range(3):
            with observe.trace("compress", doc_id=f"doc{i}"):
                print(f"    Compressing document {i}")
                observe.set_attribute("compression_ratio", 7.5 + i)

        observe.set_attribute("status", "completed")
        print("  Batch ingestion completed")


async def compress_document_async(doc_id: str, observe):
    """Async compression simulation."""
    with observe.trace("compress", doc_id=doc_id):
        print(f"    Compressing {doc_id}...")
        await asyncio.sleep(0.1)  # Simulate work
        observe.set_attribute("compression_ratio", 8.2)
        observe.set_attribute("status", "success")


async def demo_async_context_propagation():
    """Demo 4: Async context propagation."""
    print("\n=== Demo 4: Async Context Propagation ===")

    observe = get_observability()

    with observe.trace("batch_ingest_async", batch_id="async-batch", batch_size=5):
        print("  Async batch ingestion started")

        # Context propagates to all async tasks
        tasks = [compress_document_async(f"doc{i}", observe) for i in range(5)]
        await asyncio.gather(*tasks)

        observe.set_attribute("status", "completed")
        print("  Async batch ingestion completed")


def demo_logging_integration():
    """Demo 5: Integration with structured logging."""
    print("\n=== Demo 5: Logging Integration ===")

    observe = get_observability()
    logger = get_logger()

    with observe.trace("compress", doc_id="log123", fidelity_level="HIGH"):
        # Get trace context for log correlation
        trace_context = observe.get_current_trace_context()

        # Logs will include trace_id and span_id
        logger.info("Compression started", **trace_context, doc_id="log123")

        # Simulate work
        observe.set_attribute("compression_ratio", 9.1)
        observe.set_attribute("token_count", 450)

        logger.info(
            "Compression completed",
            **trace_context,
            compression_ratio=9.1,
            token_count=450,
        )

        print("  Check logs above for trace_id and span_id correlation")


def demo_events():
    """Demo 6: Span events."""
    print("\n=== Demo 6: Span Events ===")

    observe = get_observability()

    with observe.trace("compress_with_events", doc_id="events123"):
        print("  Starting compression with events...")

        # Add events to mark milestones
        observe.add_event("processing_started")
        observe.add_event("cache_hit", {"cache_key": "doc123"})
        observe.add_event("embedding_generated", {"embedding_size": 384})
        observe.add_event("graph_constructed", {"node_count": 25})
        observe.add_event("processing_completed", {"duration_ms": 150})

        observe.set_attribute("status", "success")
        print("  Events added to trace")


def demo_multiple_attributes():
    """Demo 7: Setting multiple attributes efficiently."""
    print("\n=== Demo 7: Multiple Attributes ===")

    observe = get_observability()

    with observe.trace("compress_bulk_attrs"):
        # Set multiple attributes at once
        observe.set_attributes(
            {
                "doc_id": "bulk123",
                "fidelity_level": "BALANCED",
                "compression_ratio": 7.5,
                "token_count": 500,
                "graph_nodes": 30,
                "graph_edges": 45,
                "processing_time_ms": 125,
                "cache_hit": True,
                "status": "success",
            }
        )

        print("  Multiple attributes set efficiently")


def main():
    """Run all demos."""
    print("=" * 60)
    print("OpenTelemetry Observability Demo")
    print("=" * 60)

    # Configure observability
    configure_observability(
        service_name="token-saver-5000",
        environment="development",
        sampling_rate=1.0,  # Sample 100% in development
        enable_console_export=True,  # Enable console export for demo
    )

    # Configure structured logging (JSON format for production)
    configure_structlog(
        log_level="INFO",
        format="json",
        service="token-saver-5000",
        environment="development",
    )

    # Run demos
    demo_basic_tracing()
    demo_error_handling()
    demo_nested_spans()
    demo_events()
    demo_multiple_attributes()
    demo_logging_integration()

    # Run async demo
    print("\n=== Running Async Demo ===")
    asyncio.run(demo_async_context_propagation())

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)
    print("\nNote: Spans are exported to console (JSON format)")
    print("In production, configure OTLP endpoint to send to trace backend")


if __name__ == "__main__":
    main()
