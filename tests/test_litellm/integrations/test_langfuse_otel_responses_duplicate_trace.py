"""
Test for issue #16700: Duplicate root traces with Responses API and langfuse_otel

This test verifies that when using the Responses API with langfuse_otel callback
and an active OpenTelemetry span (simulating Langfuse @observe() decorator),
the litellm spans are created as children of the active span, not as separate root traces.
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime


class TestLangfuseOtelResponsesNoDuplicateTrace:
    """Test suite to verify no duplicate root traces are created"""

    def test_responses_api_respects_active_span_context(self):
        """
        Test that when there's an active OTEL span (from @observe()),
        the langfuse_otel logger creates a child span, not a separate root trace.
        """
        from litellm.integrations.langfuse.langfuse_otel import LangfuseOtelLogger
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        from opentelemetry import trace

        # Set up in-memory span exporter to capture spans
        span_exporter = InMemorySpanExporter()
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
        trace.set_tracer_provider(tracer_provider)

        tracer = trace.get_tracer(__name__)

        # Create a mock active span (simulating @observe() decorator)
        with tracer.start_as_current_span("user_function_with_observe") as parent_span:
            parent_trace_id = parent_span.get_span_context().trace_id
            parent_span_id = parent_span.get_span_context().span_id

            # Now call langfuse_otel's _handle_success (simulating a litellm.responses() call)
            logger = LangfuseOtelLogger(
                config=None,  # Will use defaults
                tracer_provider=tracer_provider
            )

            kwargs = {
                "call_type": "responses",
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}],
                "litellm_params": {"metadata": {}},
            }

            response_obj = {
                "id": "resp-123",
                "model": "gpt-4o",
                "choices": [{"message": {"role": "assistant", "content": "Hi there"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            }

            start_time = datetime.now()
            end_time = datetime.now()

            # Call _handle_success which should create a child span
            logger._handle_success(kwargs, response_obj, start_time, end_time)

        # Force flush to ensure all spans are exported
        tracer_provider.force_flush()

        # Get all exported spans
        spans = span_exporter.get_finished_spans()

        # Verify we have spans
        assert len(spans) > 0, "No spans were created"

        # Find the litellm_request span
        litellm_spans = [s for s in spans if s.name == "litellm_request"]
        assert len(litellm_spans) > 0, "No litellm_request span was created"

        litellm_span = litellm_spans[0]
        litellm_trace_id = litellm_span.context.trace_id
        litellm_parent_span_id = litellm_span.parent.span_id if litellm_span.parent else None

        # CRITICAL ASSERTIONS:
        # 1. The litellm span should be in the SAME trace as the parent
        assert litellm_trace_id == parent_trace_id, (
            f"Duplicate root trace detected! "
            f"litellm_request span has trace_id {format(litellm_trace_id, '032x')} "
            f"but parent span has trace_id {format(parent_trace_id, '032x')}. "
            f"They should be the same to avoid duplicate traces."
        )

        # 2. The litellm span should have the parent span as its parent
        assert litellm_parent_span_id == parent_span_id, (
            f"litellm_request span is not a child of the active span! "
            f"Expected parent_span_id {format(parent_span_id, '016x')} "
            f"but got {format(litellm_parent_span_id, '016x') if litellm_parent_span_id else 'None'}"
        )

        print(f"✓ Test passed: litellm_request is a child span in the same trace")
        print(f"  Trace ID: {format(litellm_trace_id, '032x')}")
        print(f"  Parent Span ID: {format(litellm_parent_span_id, '016x')}")

    def test_responses_api_without_active_span_creates_root(self):
        """
        Test that when there's NO active OTEL span,
        the langfuse_otel logger creates a root span (existing behavior).
        """
        from litellm.integrations.langfuse.langfuse_otel import LangfuseOtelLogger
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        from opentelemetry import trace

        # Set up in-memory span exporter to capture spans
        span_exporter = InMemorySpanExporter()
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
        trace.set_tracer_provider(tracer_provider)

        # NO active span - just call _handle_success directly
        logger = LangfuseOtelLogger(
            config=None,
            tracer_provider=tracer_provider
        )

        kwargs = {
            "call_type": "responses",
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "litellm_params": {"metadata": {}},
        }

        response_obj = {
            "id": "resp-123",
            "model": "gpt-4o",
            "choices": [{"message": {"role": "assistant", "content": "Hi there"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }

        start_time = datetime.now()
        end_time = datetime.now()

        logger._handle_success(kwargs, response_obj, start_time, end_time)

        # Force flush
        tracer_provider.force_flush()

        # Get all exported spans
        spans = span_exporter.get_finished_spans()

        # Verify we have spans
        assert len(spans) > 0, "No spans were created"

        # Find the litellm_request span
        litellm_spans = [s for s in spans if s.name == "litellm_request"]
        assert len(litellm_spans) > 0, "No litellm_request span was created"

        litellm_span = litellm_spans[0]

        # This should be a root span (no parent)
        assert litellm_span.parent is None, (
            "Expected litellm_request to be a root span when no active span exists"
        )

        print(f"✓ Test passed: litellm_request is a root span when no parent exists")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
