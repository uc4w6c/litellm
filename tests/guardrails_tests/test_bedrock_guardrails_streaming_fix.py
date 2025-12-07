"""
Test for Bedrock Guardrails streaming fix - ensuring original chunks are preserved
when no masking is needed.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import BedrockGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from unittest.mock import AsyncMock, MagicMock, patch
from litellm.types.utils import ModelResponseStream


@pytest.mark.asyncio
async def test_streaming_preserves_original_chunks_when_no_masking():
    """
    Test that when guardrails pass with no masking, the original streaming chunks
    are returned to preserve the real-time streaming behavior.

    This is the fix for the issue where streaming was being converted to a single
    chunk even when no masking was needed.
    """
    # Create proper mock objects
    mock_user_api_key_dict = UserAPIKeyAuth()

    # Create guardrail instance
    guardrail = BedrockGuardrail(
        guardrailIdentifier="test-guardrail",
        guardrailVersion="DRAFT"
    )

    # Mock streaming chunks with NO PII
    original_chunks = [
        ModelResponseStream(
            id="test-id",
            choices=[
                litellm.utils.StreamingChoices(
                    index=0,
                    delta=litellm.utils.Delta(content="Hello, "),
                    finish_reason=None
                )
            ],
            created=1234567890,
            model="gpt-4o",
            object="chat.completion.chunk"
        ),
        ModelResponseStream(
            id="test-id",
            choices=[
                litellm.utils.StreamingChoices(
                    index=0,
                    delta=litellm.utils.Delta(content="how "),
                    finish_reason=None
                )
            ],
            created=1234567890,
            model="gpt-4o",
            object="chat.completion.chunk"
        ),
        ModelResponseStream(
            id="test-id",
            choices=[
                litellm.utils.StreamingChoices(
                    index=0,
                    delta=litellm.utils.Delta(content="can "),
                    finish_reason=None
                )
            ],
            created=1234567890,
            model="gpt-4o",
            object="chat.completion.chunk"
        ),
        ModelResponseStream(
            id="test-id",
            choices=[
                litellm.utils.StreamingChoices(
                    index=0,
                    delta=litellm.utils.Delta(content="I help you?"),
                    finish_reason="stop"
                )
            ],
            created=1234567890,
            model="gpt-4o",
            object="chat.completion.chunk"
        )
    ]

    async def mock_streaming_response():
        for chunk in original_chunks:
            yield chunk

    # Mock Bedrock API response with NO masking (action=NONE or no outputs)
    mock_bedrock_response_no_masking = MagicMock()
    mock_bedrock_response_no_masking.status_code = 200
    mock_bedrock_response_no_masking.json.return_value = {
        "action": "NONE",
        "outputs": [],  # No masking outputs
        "assessments": []
    }

    request_data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "stream": True
    }

    # Mock the make_bedrock_api_request method
    async def mock_make_bedrock_api_request(source, messages=None, response=None, request_data=None):
        from litellm.types.proxy.guardrails.guardrail_hooks.bedrock_guardrails import BedrockGuardrailResponse
        return BedrockGuardrailResponse(**mock_bedrock_response_no_masking.json())

    # Patch the bedrock API request method
    with patch.object(guardrail, 'make_bedrock_api_request', side_effect=mock_make_bedrock_api_request):

        # Call the streaming hook
        result_generator = guardrail.async_post_call_streaming_iterator_hook(
            user_api_key_dict=mock_user_api_key_dict,
            response=mock_streaming_response(),
            request_data=request_data
        )

        # Collect all chunks from the result
        result_chunks = []
        async for chunk in result_generator:
            result_chunks.append(chunk)

        # CRITICAL: Verify that we got the same NUMBER of chunks
        assert len(result_chunks) == len(original_chunks), \
            f"Expected {len(original_chunks)} chunks to preserve streaming, but got {len(result_chunks)}"

        # Verify each chunk's content matches the original
        for i, (original, result) in enumerate(zip(original_chunks, result_chunks)):
            original_content = original.choices[0].delta.content
            result_content = result.choices[0].delta.content if hasattr(result.choices[0], 'delta') else None

            assert original_content == result_content, \
                f"Chunk {i}: Expected '{original_content}', got '{result_content}'"

        # Verify full content is correct
        full_content = ""
        for chunk in result_chunks:
            if hasattr(chunk, 'choices') and chunk.choices:
                if hasattr(chunk.choices[0], 'delta') and chunk.choices[0].delta.content:
                    full_content += chunk.choices[0].delta.content

        expected_content = "Hello, how can I help you?"
        assert full_content == expected_content, \
            f"Expected '{expected_content}', got '{full_content}'"

        print("✅ Streaming preserves original chunks when no masking - TEST PASSED")
        print(f"✅ Original chunks: {len(original_chunks)}")
        print(f"✅ Result chunks: {len(result_chunks)}")
        print(f"✅ Content: {full_content}")


@pytest.mark.asyncio
async def test_streaming_rebuilds_to_single_chunk_when_masking_needed():
    """
    Test that when guardrails apply masking, the stream is rebuilt as a single chunk
    with the masked content (this is expected behavior when masking is needed).
    """
    # Create proper mock objects
    mock_user_api_key_dict = UserAPIKeyAuth()

    # Create guardrail instance
    guardrail = BedrockGuardrail(
        guardrailIdentifier="test-guardrail",
        guardrailVersion="DRAFT"
    )

    # Mock streaming chunks WITH PII that will be masked
    original_chunks = [
        ModelResponseStream(
            id="test-id",
            choices=[
                litellm.utils.StreamingChoices(
                    index=0,
                    delta=litellm.utils.Delta(content="My email is "),
                    finish_reason=None
                )
            ],
            created=1234567890,
            model="gpt-4o",
            object="chat.completion.chunk"
        ),
        ModelResponseStream(
            id="test-id",
            choices=[
                litellm.utils.StreamingChoices(
                    index=0,
                    delta=litellm.utils.Delta(content="john@example.com"),
                    finish_reason="stop"
                )
            ],
            created=1234567890,
            model="gpt-4o",
            object="chat.completion.chunk"
        )
    ]

    async def mock_streaming_response():
        for chunk in original_chunks:
            yield chunk

    # Mock Bedrock API response WITH masking
    mock_bedrock_response_with_masking = MagicMock()
    mock_bedrock_response_with_masking.status_code = 200
    mock_bedrock_response_with_masking.json.return_value = {
        "action": "GUARDRAIL_INTERVENED",
        "outputs": [{
            "text": "My email is {EMAIL}"
        }],
        "assessments": [{
            "sensitiveInformationPolicy": {
                "piiEntities": [{
                    "type": "EMAIL",
                    "match": "john@example.com",
                    "action": "ANONYMIZED"
                }]
            }
        }]
    }

    request_data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "What's your email?"}
        ],
        "stream": True
    }

    # Mock the make_bedrock_api_request method
    async def mock_make_bedrock_api_request(source, messages=None, response=None, request_data=None):
        from litellm.types.proxy.guardrails.guardrail_hooks.bedrock_guardrails import BedrockGuardrailResponse
        return BedrockGuardrailResponse(**mock_bedrock_response_with_masking.json())

    # Patch the bedrock API request method
    with patch.object(guardrail, 'make_bedrock_api_request', side_effect=mock_make_bedrock_api_request):

        # Call the streaming hook
        result_generator = guardrail.async_post_call_streaming_iterator_hook(
            user_api_key_dict=mock_user_api_key_dict,
            response=mock_streaming_response(),
            request_data=request_data
        )

        # Collect all chunks from the result
        result_chunks = []
        async for chunk in result_generator:
            result_chunks.append(chunk)

        # When masking is needed, we expect a SINGLE chunk (rebuilt via MockResponseIterator)
        assert len(result_chunks) == 1, \
            f"Expected 1 chunk when masking is applied (MockResponseIterator), but got {len(result_chunks)}"

        # Verify the content is masked
        full_content = ""
        for chunk in result_chunks:
            if hasattr(chunk, 'choices') and chunk.choices:
                if hasattr(chunk.choices[0], 'delta') and chunk.choices[0].delta.content:
                    full_content += chunk.choices[0].delta.content

        # The content should be masked
        assert "{EMAIL}" in full_content, \
            f"Expected masked content with {{EMAIL}}, got: {full_content}"
        assert "john@example.com" not in full_content, \
            f"Original email should be masked, got: {full_content}"

        print("✅ Streaming rebuilds to single chunk when masking needed - TEST PASSED")
        print(f"✅ Result chunks: {len(result_chunks)}")
        print(f"✅ Masked content: {full_content}")


@pytest.mark.asyncio
async def test_streaming_chunk_count_comparison():
    """
    Compare chunk counts between no-masking and masking scenarios to demonstrate
    the fix preserves streaming performance.
    """
    mock_user_api_key_dict = UserAPIKeyAuth()
    guardrail = BedrockGuardrail(
        guardrailIdentifier="test-guardrail",
        guardrailVersion="DRAFT"
    )

    # Create a stream with many chunks (simulating real streaming)
    num_original_chunks = 10
    original_chunks = [
        ModelResponseStream(
            id="test-id",
            choices=[
                litellm.utils.StreamingChoices(
                    index=0,
                    delta=litellm.utils.Delta(content=f"chunk{i} "),
                    finish_reason="stop" if i == num_original_chunks - 1 else None
                )
            ],
            created=1234567890,
            model="gpt-4o",
            object="chat.completion.chunk"
        )
        for i in range(num_original_chunks)
    ]

    async def mock_streaming_response():
        for chunk in original_chunks:
            yield chunk

    # Test 1: No masking - should preserve all chunks
    mock_bedrock_no_masking = MagicMock()
    mock_bedrock_no_masking.status_code = 200
    mock_bedrock_no_masking.json.return_value = {
        "action": "NONE",
        "outputs": [],
        "assessments": []
    }

    async def mock_make_bedrock_no_masking(source, messages=None, response=None, request_data=None):
        from litellm.types.proxy.guardrails.guardrail_hooks.bedrock_guardrails import BedrockGuardrailResponse
        return BedrockGuardrailResponse(**mock_bedrock_no_masking.json())

    with patch.object(guardrail, 'make_bedrock_api_request', side_effect=mock_make_bedrock_no_masking):
        result_generator = guardrail.async_post_call_streaming_iterator_hook(
            user_api_key_dict=mock_user_api_key_dict,
            response=mock_streaming_response(),
            request_data={"messages": [], "stream": True}
        )

        no_masking_chunks = []
        async for chunk in result_generator:
            no_masking_chunks.append(chunk)

    # Test 2: With masking - should return 1 chunk
    mock_bedrock_with_masking = MagicMock()
    mock_bedrock_with_masking.status_code = 200
    mock_bedrock_with_masking.json.return_value = {
        "action": "GUARDRAIL_INTERVENED",
        "outputs": [{"text": "masked content"}],
        "assessments": []
    }

    async def mock_streaming_response2():
        for chunk in original_chunks:
            yield chunk

    async def mock_make_bedrock_with_masking(source, messages=None, response=None, request_data=None):
        from litellm.types.proxy.guardrails.guardrail_hooks.bedrock_guardrails import BedrockGuardrailResponse
        return BedrockGuardrailResponse(**mock_bedrock_with_masking.json())

    with patch.object(guardrail, 'make_bedrock_api_request', side_effect=mock_make_bedrock_with_masking):
        result_generator = guardrail.async_post_call_streaming_iterator_hook(
            user_api_key_dict=mock_user_api_key_dict,
            response=mock_streaming_response2(),
            request_data={"messages": [], "stream": True}
        )

        masking_chunks = []
        async for chunk in result_generator:
            masking_chunks.append(chunk)

    # Assertions
    assert len(no_masking_chunks) == num_original_chunks, \
        f"No masking: Expected {num_original_chunks} chunks, got {len(no_masking_chunks)}"

    assert len(masking_chunks) == 1, \
        f"With masking: Expected 1 chunk, got {len(masking_chunks)}"

    print("✅ Chunk count comparison - TEST PASSED")
    print(f"✅ Original chunks: {num_original_chunks}")
    print(f"✅ No masking result: {len(no_masking_chunks)} chunks (preserves streaming)")
    print(f"✅ With masking result: {len(masking_chunks)} chunk (rebuilt)")
