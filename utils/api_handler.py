"""
API Error Handler: Robust error handling for OpenAI API calls.
Implements retry logic, error classification, and proper fallback.
"""

import logging
import time
from typing import Generator, Optional

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Base API error."""

    pass


class APIRetryableError(APIError):
    """Error that might succeed on retry (transient)."""

    pass


class APIPermanentError(APIError):
    """Error that won't succeed on retry (permanent)."""

    pass


class APITimeoutError(APIRetryableError):
    """API request timed out."""

    pass


class OpenAIHandler:
    """Handles OpenAI API interactions with retry logic and error handling."""

    # Errors that should trigger a retry
    RETRYABLE_ERROR_TYPES = (
        "RateLimitError",
        "APIError",
        "APIConnectionError",
        "Timeout",
        "TimeoutError",
        "ConnectionError",
    )

    # Errors that are permanent
    PERMANENT_ERROR_TYPES = (
        "AuthenticationError",
        "PermissionError",
        "InvalidRequestError",
        "NotFoundError",
    )

    DEFAULT_MAX_RETRIES = 2
    DEFAULT_TIMEOUT = 30
    INITIAL_RETRY_WAIT = 2  # seconds

    @staticmethod
    def classify_error(error: Exception) -> str:
        """Classify error as retryable, permanent, or other."""
        error_type = type(error).__name__
        error_str = str(error).lower()

        # Check by exception type
        if error_type in OpenAIHandler.RETRYABLE_ERROR_TYPES:
            return "retryable"

        if error_type in OpenAIHandler.PERMANENT_ERROR_TYPES:
            return "permanent"

        # Check by error message content
        if any(
            phrase in error_str
            for phrase in ["timeout", "connection", "rate limit", "429"]
        ):
            return "retryable"

        return "unknown"

    @staticmethod
    def should_retry(error: Exception, retry_count: int, max_retries: int) -> bool:
        """Determine if error warrants a retry."""
        if retry_count >= max_retries:
            return False

        classification = OpenAIHandler.classify_error(error)
        return classification == "retryable"

    @staticmethod
    def get_retry_wait(retry_count: int, base_wait: int = 2) -> int:
        """Calculate exponential backoff wait time."""
        # Wait: 2, 4, 8 seconds for retries 1, 2, 3
        return base_wait * (2 ** retry_count)

    @staticmethod
    def openai_complete(
        history: list,
        system_text: str,
        client,
        model_name: str = "gpt-4o-mini",
        stream: bool = False,
        max_tokens: int = 512,
        max_retries: int = 2,
        timeout: int = 30,
    ) -> Optional[Generator[str, None, None]] or str:
        """
        Complete a chat using OpenAI API with robust error handling.

        Args:
            history: Chat history
            system_text: System prompt
            client: OpenAI client instance
            model_name: Model to use
            stream: Whether to stream response
            max_tokens: Max tokens in response
            max_retries: Number of retries on transient errors
            timeout: Request timeout in seconds

        Returns:
            Generator (if stream=True) or string (if stream=False)

        Raises:
            APIPermanentError: For permanent failures
            APIRetryableError: For transient failures after retries exhausted
        """

        if client is None:
            raise APIPermanentError("OpenAI client not initialized")

        # Build messages
        messages = []
        if system_text:
            messages.append({"role": "system", "content": system_text})

        for m in history:
            if m.get("role") in ("user", "assistant"):
                messages.append({"role": m["role"], "content": m["content"]})

        retry_count = 0
        last_error = None

        while retry_count <= max_retries:
            try:
                if stream:
                    def stream_response():
                        """Generator for streaming responses."""
                        try:
                            resp = client.chat.completions.create(
                                model=model_name,
                                messages=messages,
                                stream=True,
                                max_tokens=max_tokens,
                                temperature=0.3,
                                timeout=timeout,
                            )
                            for chunk in resp:
                                delta = getattr(
                                    chunk.choices[0].delta, "content", None
                                )
                                if delta:
                                    yield delta
                        except Exception as e:
                            logger.error(f"Streaming error: {type(e).__name__}: {e}")
                            raise

                    return stream_response()
                else:
                    resp = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=0.3,
                        timeout=timeout,
                    )
                    return (resp.choices[0].message.content or "").strip()

            except Exception as e:
                error_type = type(e).__name__
                last_error = e
                should_retry = OpenAIHandler.should_retry(
                    e, retry_count, max_retries
                )

                if should_retry:
                    retry_count += 1
                    wait_time = OpenAIHandler.get_retry_wait(retry_count - 1)

                    logger.warning(
                        f"[Attempt {retry_count}/{max_retries}] "
                        f"Retryable error ({error_type}): {str(e)[:100]}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    # Permanent error or no retries left
                    error_msg = f"{error_type}: {str(e)}"
                    logger.error(f"API error (not retrying): {error_msg}")

                    if retry_count >= max_retries:
                        raise APIRetryableError(
                            f"API request failed after {max_retries} retries: {error_msg}"
                        ) from e
                    else:
                        raise APIPermanentError(error_msg) from e

        # Should not reach here
        raise APIRetryableError(f"Unexpected error after retries: {last_error}")

    @staticmethod
    def format_error_message(error: Exception) -> str:
        """Format error for user display."""
        error_type = type(error).__name__
        error_msg = str(error)

        if isinstance(error, APIRetryableError):
            if "after" in error_msg and "retries" in error_msg:
                return "⏱️ The AI took too long to respond. Please try again."
            return "⚠️ Temporary error. Please try again in a moment."

        elif isinstance(error, APIPermanentError):
            if "auth" in error_msg.lower():
                return "⚠️ Authentication error. Please contact support."
            if "key" in error_msg.lower():
                return "⚠️ API key not configured. Please contact support."
            if "invalid" in error_msg.lower():
                return "⚠️ Invalid request. Please try rephrasing your message."
            return "⚠️ Cannot generate response. Please contact support."

        return f"⚠️ Error: {error_type}. Please try again or contact support."
