"""
Analytics Logger: Session event tracking and logging.
Enables monitoring, debugging, and optimization of guided interactions.
"""

import logging
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Create logs directory
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)


class AnalyticsLogger:
    """Logs session events for analytics and monitoring."""

    def __init__(self, log_file: str = "session_analytics.jsonl"):
        self.log_path = LOGS_DIR / log_file
        self.setup_logger()

    def setup_logger(self):
        """Set up analytics logger with JSON formatting."""
        self.logger = logging.getLogger("analytics")
        self.logger.setLevel(logging.INFO)

        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # File handler with JSON formatting
        handler = logging.FileHandler(self.log_path)
        handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(handler)

        # Also log to stdout in debug mode
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(console_handler)

    def log_event(
        self,
        event_type: str,
        username: Optional[str] = None,
        session_id: Optional[str] = None,
        elapsed_seconds: Optional[float] = None,
        data: Optional[Dict[str, Any]] = None,
        level: str = "info",
    ):
        """Log a session event."""
        event_record = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "username": username,
            "session_id": session_id,
            "elapsed_seconds": elapsed_seconds,
        }

        if data:
            event_record.update(data)

        # Log as JSON
        log_method = getattr(self.logger, level, self.logger.info)
        log_method(json.dumps(event_record))

    def session_started(
        self,
        username: str,
        session_id: str,
    ):
        """Log session start."""
        self.log_event(
            "session_started",
            username=username,
            session_id=session_id,
            data={
                "phase": "intro",
            },
        )

    def phase_transition(
        self,
        username: str,
        session_id: str,
        from_phase: str,
        to_phase: str,
        elapsed_seconds: float,
        data: Optional[Dict] = None,
    ):
        """Log phase transition."""
        event_data = {
            "from_phase": from_phase,
            "to_phase": to_phase,
        }
        if data:
            event_data.update(data)

        self.log_event(
            "phase_transition",
            username=username,
            session_id=session_id,
            elapsed_seconds=elapsed_seconds,
            data=event_data,
        )

    def observation_started(
        self,
        username: str,
        session_id: str,
        observation_idx: int,
        elapsed_seconds: float,
        assertion: Optional[str] = None,
    ):
        """Log when user starts discussing an observation."""
        self.log_event(
            "observation_started",
            username=username,
            session_id=session_id,
            elapsed_seconds=elapsed_seconds,
            data={
                "observation_idx": observation_idx,
                "assertion": assertion[:100] if assertion else None,
            },
        )

    def user_message(
        self,
        username: str,
        session_id: str,
        observation_idx: int,
        elapsed_seconds: float,
        message_length: int,
        input_type: str,
    ):
        """Log user message."""
        self.log_event(
            "user_message",
            username=username,
            session_id=session_id,
            elapsed_seconds=elapsed_seconds,
            data={
                "observation_idx": observation_idx,
                "message_length": message_length,
                "input_type": input_type,
            },
        )

    def ai_response(
        self,
        username: str,
        session_id: str,
        observation_idx: int,
        elapsed_seconds: float,
        response_length: int,
        tokens_estimated: int,
        generation_time_seconds: float,
    ):
        """Log AI response."""
        self.log_event(
            "ai_response",
            username=username,
            session_id=session_id,
            elapsed_seconds=elapsed_seconds,
            data={
                "observation_idx": observation_idx,
                "response_length": response_length,
                "tokens_estimated": tokens_estimated,
                "generation_time_seconds": generation_time_seconds,
            },
        )

    def observation_advanced(
        self,
        username: str,
        session_id: str,
        from_idx: int,
        to_idx: int,
        elapsed_seconds: float,
        messages_count: int,
        time_spent_seconds: float,
    ):
        """Log when user advances to next observation."""
        self.log_event(
            "observation_advanced",
            username=username,
            session_id=session_id,
            elapsed_seconds=elapsed_seconds,
            data={
                "from_idx": from_idx,
                "to_idx": to_idx,
                "messages_count": messages_count,
                "time_spent_seconds": time_spent_seconds,
            },
        )

    def observation_revisited(
        self,
        username: str,
        session_id: str,
        from_idx: int,
        to_idx: int,
        elapsed_seconds: float,
    ):
        """Log when user revisits an observation in review phase."""
        self.log_event(
            "observation_revisited",
            username=username,
            session_id=session_id,
            elapsed_seconds=elapsed_seconds,
            data={
                "from_idx": from_idx,
                "to_idx": to_idx,
            },
        )

    def session_time_expired(
        self,
        username: str,
        session_id: str,
        observation_idx: int,
        observations_completed: int,
        total_messages: int,
        elapsed_seconds: float,
    ):
        """Log when session time expires."""
        self.log_event(
            "session_time_expired",
            username=username,
            session_id=session_id,
            elapsed_seconds=elapsed_seconds,
            data={
                "observation_idx": observation_idx,
                "observations_completed": observations_completed,
                "total_messages": total_messages,
            },
        )

    def session_completed(
        self,
        username: str,
        session_id: str,
        observations_reviewed: int,
        total_messages: int,
        elapsed_seconds: float,
        status: str = "success",
    ):
        """Log session completion."""
        self.log_event(
            "session_completed",
            username=username,
            session_id=session_id,
            elapsed_seconds=elapsed_seconds,
            data={
                "observations_reviewed": observations_reviewed,
                "total_messages": total_messages,
                "status": status,
                "avg_messages_per_observation": (
                    total_messages / observations_reviewed
                    if observations_reviewed > 0
                    else 0
                ),
            },
        )

    def error_occurred(
        self,
        username: str,
        session_id: str,
        error_type: str,
        error_message: str,
        elapsed_seconds: float,
        context: Optional[Dict] = None,
    ):
        """Log error event."""
        event_data = {
            "error_type": error_type,
            "error_message": error_message[:200],
        }
        if context:
            event_data.update(context)

        self.log_event(
            "error_occurred",
            username=username,
            session_id=session_id,
            elapsed_seconds=elapsed_seconds,
            data=event_data,
            level="error",
        )

    def rag_load_status(
        self,
        username: str,
        session_id: str,
        reference_loaded: bool,
        supporting_count: int,
        total_documents: int,
        errors: Optional[list] = None,
    ):
        """Log RAG document loading status."""
        self.log_event(
            "rag_load_status",
            username=username,
            session_id=session_id,
            data={
                "reference_loaded": reference_loaded,
                "supporting_count": supporting_count,
                "total_documents": total_documents,
                "errors": errors or [],
            },
        )


# Global analytics instance
_analytics = None


def get_analytics() -> AnalyticsLogger:
    """Get or create global analytics instance."""
    global _analytics
    if _analytics is None:
        _analytics = AnalyticsLogger()
    return _analytics
