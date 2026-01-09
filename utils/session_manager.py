"""
Session Manager: Persistent storage for guided interaction sessions.
Handles session initialization, state persistence, and recovery.
"""

import json
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages persistent session storage for guided interactions."""

    STORAGE_DIR = Path("user_sessions")
    CLEANUP_AGE_HOURS = 48
    SESSION_TIMEOUT_HOURS = 2

    @classmethod
    def _ensure_dir(cls, username: str) -> Path:
        """Ensure user directory exists."""
        user_dir = cls.STORAGE_DIR / username
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir

    @classmethod
    def create_session(cls, username: str, session_id: str) -> Dict:
        """Create and initialize a new session."""
        metadata = {
            "session_id": session_id,
            "username": username,
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "phase": "intro",
            "current_question_idx": 0,
            "total_messages": 0,
            "status": "active",
            "observations_visited": [],
        }

        user_dir = cls._ensure_dir(username)

        # Save metadata
        metadata_path = user_dir / f"{session_id}_metadata.json"
        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to create session metadata: {e}")
            raise

        # Initialize conversation storage (3 observations in Module 2)
        conversations = {str(i): [] for i in range(3)}
        conversations_path = user_dir / f"{session_id}_conversations.json"
        try:
            with open(conversations_path, "w") as f:
                json.dump(conversations, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to create conversations file: {e}")
            raise

        logger.info(f"Created session {session_id} for user {username}")
        return metadata

    @classmethod
    def load_session_metadata(cls, username: str, session_id: str) -> Optional[Dict]:
        """Load session metadata."""
        user_dir = cls._ensure_dir(username)
        metadata_path = user_dir / f"{session_id}_metadata.json"

        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                
                # Check if session has timed out
                last_activity = datetime.fromisoformat(metadata["last_activity"])
                age = datetime.now() - last_activity
                
                if age > timedelta(hours=cls.SESSION_TIMEOUT_HOURS):
                    metadata["status"] = "timeout"
                    cls.save_session_metadata(username, session_id, metadata)
                    return None
                
                return metadata
        except Exception as e:
            logger.error(f"Failed to load session metadata: {e}")
            return None

    @classmethod
    def save_session_metadata(
        cls, username: str, session_id: str, metadata: Dict
    ) -> bool:
        """Save session metadata."""
        user_dir = cls._ensure_dir(username)
        metadata_path = user_dir / f"{session_id}_metadata.json"

        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save session metadata: {e}")
            return False

    @classmethod
    def load_conversations(cls, username: str, session_id: str) -> Dict[str, List]:
        """Load all conversation history."""
        user_dir = cls._ensure_dir(username)
        conversations_path = user_dir / f"{session_id}_conversations.json"

        if not conversations_path.exists():
            return {str(i): [] for i in range(3)}

        try:
            with open(conversations_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load conversations: {e}")
            return {str(i): [] for i in range(3)}

    @classmethod
    def save_conversations(
        cls, username: str, session_id: str, conversations: Dict
    ) -> bool:
        """Save all conversation history."""
        user_dir = cls._ensure_dir(username)
        conversations_path = user_dir / f"{session_id}_conversations.json"

        try:
            # Convert int keys to strings for JSON
            conversations_to_save = {
                str(k): v for k, v in conversations.items()
            }
            with open(conversations_path, "w") as f:
                json.dump(conversations_to_save, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save conversations: {e}")
            return False

    @classmethod
    def update_session_activity(cls, username: str, session_id: str) -> bool:
        """Update last activity timestamp."""
        metadata = cls.load_session_metadata(username, session_id)
        if metadata and metadata.get("status") == "active":
            metadata["last_activity"] = datetime.now().isoformat()
            return cls.save_session_metadata(username, session_id, metadata)
        return False

    @classmethod
    def complete_session(
        cls, username: str, session_id: str, final_state: Dict
    ) -> bool:
        """Mark session as complete and save final state."""
        metadata = cls.load_session_metadata(username, session_id)
        if metadata:
            metadata["status"] = "completed"
            metadata["completed_at"] = datetime.now().isoformat()
            metadata["final_state"] = final_state
            return cls.save_session_metadata(username, session_id, metadata)
        return False

    @classmethod
    def cleanup_old_sessions(cls, username: str, max_age_hours: int = 48) -> int:
        """Delete sessions older than max_age_hours. Returns count deleted."""
        user_dir = cls._ensure_dir(username)
        now = datetime.now()
        deleted_count = 0

        try:
            for metadata_file in user_dir.glob("*_metadata.json"):
                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)

                    created = datetime.fromisoformat(metadata["created_at"])
                    age_hours = (now - created).total_seconds() / 3600

                    # Only delete completed/timed-out sessions older than threshold
                    if (
                        age_hours > max_age_hours
                        and metadata.get("status") != "active"
                    ):
                        session_id = metadata["session_id"]
                        metadata_file.unlink()
                        conversations_file = (
                            user_dir / f"{session_id}_conversations.json"
                        )
                        if conversations_file.exists():
                            conversations_file.unlink()
                        logger.info(f"Cleaned up old session {session_id}")
                        deleted_count += 1
                except Exception as e:
                    logger.error(f"Error cleaning up session file: {e}")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

        return deleted_count

    @classmethod
    def session_exists(cls, username: str, session_id: str) -> bool:
        """Check if a session exists and is still valid."""
        metadata = cls.load_session_metadata(username, session_id)
        return metadata is not None and metadata.get("status") == "active"

    @classmethod
    def get_session_summary(cls, username: str, session_id: str) -> Optional[Dict]:
        """Get session summary for display."""
        metadata = cls.load_session_metadata(username, session_id)
        conversations = cls.load_conversations(username, session_id)

        if not metadata:
            return None

        return {
            "session_id": session_id,
            "status": metadata.get("status"),
            "created_at": metadata.get("created_at"),
            "last_activity": metadata.get("last_activity"),
            "phase": metadata.get("phase"),
            "current_question": metadata.get("current_question_idx"),
            "total_messages": sum(len(c) for c in conversations.values()),
            "elapsed_seconds": (
                datetime.now()
                - datetime.fromisoformat(metadata.get("created_at", datetime.now().isoformat()))
            ).total_seconds(),
        }
