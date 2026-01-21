"""
Session Manager: Persistent storage for chat sessions.
Simplified version for the core TeamMait application.
"""

import json
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages persistent session storage for chat sessions."""

    STORAGE_DIR = Path("user_sessions")
    SESSION_TIMEOUT_HOURS = 24

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
            "status": "active",
        }

        user_dir = cls._ensure_dir(username)
        metadata_path = user_dir / f"{session_id}_metadata.json"
        
        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to create session metadata: {e}")
            raise

        # Initialize conversation storage
        conversations_path = user_dir / f"{session_id}_conversations.json"
        try:
            with open(conversations_path, "w") as f:
                json.dump({"messages": []}, f, indent=2)
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
    def save_session_metadata(cls, username: str, session_id: str, metadata: Dict) -> bool:
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
        """Load conversation history."""
        user_dir = cls._ensure_dir(username)
        conversations_path = user_dir / f"{session_id}_conversations.json"

        if not conversations_path.exists():
            return {"messages": []}

        try:
            with open(conversations_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load conversations: {e}")
            return {"messages": []}

    @classmethod
    def save_conversations(cls, username: str, session_id: str, conversations: Dict) -> bool:
        """Save conversation history."""
        user_dir = cls._ensure_dir(username)
        conversations_path = user_dir / f"{session_id}_conversations.json"

        try:
            with open(conversations_path, "w") as f:
                json.dump(conversations, f, indent=2)
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
    def session_exists(cls, username: str, session_id: str) -> bool:
        """Check if a session exists and is still valid."""
        metadata = cls.load_session_metadata(username, session_id)
        return metadata is not None and metadata.get("status") == "active"
