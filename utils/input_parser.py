"""
Input Parser: Utility module for message buffer, help text, and intent detection.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class InputParser:
    """Provides help text and intent detection for users."""

    # Keywords that indicate intent to move to next observation
    NAVIGATION_INTENT_KEYWORDS = {
        "ready", "prepared", "next", "move", "proceed", "advance",
        "forward", "go", "continue", "let's", "lets", "done", "finished",
        "complete", "done with"
    }

    # High-confidence phrases that indicate intent to move to next observation
    NAVIGATION_INTENT_PHRASES = {
        "ready to move on",
        "ready to proceed",
        "ready for next",
        "let's move on",
        "move on",
        "go to next",
        "go next",
        "that's it",
        "done with this",
        "finished with this",
        "moving on",
        "next please",
        "ok next",
        "alright next",
        "ready for the next",
        "ready for the next one",
        "next one",
    }

    @staticmethod
    def detect_navigation_intent(text: str) -> bool:
        """
        Detect if user is expressing intent to move to the next observation.
        Uses keyword-based semantic detection rather than hardcoded phrases.
        
        Examples that would be detected:
        - "im ready for the next one"
        - "lets move to the next"
        - "ready to proceed"
        - "done, moving on"
        - "ok next"
        - "advance to the next observation"
        
        Args:
            text: Lowercase user input
            
        Returns:
            True if navigation intent is detected, False otherwise
        """
        cleaned_lower = text.strip().lower()
        
        # First pass: Check high-confidence phrase list
        for phrase in InputParser.NAVIGATION_INTENT_PHRASES:
            if phrase in cleaned_lower:
                return True
        
        # Second pass: keyword-based semantic detection
        # Split text into words for analysis
        words = set(cleaned_lower.split())
        
        # Count how many navigation keywords are present
        keyword_matches = words & InputParser.NAVIGATION_INTENT_KEYWORDS
        
        # Need at least 2 intent keywords to avoid false positives
        # This prevents triggering on single words like "ready" or "next"
        if len(keyword_matches) < 2:
            return False
        
        # Additional check: avoid triggering on "help" related queries
        if "help" in cleaned_lower or "how" in cleaned_lower or "can" in cleaned_lower and "do" in cleaned_lower:
            return False
        
        # Avoid triggering if user is clearly asking a question about observation
        if "?" in cleaned_lower or "tell me" in cleaned_lower or "explain" in cleaned_lower:
            return False
        
        return True

    @staticmethod
    def get_help_message() -> str:
        """Get help text for user."""
        return """
**Navigation:**
- **⏭️ Next button** → Move to the next observation
- **ℹ️ Help button** → Show this help message

**How to use:**
- Type a question or response to discuss the current observation
- Your responses can be as detailed or brief as you like
- The AI will provide feedback based on the therapy session transcript
- Click **Next** when you're ready to move forward

**Important rules:**
- You can only move **forward** through observations (required for the study)
- You can revisit previous observations during the review phase
- You have **20 minutes total** for the entire session across all observations
- Manage your time strategically

**All your responses are automatically saved.**
"""

    @staticmethod
    def get_navigation_redirect_message() -> str:
        """Get message to redirect user to Next button when they express navigation intent."""
        return """
Looking to move forward? Click the **⏭️ Next** button in the sidebar to proceed to the next observation!

Or, if you have more thoughts on this observation, feel free to share them below.
"""


class MessageBuffer:
    """Manages user messages to prevent duplicates."""

    def __init__(self):
        self.last_message = None
        self.last_message_hash = None

    def add_message(self, message: str) -> bool:
        """
        Add message to buffer.

        Returns: False if duplicate, True if new
        """
        msg_hash = hash(message.strip())

        if msg_hash == self.last_message_hash:
            logger.warning("Duplicate message detected, skipping")
            return False

        self.last_message = message
        self.last_message_hash = msg_hash
        return True

    def clear(self):
        """Clear buffer."""
        self.last_message = None
        self.last_message_hash = None
