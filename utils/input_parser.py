"""
Input Parser: User input parsing and command recognition.
Handles typos, fuzzy matching, and command interpretation.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class InputParser:
    """Parses user input and identifies commands vs messages."""

    # Commands that advance to next observation
    NEXT_COMMANDS = {
        "next",
        "next question",
        "skip",
        "proceed",
        "continue",
        "advance",
        "forward",
    }

    # Commands that show help
    HELP_COMMANDS = {
        "help",
        "?",
        "commands",
        "h",
        "info",
    }

    # Commands that exit
    EXIT_COMMANDS = {
        "exit",
        "quit",
        "leave",
        "back",
        "q",
    }

    # Common typos of "next"
    NEXT_TYPOS = {
        "nxt",
        "ne xt",
        "nextt",
        "net",
        "nect",
        "nexy",
        "nect",
        "nx",
        "nex",
    }

    @staticmethod
    def parse(
        text: str,
        phase: str = "active",
        suggestion_threshold: int = 1,
    ) -> tuple[str, str, Optional[str]]:
        """
        Parse user input and identify command vs message.

        Args:
            text: Raw user input
            phase: Current phase (affects available commands)
            suggestion_threshold: Edit distance for typo suggestions

        Returns:
            (input_type, content, suggestion)
            input_type: "command", "message", or "probable_typo"
            content: The parsed command or original message
            suggestion: Suggested correction if probable_typo
        """
        if not text or not text.strip():
            return "empty", "", None

        cleaned = text.strip()
        cleaned_lower = cleaned.lower()

        # ==================== EXACT COMMAND MATCH ====================

        # Check for "next" command (most common)
        if cleaned_lower in InputParser.NEXT_COMMANDS:
            return "command", "next", None

        # Check for help commands
        if cleaned_lower in InputParser.HELP_COMMANDS:
            return "command", "help", None

        # Check for exit commands (only in active/review)
        if phase in ("active", "review") and cleaned_lower in InputParser.EXIT_COMMANDS:
            return "command", "exit", None

        # ==================== FUZZY MATCH FOR "NEXT" ====================

        # Check if starts with "next" (handles "next " with trailing space)
        if cleaned_lower.startswith("next") and len(cleaned_lower) <= 10:
            return "command", "next", None

        # Check for common typos of "next"
        if cleaned_lower in InputParser.NEXT_TYPOS:
            return "probable_typo", cleaned_lower, "next"

        # ==================== EDIT DISTANCE CHECK ====================

        # Check for typos with edit distance <= 1
        suggestion = InputParser.find_typo_suggestion(
            cleaned_lower, threshold=suggestion_threshold
        )
        if suggestion:
            return "probable_typo", cleaned_lower, suggestion

        # ==================== DEFAULT TO MESSAGE ====================

        return "message", cleaned, None

    @staticmethod
    def find_typo_suggestion(text: str, threshold: int = 1) -> Optional[str]:
        """
        Find probable command suggestion using edit distance.

        Returns: Suggested command or None
        """
        if not text or len(text) < 2:
            return None

        all_commands = (
            InputParser.NEXT_COMMANDS
            | InputParser.HELP_COMMANDS
            | InputParser.EXIT_COMMANDS
        )

        # Check each known command
        for cmd in all_commands:
            if InputParser.edit_distance(text, cmd) <= threshold:
                return cmd

        return None

    @staticmethod
    def edit_distance(s1: str, s2: str) -> int:
        """
        Calculate Levenshtein edit distance between two strings.
        Used for typo detection.
        """
        if len(s1) < len(s2):
            return InputParser.edit_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # j+1 instead of j since previous_row and current_row are one character longer
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    @staticmethod
    def get_help_message() -> str:
        """Get help text for user."""
        return """
**Commands:**
- **next** → Move to the next observation
- **help** or **?** → Show this help message

**Navigation Rules:**
- ✅ You can only move **forward** through observations (required for the study)
- ✅ You can revisit previous observations during the review phase
- ✅ You can discuss each observation as long as you want
- ✅ No time limit on individual observations (but 20 minutes total)

**Tips:**
- Type a question or response to discuss the current observation
- Your responses can be as detailed or brief as you like
- The AI will provide feedback based on the therapy session transcript
"""

    @staticmethod
    def get_typo_warning(typo: str, suggestion: str) -> str:
        """Get user-friendly typo warning."""
        return f"""
❓ Did you mean **'{suggestion}'**? 

You typed: `{typo}`

If you want to move to the next observation, type **'next'** exactly.
Otherwise, I'll treat your input as a message to discuss. What would you like to do?
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
