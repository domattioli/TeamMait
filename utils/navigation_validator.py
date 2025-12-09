"""
Navigation Validator: Server-side validation for forward-only navigation.
Enforces experimental protocol constraints.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class NavigationValidator:
    """Validates navigation according to experiment protocol."""

    # Define valid phase transitions
    PHASES_TIMELINE = {
        "intro": ["active"],
        "active": ["active", "expired", "review"],
        "review": ["review", "active", "complete"],
        "expired": ["complete"],
        "complete": [],
    }

    @staticmethod
    def can_transition(
        current_phase: str, new_phase: str, reason: Optional[str] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Check if phase transition is allowed.

        Returns:
            (is_allowed, error_message)
        """
        allowed = NavigationValidator.PHASES_TIMELINE.get(current_phase, [])

        if new_phase not in allowed:
            error_msg = (
                f"Invalid phase transition: {current_phase} → {new_phase}. "
                f"Allowed transitions from {current_phase}: {allowed}"
            )
            logger.warning(f"{error_msg} (reason: {reason})")
            return False, error_msg

        return True, None

    @staticmethod
    def can_advance_question(
        current_idx: int,
        target_idx: int,
        phase: str,
        total_questions: int = 4,
    ) -> tuple[bool, Optional[str]]:
        """
        Validate question advancement.

        Rules:
        - In 'active' phase: can only go forward by exactly 1
        - In 'review' phase: can jump to any previous or current question
        - Cannot go backward in active phase
        - Cannot go past end

        Returns:
            (is_allowed, error_message)
        """
        # Check bounds
        if target_idx < 0:
            error_msg = f"Cannot navigate to question {target_idx} (negative index)"
            logger.warning(error_msg)
            return False, error_msg

        if target_idx >= total_questions:
            error_msg = (
                f"Cannot navigate to question {target_idx} (only {total_questions} available)"
            )
            logger.warning(error_msg)
            return False, error_msg

        if phase == "active":
            # In active phase: only move forward by 1, or stay on same question
            if target_idx > current_idx + 1:
                error_msg = (
                    f"Cannot skip questions in active phase: {current_idx} → {target_idx}"
                )
                logger.warning(error_msg)
                return False, error_msg

            if target_idx < current_idx:
                error_msg = (
                    f"Cannot go backward in active phase: {current_idx} → {target_idx}"
                )
                logger.warning(error_msg)
                return False, error_msg

            return True, None

        elif phase == "review":
            # In review phase: can revisit any question up to max completed
            if target_idx < current_idx:
                # Revisiting earlier question - allowed
                return True, None
            elif target_idx == current_idx:
                # Staying on same - allowed
                return True, None
            elif target_idx == current_idx + 1:
                # Moving forward in review (shouldn't happen but allow)
                return True, None
            else:
                error_msg = (
                    f"Invalid navigation in review phase: {current_idx} → {target_idx}"
                )
                logger.warning(error_msg)
                return False, error_msg

        else:
            error_msg = f"Question navigation not applicable in {phase} phase"
            logger.warning(error_msg)
            return False, error_msg

    @staticmethod
    def validate_navigation(
        current_phase: str,
        current_question_idx: int,
        target_phase: str,
        target_question_idx: int,
        total_questions: int = 4,
    ) -> tuple[bool, Optional[str]]:
        """
        Comprehensive navigation validation.

        Returns:
            (is_allowed, error_message)
        """
        # First check phase transition
        is_valid, error_msg = NavigationValidator.can_transition(
            current_phase, target_phase, f"from Q{current_question_idx}"
        )
        if not is_valid:
            return False, error_msg

        # Then check question navigation if applicable
        if target_phase in ("active", "review"):
            is_valid, error_msg = NavigationValidator.can_advance_question(
                current_question_idx,
                target_question_idx,
                target_phase,
                total_questions,
            )
            if not is_valid:
                return False, error_msg

        return True, None

    @staticmethod
    def get_next_valid_question(
        current_idx: int, phase: str, total_questions: int = 4
    ) -> Optional[int]:
        """Get the next valid question index after current."""
        if phase == "active":
            next_idx = current_idx + 1
            if next_idx < total_questions:
                return next_idx
            return None  # No next question, go to review
        elif phase == "review":
            return current_idx  # Stay on same question in review
        return None
