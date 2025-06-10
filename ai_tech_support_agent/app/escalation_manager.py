import logging
import uuid
from typing import List, Tuple, Optional

from app import config # For ESCALATION_KEYWORDS and MAX_TURNS_BEFORE_ESCALATION_CHECK

logger = logging.getLogger(__name__)

def check_for_escalation(
    session_id: str,
    user_query: str,
    conversation_history: List[str]
) -> Tuple[bool, Optional[str]]:
    """
    Checks if the current interaction meets criteria for escalation to a human agent.

    Args:
        session_id: The ID of the current session.
        user_query: The latest query from the user.
        conversation_history: The list of all turns in the current conversation.

    Returns:
        A tuple (should_escalate: bool, reason: Optional[str]).
        'reason' is a string explaining why escalation is suggested, or None if no escalation.
    """
    normalized_query = user_query.lower()

    # 1. Check for explicit escalation keywords in the user's query
    for keyword in config.ESCALATION_KEYWORDS:
        if keyword in normalized_query:
            reason = f"User explicitly requested escalation with keyword: '{keyword}'."
            logger.info(f"Escalation check for session {session_id}: True. Reason: {reason}")
            return True, reason

    # 2. Check for repeated failures (conversation length heuristic)
    # Note: conversation_history contains pairs of "User: ..." and "AI: ..."
    # So, MAX_TURNS_BEFORE_ESCALATION_CHECK refers to total items in history.
    if len(conversation_history) >= config.MAX_TURNS_BEFORE_ESCALATION_CHECK:
        # This is a simple heuristic. More sophisticated checks could analyze sentiment,
        # repetition of issues, or lack of positive feedback markers if those were implemented.
        reason = (f"Conversation length ({len(conversation_history)} items) met or exceeded "
                  f"threshold ({config.MAX_TURNS_BEFORE_ESCALATION_CHECK}) without resolution.")
        logger.info(f"Escalation check for session {session_id}: True. Reason: {reason}")
        # For this heuristic to be more effective, we might also want to check if the *current* query
        # indicates frustration or is a repeat of a previous one. For now, length is the trigger.
        return True, reason

    # logger.debug(f"Escalation check for session {session_id}: False.")
    return False, None


def create_escalation_ticket(
    session_id: str,
    user_query: str,
    conversation_history: List[str],
    reason: str
) -> str:
    """
    Simulates the creation of an escalation ticket.

    In a real system, this would integrate with a ticketing system (e.g., Jira, Zendesk).
    For now, it logs the intent and returns a fake ticket ID.

    Args:
        session_id: The ID of the session being escalated.
        user_query: The user query that triggered the escalation.
        conversation_history: The full conversation history.
        reason: The reason for escalation.

    Returns:
        A string representing the fake ticket ID.
    """
    ticket_id = f"ESCALATED_TICKET_{uuid.uuid4().hex[:10].upper()}"

    log_message = (
        f"Escalation Ticket Simulation:\n"
        f"  Ticket ID: {ticket_id}\n"
        f"  Session ID: {session_id}\n"
        f"  Reason: {reason}\n"
        f"  Triggering User Query: {user_query}\n"
        f"  Conversation History Length: {len(conversation_history)} items\n"
        f"  Full History (last few turns for brevity in log, actual ticket would have all):\n"
    )

    # Log a few turns of history for context in main logs
    # The full history would be attached to a real ticket.
    history_snippet = "\n".join([f"    - {turn}" for turn in conversation_history[-6:]]) # Log last 3 user/AI turns
    log_message += history_snippet

    logger.info(log_message)

    # Here, you would add code to interact with an actual ticketing system API.
    # For example:
    # try:
    #     response = ticketing_system_api.create_ticket(
    #         summary=f"User escalation: {user_query[:50]}...",
    #         description=f"Session ID: {session_id}\nReason: {reason}\nQuery: {user_query}\n\nHistory:\n" + "\n".join(conversation_history),
    #         tags=["ai_escalation", "tech_support"]
    #     )
    #     actual_ticket_id = response.get("id")
    #     logger.info(f"Successfully created real ticket: {actual_ticket_id}")
    #     return actual_ticket_id
    # except Exception as e:
    #     logger.error(f"Failed to create real ticket for session {session_id}: {e}", exc_info=True)
    #     return f"FAILED_TO_CREATE_TICKET_FOR_{session_id}"

    return ticket_id


if __name__ == "__main__":
    # Basic test for escalation_manager
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

    # Mock app.config for testing
    class MockConfig:
        ESCALATION_KEYWORDS = ["escalate", "human", "agent"]
        MAX_TURNS_BEFORE_ESCALATION_CHECK = 6 # 3 user turns, 3 AI turns

    config.ESCALATION_KEYWORDS = MockConfig.ESCALATION_KEYWORDS
    config.MAX_TURNS_BEFORE_ESCALATION_CHECK = MockConfig.MAX_TURNS_BEFORE_ESCALATION_CHECK

    logger.info("--- Testing EscalationManager ---")

    test_session_id = "test_session_escalate"

    # Test 1: Explicit keyword escalation
    query1 = "I need to talk to a human."
    history1: List[str] = []
    should_escalate, reason = check_for_escalation(test_session_id, query1, history1)
    logger.info(f"Test 1: Escalation: {should_escalate}, Reason: {reason}")
    assert should_escalate
    assert "human" in reason

    # Test 2: No keyword, short history
    query2 = "My printer is broken."
    history2 = ["User: My screen is blank", "AI: Try restarting"]
    should_escalate, reason = check_for_escalation(test_session_id, query2, history2)
    logger.info(f"Test 2: Escalation: {should_escalate}, Reason: {reason}")
    assert not should_escalate

    # Test 3: No keyword, long history (meeting threshold)
    query3 = "It's still not working, I'm frustrated!"
    history3 = [
        "User: Problem A", "AI: Solution A",
        "User: Still Problem A", "AI: Solution B",
        "User: Still Problem A!!", "AI: Solution C", # This makes history length 6, meeting threshold
    ]
    should_escalate, reason = check_for_escalation(test_session_id, query3, history3)
    logger.info(f"Test 3: Escalation: {should_escalate}, Reason: {reason}")
    assert should_escalate
    assert "Conversation length" in reason

    # Test 4: No keyword, history just under threshold
    query4 = "One last try."
    history4 = [
        "User: Problem A", "AI: Solution A",
        "User: Still Problem A", "AI: Solution B",
    ] # Length 4, threshold 6
    should_escalate, reason = check_for_escalation(test_session_id, query4, history4)
    logger.info(f"Test 4: Escalation: {should_escalate}, Reason: {reason}")
    assert not should_escalate

    # Test 5: Create a ticket
    if should_escalate: # From Test 3
        ticket_id = create_escalation_ticket(test_session_id, query3, history3, reason)
        logger.info(f"Test 5: Created ticket ID: {ticket_id}")
        assert ticket_id.startswith("ESCALATED_TICKET_")

    logger.info("--- EscalationManager tests complete ---")
```
