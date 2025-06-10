import logging
import uuid
import json # For payload
import requests # For making HTTP requests
from typing import List, Tuple, Optional

from app import config # For ESCALATION_KEYWORDS, MAX_TURNS_BEFORE_ESCALATION_CHECK, TICKETING_SYSTEM_API_URL, TICKETING_SYSTEM_API_KEY

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

    # --- Conditional Mocking & Actual API Call ---
    api_url = config.TICKETING_SYSTEM_API_URL
    api_key = config.TICKETING_SYSTEM_API_KEY

    if api_url == "MOCK_API_SUCCESS":
        logger.info(f"MOCKING API: Simulating successful ticket creation for session {session_id}.")
        mock_ticket_id = f"MOCK_SYS_TICKET_{uuid.uuid4().hex[:7].upper()}"
        logger.info(f"Mock ticket ID {mock_ticket_id} generated for session {session_id} due to MOCK_API_SUCCESS setting.")
        return mock_ticket_id

    if api_url == "MOCK_API_FAILURE":
        logger.warning(f"MOCKING API: Simulating failed ticket creation for session {session_id}.")
        return "ESCALATION_FAILED_COULD_NOT_CREATE_TICKET"

    # --- Actual API Call Logic (if not mocked) ---
    payload = {
        "summary": f"AI Escalation: {user_query[:100]}", # Truncate summary if too long
        "description": (
            f"Reason for Escalation: {reason}\n\n"
            f"User Query: {user_query}\n\n"
            f"Session ID: {session_id}\n\n"
            f"Conversation History:\n" + "\n".join(conversation_history)
        ),
        "user_session_id": session_id,
        "source": "AI_SUPPORT_AGENT",
        # Add any other fields required by your ticketing system
        # "priority": "High",
        # "tags": ["ai_escalation", "tech_support"]
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    logger.info(f"Attempting to create escalation ticket via API for session {session_id} at URL {api_url}.")
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=10) # 10s timeout
        response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)

        # Assuming the API returns JSON with a 'ticket_id' field on success
        # Example: {"ticket_id": "SYS-12345", "status": "created"}
        response_data = response.json()
        actual_ticket_id = response_data.get("ticket_id")

        if actual_ticket_id:
            logger.info(f"Successfully created ticket via API: {actual_ticket_id} (HTTP {response.status_code})")
            return str(actual_ticket_id) # Ensure it's a string
        else:
            logger.error(f"Ticket API call successful (HTTP {response.status_code}) but response missing 'ticket_id'. Response: {response.text[:200]}")
            return "ESCALATION_FAILED_TICKET_ID_MISSING_IN_RESPONSE"

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error creating ticket for session {session_id}: {e.response.status_code} - {e.response.text[:200]}", exc_info=True)
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error creating ticket for session {session_id} to {api_url}: {e}", exc_info=True)
    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout error creating ticket for session {session_id} to {api_url}: {e}", exc_info=True)
    except requests.exceptions.RequestException as e: # Catch any other requests-related error
        logger.error(f"Error creating ticket for session {session_id}: {e}", exc_info=True)
    except json.JSONDecodeError as e: # If API response is not valid JSON
        logger.error(f"Error decoding JSON response from ticketing API for session {session_id}. Status: {response.status_code if 'response' in locals() else 'N/A'}. Response text: {response.text[:200] if 'response' in locals() else 'N/A'}. Error: {e}", exc_info=True)

    return "ESCALATION_FAILED_COULD_NOT_CREATE_TICKET" # Fallback for any failure


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
    should_escalate_test3, reason_test3 = check_for_escalation(test_session_id, query3, history3)
    logger.info(f"Test 3: Escalation: {should_escalate_test3}, Reason: {reason_test3}")
    assert should_escalate_test3
    assert "Conversation length" in reason_test3

    # Test 4: No keyword, history just under threshold
    query4 = "One last try."
    history4 = [
        "User: Problem A", "AI: Solution A",
        "User: Still Problem A", "AI: Solution B",
    ] # Length 4, threshold 6
    should_escalate, reason = check_for_escalation(test_session_id, query4, history4)
    logger.info(f"Test 4: Escalation: {should_escalate}, Reason: {reason}")
    assert not should_escalate

    # Test 5: Create a ticket (Mock Success)
    logger.info("\n--- Test 5: Create Ticket (Mock Success) ---")
    original_api_url = config.TICKETING_SYSTEM_API_URL # Save to restore
    config.TICKETING_SYSTEM_API_URL = "MOCK_API_SUCCESS"
    if should_escalate_test3:
        ticket_id_success = create_escalation_ticket(test_session_id, query3, history3, reason_test3)
        logger.info(f"Test 5: Mock Success Ticket ID: {ticket_id_success}")
        assert ticket_id_success.startswith("MOCK_SYS_TICKET_")

    # Test 6: Create a ticket (Mock Failure)
    logger.info("\n--- Test 6: Create Ticket (Mock Failure) ---")
    config.TICKETING_SYSTEM_API_URL = "MOCK_API_FAILURE"
    if should_escalate_test3:
        ticket_id_failure = create_escalation_ticket(test_session_id, query3, history3, reason_test3)
        logger.info(f"Test 6: Mock Failure Ticket ID: {ticket_id_failure}")
        assert ticket_id_failure == "ESCALATION_FAILED_COULD_NOT_CREATE_TICKET"

    # Test 7: Create a ticket (Simulate Real API - will fail if URL is not live, but tests path)
    # This test is more for verifying the requests call structure.
    # It's expected to fail if the URL is "http://mock-ticketing-api.example.com/api/tickets"
    logger.info("\n--- Test 7: Create Ticket (Simulate Real API Call - Expect Failure for mock URL) ---")
    config.TICKETING_SYSTEM_API_URL = "http://mock-ticketing-api.example.com/api/tickets" # A non-mock URL
    config.TICKETING_SYSTEM_API_KEY = "test_key_for_simulated_call"
    if should_escalate_test3:
        ticket_id_real_attempt = create_escalation_ticket(test_session_id, query3, history3, reason_test3)
        logger.info(f"Test 7: Real API Call Attempt Ticket ID: {ticket_id_real_attempt}")
        assert "ESCALATION_FAILED" in ticket_id_real_attempt # Expecting failure for this URL

    config.TICKETING_SYSTEM_API_URL = original_api_url # Restore for other potential tests or default behavior
    logger.info("\n--- EscalationManager tests complete ---")
```
