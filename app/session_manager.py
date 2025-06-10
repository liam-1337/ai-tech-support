import uuid
from typing import Dict, List, Optional

from app import config # For MAX_HISTORY_LENGTH

# Get a logger for this module if verbose logging is needed here
# import logging
# logger = logging.getLogger(__name__)

class SessionManager:
    """
    Manages in-memory conversation sessions and their history.
    """
    def __init__(self):
        # sessions stores: session_id -> List of conversation turns
        # Each turn can be a simple string, e.g., "User: How do I reset my password?" or "AI: Follow these steps..."
        self.sessions: Dict[str, List[str]] = {}
        self.max_history_length = config.SESSION_MAX_HISTORY_LENGTH * 2 # User + AI is 2 items per "turn"

    def create_session(self) -> str:
        """
        Creates a new session with an empty history and returns the session ID.
        """
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = []
        # logger.debug(f"Created new session: {session_id}")
        return session_id

    def get_session_history(self, session_id: str) -> List[str]:
        """
        Retrieves the conversation history for a given session ID.
        Returns an empty list if the session ID is not found.
        """
        history = self.sessions.get(session_id, [])
        # logger.debug(f"Retrieved history for session {session_id}: {len(history)} items.")
        return history

    def add_turn_to_history(self, session_id: str, user_query: str, ai_response: str):
        """
        Adds a user query and its corresponding AI response to the session's history.
        Manages history length according to MAX_HISTORY_LENGTH.

        Args:
            session_id: The ID of the session.
            user_query: The user's query string.
            ai_response: The AI's response string.
        """
        if session_id not in self.sessions:
            # This case should ideally be handled by create_session first if session_id is new
            # For robustness, one might create it here, or log a warning.
            # logger.warning(f"Session ID {session_id} not found while trying to add history. This might indicate an issue.")
            # For now, let's assume session_id is valid and exists.
            # If it's critical it exists, self.sessions[session_id] will raise KeyError appropriately.
            # Or, to be safe for this method:
            self.sessions[session_id] = [] # Initialize if somehow missed


        # Add new turns
        # Prefixing helps LLM differentiate, though the prompt template might also do this.
        # The current LLM prompt template just lists turns, so prefixes here are good.
        self.sessions[session_id].append(f"User: {user_query}")
        self.sessions[session_id].append(f"AI: {ai_response}")

        # Trim history if it exceeds max length
        if len(self.sessions[session_id]) > self.max_history_length:
            # Keep the most recent N turns (N = max_history_length)
            self.sessions[session_id] = self.sessions[session_id][-self.max_history_length:]
            # logger.debug(f"Trimmed history for session {session_id} to {self.max_history_length} items.")

        # logger.debug(f"Added turn to session {session_id}. History length: {len(self.sessions[session_id])}")

    def session_exists(self, session_id: str) -> bool:
        """Checks if a session ID exists."""
        return session_id in self.sessions

# Global instance (simple approach for in-memory store)
# For a multi-worker setup (like Gunicorn with Uvicorn workers), a simple global dict
# like this won't be shared across workers. A proper external store (Redis, DB)
# would be needed for that. For this subtask, in-memory is fine.
session_manager = SessionManager()

if __name__ == '__main__':
    # Basic test for SessionManager
    # Configure a simple logger for testing this module directly
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__) # Logger for this test block

    # Assuming config.py can be loaded or SESSION_MAX_HISTORY_LENGTH is mocked in config
    if not hasattr(config, 'SESSION_MAX_HISTORY_LENGTH'):
        config.SESSION_MAX_HISTORY_LENGTH = 3 # Mock for testing: 3 pairs of User/AI turns (6 items)
        logger.info(f"Mocked config.SESSION_MAX_HISTORY_LENGTH to {config.SESSION_MAX_HISTORY_LENGTH}")
        session_manager.max_history_length = config.SESSION_MAX_HISTORY_LENGTH * 2


    test_sm = SessionManager() # Use a local instance for isolated testing
    test_sm.max_history_length = config.SESSION_MAX_HISTORY_LENGTH * 2


    logger.info("--- Testing SessionManager ---")

    # Test session creation
    sid1 = test_sm.create_session()
    logger.info(f"Created session 1: {sid1}")
    assert test_sm.session_exists(sid1)
    assert test_sm.get_session_history(sid1) == []

    sid2 = test_sm.create_session()
    logger.info(f"Created session 2: {sid2}")
    assert sid1 != sid2

    # Test adding turns to history (session 1)
    test_sm.add_turn_to_history(sid1, "Hello AI", "Hello User!")
    history1 = test_sm.get_session_history(sid1)
    logger.info(f"Session 1 History (1 turn): {history1}")
    assert len(history1) == 2
    assert history1[0] == "User: Hello AI"
    assert history1[1] == "AI: Hello User!"

    test_sm.add_turn_to_history(sid1, "How are you?", "I am fine, thank you.")
    history1 = test_sm.get_session_history(sid1)
    logger.info(f"Session 1 History (2 turns): {history1}")
    assert len(history1) == 4

    # Test history trimming (assuming max_history_length is 6 for 3 turns)
    logger.info(f"Testing history trimming (max_history_length = {test_sm.max_history_length} items / {config.SESSION_MAX_HISTORY_LENGTH} turns)")
    test_sm.add_turn_to_history(sid1, "Question 3", "Answer 3") # Total 3 turns (6 items)
    history1 = test_sm.get_session_history(sid1)
    logger.info(f"Session 1 History (3 turns): {history1}")
    assert len(history1) == 6

    # This turn should cause trimming if max_history_length is 6 (3 turns)
    test_sm.add_turn_to_history(sid1, "Question 4 that will push out oldest", "Answer 4")
    history1 = test_sm.get_session_history(sid1)
    logger.info(f"Session 1 History (after 4th turn, should be trimmed): {history1}")
    assert len(history1) == test_sm.max_history_length # Should be trimmed to max items
    assert "User: Hello AI" not in history1 # Oldest user query should be gone
    assert "AI: Hello User!" not in history1 # Oldest AI response should be gone
    assert "User: How are you?" in history1 # Second user query should remain
    assert "AI: Answer 4" in history1 # Newest AI response should be present

    # Test getting history for non-existent session
    non_existent_history = test_sm.get_session_history("non-existent-id")
    logger.info(f"History for non-existent session: {non_existent_history}")
    assert non_existent_history == []

    logger.info("--- SessionManager tests complete ---")

# To make session_manager instance available for import from other modules:
# from app.session_manager import session_manager
# No, it's already a global in this file.
# Other modules will do: from app.session_manager import session_manager
# (if this file is app/session_manager.py)
# For clarity, could also do:
# global_session_manager = SessionManager()
# and then other files import global_session_manager.
# The current `session_manager = SessionManager()` is fine.

```
