import json
import sys
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any

# --- Path Setup ---
# This allows the script to import modules from the 'app' directory,
# assuming the script is run from the project root (e.g., python scripts/generate_analytics_report.py)
# or if 'scripts' is a direct subdirectory of the project root.
try:
    # Calculate the project root directory (assuming scripts/ is one level down from root)
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(PROJECT_ROOT))
    from app import config # Now we can import config
except ImportError as e:
    print(f"Error: Could not import app.config. Ensure PROJECT_ROOT is correct and script is run from project root. Details: {e}")
    sys.exit(1)
except IndexError:
    print("Error: Could not determine PROJECT_ROOT. Script may not be in the expected 'scripts' subdirectory.")
    sys.exit(1)


def parse_feedback_log(log_file_path: Path) -> List[Dict[str, Any]]:
    """
    Reads and parses a log file where each line is expected to be a JSON object.

    Args:
        log_file_path: Path to the log file.

    Returns:
        A list of successfully parsed log entries (dictionaries).
    """
    parsed_entries: List[Dict[str, Any]] = []
    if not log_file_path.exists():
        print(f"Warning: Log file not found at {log_file_path}")
        return parsed_entries

    with open(log_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line: # Skip empty lines
                continue
            # The first line of the feedback log is an info message, not JSON
            if "Interaction feedback logger initialized" in line:
                continue
            try:
                log_entry = json.loads(line)
                parsed_entries.append(log_entry)
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed JSON line {i+1} in {log_file_path}: '{line[:100]}...'")
    return parsed_entries


def generate_report(log_entries: List[Dict[str, Any]], top_n_queries: int = 10) -> Dict[str, Any]:
    """
    Generates basic analytics from a list of parsed log entries.

    Args:
        log_entries: A list of log entries (dictionaries).
        top_n_queries: The number of most common queries to include in the report.

    Returns:
        A dictionary containing the calculated analytics.
    """
    analytics: Dict[str, Any] = {}
    if not log_entries:
        analytics["total_interactions"] = 0
        return analytics

    analytics["total_interactions"] = len(log_entries)

    # Escalations
    escalated_count = 0
    for entry in log_entries:
        if "llm_response" in entry and isinstance(entry["llm_response"], str) and \
           "ESCALATED_TICKET_" in entry["llm_response"]:
            escalated_count += 1
    analytics["escalated_interactions"] = escalated_count

    # Common user queries
    query_counter = Counter()
    for entry in log_entries:
        if "user_query" in entry and isinstance(entry["user_query"], str):
            query_counter[entry["user_query"].lower().strip()] += 1
    analytics[f"top_{top_n_queries}_common_queries"] = query_counter.most_common(top_n_queries)

    # Interactions with no context retrieved
    no_context_count = 0
    for entry in log_entries:
        # Check if 'retrieved_context_summary' is present and an empty list
        if "retrieved_context_summary" in entry and isinstance(entry["retrieved_context_summary"], list) \
           and not entry["retrieved_context_summary"]:
            no_context_count += 1
    analytics["interactions_with_no_context_retrieved"] = no_context_count

    # Interactions per session_id
    session_interaction_counter = Counter()
    for entry in log_entries:
        if "session_id" in entry:
            session_interaction_counter[entry["session_id"]] += 1
    analytics["interactions_per_session"] = dict(session_interaction_counter) # Convert Counter to dict for JSON if needed later
    analytics["average_interactions_per_session"] = \
        len(log_entries) / len(session_interaction_counter) if session_interaction_counter else 0


    # Placeholder for feedback analysis (if feedback fields were populated)
    helpful_feedback_count = 0
    not_helpful_feedback_count = 0
    corrections_provided_count = 0
    for entry in log_entries:
        feedback_data = entry.get("feedback", {})
        if feedback_data.get("was_helpful") is True:
            helpful_feedback_count +=1
        elif feedback_data.get("was_helpful") is False: # Explicitly False
            not_helpful_feedback_count +=1
        if feedback_data.get("corrected_answer"):
            corrections_provided_count +=1

    analytics["feedback_summary"] = {
        "marked_helpful": helpful_feedback_count,
        "marked_not_helpful": not_helpful_feedback_count,
        "corrections_provided": corrections_provided_count
    }

    return analytics


def print_report(report_data: Dict[str, Any]):
    """
    Prints the generated analytics report to the console in a readable format.
    """
    if not report_data or report_data.get("total_interactions", 0) == 0:
        print("No interactions found or report data is empty. Nothing to print.")
        return

    print("\n--- Interaction Analytics Report ---")
    print(f"Total Interactions Logged: {report_data.get('total_interactions', 0)}")
    print(f"Escalated Interactions: {report_data.get('escalated_interactions', 0)}")
    print(f"Interactions with No Context Retrieved: {report_data.get('interactions_with_no_context_retrieved', 0)}")

    avg_interactions = report_data.get('average_interactions_per_session', 0)
    print(f"Average Interactions per Session: {avg_interactions:.2f}")

    print("\nFeedback Summary:")
    feedback_summary = report_data.get("feedback_summary", {})
    print(f"  Marked as Helpful: {feedback_summary.get('marked_helpful',0)}")
    print(f"  Marked as Not Helpful: {feedback_summary.get('marked_not_helpful',0)}")
    print(f"  Corrections Provided: {feedback_summary.get('corrections_provided',0)}")

    top_n = 0
    for key, value in report_data.items():
        if key.startswith("top_") and key.endswith("_common_queries"):
            top_n = int(key.split('_')[1]) # Extract N from the key name
            print(f"\nTop {top_n} Most Common User Queries:")
            if value:
                for i, (query, count) in enumerate(value):
                    print(f"  {i+1}. \"{query}\" (Count: {count})")
            else:
                print("  No query data to display.")
            break # Assuming only one such key

    # Optionally print interactions per session if it's not too verbose
    # num_sessions_to_print = 5
    # print(f"\nInteractions per Session (sample of first {num_sessions_to_print} sessions):")
    # interactions_per_session = report_data.get("interactions_per_session", {})
    # if interactions_per_session:
    #     for i, (session_id, count) in enumerate(interactions_per_session.items()):
    #         if i >= num_sessions_to_print:
    #             print(f"  ... and {len(interactions_per_session) - num_sessions_to_print} more sessions.")
    #             break
    #         print(f"  Session {session_id}: {count} interactions")
    # else:
    #     print("  No session data.")

    print("--- End of Report ---")


if __name__ == "__main__":
    # Construct the full path to the feedback log file
    # config.FEEDBACK_LOG_FILE is relative to PROJECT_ROOT in feedback_logger.py
    # So, here we need config.PROJECT_ROOT / config.FEEDBACK_LOG_FILE
    log_file_name = getattr(config, 'FEEDBACK_LOG_FILE', 'interaction_feedback.log')
    log_file_path = config.PROJECT_ROOT / log_file_name

    print(f"Attempting to read feedback log from: {log_file_path}")

    parsed_logs = parse_feedback_log(log_file_path)

    if not parsed_logs:
        print("No log entries parsed. Ensure the log file exists and contains valid JSON entries (one per line).")
    else:
        print(f"Successfully parsed {len(parsed_logs)} log entries.")
        analytics_report = generate_report(parsed_logs, top_n_queries=10)
        print_report(analytics_report)
```
