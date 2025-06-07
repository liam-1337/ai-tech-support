import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, field_validator, ValidationError

# Configure a logger for this module
# Assumes the main application (e.g., FastAPI app or script using this module)
# will set up the basic logging configuration (level, format).
logger = logging.getLogger(__name__)

class TicketData(BaseModel):
    """
    Represents a single ticket's data, validated using Pydantic.
    Mirrors the JSON schema defined for ticket data uploads.
    """
    ticket_id: str
    creation_date: Optional[str] = None # Could be Optional[datetime] with a pre-validator if parsing to datetime is desired here
    status: str
    problem_description: str
    resolution_steps: str
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    affected_software: Optional[List[str]] = None
    affected_hardware: Optional[List[str]] = None

    @field_validator('status')
    @classmethod
    def normalize_status(cls, v: str) -> str:
        """
        Normalizes the status field to lowercase.
        Can be expanded to validate against a known set of statuses if required.
        """
        # Example of stricter validation (can be enabled if needed):
        # known_statuses = {"resolved", "closed", "open", "in progress", "pending user"}
        # if v.lower() not in known_statuses:
        #     raise ValueError(f"Status '{v}' is not a recognized status. Known statuses are: {known_statuses}")
        return v.lower()

    @field_validator('problem_description', 'resolution_steps', 'ticket_id')
    @classmethod
    def field_must_not_be_empty_or_whitespace(cls, v: str, info) -> str:
        """
        Validates that critical string fields (ticket_id, problem_description, resolution_steps)
        are not empty or contain only whitespace.
        """
        if not v or not v.strip():
            raise ValueError(f"Field '{info.field_name}' must not be empty or contain only whitespace.")
        return v.strip() # Also strip whitespace from these fields

    # Example of a root validator if cross-field validation was needed:
    # from pydantic import root_validator
    # @root_validator(pre=True) # or pre=False for after individual field validation
    # @classmethod
    # def check_something_across_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
    #     # ... validation logic ...
    #     return values

def load_and_parse_ticket_data(file_path: Path) -> List[TicketData]:
    """
    Loads ticket data from a specified JSON file, parses it, and validates each
    ticket against the TicketData Pydantic model.

    Args:
        file_path: The path (pathlib.Path object) to the JSON file containing ticket data.
                   The JSON file should contain a list of ticket objects.

    Returns:
        A list of validated TicketData objects.

    Raises:
        FileNotFoundError: If the specified file_path does not exist or is not a file.
        ValueError: If the file content is not valid JSON, or if the top-level JSON structure
                    is not a list, or if critical validation errors occur that prevent processing.
        IOError: If there's an error reading the file beyond JSON decoding.
    """
    logger.info(f"Attempting to load and parse ticket data from: {file_path}")

    if not file_path.is_file():
        logger.error(f"Ticket data file not found or is not a file: {file_path}")
        raise FileNotFoundError(f"Ticket data file not found: {file_path}")

    raw_tickets_data: Any
    try:
        with file_path.open('r', encoding='utf-8') as f:
            raw_tickets_data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {file_path}: {e.msg} (line {e.lineno}, col {e.colno})")
        raise ValueError(f"Invalid JSON file: {file_path}. Details: {e.msg}") from e
    except Exception as e: # Catch other potential I/O errors
        logger.error(f"Error reading file {file_path}: {e}", exc_info=True)
        # Use exc_info=True to log the traceback for unexpected I/O errors
        raise IOError(f"Could not read file: {file_path}") from e

    if not isinstance(raw_tickets_data, list):
        logger.error(f"Ticket data in {file_path} is not a list as expected. Found type: {type(raw_tickets_data)}")
        raise ValueError(f"Invalid ticket data format: expected a list of tickets in {file_path}")

    validated_tickets: List[TicketData] = []
    skipped_tickets_count = 0

    for i, ticket_dict in enumerate(raw_tickets_data):
        if not isinstance(ticket_dict, dict):
            logger.warning(f"Skipping item at index {i} in {file_path}: item is not a dictionary (found type: {type(ticket_dict)}).")
            skipped_tickets_count += 1
            continue

        try:
            ticket = TicketData(**ticket_dict)
            validated_tickets.append(ticket)
        except ValidationError as e:
            # Log detailed validation errors for the specific ticket
            error_details = e.errors(include_input=False, include_url=False) # Exclude input value and Pydantic URL
            logger.warning(f"Skipping ticket at index {i} (ID: {ticket_dict.get('ticket_id', 'N/A')}) due to validation error(s) in {file_path}: {error_details}")
            skipped_tickets_count += 1
        except Exception as e: # Catch other unexpected errors during Pydantic model instantiation
            logger.warning(f"Skipping ticket at index {i} (ID: {ticket_dict.get('ticket_id', 'N/A')}) due to an unexpected error during data parsing in {file_path}: {str(e)}", exc_info=True)
            skipped_tickets_count += 1

    total_processed = len(raw_tickets_data)
    logger.info(f"Successfully validated {len(validated_tickets)} tickets.")
    if skipped_tickets_count > 0:
        logger.warning(f"Skipped {skipped_tickets_count} out of {total_processed} total records due to validation or parsing issues from {file_path}.")
    else:
        logger.info(f"All {total_processed} records from {file_path} were processed successfully.")

    return validated_tickets


if __name__ == '__main__':
    # This basicConfig is for testing the module directly.
    # In a larger app, logging would be configured centrally by the main entry point.
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger.info("Running ticket_loader.py module directly for testing.")

    # Create a dummy JSON file for testing
    test_data_dir = Path(__file__).parent.parent / "data_for_testing" # Assuming a 'data_for_testing' dir at project root
    test_data_dir.mkdir(parents=True, exist_ok=True)
    dummy_file_path = test_data_dir / "dummy_tickets.json"

    sample_tickets = [
        {
            "ticket_id": "TICK001",
            "creation_date": "2023-10-22T14:35:12Z",
            "status": "Resolved",
            "problem_description": "User reports being unable to log in to the VPN.",
            "resolution_steps": "Guided user to install latest VPN client.",
            "category": "Network",
            "tags": ["vpn", "login_error"],
        },
        { # Valid ticket
            "ticket_id": "TICK002",
            "status": "Closed",
            "problem_description": "Printer not working.",
            "resolution_steps": "Cleared paper jam.",
            "affected_hardware": ["HP LaserJet Pro M404dn"]
        },
        { # Invalid: missing required 'status'
            "ticket_id": "TICK003",
            "problem_description": "Software X crashes on startup.",
            "resolution_steps": "Advised to reinstall."
        },
        { # Invalid: 'problem_description' is empty
            "ticket_id": "TICK004",
            "status": "Open",
            "problem_description": "   ",
            "resolution_steps": "Awaiting user feedback."
        },
        "not_a_dictionary", # Invalid record type
        { # Valid again
            "ticket_id": "TICK005",
            "status": "IN PROGRESS", # Will be lowercased by validator
            "problem_description": "Cannot access shared drive.",
            "resolution_steps": "Checking permissions."
        }
    ]

    with open(dummy_file_path, 'w', encoding='utf-8') as f:
        json.dump(sample_tickets, f, indent=2)

    logger.info(f"Created dummy ticket file at: {dummy_file_path}")

    # Test loading valid file
    try:
        logger.info(f"\n--- Attempting to load tickets from: {dummy_file_path} ---")
        loaded_tickets = load_and_parse_ticket_data(dummy_file_path)
        logger.info(f"Successfully loaded {len(loaded_tickets)} tickets.")
        for i, ticket in enumerate(loaded_tickets):
            logger.debug(f"Ticket {i+1}: ID={ticket.ticket_id}, Status={ticket.status}, Category={ticket.category or 'N/A'}")
        assert len(loaded_tickets) == 3 # TICK001, TICK002, TICK005 should pass
    except Exception as e:
        logger.error(f"Error during test loading: {e}", exc_info=True)

    # Test loading non-existent file
    non_existent_file = Path("non_existent_tickets.json")
    try:
        logger.info(f"\n--- Attempting to load tickets from non-existent file: {non_existent_file} ---")
        load_and_parse_ticket_data(non_existent_file)
    except FileNotFoundError:
        logger.info(f"Correctly caught FileNotFoundError for {non_existent_file}.")
    except Exception as e:
        logger.error(f"Unexpected error for non-existent file test: {e}", exc_info=True)

    # Test loading invalid JSON file
    invalid_json_file = test_data_dir / "invalid_tickets.json"
    with open(invalid_json_file, 'w', encoding='utf-8') as f:
        f.write("this is not valid json {")
    try:
        logger.info(f"\n--- Attempting to load tickets from invalid JSON file: {invalid_json_file} ---")
        load_and_parse_ticket_data(invalid_json_file)
    except ValueError as e:
        logger.info(f"Correctly caught ValueError for invalid JSON: {e}")
        assert "Invalid JSON file" in str(e)
    except Exception as e:
        logger.error(f"Unexpected error for invalid JSON test: {e}", exc_info=True)

    # Test loading JSON that is not a list
    not_a_list_file = test_data_dir / "not_a_list_tickets.json"
    with open(not_a_list_file, 'w', encoding='utf-8') as f:
        json.dump({"oops": "this is a dict, not a list"}, f)
    try:
        logger.info(f"\n--- Attempting to load tickets from JSON that is not a list: {not_a_list_file} ---")
        load_and_parse_ticket_data(not_a_list_file)
    except ValueError as e:
        logger.info(f"Correctly caught ValueError for non-list JSON: {e}")
        assert "expected a list of tickets" in str(e)
    except Exception as e:
        logger.error(f"Unexpected error for non-list JSON test: {e}", exc_info=True)


    logger.info("\nTicket loader module test run complete.")
    # Clean up dummy files
    # dummy_file_path.unlink(missing_ok=True)
    # invalid_json_file.unlink(missing_ok=True)
    # not_a_list_file.unlink(missing_ok=True)
    # logger.info("Cleaned up dummy test files.")
