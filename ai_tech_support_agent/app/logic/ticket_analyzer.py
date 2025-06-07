import logging
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field

# Assuming these modules are in sys.path or the project is structured as a package
from app.logic.ticket_loader import TicketData # Input Pydantic model
from app.logic.llm_handler import LLMGenerator   # For using the LLM
from app import config # For LLM model name, etc., though LLMGenerator handles its own config

# Configure a logger for this module
logger = logging.getLogger(__name__)

class ExtractedTicketInfo(BaseModel):
    """
    Represents the structured information extracted and summarized from a single ticket.
    This will be the input for generating more detailed documentation or knowledge base articles.
    """
    ticket_id: str
    original_status: str = Field(description="The status of the ticket when it was analyzed (e.g., resolved, closed).")
    problem_summary: str = Field(description="A concise summary of the problem described in the ticket.")
    solution_summary: str = Field(description="A concise summary of the solution or steps taken to resolve the issue.")
    category: Optional[str] = None # Carry over from original ticket
    tags: Optional[List[str]] = None # Carry over from original ticket
    affected_software: Optional[List[str]] = None # Carry over
    affected_hardware: Optional[List[str]] = None # Carry over

    # Optional future fields:
    # extracted_keywords: Optional[List[str]] = None
    # confidence_score: Optional[float] = None # For the summaries generated

def _create_summarization_prompt(text_to_summarize: str, task_description: str, output_format: str = "a concise summary") -> str:
    """
    Helper function to create a detailed prompt for summarization tasks.
    """
    # This prompt structure assumes that LLMGenerator.generate_answer will primarily use the 'query' part
    # and that the existing RAG prompt template might wrap this.
    # A more direct call to the LLM would be cleaner if LLMGenerator is refactored for it.
    # For now, we embed the full instruction and text within the "query" passed to generate_answer.

    # The LLM_PROMPT_TEMPLATE is:
    # """
    # You are an AI technical support assistant. Your task is to answer the user's question based *only* on the provided context.
    # Read the context carefully before answering.
    # Context:
    # {context_chunks}
    # Question: {question}
    # Based *solely* on the context above, provide a helpful and concise answer.
    # If the context does not contain the information needed to answer the question, state clearly: "The provided information does not directly answer this question."
    # Do not make assumptions or use any external knowledge.
    # Helpful Answer:"""

    # We will make our "question" be the summarization instruction and the text.
    # The {context_chunks} will be empty.

    prompt = f"""Please act as a text summarization expert. {task_description}.
The output should be {output_format}.
Do not add any preamble or conversational fluff. Output only the summary itself.

Text to summarize:
---
{text_to_summarize}
---

Summary:"""
    return prompt


def analyze_ticket(ticket: TicketData) -> Optional[ExtractedTicketInfo]:
    """
    Analyzes a single TicketData object to extract and summarize key information
    using an LLM.

    Args:
        ticket: A TicketData object representing a single parsed ticket.

    Returns:
        An ExtractedTicketInfo object if analysis is successful and ticket is relevant,
        otherwise None.
    """
    logger.debug(f"Analyzing ticket_id: {ticket.ticket_id}, status: {ticket.status}")

    # 1. Filter by Status: Only process resolved or closed tickets for knowledge base generation.
    #    (Adjust this set as per requirements for what constitutes a "finalized" ticket)
    relevant_statuses = {"resolved", "closed", "completed"}
    if ticket.status.lower() not in relevant_statuses:
        logger.info(f"Skipping ticket_id: {ticket.ticket_id} due to irrelevant status: '{ticket.status}'.")
        return None

    # 2. Summarization using LLM
    problem_summary_text = ""
    solution_summary_text = ""

    # Summarize Problem Description
    try:
        if not ticket.problem_description.strip():
            logger.warning(f"Ticket {ticket.ticket_id} has empty problem description. Skipping problem summary.")
            problem_summary_text = "No problem description provided." # Or skip ticket entirely
        else:
            problem_prompt = _create_summarization_prompt(
                ticket.problem_description,
                "Summarize the following IT support problem description",
                "one or two concise sentences focusing on the main issue reported"
            )
            logger.debug(f"Problem summarization prompt for {ticket.ticket_id}:\n{problem_prompt[:200]}...") # Log snippet
            # Pass empty context_chunks as the text is in the "query" (prompt)
            problem_summary_text = LLMGenerator.generate_answer(query=problem_prompt, context_chunks=[])
            if problem_summary_text.startswith("Error:") or not problem_summary_text.strip():
                logger.warning(f"LLM failed to generate problem summary for ticket {ticket.ticket_id}. Response: {problem_summary_text}")
                problem_summary_text = ticket.problem_description[:250] + "..." # Fallback: Truncated original
    except Exception as e:
        logger.error(f"Error during problem summarization for ticket {ticket.ticket_id}: {e}", exc_info=True)
        problem_summary_text = ticket.problem_description[:250] + "..." # Fallback

    # Summarize Resolution Steps
    try:
        if not ticket.resolution_steps.strip():
            logger.warning(f"Ticket {ticket.ticket_id} has empty resolution steps. Skipping solution summary.")
            solution_summary_text = "No resolution steps provided." # Or skip ticket entirely
        else:
            solution_prompt = _create_summarization_prompt(
                ticket.resolution_steps,
                "Summarize the following IT support resolution steps, or list the key actions taken",
                "a concise summary or a short, numbered list of key actions"
            )
            logger.debug(f"Solution summarization prompt for {ticket.ticket_id}:\n{solution_prompt[:200]}...") # Log snippet
            solution_summary_text = LLMGenerator.generate_answer(query=solution_prompt, context_chunks=[])
            if solution_summary_text.startswith("Error:") or not solution_summary_text.strip() :
                logger.warning(f"LLM failed to generate solution summary for ticket {ticket.ticket_id}. Response: {solution_summary_text}")
                solution_summary_text = ticket.resolution_steps[:350] + "..." # Fallback: Truncated original
    except Exception as e:
        logger.error(f"Error during solution summarization for ticket {ticket.ticket_id}: {e}", exc_info=True)
        solution_summary_text = ticket.resolution_steps[:350] + "..." # Fallback

    # If both summaries ended up as fallbacks or errors, it might not be useful.
    # However, for PoC, we'll proceed even with partial success / fallbacks.
    if (problem_summary_text == "No problem description provided." and solution_summary_text == "No resolution steps provided."):
        logger.warning(f"Ticket {ticket.ticket_id} resulted in no usable summaries. Skipping.")
        return None

    logger.info(f"Successfully generated summaries for ticket_id: {ticket.ticket_id}")
    return ExtractedTicketInfo(
        ticket_id=ticket.ticket_id,
        original_status=ticket.status,
        problem_summary=problem_summary_text.strip(),
        solution_summary=solution_summary_text.strip(),
        category=ticket.category,
        tags=ticket.tags,
        affected_software=ticket.affected_software,
        affected_hardware=ticket.affected_hardware,
    )

def analyze_tickets_batch(tickets: List[TicketData]) -> List[ExtractedTicketInfo]:
    """
    Analyzes a batch of TicketData objects and returns structured information for each.

    Args:
        tickets: A list of TicketData objects.

    Returns:
        A list of ExtractedTicketInfo objects for successfully analyzed tickets.
    """
    if not tickets:
        logger.info("No tickets provided for batch analysis.")
        return []

    logger.info(f"Starting batch analysis for {len(tickets)} tickets.")
    extracted_info_list: List[ExtractedTicketInfo] = []
    successfully_analyzed_count = 0

    # Ensure LLM is initialized once before starting the batch
    try:
        LLMGenerator.get_llm_resources() # This will initialize if not already done
        logger.info(f"LLM ({config.LLM_MODEL_NAME}) initialized and ready for batch processing.")
    except RuntimeError as e:
        logger.error(f"LLM failed to initialize. Cannot proceed with batch ticket analysis: {e}")
        return [] # Or raise the error, depending on desired behavior

    for i, ticket in enumerate(tickets):
        logger.debug(f"Analyzing ticket {i+1}/{len(tickets)}: ID {ticket.ticket_id}")
        extracted_info = analyze_ticket(ticket)
        if extracted_info:
            extracted_info_list.append(extracted_info)
            successfully_analyzed_count += 1
        # analyze_ticket already logs reasons for returning None
        if (i + 1) % 10 == 0: # Log progress every 10 tickets
             logger.info(f"Batch analysis progress: Processed {i+1}/{len(tickets)} tickets. Successfully analyzed: {successfully_analyzed_count}.")


    logger.info(f"Batch analysis complete. Successfully analyzed {successfully_analyzed_count} out of {len(tickets)} tickets.")
    return extracted_info_list


if __name__ == '__main__':
    # Setup basic logging for standalone testing
    logging.basicConfig(
        level=logging.DEBUG, # Set to DEBUG to see detailed logs from this module and LLMHandler
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger.info("Running ticket_analyzer.py module directly for testing.")

    # Ensure app.config settings (especially LLM_MODEL_NAME) are appropriate for testing.
    # The LLMGenerator will use settings from app.config.
    logger.info(f"Using LLM: {config.LLM_MODEL_NAME} for summarization tests.")
    logger.warning("This test will invoke the LLM. Ensure it's configured and you have resources/API access.")

    sample_ticket_list = [
        TicketData(
            ticket_id="TICK_RESOLVED_001",
            status="resolved",
            problem_description="The user is experiencing very slow login times. It takes almost 2 minutes to get to the desktop after entering their password. This started yesterday. They have tried rebooting multiple times. Other applications seem slow too once logged in.",
            resolution_steps="1. Checked for pending Windows updates - none found.\n2. Ran disk cleanup and defragmentation - marginal improvement.\n3. Scanned for malware using corporate AV - clean.\n4. Investigated startup programs - found several unnecessary heavy applications.\n5. Disabled bloatware 'SuperSystemOptimizer' and 'SearchEverywhere Toolbar' from startup.\n6. User rebooted. Login time reduced to 30 seconds. System performance noticeably better."
        ),
        TicketData(
            ticket_id="TICK_CLOSED_002",
            status="closed",
            category="Software",
            tags=["outlook", "crash"],
            problem_description="Microsoft Outlook client crashes every time the user tries to open an email with a large PDF attachment (around 20MB). User can open other emails fine. Issue started this morning.",
            resolution_steps="1. Asked user to try opening Outlook in Safe Mode (outlook.exe /safe). Crashing persisted.\n2. Checked for Outlook add-ins. Disabled a third-party PDF previewer add-in.\n3. Restarted Outlook normally. User was able to open the email with large PDF without crashing.\n4. Advised user to keep the problematic add-in disabled and use dedicated PDF software for large files."
        ),
        TicketData(
            ticket_id="TICK_OPEN_003", # This one should be skipped by analyze_ticket
            status="open",
            problem_description="User needs access to the new marketing shared drive.",
            resolution_steps="Pending approval from manager."
        ),
         TicketData(
            ticket_id="TICK_RESOLVED_004_EMPTY",
            status="resolved",
            problem_description="", # Empty problem
            resolution_steps=""     # Empty solution
        ),
    ]

    logger.info(f"\n--- Analyzing a batch of {len(sample_ticket_list)} tickets ---")
    extracted_results = analyze_tickets_batch(sample_ticket_list)

    if extracted_results:
        logger.info(f"\n--- Successfully extracted information for {len(extracted_results)} tickets: ---")
        for i, info in enumerate(extracted_results):
            logger.info(f"Result {i+1}:")
            logger.info(f"  Ticket ID: {info.ticket_id} (Original Status: {info.original_status})")
            logger.info(f"  Category: {info.category or 'N/A'}")
            logger.info(f"  Problem Summary: {info.problem_summary}")
            logger.info(f"  Solution Summary: {info.solution_summary}")
            logger.info(f"  Tags: {info.tags}")
            logger.info("-" * 30)

        # We expect 2 successfully analyzed tickets (TICK_RESOLVED_001, TICK_CLOSED_002)
        # TICK_OPEN_003 is skipped due to status.
        # TICK_RESOLVED_004_EMPTY is skipped as both summaries would be fallbacks to empty/placeholder.
        assert len(extracted_results) == 2, f"Expected 2 successful analyses, got {len(extracted_results)}"
        assert extracted_results[0].ticket_id == "TICK_RESOLVED_001"
        assert extracted_results[1].ticket_id == "TICK_CLOSED_002"
        assert "login times" in extracted_results[0].problem_summary.lower()
        assert "outlook client crashes" in extracted_results[1].problem_summary.lower()

    else:
        logger.warning("No information was extracted from the batch. Check logs for errors (e.g., LLM initialization).")

    logger.info("\nTicket analyzer module test run complete.")
