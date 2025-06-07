import logging
from typing import List, Optional

# Assuming ExtractedTicketInfo is in ticket_analyzer.py and accessible
from app.logic.ticket_analyzer import ExtractedTicketInfo

# Configure a logger for this module
logger = logging.getLogger(__name__)

def generate_draft_document(extracted_info: ExtractedTicketInfo) -> str:
    """
    Generates a Markdown document string from an ExtractedTicketInfo object.

    Args:
        extracted_info: An ExtractedTicketInfo Pydantic model instance containing
                        the structured information from a ticket.

    Returns:
        A string formatted as a Markdown document.
    """
    logger.debug(f"Generating draft document for ticket_id: {extracted_info.ticket_id}")

    lines: List[str] = []

    # 1. Title Generation
    title_snippet = extracted_info.problem_summary
    if len(title_snippet) > 60:
        # Try to find a space to break nicely for snippet, otherwise hard cut.
        last_space_index = title_snippet[:60].rfind(' ')
        if last_space_index != -1 and last_space_index > 40: # Ensure snippet is not too short
             title_snippet = title_snippet[:last_space_index] + "..."
        else:
            title_snippet = title_snippet[:57] + "..." # 57 + 3 dots
    elif not title_snippet and extracted_info.ticket_id: # Fallback if problem_summary is empty
        title_snippet = f"Details for Ticket {extracted_info.ticket_id}"
    elif not title_snippet:
        title_snippet = "Untitled Solution Document"


    lines.append(f"# Solution for: {title_snippet}")
    lines.append("")  # Blank line after H1 for spacing

    # 2. Metadata Block Construction
    lines.append(f"**Ticket ID:** {extracted_info.ticket_id}")

    # original_status should always be present based on ExtractedTicketInfo model definition
    lines.append(f"**Original Status:** {extracted_info.original_status.capitalize()}")

    if extracted_info.category and extracted_info.category.strip():
        lines.append(f"**Category:** {extracted_info.category}")

    if extracted_info.tags and len(extracted_info.tags) > 0:
        lines.append(f"**Tags:** {', '.join(extracted_info.tags)}")

    if extracted_info.affected_software and len(extracted_info.affected_software) > 0:
        lines.append(f"**Affected Software:** {', '.join(extracted_info.affected_software)}")

    if extracted_info.affected_hardware and len(extracted_info.affected_hardware) > 0:
        lines.append(f"**Affected Hardware:** {', '.join(extracted_info.affected_hardware)}")

    # 3. Horizontal Rule after Metadata
    lines.append("") # Ensure a blank line before a horizontal rule if metadata was short
    lines.append("---")
    lines.append("") # Blank line after HR

    # 4. Problem Summary Section
    lines.append("## Problem Summary")
    lines.append("") # Blank line
    # problem_summary might contain newlines which should be preserved by Markdown
    lines.append(extracted_info.problem_summary if extracted_info.problem_summary.strip() else "No problem summary provided.")
    lines.append("") # Blank line

    # 5. Horizontal Rule and Solution Section
    lines.append("---")
    lines.append("")
    lines.append("## Solution")
    lines.append("")
    # solution_summary might contain newlines (e.g., if LLM generated a list)
    lines.append(extracted_info.solution_summary if extracted_info.solution_summary.strip() else "No solution summary provided.")
    lines.append("")

    # 6. Footer
    lines.append("---")
    lines.append("")
    lines.append("_Generated from support ticket analysis._")

    # Join all parts with a single newline, as blank lines for spacing are already added.
    markdown_output = "\n".join(lines)
    logger.debug(f"Generated Markdown for {extracted_info.ticket_id} (length: {len(markdown_output)}).")
    return markdown_output


if __name__ == '__main__':
    # Setup basic logging for standalone testing
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger.info("Running draft_generator.py module directly for testing.")

    # Create a sample ExtractedTicketInfo object for testing
    sample_extracted_info_1 = ExtractedTicketInfo(
        ticket_id="TICK_VPN_001",
        original_status="resolved",
        problem_summary="User cannot connect to the company VPN after a recent system update, receiving 'Error 619'. Other network functions seem okay.",
        solution_summary="The issue was resolved by guiding the user to download and install the latest version of the VPN client software (v6.2.1) from the company's internal software portal. Connection was successful post-update.\nKey steps:\n1. Navigate to Software Portal.\n2. Search for 'GlobalProtect VPN'.\n3. Download and run installer.\n4. Reboot if prompted.",
        category="Network Issues",
        tags=["vpn", "connectivity", "error_619", "software_update", "windows 11"],
        affected_software=["GlobalProtect VPN Client v5.x", "Windows 11 Enterprise"],
        affected_hardware=[] # Example of empty list
    )

    sample_extracted_info_2 = ExtractedTicketInfo(
        ticket_id="TICK_PRN_005",
        original_status="closed",
        problem_summary="Printer 'PRN-FINANCE-01' is offline and not responding to pings.",
        solution_summary="Technician performed an on-site check.\nFound the printer was unplugged from the power outlet.\nPlugged the printer back in and powered it on.\nSent a test page successfully.\nAdvised user to check power connections first for similar issues.",
        category="Hardware",
        tags=["printer", "offline"],
        affected_hardware=["Printer HP-XYZ-123"],
        affected_software=None # Example of None
    )

    sample_extracted_info_3 = ExtractedTicketInfo(
        ticket_id="TICK_EMPTY_SUM",
        original_status="resolved",
        problem_summary="", # Empty problem summary
        solution_summary="   ", # Solution summary with only whitespace
        category="General"
    )


    logger.info("\n--- Generating Markdown for Sample Ticket 1 (VPN Issue) ---")
    markdown_doc_1 = generate_draft_document(sample_extracted_info_1)
    print(markdown_doc_1)

    logger.info("\n--- Generating Markdown for Sample Ticket 2 (Printer Offline) ---")
    markdown_doc_2 = generate_draft_document(sample_extracted_info_2)
    print(markdown_doc_2)

    logger.info("\n--- Generating Markdown for Sample Ticket 3 (Empty Summaries) ---")
    markdown_doc_3 = generate_draft_document(sample_extracted_info_3)
    print(markdown_doc_3)


    # Basic assertion checks for content
    assert f"**Ticket ID:** {sample_extracted_info_1.ticket_id}" in markdown_doc_1
    assert "## Problem Summary" in markdown_doc_1
    assert sample_extracted_info_1.problem_summary in markdown_doc_1
    assert "## Solution" in markdown_doc_1
    assert "1. Navigate to Software Portal." in markdown_doc_1 # Check if newline list is preserved
    assert "_Generated from support ticket analysis._" in markdown_doc_1
    assert "Tags: vpn, connectivity" in markdown_doc_1
    assert "Affected Hardware" not in markdown_doc_1 # Since it was empty list

    assert f"**Ticket ID:** {sample_extracted_info_2.ticket_id}" in markdown_doc_2
    assert "Category: Hardware" in markdown_doc_2
    assert "Affected Software" not in markdown_doc_2 # Since it was None
    assert "Printer HP-XYZ-123" in markdown_doc_2

    assert "No problem summary provided." in markdown_doc_3
    assert "No solution summary provided." in markdown_doc_3
    assert "# Solution for: Details for Ticket TICK_EMPTY_SUM" in markdown_doc_3


    logger.info("\nDraft generator module test run complete.")
