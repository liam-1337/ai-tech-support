# 4. What the AI Knows: Understanding Its Knowledge Sources

To use the AI Tech Support Agent effectively, it's helpful to understand where it gets its information. The AI doesn't "think" like a human or browse the public internet in real-time. Instead, its answers are based entirely on the information that has been specifically loaded into its dedicated knowledge base.

## Main Knowledge Sources

The AI agent's knowledge comes primarily from two types of internal company information:

1.  **Internal IT Documents & Manuals:**
    *   A curated collection of official IT documents forms a core part of the AI's knowledge. This can include:
        *   Setup guides for standard company software (e.g., Microsoft Office, VPN client, specialized internal applications).
        *   Troubleshooting procedures for common network connectivity or hardware issues.
        *   Company IT policies (e.g., security guidelines, data backup procedures).
        *   Best practice documents.
        *   Frequently Asked Questions (FAQs) related to IT services and systems.
    *   The accuracy and usefulness of the AI heavily depend on these documents being comprehensive and up-to-date.

2.  **Analyzed IT Support Tickets (Historical Data):**
    *   The AI also learns from past IT support tickets, specifically those that have been successfully "Resolved" or "Closed."
    *   The problem descriptions and, crucially, the resolution steps from these historical tickets are processed. The system uses AI (an LLM) to summarize these details, and this summarized information is then transformed into a structured Markdown format.
    *   These Markdown documents, derived from past solutions, are then added to the AI's knowledge base.
    *   This allows the AI to provide solutions based on real-world issues that have been successfully addressed by our IT team.

## What This Means for You

Understanding these sources helps set the right expectations:

*   **Scope of Knowledge:** The AI's expertise is strictly limited to the information contained within the documents and processed ticket data loaded into its knowledge base. It cannot answer questions about topics not covered in these materials.
*   **Information Freshness (Data Cutoff):** The AI's knowledge reflects the state of the documents and tickets at the time they were last ingested or processed. For example, if a new software version was released yesterday and the relevant documentation hasn't been added to the AI's knowledge base yet, it won't know about the very latest changes. It does not have live access to external internet information or real-time system status.
*   **Importance of Data Quality:** The accuracy and helpfulness of the AI's answers directly depend on the quality, clarity, and comprehensiveness of the source documents and the details recorded in past IT support tickets. If the source information is ambiguous or incorrect, the AI may reflect these limitations.

## Helping Improve the AI's Knowledge

The AI Tech Support Agent is designed to be an evolving tool.
*   If you consistently find that the AI is unable to answer questions about a specific important topic, it might indicate a gap in its current knowledge base.
*   If you believe key internal documents are missing or that information from certain types of resolved tickets would be beneficial, please provide this feedback.

Suggestions for new knowledge sources or areas for improvement can be directed to the **[Specify Contact Person/Team/Channel from `05_troubleshooting_tips.md` here]**. Your input is crucial for making the AI agent more effective.

## Next Steps

Sometimes, you might not get the answer you expect, or you might have questions about using the AI itself. The next section provides some tips.

---
*   **Previous: [3. Understanding AI Responses](./03_understanding_ai_responses.md)**
*   **Next: [5. Troubleshooting Tips](./05_troubleshooting_tips.md)**
