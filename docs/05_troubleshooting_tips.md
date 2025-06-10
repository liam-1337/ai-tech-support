# 5. Troubleshooting & Getting Help

This section provides tips for what to do if the AI Tech Support Agent isn't providing the answers you expect, or if you encounter issues while using it.

## When AI Answers Seem Off or Unexpected

If an answer from the AI doesn't seem quite right, or if it's not what you were looking for, here are a few things you can try:

1.  **Rephrase Your Question:**
    *   **Try different keywords:** If your initial query didn't yield good results, think of synonyms or alternative ways to describe the issue. For example, instead of "VPN won't connect," try "VPN connection error [specific error number if you have one]."
    *   **Be more specific:** If your question was broad ("My computer is slow"), try adding more details ("My Dell laptop is slow when opening Outlook").
    *   **Be less specific:** If you were very specific and got no results (or an "I don't know" response), try broadening your query slightly, as the exact phrasing might not match the knowledge base.
    *   **Break down complex questions:** Ask a series of simpler, related questions instead of one very complex one.

2.  **Leverage Conversation Context (Session Memory):**
    *   The AI remembers the previous turns within your current conversation session (see "[Understanding Conversation Context & Sessions](./02_using_the_chat_interface.md#understanding-conversation-context--sessions-session-persistence)" for more details).
    *   If the AI seems to have misunderstood a follow-up or missed a detail you mentioned earlier *in the current session*, gently remind it by restating the key context. For example, "Regarding the VPN steps you gave me, when I tried step 2, I got error X. What should I do next?"

3.  **Interacting with Automated Guidance:**
    *   For common issues (like password resets or basic connectivity), the AI might append a structured, step-by-step automated guide to its response.
    *   **Follow these steps carefully.** They are designed to be clear and actionable.
    *   If the automated guide doesn't resolve your issue, inform the AI. For example, "I followed the connectivity guide, but pinging 8.8.8.8 still fails." This gives the AI more information to help you further or determine if escalation is needed.

4.  **Review the Source Documents:**
    *   As detailed in "[Understanding AI Responses](./03_understanding_ai_responses.md)", the AI often provides "Sources" with its answers. These are snippets from the knowledge base it used.
    *   **Expand and read these sources.** The AI's summary might be missing a nuance that the original text provides, or the source itself might not be a perfect match for your specific scenario.
    *   This can help you understand *why* the AI gave a particular answer.

5.  **Consider Knowledge Gaps:**
    *   As mentioned in the "[Knowledge Sources](./04_knowledge_sources.md)" section, the AI only knows what's in its loaded and continuously updated knowledge base. If you're asking about a very new system not yet documented for the AI, or a topic outside its intended scope, it may correctly state that it doesn't have the information.

## Common Usage Issues & Tips

*   **AI Doesn't Seem to Understand My Question:**
    *   Refer back to the "[Tips for Phrasing Effective Questions](./02_using_the_chat_interface.md#tips-for-phrasing-effective-questions)". Clear, specific questions with relevant context usually get the best results.
    *   Utilize the session context as described above to build on previous interactions.

*   **AI is Slow to Respond:**
    *   Generating answers, especially if they involve searching a large knowledge base and using a complex language model, can take a few moments.
    *   Please be patient. If it seems excessively slow (e.g., more than 30 seconds), you could try sending a simpler version of your query or reloading the interface (which might start a new session if the UI doesn't preserve it) and trying again after a short break. If slowness is persistent, please report it.

*   **Chat Interface Not Loading or Displaying Correctly:**
    *   **Refresh the page:** This is often the quickest fix for minor display glitches (Ctrl+R or Cmd+R).
    *   **Check your internet/company network connection.**
    *   **Try clearing your browser's cache and cookies** (for this specific site, if your browser allows), then reload the page. Instructions for this vary by browser.
    *   **Try a different web browser:** This can help determine if the issue is specific to one browser. If it works in another browser, please report this detail.

## Providing Feedback & Reporting Issues

Your feedback is crucial for identifying knowledge gaps, improving the AI's accuracy, and fixing any technical problems.

*   **For issues with the AI's answers (e.g., incorrect, irrelevant, missing information where you expect it to exist):**
    *   Note down:
        *   The **`session_id`** of your conversation (usually provided with each AI response). This is very helpful for us to find the specific interaction log.
        *   The exact question(s) you asked.
        *   The AI's full answer.
        *   The source documents provided (if any).
        *   Why you believe the answer is problematic or what information was missing.
*   **For issues with the AI agent's functionality (e.g., bugs in the chat interface, errors appearing, very slow performance):**
    *   Note down:
        *   The **`session_id`** if the issue occurred during a conversation.
        *   What you were doing when the issue occurred.
        *   Any error messages you saw (screenshots can be very helpful).
        *   The date and time of the issue.

**How to Report:**
Please send your feedback or report any issues to the **IT Helpdesk via a new ticket under category 'AI Support Agent Feedback'**. Providing specific examples, including your original question, the AI's response, and the `session_id`, will help us address the problem more effectively.

## Requesting Escalation to Human Support

If you've tried troubleshooting with the AI and are still stuck, or if you believe your issue requires human intervention immediately, you can request an escalation.

*   **How to Request:** Simply tell the AI you'd like to escalate. Try phrases like:
    *   "Please escalate this issue."
    *   "I need to talk to a human."
    *   "Can you connect me to an agent?"
*   **What Happens:** As described in "[When the AI Can't Help: Issue Escalation](./03_understanding_ai_responses.md#when-the-ai-cant-help-issue-escalation-to-human-support)", the AI should recognize your request and attempt to create a support ticket. It will provide you with a ticket ID. The conversation history from your session will be attached to this ticket for the human support team.

## Conclusion

We hope the AI Tech Support Agent becomes a valuable and efficient tool in your daily IT support tasks. Like any advanced technology, it has its strengths and areas for ongoing improvement. By understanding how it works, providing constructive feedback, and using it critically, you can help make it an even more powerful resource for our team.

Thank you for using and helping to refine the AI Tech Support Agent!

---
*   **Previous: [4. Knowledge Sources](./04_knowledge_sources.md)**
