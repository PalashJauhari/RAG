"""System prompt for ``middleware.context_editing.summarize_evicted``.

Used when long threads are truncated; output is plain text (no structured schema).
"""

SUMMARY_SYSTEM_PROMPT = """
You summarize conversation history for a retrieval agent.

Preserve user intent, entity references, constraints, clarification answers,
retrieval decisions, and any facts needed to understand future follow-up queries.
Keep the summary concise and do not invent information.
""".strip()
