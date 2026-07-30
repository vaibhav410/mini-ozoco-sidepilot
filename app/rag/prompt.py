"""Prompt templates for both agents.

Keeping every prompt in one file makes the AI behavior reviewable at a
glance -- prompts are configuration, not logic.
"""

from langchain_core.prompts import ChatPromptTemplate

# Sentinel Agent 2 must output when the context does not contain the
# answer. The service layer detects it and returns found=False.
NOT_FOUND_TOKEN = "ANSWER_NOT_FOUND"

# --- Agent 1: Document Understanding / Classification -------------------

CLASSIFICATION_PROMPT = ChatPromptTemplate.from_template(
    """You are a document analysis agent.

Analyze the document below and respond with ONLY a valid JSON object,
no markdown fences, no extra text.

Filename: {filename}

Document text (may be truncated):
---
{text}
---

JSON format:
{{"category": "<one of: Resume, Invoice, Research Paper, Report, General Document>",
 "summary": "<2-3 sentence summary of the document>",
 "topics": ["<3-6 short key topics>"]}}"""
)

# --- Agent 2: Follow-up condensation ------------------------------------

# Conversational RAG: a follow-up like "what about his education?" embeds
# poorly on its own. This prompt rewrites it into a standalone question
# using the recent chat history, and that standalone version is what gets
# routed, embedded, and answered.
CONDENSE_PROMPT = ChatPromptTemplate.from_template(
    """Rewrite the user's new question as a single standalone question,
using the conversation history to resolve references like "he", "it",
"that document", "what about...".

Conversation history:
{history}

New question: {question}

Reply with ONLY the rewritten standalone question. If the question is
already standalone, reply with it unchanged."""
)

# --- Agent 2: Routing ----------------------------------------------------

ROUTING_PROMPT = ChatPromptTemplate.from_template(
    """You are a routing agent. Decide which uploaded document a user's
question is about.

Available documents, in upload order (the last one is the most recent):
{documents}

Question: {question}

Rules:
- Reply with ONLY the doc_id of the single most relevant document.
- If the question says "this document", "the document", "this file" or
  similar without naming a specific one, choose the MOST RECENTLY
  uploaded document (the last in the list).
- If the question is not related to any of the documents, reply with
  exactly: NONE"""
)

# --- Agent 2: Grounded answer generation ---------------------------------

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are a precise question-answering agent. Answer the user's
question using ONLY the context below. Do not use any outside knowledge.

Context from the uploaded documents:
---
{context}
---

Question: {question}

Rules:
- Base the answer strictly on the context above.
- Be clear and concise; use short bullet points when listing items.
- If the context does not contain the information needed to answer,
  reply with exactly: """
    + NOT_FOUND_TOKEN
)

# --- Agent 3: Answer validation ------------------------------------------

# The validation agent double-checks Agent 2's draft before it reaches
# the user: every claim must be supported by the retrieved context.
VALIDATION_PROMPT = ChatPromptTemplate.from_template(
    """You are a strict fact-checking agent. Verify whether the draft
answer below is fully supported by the provided context.

Context from the documents:
---
{context}
---

Question: {question}

Draft answer: {answer}

Respond with ONLY a valid JSON object, no markdown fences:
{{"supported": true or false,
 "confidence": "high" or "medium" or "low",
 "reason": "<one short sentence>"}}

Rules:
- "supported" is true only if every factual claim in the draft answer
  appears in the context.
- Minor rephrasing is fine; invented facts are not."""
)
