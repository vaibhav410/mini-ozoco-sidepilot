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

# --- Agent 4: Screen understanding (vision) ------------------------------

# Sent together with the screenshot in a single multimodal Gemini call.
# A plain string, not a ChatPromptTemplate: the image travels as its own
# content block next to this text, so there is nothing to template.
SCREEN_UNDERSTANDING_INSTRUCTIONS = """You are a screen understanding agent \
inside an AI SidePilot assistant. You are shown a screenshot of the user's \
screen.

Analyze the screenshot and respond with ONLY a valid JSON object, no
markdown fences, no extra text:

{"application": "<the application or website visible, e.g. 'Gmail', 'VS Code', 'Excel'>",
 "activity": "<one short sentence: what the user is doing on screen>",
 "detected_text": "<the most important text visible on screen, condensed>",
 "summary": "<2-3 sentence summary of what is happening on screen>",
 "user_intent": "<the user's most likely goal right now>",
 "suggested_actions": ["<3-5 short, concrete next actions the user could take>"]}

Rules:
- Describe only what is actually visible; never invent content.
- Keep detected_text faithful to the screen (quote key lines, skip chrome
  like menus and scrollbars).
- suggested_actions must be actionable ("Reply to the email from X"),
  not generic ("continue working")."""

# --- Agent 4 fallback: OCR text interpretation ---------------------------

# Used when the vision call fails: PyTesseract extracts the raw text and
# this text-only prompt reconstructs the same structured understanding.
OCR_SCREEN_PROMPT = ChatPromptTemplate.from_template(
    """You are a screen understanding agent inside an AI SidePilot
assistant. A screenshot of the user's screen could not be analyzed
visually, but OCR extracted the raw on-screen text below (it may be
noisy or out of order).

OCR text:
---
{ocr_text}
---

Infer what the user is doing and respond with ONLY a valid JSON object,
no markdown fences, no extra text:

{{"application": "<best guess at the application or website, or 'Unknown'>",
 "activity": "<one short sentence: what the user is doing on screen>",
 "detected_text": "<the most important lines from the OCR text, cleaned up>",
 "summary": "<2-3 sentence summary of what is happening on screen>",
 "user_intent": "<the user's most likely goal right now>",
 "suggested_actions": ["<3-5 short, concrete next actions the user could take>"]}}

Rules:
- Base everything strictly on the OCR text; never invent content.
- If the text is too sparse to tell, say so in the summary rather than
  guessing confidently."""
)

# --- Agent 5: Intent detection -------------------------------------------

INTENT_PROMPT = ChatPromptTemplate.from_template(
    """You are an intent detection agent for an AI SidePilot assistant.

Classify the user's request into exactly ONE of these intents:
- question_answering: asking about the content of uploaded documents
- summarization: wants a summary or overview of a document
- screen_help: wants help with what is currently on their screen
- automation: wants a multi-step action performed for them
- email: wants an email, reply or message written or sent
- export: wants content exported, downloaded or saved as a file
- search: wants to find or locate specific information or a document
- navigation: asking how to get somewhere or open something
- classification: asking what kind or category of document something is

Uploaded documents:
{documents}

Screen context:
{screen}

User request: {question}

Respond with ONLY a valid JSON object, no markdown fences:
{{"intent": "<one intent id from the list above>",
 "confidence": <number between 0.0 and 1.0>,
 "reason": "<one short sentence>"}}"""
)

# --- Agent 6: Automation -------------------------------------------------

EMAIL_DRAFT_PROMPT = ChatPromptTemplate.from_template(
    """You are an email drafting agent inside an AI SidePilot assistant.
Write a professional email that fulfils the user's request, grounded in
the context below when relevant.

Context (documents / screen):
---
{context}
---

User request: {question}

Respond with ONLY a valid JSON object, no markdown fences:
{{"to": "<recipient address if identifiable, else empty string>",
 "subject": "<concise subject line>",
 "body": "<the full email body, plain text, professional tone>"}}"""
)

EXPORT_SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    """You are a summary export agent. Write a clean, well-structured
Markdown summary of the content below, suitable for saving as a report.

Use headings, short paragraphs and bullet points. Do not invent facts.

Content:
---
{context}
---

User request: {question}

Reply with ONLY the Markdown document, starting with a # title line."""
)

ACTION_PLAN_PROMPT = ChatPromptTemplate.from_template(
    """You are a planning agent. Create a concrete, step-by-step action
plan that fulfils the user's request, grounded in the context when
relevant.

Context (documents / screen):
---
{context}
---

User request: {question}

Reply with ONLY a Markdown document: a # title line, then numbered
steps, each with one short explanation line."""
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
