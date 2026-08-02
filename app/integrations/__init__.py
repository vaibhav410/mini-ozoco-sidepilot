"""Integrations package: modular bridges to the outside world.

Each integration is a small, independent module with plain functions --
no LLM logic here. Agents decide *what* to do; integrations do *how*.

- gmail:      email draft creation (.eml + mailto, Gmail API-ready)
- filesystem: safe management of the exports directory
- export:     Markdown and PDF file generation
"""
