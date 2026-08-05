"""Third-party engine integrations.

Each subpackage is an integration shim for a specific external
serving engine (vLLM, SGLang, ...). They are intentionally
self-contained so the rest of WiSP can be imported on a box without
the integration target installed.
"""
