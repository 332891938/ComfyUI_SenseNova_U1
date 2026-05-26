"""Cooperative cancel checks for long SenseNova inference inside ComfyUI.

Kept outside ``utils/`` so importing it does not load ``utils/__init__.py``
(which would circular-import ``sensenova_u1`` via ``param_count``).
"""

from __future__ import annotations


def throw_if_interrupted() -> None:
    """Raise ``InterruptProcessingException`` when the user cancelled the prompt."""
    try:
        import comfy.model_management as mm

        mm.throw_exception_if_processing_interrupted()
    except ImportError:
        return
