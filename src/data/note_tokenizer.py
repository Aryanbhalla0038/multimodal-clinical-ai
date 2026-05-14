"""BioBERT tokenizer for clinical notes.

We truncate from the *front* (keep the tail) because the Assessment & Plan,
which is the most clinically informative section, sits at the end of MIMIC
discharge summaries.
"""
from __future__ import annotations
from typing import Optional

from transformers import AutoTokenizer, BertTokenizer

_DEFAULT_MODEL = "dmis-lab/biobert-base-cased-v1.1"
_TOKENIZER = None


def get_tokenizer(model_name: str = _DEFAULT_MODEL):
    """Try fast tokenizer first; fall back to slow BertTokenizer.

    Newer transformers (>=5) dropped auto-conversion for legacy BioBERT,
    which ships only a vocab.txt and no tokenizer.json.
    """
    global _TOKENIZER
    if _TOKENIZER is not None and getattr(_TOKENIZER, "name_or_path", None) == model_name:
        return _TOKENIZER
    try:
        _TOKENIZER = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except (ValueError, OSError):
        try:
            _TOKENIZER = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        except (ValueError, OSError):
            _TOKENIZER = BertTokenizer.from_pretrained(model_name)
    return _TOKENIZER


def tokenize_note(
    text: str,
    max_len: int = 512,
    model_name: str = _DEFAULT_MODEL,
    keep_tail: bool = True,
) -> dict:
    """
    Returns a dict with `input_ids` and `attention_mask` tensors of shape (1, max_len).

    If `keep_tail` is True (recommended for MIMIC discharge summaries), we
    pre-truncate the raw text at the character level to keep the END of the note,
    then let HF tokenizer pad/truncate.
    """
    tokenizer = get_tokenizer(model_name)
    if keep_tail:
        # Cheap heuristic: ~6 chars/token. Keep ~6x the budget from the END.
        approx_chars = max_len * 6
        if len(text) > approx_chars:
            text = text[-approx_chars:]
    return tokenizer(
        text,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
