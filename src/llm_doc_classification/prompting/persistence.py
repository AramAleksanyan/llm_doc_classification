"""Serialize prompts under ``prompts/generated/`` for experiment tracking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from llm_doc_classification.prompting.schemas import BuiltPrompt

SaveFormat = Literal["json", "txt"]


def save_prompt(
    built: BuiltPrompt,
    destination: Path | str,
    *,
    format: SaveFormat = "json",
    mkdir: bool = True,
) -> Path:
    """
    Write a rendered prompt to disk.

    * ``json`` — full :meth:`BuiltPrompt.to_serializable_dict` (text + metadata + examples).
    * ``txt`` — prompt text only (for quick inspection or non-JSON tooling).
    """

    dest = Path(destination)
    if mkdir:
        dest.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        payload = built.to_serializable_dict()
        dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    elif format == "txt":
        dest.write_text(built.text, encoding="utf-8")
    else:
        raise ValueError(f"Unknown format: {format!r}")
    return dest


def default_generated_path(
    generated_dir: Path | str,
    *,
    experiment_id: str,
    split: str,
    row_key: str,
    n_shots: int,
    extension: str = "json",
) -> Path:
    """Convention: ``{generated_dir}/{experiment_id}/{split}_{row_key}_k{n_shots}.{ext}``."""

    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in row_key)[:120]
    return Path(generated_dir) / experiment_id / f"{split}_{safe}_k{n_shots}.{extension}"
