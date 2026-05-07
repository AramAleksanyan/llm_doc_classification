#!/usr/bin/env python3
"""
Build zero-/few-shot prompts from a JSON task description and a training CSV.

Usage (from repository root)::

    pip install jinja2 pandas  # plus project requirements
    python examples/prompting/build_prompts_demo.py \\
        --train-csv data/multidomain_documents/train.csv \\
        --task-json configs/prompting/7domain_labels.json \\
        --n-examples 3 \\
        --target-row-iloc 0

``PYTHONPATH=src`` must include the library (this script prepends ``src/`` automatically).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_src_on_path() -> Path:
    repo = Path(__file__).resolve().parents[2]
    src = repo / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return repo


def main() -> None:
    repo = _ensure_src_on_path()
    from llm_doc_classification.prompting import (
        FewShotSpec,
        PromptBuilder,
        PromptTaskSpec,
        default_generated_path,
    )

    p = argparse.ArgumentParser(description="Build classification prompts for LLM experiments.")
    p.add_argument(
        "--task-json",
        type=Path,
        default=repo / "configs" / "prompting" / "7domain_labels.json",
        help="JSON with keys text_column, label_column, labels (optional instruction).",
    )
    p.add_argument(
        "--train-csv",
        type=Path,
        default=repo / "data" / "multidomain_documents" / "train.csv",
        help="Training split CSV (few-shot rows are sampled only from here).",
    )
    p.add_argument(
        "--n-examples",
        type=int,
        choices=(0, 1, 3, 5),
        default=3,
        help="In-context example count (balanced across labels when using default sampler).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    p.add_argument(
        "--target-row-iloc",
        type=int,
        default=0,
        help="Row position in --train-csv used only as target text (still sampled few-shot from full train pool).",
    )
    p.add_argument(
        "--exclude-target-from-shots",
        action="store_true",
        help="If set, exclude the target row index from few-shot sampling (recommended if target comes from train).",
    )
    p.add_argument(
        "--strategy",
        default="random_balanced",
        choices=("random_balanced", "random_pool"),
        help="Example selection strategy (semantic/hard require a custom ExampleSampler in code).",
    )
    p.add_argument(
        "--experiment-id",
        default="demo",
        help="Subfolder under prompts/generated/",
    )
    args = p.parse_args()

    if not args.task_json.is_file():
        raise SystemExit(f"Missing task JSON: {args.task_json}")
    if not args.train_csv.is_file():
        raise SystemExit(
            f"Missing training CSV: {args.train_csv}. "
            "Export splits from the notebook or pass --train-csv."
        )

    raw = json.loads(args.task_json.read_text(encoding="utf-8"))
    text_col = raw["text_column"]
    label_col = raw["label_column"]
    labels: list[str] = list(raw["labels"])
    instruction = raw.get("instruction")

    import pandas as pd

    train_df = pd.read_csv(args.train_csv)

    tgt_pos = args.target_row_iloc
    if not (0 <= tgt_pos < len(train_df)):
        raise SystemExit(f"--target-row-iloc {tgt_pos} out of range for len {len(train_df)}")
    target_row = train_df.iloc[tgt_pos]
    target_text = str(target_row[text_col])
    target_index = train_df.index[tgt_pos]

    task = PromptTaskSpec(
        text_column=text_col,
        label_column=label_col,
        label_names=tuple(labels),
        instruction=instruction,
    )
    few = FewShotSpec(n_examples=args.n_examples, strategy=args.strategy, seed=args.seed)
    builder = PromptBuilder(task, few)

    exclude = {target_index} if args.exclude_target_from_shots else None
    built = builder.build(train_df, target_text, exclude_train_index=exclude)

    out_dir = repo / "prompts" / "generated"
    dest = default_generated_path(
        out_dir,
        experiment_id=args.experiment_id,
        split="train",
        row_key=str(target_index),
        n_shots=args.n_examples,
        extension="json",
    )
    builder.save(built, dest, format="json")
    print(f"Wrote {dest}")
    print("--- prompt preview (800 chars) ---")
    print(built.text[:800] + ("…" if len(built.text) > 800 else ""))


if __name__ == "__main__":
    main()
