import os
import random
import re
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llm_doc_classification.config import employee_reviews_paths, get_seed

REMOTE_PAT = re.compile(r"\b(remote|wfh|work from home|telework|virtual office|home[- ]office)\b", re.I)
ONSITE_PAT = re.compile(r"\b(on[- ]?site|office[- ]only|no remote|not remote|in[- ]office|commute)\b", re.I)

def label_review(text: str) -> str:
    t = (text or "").lower()
    if REMOTE_PAT.search(t):
        return "working remotely"
    if ONSITE_PAT.search(t):
        return "not working remotely"
    return "not mentioned"

def main(
    out_path: str | os.PathLike[str] | None = None,
    seed: int | None = None,
):
    er = employee_reviews_paths()
    if out_path is None:
        out_path = str(er["out_csv"])
    if seed is None:
        seed = get_seed()
    random.seed(seed)

    ds = load_dataset(str(er["hf_dataset"]), split="train")

    # choose a text field if present; otherwise join string fields
    col_candidates = ["review", "text", "content", "pros", "cons"]
    cols = ds.column_names
    text_col = next((c for c in col_candidates if c in cols), None)

    if text_col is None:
        def row_to_text(row):
            parts = []
            for k, v in row.items():
                if isinstance(v, str) and v.strip():
                    parts.append(f"{k}: {v.strip()}")
            return "\n".join(parts)
        texts = [row_to_text(ds[i]) for i in range(len(ds))]
    else:
        texts = [ds[i][text_col] for i in range(len(ds))]

    df = pd.DataFrame({"text": texts})
    df["label"] = df["text"].apply(label_review)

    # try to balance ~333 per class (total ~1000 like the paper)
    target_per_class = 333
    parts = []
    for lbl in ["working remotely", "not working remotely", "not mentioned"]:
        sub = df[df["label"] == lbl]
        if len(sub) == 0:
            continue
        parts.append(sub.sample(n=min(target_per_class, len(sub)), random_state=seed))

    out = pd.concat(parts, ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # force exactly 1000 rows (pad/truncate)
    if len(out) < 1000:
        remaining = 1000 - len(out)
        extra = df.sample(n=remaining, random_state=seed)
        out = pd.concat([out, extra], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    elif len(out) > 1000:
        out = out.sample(n=1000, random_state=seed).reset_index(drop=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print("Label counts:")
    print(out["label"].value_counts())

if __name__ == "__main__":
    main()
    