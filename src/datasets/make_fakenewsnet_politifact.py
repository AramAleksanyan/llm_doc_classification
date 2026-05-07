import os
import sys
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llm_doc_classification.config import fakenewsnet_paths, get_seed

def pick_text_col(df: pd.DataFrame) -> str:
    # common candidates; fallback to first object column
    for c in ["title", "text", "content", "news", "article", "statement", "claim", "url"]:
        if c in df.columns:
            return c
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if obj_cols:
        return obj_cols[0]
    return df.columns[0]

def main(
    fake_path: str | os.PathLike[str] | None = None,
    real_path: str | os.PathLike[str] | None = None,
    out_path: str | os.PathLike[str] | None = None,
    seed: int | None = None,
):
    p = fakenewsnet_paths()
    if fake_path is None:
        fake_path = p["fake"]
    if real_path is None:
        real_path = p["real"]
    if out_path is None:
        out_path = p["out"]
    if seed is None:
        seed = get_seed()
    fake = pd.read_csv(str(fake_path))
    real = pd.read_csv(str(real_path))

    fake_col = pick_text_col(fake)
    real_col = pick_text_col(real)

    fake_df = pd.DataFrame({"text": fake[fake_col].astype(str), "label": "fake"})
    real_df = pd.DataFrame({"text": real[real_col].astype(str), "label": "real"})

    # sample to match paper's 214 total (107+107)
    fake_df = fake_df.sample(n=min(107, len(fake_df)), random_state=seed)
    real_df = real_df.sample(n=min(107, len(real_df)), random_state=seed)

    out = pd.concat([fake_df, real_df], ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    out_s = str(out_path)
    os.makedirs(os.path.dirname(out_s), exist_ok=True)
    out.to_csv(out_s, index=False)

    print(f"Saved: {out_s}")
    print("Label counts:")
    print(out["label"].value_counts())
    print(f"Used fake text column: {fake_col}")
    print(f"Used real text column: {real_col}")

if __name__ == "__main__":
    main()
    