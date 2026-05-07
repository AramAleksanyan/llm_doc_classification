import sys
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llm_doc_classification.config import employee_reviews_paths, fakenewsnet_paths


def main():
    fn_path = fakenewsnet_paths()["out"]
    er_path = employee_reviews_paths()["out_csv"]
    fn = pd.read_csv(fn_path)
    er = pd.read_csv(er_path)

    print("=== FakeNewsNet Politifact ===")
    print(fn.head(3))
    print(fn["label"].value_counts())
    print("empty texts:", (fn["text"].astype(str).str.strip() == "").sum())

    print("\n=== Employee Reviews ===")
    print(er.head(3))
    print(er["label"].value_counts())
    print("empty texts:", (er["text"].astype(str).str.strip() == "").sum())

if __name__ == "__main__":
    main()
    