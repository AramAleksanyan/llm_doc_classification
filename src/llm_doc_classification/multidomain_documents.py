"""Create multidomain train/validation/test CSVs for LLM experiments (seven nvidia_domain labels).

The files are not committed to git (large). This module reproduces the split from
``notebooks/exploration/multilabel_documents.ipynb`` using
``agentlans/multilingual-document-classification``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

HF_DATASET_ID = "agentlans/multilingual-document-classification"

ALLOWED_NVIDIA_DOMAINS = frozenset(
    {
        "News",
        "People_and_Society",
        "Sensitive_Subjects",
        "Arts_and_Entertainment",
        "Sports",
        "Travel_and_Transportation",
        "Law_and_Government",
    }
)

DROP_FOR_LLM = [
    "id",
    "original",
    "language",
    "bloom_cognitive_primary",
    "bloom_cognitive_secondary",
    "bloom_knowledge_primary",
    "bloom_knowledge_secondary",
    "reasoning_depth_primary",
    "reasoning_depth_secondary",
    "technical_correctness_primary",
    "technical_correctness_secondary",
    "extraction_artifacts_secondary",
]


def _build_llm_splits(df_llm: pd.DataFrame, *, rng: int = 42, holdout_frac: float = 0.30):
    _SPLIT_RNG = rng
    _HOLDOUT_FRAC = holdout_frac
    _n_by_domain = df_llm.groupby("nvidia_domain", sort=False).size()
    _k_train = int((_n_by_domain * (1.0 - _HOLDOUT_FRAC)).min())
    _k_train = max(_k_train, 1)
    _train_parts = []
    _remain_parts = []
    for _dom in _n_by_domain.index:
        _g = df_llm.loc[df_llm["nvidia_domain"] == _dom]
        _g = _g.sample(frac=1.0, random_state=_SPLIT_RNG).reset_index(drop=True)
        _train_parts.append(_g.iloc[:_k_train])
        _remain_parts.append(_g.iloc[_k_train:])
    train_llm = (
        pd.concat(_train_parts, ignore_index=True)
        .sample(frac=1.0, random_state=_SPLIT_RNG)
        .reset_index(drop=True)
    )
    remain_df = pd.concat(_remain_parts, ignore_index=True)
    try:
        val_llm, test_llm = train_test_split(
            remain_df,
            test_size=0.5,
            random_state=_SPLIT_RNG,
            stratify=remain_df["nvidia_domain"],
        )
    except ValueError:
        val_llm, test_llm = train_test_split(
            remain_df, test_size=0.5, random_state=_SPLIT_RNG
        )
    return (
        train_llm.reset_index(drop=True),
        val_llm.reset_index(drop=True),
        test_llm.reset_index(drop=True),
    )


def write_multidomain_csvs(out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(HF_DATASET_ID)
    train_df = pd.DataFrame(dataset["train"])
    df_llm = train_df.loc[train_df["nvidia_domain"].isin(ALLOWED_NVIDIA_DOMAINS)].copy()
    _unknown = set(df_llm["nvidia_domain"].dropna().unique()) - ALLOWED_NVIDIA_DOMAINS
    if _unknown:
        raise RuntimeError(f"Unexpected domain in filtered frame: {_unknown}")
    cols_drop = [c for c in DROP_FOR_LLM if c in df_llm.columns]
    df_llm = df_llm.drop(columns=cols_drop)
    train_llm, val_llm, test_llm = _build_llm_splits(df_llm)
    for split_name, split_df in (
        ("train", train_llm),
        ("validation", val_llm),
        ("test", test_llm),
    ):
        split_df.to_csv(out_dir / f"{split_name}.csv", index=False)


def resolve_multidomain_dir(repo_root: Path, env: dict | None = None) -> Path:
    import os

    env = env if env is not None else os.environ
    raw = (env.get("LLM_DOC_MULTIDOMAIN_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(repo_root) / "data" / "multidomain_documents"


def ensure_multidomain_csvs(repo_root: Path, env: dict | None = None) -> Path:
    """Return directory containing train.csv / test.csv, downloading & building if missing."""
    d = resolve_multidomain_dir(repo_root, env=env)
    train_csv = d / "train.csv"
    test_csv = d / "test.csv"
    if train_csv.is_file() and test_csv.is_file():
        return d
    write_multidomain_csvs(d)
    if not train_csv.is_file() or not test_csv.is_file():
        raise FileNotFoundError(
            f"Failed to create {train_csv} and {test_csv}. "
            "Check network access and Hugging Face dataset availability."
        )
    return d
