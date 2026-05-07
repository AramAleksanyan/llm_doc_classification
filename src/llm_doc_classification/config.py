"""Load `configs/default.json` (repo root) and expose paths, seed, and experiment settings."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


def project_root() -> Path:
    """Repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    return project_root() / "configs" / "default.json"


@lru_cache
def load_config() -> dict[str, Any]:
    path = default_config_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Config not found: {path}. Create it or set LLM_DOC_CONFIG to another JSON path."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_seed() -> int:
    return int(load_config().get("seed", 42))


def bbc_labels() -> list[str]:
    return list(load_config()["bbc"]["labels"])


def paths() -> dict[str, Path]:
    """Resolved ``data``, ``results``, ``models`` under project root."""
    root = project_root()
    p = load_config()["paths"]
    return {
        "root": root,
        "data": root / p["data_dir"],
        "results": root / p["results_dir"],
        "models": root / p["models_dir"],
        "checkpoints": root / p["models_dir"] / p.get("checkpoints_subdir", "checkpoints"),
    }


def bbc_data_dir() -> Path:
    return paths()["data"] / load_config()["bbc"]["subdir"]


def bert_config() -> dict[str, Any]:
    return dict(load_config()["bert"])


def gpt2_profile(name: str) -> dict[str, Any]:
    """``name`` is ``\"124m\"`` or ``\"355m\"`` (keys under ``gpt2``)."""
    g = load_config()["gpt2"]
    if name not in g or name == "training":
        raise KeyError(f'Unknown gpt2 profile: {name!r}. Use "124m" or "355m".')
    out = dict(g[name])
    out["_training"] = g.get("training", {})
    return out


def gpt2_training_defaults() -> dict[str, Any]:
    return dict(load_config()["gpt2"].get("training", {}))


def baselines_config() -> dict[str, Any]:
    return dict(load_config()["baselines"])


def baselines_task_paths() -> list[dict[str, str]]:
    """Tasks with ``path`` resolved under ``data/``."""
    root = paths()["data"]
    tasks = []
    for t in load_config()["baselines"]["tasks"]:
        tasks.append(
            {
                "name": t["name"],
                "path": str(root / t["path"]),
            }
        )
    return tasks


def baselines_results_csv() -> Path:
    return paths()["results"] / load_config()["baselines"]["summary_csv"]


def baselines_detail_json(task_name: str) -> Path:
    pat = load_config()["baselines"]["detail_json_pattern"]
    return paths()["results"] / pat.format(task_name=task_name)


def fakenewsnet_paths() -> dict[str, Path]:
    root = paths()["data"]
    fn = load_config()["fakenewsnet"]
    return {
        "fake": root / fn["fake_csv"],
        "real": root / fn["real_csv"],
        "out": root / fn["out_csv"],
    }


def employee_reviews_paths() -> dict[str, str | Path]:
    er = load_config()["employee_reviews"]
    return {
        "out_csv": paths()["data"] / er["out_csv"],
        "hf_dataset": er["hf_dataset"],
    }


def bert_best_head_weights() -> Path:
    """Path to best BERT linear head checkpoint (``models/...`` under project root)."""
    rel = load_config()["bert"]["best_head_weights"].replace("\\", "/")
    return project_root() / rel


def gpt2_124m_preds_csv() -> Path:
    return paths()["results"] / load_config()["gpt2"]["124m"]["preds_csv"]


def config_path_display() -> str:
    return str(default_config_path())
