import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from llm_doc_classification.config import (
    baselines_config,
    baselines_detail_json,
    baselines_results_csv,
    baselines_task_paths,
    get_seed,
    paths,
)

SEED = get_seed()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{path} must contain columns: text, label")

    # Keep only needed columns
    df = df[["text", "label"]]

    # Drop rows with missing labels
    df = df.dropna(subset=["label"])

    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(str)

    # Remove rows that become empty after cleaning
    df["text"] = df["text"].str.strip()
    df = df[df["text"] != ""]

    df = df.reset_index(drop=True)
    return df


def eval_metrics(y_true, y_pred, labels):
    acc = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    prec, rec, f1_per, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": acc,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "per_class": {
            lbl: {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(s),
            }
            for lbl, p, r, f, s in zip(labels, prec, rec, f1_per, sup)
        },
        "confusion_matrix": cm.tolist(),
        "labels_order": labels,
    }


def run_cv_gridsearch(
    X, y, labels, model_name: str, pipeline: Pipeline, param_grid: dict, n_splits: int = 5
):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    # Use weighted F1 as the selection metric (paper emphasizes weighted F1)
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=skf,
        n_jobs=1,
        verbose=0,
    )

    t0 = time.time()
    grid.fit(X, y)
    train_time = time.time() - t0

    best = grid.best_estimator_
    best_params = grid.best_params_
    best_cv_score = grid.best_score_

    # Evaluate via CV predictions with best model: manual fold loop
    all_true = []
    all_pred = []
    fold_times = []

    for fold_idx, (tr, te) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        t1 = time.time()
        best.fit(X_tr, y_tr)
        y_hat = best.predict(X_te)
        fold_times.append(time.time() - t1)

        all_true.extend(list(y_te))
        all_pred.extend(list(y_hat))

    metrics = eval_metrics(all_true, all_pred, labels)

    return {
        "model": model_name,
        "best_params": best_params,
        "best_cv_f1_weighted": float(best_cv_score),
        "grid_fit_time_sec": float(train_time),
        "mean_fold_fit_predict_time_sec": float(np.mean(fold_times)),
        **metrics,
    }


def main():
    ensure_dir(str(paths()["results"]))

    tasks = baselines_task_paths()

    all_rows = []

    for task in tasks:
        df = load_csv(task["path"])
        X = df["text"].astype(str).to_numpy(dtype=object)
        y = df["label"].astype(str).to_numpy(dtype=object)
        labels = sorted(df["label"].unique().tolist())

        print(f"\n=== Task: {task['name']} ===")
        print("Label counts:\n", df["label"].value_counts())

        # --- Naive Bayes pipeline ---
        nb_pipe = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer()),
                ("clf", MultinomialNB()),
            ]
        )
        nb_grid = {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__min_df": [1, 2, 5],
            "tfidf__max_df": [0.9, 1.0],
            "clf__alpha": [0.1, 0.5, 1.0],
        }

        nb_res = run_cv_gridsearch(
            X,
            y,
            labels,
            "TFIDF+MultinomialNB",
            nb_pipe,
            nb_grid,
            n_splits=baselines_config()["cv_splits"],
        )
        nb_res["task"] = task["name"]
        all_rows.append(nb_res)
        print("NB weighted F1:", nb_res["f1_weighted"])

        # --- Linear SVM pipeline ---
        svm_pipe = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer()),
                ("clf", LinearSVC()),
            ]
        )
        svm_grid = {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__min_df": [1, 2, 5],
            "tfidf__max_df": [0.9, 1.0],
            "clf__C": [0.1, 1.0, 10.0],
        }

        svm_res = run_cv_gridsearch(
            X,
            y,
            labels,
            "TFIDF+LinearSVC",
            svm_pipe,
            svm_grid,
            n_splits=baselines_config()["cv_splits"],
        )
        svm_res["task"] = task["name"]
        all_rows.append(svm_res)
        print("SVM weighted F1:", svm_res["f1_weighted"])

        # Save detailed JSON per task
        detail_path = str(baselines_detail_json(task["name"]))
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(
                {"task": task["name"], "results": [nb_res, svm_res]},
                f,
                indent=2,
                ensure_ascii=False,
            )
        print("Saved detail:", detail_path)

    # Save summary CSV
    out_df = pd.DataFrame(all_rows)

    # Keep summary columns first (JSON columns stay but you can ignore)
    cols_first = [
        "task",
        "model",
        "accuracy",
        "f1_weighted",
        "f1_macro",
        "best_cv_f1_weighted",
        "grid_fit_time_sec",
        "mean_fold_fit_predict_time_sec",
        "best_params",
    ]
    cols_first = [c for c in cols_first if c in out_df.columns]
    out_df = out_df[cols_first + [c for c in out_df.columns if c not in cols_first]]

    out_csv = str(baselines_results_csv())
    out_df.to_csv(out_csv, index=False)
    print("\nSaved summary:", out_csv)


if __name__ == "__main__":
    main()
    