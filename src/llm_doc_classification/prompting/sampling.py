"""Training-pool example selection for in-context learning."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping

import numpy as np
import pandas as pd

from llm_doc_classification.prompting.schemas import ExampleStrategy, FewShotExample, FewShotSpec


class ExampleSampler(ABC):
    """Strategy that picks in-context rows from a training dataframe."""

    @property
    @abstractmethod
    def name(self) -> ExampleStrategy | str:
        """Identifier stored in prompt metadata."""

    @abstractmethod
    def sample(
        self,
        pool: pd.DataFrame,
        *,
        text_column: str,
        label_column: str,
        allowed_labels: Collection[str],
        n_total: int,
        rng: np.random.Generator,
        exclude_positions: Collection[int] | None = None,
    ) -> list[FewShotExample]:
        """
        :param pool: Training data only; must contain ``text_column`` and ``label_column``.
        :param exclude_positions: **Positions** (0..len-1) into ``pool.iloc`` to skip,
            e.g. to avoid leaking the evaluation row when train/eval share a frame.
        """


class RandomBalancedSampler(ExampleSampler):
    """
    Draw ``n_total`` examples with as-even-as-possible counts per label.

    Allocation: ``base, rem = divmod(n_total, n_labels)`` — each label gets ``base``
    samples; the first ``rem`` labels (sorted alphabetically) receive one extra.
    """

    @property
    def name(self) -> str:
        return "random_balanced"

    def sample(
        self,
        pool: pd.DataFrame,
        *,
        text_column: str,
        label_column: str,
        allowed_labels: Collection[str],
        n_total: int,
        rng: np.random.Generator,
        exclude_positions: Collection[int] | None = None,
    ) -> list[FewShotExample]:
        if n_total == 0:
            return []

        labels_sorted = sorted(set(str(x) for x in allowed_labels), key=str.lower)
        n_labels = len(labels_sorted)
        if n_labels == 0:
            raise ValueError("allowed_labels must be non-empty.")

        exclude_pos = set(exclude_positions or ())

        base, rem = divmod(n_total, n_labels)
        quotas = {lab: base + (i < rem) for i, lab in enumerate(labels_sorted)}

        results: list[FewShotExample] = []
        for lab in labels_sorted:
            need = quotas[lab]
            if need == 0:
                continue
            mask = pool[label_column].astype(str) == lab
            positions = np.flatnonzero(mask.to_numpy())
            # map to candidates excluding excluded iloc positions
            candidates = [int(p) for p in positions if p not in exclude_pos]
            if len(candidates) < need:
                raise ValueError(
                    f"Not enough training rows for label {lab!r}: need {need}, "
                    f"have {len(candidates)} after exclusions."
                )
            pick_pos = rng.choice(candidates, size=need, replace=False)
            for pos in pick_pos:
                row = pool.iloc[int(pos)]
                results.append(
                    FewShotExample(
                        text=str(row[text_column]),
                        label=str(row[label_column]),
                        source_index=int(pool.index[int(pos)]),
                    )
                )

        rng.shuffle(results)
        return results


class RandomPoolSampler(ExampleSampler):
    """Uniform sample over all rows (ignores label balance)."""

    @property
    def name(self) -> str:
        return "random_pool"

    def sample(
        self,
        pool: pd.DataFrame,
        *,
        text_column: str,
        label_column: str,
        allowed_labels: Collection[str],
        n_total: int,
        rng: np.random.Generator,
        exclude_positions: Collection[int] | None = None,
    ) -> list[FewShotExample]:
        if n_total == 0:
            return []

        allowed = set(str(x) for x in allowed_labels)
        exclude_pos = set(exclude_positions or ())

        positions: list[int] = []
        for pos in range(len(pool)):
            if pos in exclude_pos:
                continue
            lab = str(pool.iloc[pos][label_column])
            if lab not in allowed:
                continue
            positions.append(pos)

        if len(positions) < n_total:
            raise ValueError(
                f"Need {n_total} examples with allowed labels; only {len(positions)} candidates."
            )

        pick = rng.choice(positions, size=n_total, replace=False)
        out: list[FewShotExample] = []
        for pos in pick:
            row = pool.iloc[int(pos)]
            out.append(
                FewShotExample(
                    text=str(row[text_column]),
                    label=str(row[label_column]),
                    source_index=int(pool.index[int(pos)]),
                )
            )
        return out


class SemanticRetrievalSampler(ExampleSampler):
    """Placeholder: retrieve nearest neighbors in an embedding space (future work)."""

    @property
    def name(self) -> str:
        return "semantic"

    def sample(
        self,
        pool: pd.DataFrame,
        *,
        text_column: str,
        label_column: str,
        allowed_labels: Collection[str],
        n_total: int,
        rng: np.random.Generator,
        exclude_positions: Collection[int] | None = None,
    ) -> list[FewShotExample]:
        raise NotImplementedError(
            "SemanticRetrievalSampler: provide embedding index + query vector, then implement "
            "`sample` (or subclass ExampleSampler) for your experiment."
        )


class HardExampleSampler(ExampleSampler):
    """Placeholder: e.g. high-loss or margin examples from a baseline model."""

    @property
    def name(self) -> str:
        return "hard"

    def sample(
        self,
        pool: pd.DataFrame,
        *,
        text_column: str,
        label_column: str,
        allowed_labels: Collection[str],
        n_total: int,
        rng: np.random.Generator,
        exclude_positions: Collection[int] | None = None,
    ) -> list[FewShotExample]:
        raise NotImplementedError(
            "HardExampleSampler: wire in scores from a baseline classifier and implement selection."
        )


def sampler_for_spec(spec: FewShotSpec) -> ExampleSampler:
    """Factory mapping :class:`FewShotSpec` to the built-in sampler for that strategy."""

    mapping: Mapping[str, ExampleSampler] = {
        "random_balanced": RandomBalancedSampler(),
        "random_pool": RandomPoolSampler(),
        "semantic": SemanticRetrievalSampler(),
        "hard": HardExampleSampler(),
    }
    s = mapping.get(spec.strategy)
    if s is None:
        raise ValueError(f"Unknown strategy: {spec.strategy!r}")
    if spec.n_examples > 0 and spec.strategy in ("semantic", "hard"):
        raise NotImplementedError(
            f"Strategy {spec.strategy!r} has no default implementation; "
            "pass a custom ExampleSampler to PromptBuilder(sampler=...)."
        )
    return s


def training_pool_mask(
    df: pd.DataFrame,
    *,
    label_column: str,
    allowed_labels: Collection[str],
) -> pd.Series:
    """Boolean mask restricting to non-null text/label rows within ``allowed_labels``."""

    allowed = set(allowed_labels)
    lab = df[label_column]
    return lab.notna() & lab.astype(str).isin(allowed)
