"""High-level API: sample training examples and render classification prompts from templates."""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from llm_doc_classification.config import project_root
from llm_doc_classification.prompting.persistence import SaveFormat, save_prompt
from llm_doc_classification.prompting.sampling import ExampleSampler, sampler_for_spec, training_pool_mask
from llm_doc_classification.prompting.schemas import BuiltPrompt, FewShotSpec, PromptLayout, PromptTaskSpec
from llm_doc_classification.prompting.template_engine import TemplateRepository


class PromptBuilder:
    """
    Build zero- / few-shot document classification prompts from real training rows.

    * **Separation of concerns**: sampling (``ExampleSampler``) is independent of
      template rendering (``TemplateRepository``). Swap samplers or templates without
      changing inference clients — they only consume :class:`BuiltPrompt`.
    * **Integration**: pass ``built.text`` to Hugging Face ``text-generation`` pipelines,
      OpenAI-compatible chat/complete endpoints (as the user message), or local LLMs.
    """

    def __init__(
        self,
        task: PromptTaskSpec,
        few_shot: FewShotSpec,
        *,
        layout: PromptLayout | None = None,
        sampler: ExampleSampler | None = None,
        repo_root: Path | None = None,
    ) -> None:
        self.task = task
        self.few_shot = few_shot
        root = repo_root or project_root()
        self.layout = layout or PromptLayout(
            templates_dir=str(root / "prompts" / "templates"),
            generated_dir=str(root / "prompts" / "generated"),
        )
        self._sampler: ExampleSampler = sampler if sampler is not None else sampler_for_spec(few_shot)
        self._templates = TemplateRepository(Path(self.layout.templates_dir))

    def training_pool(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Restrict to rows with usable text/label within ``task.label_names``."""

        self.task.validate_against_frame_columns(train_df.columns)
        m = training_pool_mask(
            train_df,
            label_column=self.task.label_column,
            allowed_labels=self.task.label_names,
        )
        m &= train_df[self.task.text_column].notna()
        return train_df.loc[m].copy()

    def build(
        self,
        train_df: pd.DataFrame,
        target_text: str,
        *,
        exclude_train_index: Collection[Any] | None = None,
        template_name: str | None = None,
        extra_template_context: dict[str, Any] | None = None,
    ) -> BuiltPrompt:
        """
        :param train_df: Training split only; few-shot rows are sampled from here.
        :param target_text: Document body to classify.
        :param exclude_train_index: Row index **labels** (``train_df.index``) to omit
            from the few-shot pool (e.g. the same example as ``target_text`` if duplicated).
        """

        pool = self.training_pool(train_df)
        exclude_ilocs: list[int] | None = None
        if exclude_train_index:
            ex = set(exclude_train_index)
            exclude_ilocs = [i for i, idx in enumerate(pool.index) if idx in ex]

        rng = np.random.default_rng(self.few_shot.seed)
        examples = self._sampler.sample(
            pool,
            text_column=self.task.text_column,
            label_column=self.task.label_column,
            allowed_labels=self.task.label_names,
            n_total=self.few_shot.n_examples,
            rng=rng,
            exclude_positions=exclude_ilocs,
        )

        tpl = template_name or self.layout.default_template_name
        ctx: dict[str, Any] = {
            "task_instruction": self.task.instruction,
            "answer_preamble": self.task.answer_preamble,
            "label_names": list(self.task.sorted_labels()),
            "examples": [{"text": e.text, "label": e.label} for e in examples],
            "target_text": target_text,
        }
        if extra_template_context:
            ctx.update(extra_template_context)

        text = self._templates.render(tpl, **ctx)
        meta: dict[str, Any] = {
            "template": tpl,
            "n_examples": len(examples),
            "strategy": self._sampler.name,
            "seed": self.few_shot.seed,
            "text_column": self.task.text_column,
            "label_column": self.task.label_column,
            "label_names": list(self.task.label_names),
        }
        return BuiltPrompt(text=text, metadata=meta, examples=tuple(examples))

    def save(
        self,
        built: BuiltPrompt,
        destination: Path | str,
        *,
        format: SaveFormat = "json",
        mkdir: bool = True,
    ) -> Path:
        """Persist via :func:`~llm_doc_classification.prompting.persistence.save_prompt`."""

        return save_prompt(built, destination, format=format, mkdir=mkdir)
