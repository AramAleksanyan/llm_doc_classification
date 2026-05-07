"""Core data structures for prompt construction (task spec, examples, rendered prompts)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

# Supported shot counts for built-in experiments; larger integers remain valid for research.
ShotCount = Literal[0, 1, 3, 5]

ExampleStrategy = Literal["random_balanced", "random_pool", "semantic", "hard"]


@dataclass(frozen=True)
class FewShotExample:
    """One training-derived in-context example."""

    text: str
    label: str
    source_index: int
    """Row index in the dataframe passed to the sampler (``train_df.index`` value)."""


@dataclass(frozen=True)
class PromptTaskSpec:
    """Dataset-agnostic description of the classification task (no paths)."""

    text_column: str
    label_column: str
    label_names: tuple[str, ...]
    instruction: str | None = None
    """Override for the opening task description; default comes from the template."""

    answer_preamble: str | None = None
    """Final line(s) after the target document (e.g. \"Answer with a single label:\")."""

    def sorted_labels(self) -> tuple[str, ...]:
        return tuple(sorted(self.label_names, key=str.lower))

    def validate_against_frame_columns(self, columns: Sequence[str]) -> None:
        cols = set(columns)
        if self.text_column not in cols:
            raise KeyError(f"text_column {self.text_column!r} not in dataframe columns.")
        if self.label_column not in cols:
            raise KeyError(f"label_column {self.label_column!r} not in dataframe columns.")


@dataclass(frozen=True)
class FewShotSpec:
    """How many in-context examples to draw and with which strategy."""

    n_examples: int
    strategy: ExampleStrategy = "random_balanced"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.n_examples < 0:
            raise ValueError("n_examples must be non-negative.")


@dataclass(frozen=True)
class PromptLayout:
    """Filesystem layout for templates and generated prompts."""

    templates_dir: str
    generated_dir: str
    default_template_name: str = "document_classification.j2"


@dataclass(frozen=True)
class BuiltPrompt:
    """Final user/model-facing prompt plus serializable metadata for reproducibility."""

    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    examples: tuple[FewShotExample, ...] = ()

    def to_serializable_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "metadata": dict(self.metadata),
            "examples": [
                {"text": e.text, "label": e.label, "source_index": e.source_index}
                for e in self.examples
            ],
        }

    def openai_chat_messages(
        self,
        *,
        system_prompt: str = (
            "You classify documents into exactly one label from a provided closed set. "
            "Reply with only the label string."
        ),
    ) -> list[dict[str, str]]:
        """``[{"role":"system"|"user","content":...}]`` for OpenAI / Groq / compatible APIs."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self.text},
        ]
