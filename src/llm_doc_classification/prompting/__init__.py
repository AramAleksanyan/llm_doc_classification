"""Universal prompting utilities for zero-/few-shot document classification.

Public surface is intentionally small so experiments depend on stable types
(:class:`PromptBuilder`, :class:`BuiltPrompt`, :class:`PromptTaskSpec`) while
samplers and templates can evolve.
"""

from llm_doc_classification.prompting.builder import PromptBuilder
from llm_doc_classification.prompting.persistence import SaveFormat, default_generated_path, save_prompt
from llm_doc_classification.prompting.sampling import (
    ExampleSampler,
    HardExampleSampler,
    RandomBalancedSampler,
    RandomPoolSampler,
    SemanticRetrievalSampler,
    sampler_for_spec,
    training_pool_mask,
)
from llm_doc_classification.prompting.schemas import (
    BuiltPrompt,
    ExampleStrategy,
    FewShotExample,
    FewShotSpec,
    PromptLayout,
    PromptTaskSpec,
    ShotCount,
)
from llm_doc_classification.prompting.template_engine import TemplateRepository

__all__ = [
    "BuiltPrompt",
    "ExampleSampler",
    "ExampleStrategy",
    "FewShotExample",
    "FewShotSpec",
    "HardExampleSampler",
    "PromptBuilder",
    "PromptLayout",
    "PromptTaskSpec",
    "RandomBalancedSampler",
    "RandomPoolSampler",
    "SaveFormat",
    "SemanticRetrievalSampler",
    "ShotCount",
    "TemplateRepository",
    "default_generated_path",
    "sampler_for_spec",
    "save_prompt",
    "training_pool_mask",
]
