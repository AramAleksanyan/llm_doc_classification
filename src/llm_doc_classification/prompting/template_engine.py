"""Load and render Jinja2 templates from ``prompts/templates/``."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader


class TemplateRepository:
    """Thin wrapper around a Jinja2 ``Environment`` bound to a templates directory."""

    def __init__(self, templates_dir: Path | str) -> None:
        path = Path(templates_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"Templates directory not found: {path}")
        # Plain-text LLM prompts: do not HTML-escape document bodies.
        self._env = Environment(
            loader=FileSystemLoader(str(path)),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render(self, template_filename: str, **context: object) -> str:
        return self._env.get_template(template_filename).render(**context)
