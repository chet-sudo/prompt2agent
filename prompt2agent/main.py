"""Small helper script demonstrating the prompt → workflow lifecycle."""

from __future__ import annotations

import argparse
import logging

from .compiler import transform_prompt_to_workflow
from .persistence import load_workflow, save_workflow
from .runtime import run_workflow

logger = logging.getLogger(__name__)


def execute(prompt: str) -> str:
    """Run the full lifecycle for the provided prompt and return the answer string."""

    spec = transform_prompt_to_workflow(prompt)
    workflow_id = save_workflow(spec)
    workflow = load_workflow(workflow_id)
    return run_workflow(workflow)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compile and run a prompt-driven workflow.")
    parser.add_argument("prompt", help="User prompt to feed into the workflow compiler.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    try:
        answer = execute(args.prompt)
    except Exception as exc:  # pragma: no cover - surfaced to CLI
        logger.error("Workflow execution failed: %s", exc)
        raise SystemExit(1) from exc

    print(answer)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
