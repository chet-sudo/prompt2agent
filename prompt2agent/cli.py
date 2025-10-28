"""Command line interface for the Prompt2Agent POC."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from prompt2agent.adapters.model_adapter import ModelAdapter
from prompt2agent.config import WORKFLOWS_DIR, ensure_storage
from prompt2agent.meta.generator import generate_workflow_sync
from prompt2agent.orchestrator.coordinator import run_workflow_sync
from prompt2agent.utils.logging import get_logger
from prompt2agent.workflow.exporter import export_workflow_as_python
from prompt2agent.workflow.persistence import list_workflows, load_workflow, save_workflow

logger = get_logger(__name__)


def _resolve_workflow_path(identifier: str) -> Path:
    path = Path(identifier)
    if path.exists():
        return path
    candidate = WORKFLOWS_DIR / identifier
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Workflow file not found: {identifier}")


def command_generate(args: argparse.Namespace) -> None:
    prompt = args.prompt
    adapter = ModelAdapter(model=args.model)
    workflow = generate_workflow_sync(prompt, adapter)
    path = save_workflow(workflow, goal=prompt)
    print(f"Workflow saved to {path}")


def command_list(_: argparse.Namespace) -> None:
    ensure_storage()
    entries = list_workflows()
    if not entries:
        print("No workflows saved yet.")
        return
    for entry in entries:
        print(json.dumps(entry, indent=2))


def command_run(args: argparse.Namespace) -> None:
    workflow_path = _resolve_workflow_path(args.workflow)
    workflow = load_workflow(workflow_path)
    adapter = ModelAdapter(model=args.model)
    logs, result, artifacts = run_workflow_sync(workflow, model_adapter=adapter)
    print("Run completed. Result:")
    print(result)
    print("Logs saved to:", artifacts.log_file)
    print("Result saved to:", artifacts.result_file)
    if args.verbose:
        print("\nStepwise logs:")
        for entry in logs:
            print(json.dumps(entry, indent=2))


def command_export(args: argparse.Namespace) -> None:
    workflow_path = _resolve_workflow_path(args.workflow)
    workflow = load_workflow(workflow_path)
    output = Path(args.output)
    export_workflow_as_python(workflow, output)
    print(f"Workflow exported to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prompt2Agent workflow CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen_parser = subparsers.add_parser("generate", help="Generate workflow from prompt")
    gen_parser.add_argument("prompt", help="Free-text prompt describing the workflow goal")
    gen_parser.add_argument("--model", help="Model identifier to use", default=None)
    gen_parser.set_defaults(func=command_generate)

    list_parser = subparsers.add_parser("list", help="List saved workflows")
    list_parser.set_defaults(func=command_list)

    run_parser = subparsers.add_parser("run", help="Run a saved workflow")
    run_parser.add_argument("workflow", help="Path or filename of the workflow JSON")
    run_parser.add_argument("--model", help="Model identifier to use", default=None)
    run_parser.add_argument("--verbose", action="store_true", help="Print stepwise logs")
    run_parser.set_defaults(func=command_run)

    export_parser = subparsers.add_parser("export", help="Export workflow as a Python script")
    export_parser.add_argument("workflow", help="Path or filename of the workflow JSON")
    export_parser.add_argument("output", help="Output python file path")
    export_parser.set_defaults(func=command_export)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
