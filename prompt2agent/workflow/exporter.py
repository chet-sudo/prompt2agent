"""Export workflows into standalone Python scripts."""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, Dict

from prompt2agent.utils.logging import get_logger

logger = get_logger(__name__)


EXPORT_TEMPLATE = """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""Auto-generated workflow runner."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from prompt2agent.adapters.model_adapter import ModelAdapter
from prompt2agent.orchestrator.coordinator import run_workflow


WORKFLOW: dict[str, object] = json.loads("""{workflow_json}""")


async def main() -> None:
    adapter = ModelAdapter()
    logs, result, artifacts = await run_workflow(WORKFLOW, model_adapter=adapter)
    print("Execution result:", result)
    print("Logs saved to:", artifacts.log_file)
    print("Result saved to:", artifacts.result_file)


if __name__ == "__main__":
    asyncio.run(main())
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def export_workflow_as_python(workflow: Dict[str, Any], output_path: Path) -> Path:
    """Persist workflow as an executable Python script."""
    payload = textwrap.dedent(EXPORT_TEMPLATE).format(
        workflow_json=textwrap.indent(json_dumps(workflow), " " * 4)
    )
    output_path.write_text(payload, encoding="utf-8")
    logger.info("Exported workflow script to %s", output_path)
    return output_path


def json_dumps(data: Dict[str, Any]) -> str:
    import json

    return json.dumps(data, indent=2)
