# Prompt → Reusable Multi-Agent Workflow (POC)

```
+-------------+       +--------------------+       +-----------------+
| CLI / Meta  |  -->  | Workflow Generator |  -->  | Workflow JSON   |
| Prompt      |       |  (LLM + fallback)  |       |  + Persistence  |
+-------------+       +--------------------+       +-----------------+
        |                        |                             |
        v                        v                             v
+-------------+       +--------------------+       +-----------------+
| Orchestrator|  -->  | Agent Factory      |  -->  | OpenAI Agents   |
| (Execution) |       | + Tool Registry    |       | SDK + Tools     |
+-------------+       +--------------------+       +-----------------+
        |                        |
        v                        v
+-------------+       +--------------------+
| Runs Folder |       | Exported Runner    |
+-------------+       +--------------------+
```

## Overview

This proof-of-concept project turns a single free-text goal into a reusable multi-agent workflow powered by the [OpenAI Agents Python SDK](https://github.com/openai/openai-agents-python) and executed through the OpenRouter API.

Key capabilities:

- Generate validated workflow JSON via an LLM-backed meta agent with deterministic fallback.
- Persist workflows with versioned filenames and maintain an index.
- Instantiate agents, memory buffers, and tools, then orchestrate sequential or peer execution.
- Save run logs/results, and export workflows as standalone Python runners.
- Provide a straightforward CLI for generation, listing, execution, and export.

## Requirements

- Python 3.10+
- Environment variable `OPENROUTER_API_KEY` set to a valid OpenRouter token.
- (Optional) `PROMPT2AGENT_DEFAULT_MODEL` to override the default OpenRouter model id.

Install dependencies:

```bash
pip install -r requirements.txt
```

## CLI Usage

Generate a workflow from a prompt:

```bash
python -m prompt2agent.cli generate "Plan a weekend trip to Kyoto"
```

List saved workflows:

```bash
python -m prompt2agent.cli list
```

Execute a workflow:

```bash
python -m prompt2agent.cli run workflows/workflow_plan-a-weekend-trip-fallback-20240101010101.json --verbose
```

Export a workflow to a standalone Python script:

```bash
python -m prompt2agent.cli export workflows/workflow_plan-a-weekend-trip_*.json exported_runner.py
python exported_runner.py
```

Run artifacts are stored under `runs/` with stepwise logs and final results.

## Notes

- The meta agent uses the OpenRouter API; ensure outbound HTTPS access is available.
- Tools are pluggable; extend `prompt2agent/tools` to add new integrations.
- Memory buffers are implemented via `ShortTermMemoryBuffer`, a per-agent ring buffer.
