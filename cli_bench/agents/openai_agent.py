"""OpenAI-based reference agent for benchmarks."""

import json
import os

from cli_bench.agents.base import BenchAgent
from cli_bench.models.observation import Action, Observation


class OpenAIAgent(BenchAgent):
    """Reference agent using OpenAI API."""

    def __init__(self, model: str = "gpt-4o") -> None:
        self.model = model
        self._messages: list[dict] = []

    async def act(self, observation: Observation) -> Action:
        import httpx

        # Build system message with tool docs on first turn
        if observation.is_first_turn:
            self._messages = [
                {"role": "system", "content": self._build_system_prompt(observation)},
                {"role": "user", "content": observation.task},
            ]
        else:
            # Add previous command result
            content = ""
            if observation.stdout:
                content += f"STDOUT:\n{observation.stdout}\n"
            if observation.stderr:
                content += f"STDERR:\n{observation.stderr}\n"
            self._messages.append({"role": "user", "content": content or "Command executed."})

        # Call OpenAI API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                json={
                    "model": self.model,
                    "messages": self._messages,
                    "temperature": 0,
                },
                timeout=60,
            )
            result = response.json()

        assistant_msg = result["choices"][0]["message"]["content"]
        self._messages.append({"role": "assistant", "content": assistant_msg})

        return self._parse_action(assistant_msg)

    def _build_system_prompt(self, obs: Observation) -> str:
        tools_doc = "\n\n".join(
            f"## {t['name']}\n{t.get('description', '')}\nCommands:\n"
            + "\n".join(
                f"  {c['name']}: {c.get('description', '')}"
                for c in t.get("commands", [])
            )
            for t in obs.tools
        )

        memory_section = ""
        if obs.memory:
            memory_section = (
                f"\n\n## User Context\n```json\n{json.dumps(obs.memory, indent=2)}\n```"
            )

        return f"""You are an AI agent executing CLI commands to complete tasks.

Available tools:
{tools_doc}
{memory_section}

Respond with either:
1. A CLI command: ```command\n<command here>\n```
2. Task completion: ```finish\n<result summary>\n```

Always use the exact CLI tool binary names and argument patterns shown above."""

    def _parse_action(self, text: str) -> Action:
        """Extract command or finish from LLM response text."""
        if "```command" in text:
            cmd_text = text.split("```command")[1].split("```")[0].strip()
            parts = cmd_text.split()
            if parts:
                return Action.command(parts)
            return Action.finish("Empty command")
        elif "```finish" in text:
            result = text.split("```finish")[1].split("```")[0].strip()
            return Action.finish(result)
        return Action.finish(text)

    def reset(self) -> None:
        self._messages = []
