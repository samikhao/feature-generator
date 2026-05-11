from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from src.config import API_BASE, API_KEY, MODEL_NAME
from src.agent_tools import ToolCallRecord, build_openai_tool_specs, execute_tool_call
from openai import OpenAI


client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE,
)


class AgentRunResult(BaseModel):
    final_answer: str
    tool_calls: list[ToolCallRecord]
    raw_messages: list[dict[str, Any]]


SYSTEM_PROMPT = """
You are a data analysis agent for tabular machine learning datasets.

Your job is to inspect actual CSV column values before answering.
Rules:
1. Use the available tools whenever the user asks about the contents of columns, numeric ranges, missing values, target behavior, or data quality.
2. If the user explicitly mentions a target column or asks about target distribution, you must call the target-analysis tool before answering.
3. If the user asks about data quality, missing values, or anomalies, you must call the data-quality tool before answering.
4. Only analyze local CSV files inside the project's data directory.
5. Do not invent statistics without tool evidence.
6. Prefer concise, evidence-based answers.
7. If the user asks for feature ideas, ground them in the observed data and mention concrete risks.
""".strip()


def run_data_analysis_agent(user_message: str, max_rounds: int = 6) -> AgentRunResult:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    tool_calls: list[ToolCallRecord] = []
    tools = build_openai_tool_specs()

    for _ in range(max_rounds):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.1,
        )
        message = response.choices[0].message
        message_dict: dict[str, Any] = {"role": "assistant"}

        if message.content:
            message_dict["content"] = message.content
        else:
            message_dict["content"] = ""

        if message.tool_calls:
            message_dict["tool_calls"] = [
                {
                    "id": call.id,
                    "type": call.type,
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in message.tool_calls
            ]
            messages.append(message_dict)

            for tool_call in message.tool_calls:
                record = execute_tool_call(
                    name=tool_call.function.name,
                    arguments_json=tool_call.function.arguments,
                )
                tool_calls.append(record)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(record.output, ensure_ascii=False),
                    }
                )
            continue

        messages.append(message_dict)
        return AgentRunResult(
            final_answer=message.content or "",
            tool_calls=tool_calls,
            raw_messages=messages,
        )

    raise RuntimeError("Agent did not finish within the allowed number of rounds.")
