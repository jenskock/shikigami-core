"""Map body-control tools onto Pipecat function handlers.

Speech is the LLM text stream. These handlers only move the body. They set
``run_llm=False`` because actuators do not return conversational data — a
second completion after every emote would talk twice. Spoken words already
live in the same LLM turn as the tool call.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from pipecat.adapters.llm.schemas.direct_function import tool_options
from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.frames.frames import FunctionCallResultProperties, TTSSpeakFrame
from pipecat.services.llm_service import FunctionCallParams

from ..brain.actions import TOOL_SCHEMAS, Action, parse_tool_call

STOP_TOOL_NAME = "stop_running_tools"

_log = logging.getLogger("shikigami.voice.tools")


class ToolHooks(Protocol):
    def note_tool(self, action: Action) -> None: ...

    async def execute(self, action: Action) -> None: ...

    def tool_enter(self) -> int | None: ...

    def tool_exit(self, turn: int | None = None) -> None: ...


def speak_frame(text: str) -> TTSSpeakFrame:
    """Apology / error speech only. Normal replies use the LLM text stream."""
    frame = TTSSpeakFrame(text, append_to_context=False)
    frame.interruptible = True
    return frame


def build_tool_schemas(hooks: ToolHooks) -> list[FunctionSchema]:
    """One schema per body tool, plus stop_running_tools for long move/idle."""

    # tool_call_id -> (tool name, task). Body tools keep running when the user
    # talks over the bot; only stop_running_tools or failure ends them.
    running: dict[str, tuple[str, asyncio.Task]] = {}
    stopped: set[str] = set()

    def handler_for(name: str):
        @tool_options(cancel_on_interruption=False)
        async def handler(params: FunctionCallParams) -> None:
            turn = hooks.tool_enter()
            call_id = params.tool_call_id
            try:
                action = parse_tool_call(params.function_name, dict(params.arguments))
                hooks.note_tool(action)
                task = asyncio.create_task(hooks.execute(action))
                running[call_id] = (name, task)
                try:
                    await task
                except asyncio.CancelledError:
                    if call_id not in stopped:
                        raise
                finally:
                    running.pop(call_id, None)
                    stopped.discard(call_id)
                # Actuators are not data tools: do not run a follow-up completion.
                await params.result_callback(
                    {"ok": True},
                    properties=FunctionCallResultProperties(run_llm=False),
                )
            except Exception as exc:
                _log.warning("tool %s failed: %s", name, type(exc).__name__)
                await params.result_callback(
                    {"ok": False, "error": str(exc)},
                    properties=FunctionCallResultProperties(run_llm=False),
                )
            finally:
                hooks.tool_exit(turn)

        return handler

    async def stop_handler(params: FunctionCallParams) -> None:
        wanted = str(params.arguments.get("name") or "").strip()
        names: list[str] = []
        for call_id, (name, task) in list(running.items()):
            if wanted and wanted not in name:
                continue
            stopped.add(call_id)
            task.cancel()
            names.append(name)
        if names:
            result: dict = {"stopped": names}
        else:
            result = {"stopped": [], "message": "Nothing was running."}
        await params.result_callback(
            result,
            properties=FunctionCallResultProperties(run_llm=False),
        )

    schemas: list[FunctionSchema] = [
        FunctionSchema(
            name=STOP_TOOL_NAME,
            description=(
                "Stop body tools that are still running (long move or idle). "
                "Call only when the user clearly asks to stop or cancel. "
                "Never call it just because the user spoke."
            ),
            properties={
                "name": {
                    "type": "string",
                    "description": "Optional part of the tool name to stop. Empty stops all.",
                }
            },
            required=[],
            handler=stop_handler,
        )
    ]
    for spec in TOOL_SCHEMAS:
        fn = spec["function"]
        params = fn["parameters"]
        schemas.append(
            FunctionSchema(
                name=fn["name"],
                description=fn["description"],
                properties=params["properties"],
                required=list(params.get("required", [])),
                handler=handler_for(fn["name"]),
            )
        )
    return schemas
