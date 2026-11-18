"""Map body-control tools onto Pipecat function handlers.

Speech is the LLM text stream. These handlers only move the body. They set
``run_llm=False`` because actuators do not return conversational data — a
second completion after every emote would talk twice. Spoken words already
live in the same LLM turn as the tool call.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from pipecat.adapters.llm.schemas.direct_function import tool_options
from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.frames.frames import FunctionCallResultProperties, TTSSpeakFrame
from pipecat.services.llm_service import FunctionCallParams

from ..brain.actions import TOOL_SCHEMAS, Action, parse_tool_call

STOP_TOOL_NAME = "stop_running_tools"

# Vision tools return data, unlike body tools. look always runs the LLM again so it
# can retell what it saw; remember/forget_face only on failure, so the robot can say
# honestly what went wrong ("Ich seh grad kein Gesicht").
VisionTool = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]
VISION_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "look",
        "description": (
            "Take a picture with your camera and get a short description of what is "
            "really there. Call it whenever you are asked what you see, how someone "
            "looks, what they hold or wear, or what is in the room. Say something tiny "
            'first ("Ich schau mal!"), then retell only what the result says. '
            "Takes one or two seconds."
        ),
        "properties": {
            "question": {
                "type": "string",
                "description": "What to look for, e.g. 'what is Jens holding?'. Optional.",
            }
        },
        "required": [],
    },
    {
        "name": "remember_face",
        "description": (
            "Learn the face currently in view under a name, so you recognize this person "
            'later. Call it when someone tells you their name ("Ich bin Anna", "Merk dir '
            'mein Gesicht") while state.in_view shows exactly one face that is unknown. '
            "Takes about two seconds; the person should look at you."
        ),
        "properties": {"name": {"type": "string", "description": "The person's name, e.g. Anna."}},
        "required": ["name"],
    },
    {
        "name": "forget_face",
        "description": "Forget a learned face when someone asks you to forget it.",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    },
]

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
                # Actuators are not data tools: no follow-up completion — unless the
                # response had no words at all, then once, so the robot speaks.
                await params.result_callback(
                    {"ok": True},
                    properties=FunctionCallResultProperties(run_llm=_needs_words(hooks, params)),
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
    vision_tool: VisionTool | None = getattr(hooks, "vision_tool", None)
    offered: set[str] = set(getattr(hooks, "vision_tools", ()) or ())
    if vision_tool is not None:
        for spec in VISION_TOOL_SCHEMAS:
            if offered and spec["name"] not in offered:
                continue
            schemas.append(
                FunctionSchema(
                    name=spec["name"],
                    description=spec["description"],
                    properties=spec["properties"],
                    required=spec["required"],
                    handler=_vision_handler(vision_tool, hooks),
                )
            )
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


def _needs_words(hooks, params: FunctionCallParams) -> bool:
    """True once if this response spoke no words (tool calls only)."""
    if str(getattr(params.llm, "response_text", "") or "").strip():
        return False
    nudge = getattr(hooks, "nudge_for_words", None)
    return bool(nudge is not None and nudge())


def _vision_handler(vision_tool: VisionTool, hooks):
    @tool_options(cancel_on_interruption=False)
    async def handler(params: FunctionCallParams) -> None:
        # Count as a running tool like body tools, so the turn stays open while a
        # look (1-3 s) or an enrollment runs.
        turn = hooks.tool_enter()
        try:
            await _vision_call(vision_tool, hooks, params)
        finally:
            hooks.tool_exit(turn)

    return handler


async def _vision_call(vision_tool: VisionTool, hooks, params: FunctionCallParams) -> None:
    try:
        result = await vision_tool(params.function_name, dict(params.arguments))
    except Exception as exc:
        _log.warning("tool %s failed: %s", params.function_name, exc)
        result = {"ok": False, "error": str(exc)}
    # "say": the tool already has the spoken answer (look) — speak it here instead
    # of a second completion, which often answered with only an emote. Otherwise
    # the reply was already spoken on success; on failure the LLM says what failed.
    say = str(result.pop("say", "") or "").strip()
    if say:
        await params.llm.push_frame(speak_frame(say))
    run_llm = bool(result.pop("run_llm", not result.get("ok", False)))
    if not run_llm and not say:
        run_llm = _needs_words(hooks, params)
    elif run_llm and (expect := getattr(hooks, "expect_followup", None)) is not None:
        expect()
    await params.result_callback(result, properties=FunctionCallResultProperties(run_llm=run_llm))
