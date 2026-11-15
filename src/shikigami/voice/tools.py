"""Map the existing robot tools onto Pipecat function handlers."""

from __future__ import annotations

from typing import Protocol

from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.frames.frames import FunctionCallResultProperties, TTSSpeakFrame
from pipecat.services.llm_service import FunctionCallParams

from ..brain.actions import TOOL_SCHEMAS, Action, SayAction, parse_tool_call


class ToolHooks(Protocol):
    tts_enabled: bool

    def note_tool(self, action: Action) -> None: ...

    async def execute(self, action: Action) -> None: ...

    def expect_speech(self) -> None: ...

    def tool_enter(self) -> None: ...

    def tool_exit(self) -> None: ...


def build_tool_schemas(hooks: ToolHooks) -> list[FunctionSchema]:
    """One schema per ``TOOL_SCHEMAS`` entry, with a handler that runs the body.

    Property enums are copied from the existing schemas. Direct functions do
    not emit enums, and the motion / emote names are a closed set.
    """

    async def handle(params: FunctionCallParams) -> None:
        hooks.tool_enter()
        try:
            action = parse_tool_call(params.function_name, dict(params.arguments))
            hooks.note_tool(action)
            if isinstance(action, SayAction):
                text = action.text.strip()
                if text and hooks.tts_enabled:
                    hooks.expect_speech()
                    await params.llm.push_frame(TTSSpeakFrame(text))
            else:
                await hooks.execute(action)
            await params.result_callback(
                {"ok": True},
                properties=FunctionCallResultProperties(run_llm=False),
            )
        except Exception as exc:
            await params.result_callback(
                {"ok": False, "error": str(exc)},
                properties=FunctionCallResultProperties(run_llm=False),
            )
        finally:
            hooks.tool_exit()

    schemas: list[FunctionSchema] = []
    for spec in TOOL_SCHEMAS:
        fn = spec["function"]
        params = fn["parameters"]
        schemas.append(
            FunctionSchema(
                name=fn["name"],
                description=fn["description"],
                properties=params["properties"],
                required=list(params.get("required", [])),
                handler=handle,
            )
        )
    return schemas
