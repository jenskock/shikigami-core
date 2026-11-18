"""Phase 3: look() — snapshot -> vision model -> description; tool wiring."""

import asyncio
import base64
from types import SimpleNamespace

from shikigami.config import Config
from shikigami.perception.look import (
    Looker,
    build_messages,
    has_names,
    persona_from,
    trim_to_sentence,
)
from shikigami.voice.tools import build_tool_schemas


class Cam:
    def __init__(self, jpeg=b"\xff\xd8fakejpeg"):
        self.jpeg = jpeg
        self.max_side = None

    async def snapshot_jpeg(self, max_side=768, quality=80):
        self.max_side = max_side
        return self.jpeg


class Client:
    """Shaped like AsyncOpenAI: client.chat.completions.create(...)."""

    def __init__(self, text="Du hältst eine rote Tasse!", exc=None, replies=None):
        self.kwargs = None
        self.calls = 0
        self._exc = exc
        self._replies = list(replies or [(text, "stop")])
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        self.kwargs = kwargs
        self.calls += 1
        if self._exc:
            raise self._exc
        text, finish = self._replies[min(self.calls, len(self._replies)) - 1]
        msg = SimpleNamespace(content=text)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish)])


def looker(cam=None, client=None, view="1 face, who: Jens"):
    cfg = Config()
    return Looker(
        cam or Cam(),
        cfg.llm,
        cfg.perception,
        cfg.camera,
        view=lambda: view,
        client=client or Client(),
    )


def test_has_names():
    assert has_names("1 face, who: Jens")
    assert has_names("2 faces, who: Jens + 1 unknown")
    assert not has_names("1 face, who: unknown")
    assert not has_names("2 faces, who: unknown")
    assert not has_names("nothing")


def test_messages_carry_image_question_and_names():
    msgs = build_messages(b"jpg", "what is he holding?", "1 face, who: Jens")
    system, user = msgs[0]["content"], msgs[1]["content"]
    assert "who: Jens" in system and "Never name anyone else" in system
    assert "what is he holding?" in user[0]["text"]
    persona = persona_from(SimpleNamespace(name="Kogane", language="German", voice="cute"))
    assert "Always answer in German" in build_messages(b"j", "", "nothing", persona)[0]["content"]
    url = user[1]["image_url"]["url"]
    assert url == "data:image/jpeg;base64," + base64.b64encode(b"jpg").decode()
    anon = build_messages(b"jpg", "", "1 face, who: unknown")[0]["content"]
    assert "Never name anyone; you do not know" in anon


def test_look_returns_description_and_uses_llm_model():
    client, cam = Client(), Cam()
    res = asyncio.run(looker(cam, client).look("was hält Jens?"))
    assert res == {"ok": True, "saw": "Du hältst eine rote Tasse!"}
    assert client.kwargs["model"] == Config().llm.model
    assert cam.max_side == Config().camera.snapshot_max_side


def test_look_failures_are_reported_not_invented():
    assert asyncio.run(looker(cam=Cam(jpeg=None)).look())["ok"] is False
    res = asyncio.run(looker(client=Client(exc=TimeoutError())).look())
    assert res["ok"] is False and "TimeoutError" in res["error"]
    assert asyncio.run(looker(client=Client(text="  ")).look())["ok"] is False


def test_look_tool_always_reruns_llm_and_only_offered_tools_appear():
    from pipecat.services.llm_service import FunctionCallParams

    class Hooks:
        vision_tools = ("look",)

        def note_tool(self, a): ...
        async def execute(self, a): ...
        def tool_enter(self):
            return 0

        def tool_exit(self, turn=None): ...

        async def vision_tool(self, name, args):
            return {"ok": True, "saw": "a dog", "run_llm": True}

    schemas = build_tool_schemas(Hooks())
    names = [s.name for s in schemas]
    assert "look" in names and "remember_face" not in names
    results = []

    async def cb(result, *, properties=None):
        results.append((result, properties))

    look = next(s for s in schemas if s.name == "look")
    asyncio.run(
        look.handler(
            FunctionCallParams(
                function_name="look",
                tool_call_id="l",
                arguments={},
                llm=None,  # type: ignore[arg-type]
                pipeline_worker=None,  # type: ignore[arg-type]
                context=None,  # type: ignore[arg-type]
                result_callback=cb,
            )
        )
    )
    result, props = results[0]
    assert result == {"ok": True, "saw": "a dog"} and props.run_llm is True


def test_cut_off_answer_is_trimmed_and_empty_answer_retried():
    assert trim_to_sentence("Da liegt ein Hund. Er ist zwei H") == "Da liegt ein Hund."
    assert trim_to_sentence("zwei H") == ""
    cut = Client(replies=[("Da liegt ein Hund! Und zwei H", "length")])
    assert asyncio.run(looker(client=cut).look())["saw"] == "Da liegt ein Hund!"
    retry = Client(replies=[("", "length"), ("Ein Hund!", "stop")])
    assert asyncio.run(looker(client=retry).look())["saw"] == "Ein Hund!"
    assert retry.calls == 2
    assert Config().perception.look_max_tokens >= 1024  # reasoning eats the budget


def test_say_is_spoken_by_the_handler_without_a_second_completion():
    from pipecat.frames.frames import TTSSpeakFrame
    from pipecat.services.llm_service import FunctionCallParams

    class Hooks:
        def note_tool(self, a): ...
        async def execute(self, a): ...
        def tool_enter(self):
            return 0

        def tool_exit(self, turn=None): ...

        async def vision_tool(self, name, args):
            return {"ok": True, "saw": "Ein Hund!", "say": "Ein Hund!", "run_llm": False}

    pushed, results = [], []

    class LLM:
        async def push_frame(self, frame):
            pushed.append(frame)

    async def cb(result, *, properties=None):
        results.append((result, properties))

    look = next(s for s in build_tool_schemas(Hooks()) if s.name == "look")
    asyncio.run(
        look.handler(
            FunctionCallParams(
                function_name="look",
                tool_call_id="l",
                arguments={},
                llm=LLM(),  # type: ignore[arg-type]
                pipeline_worker=None,  # type: ignore[arg-type]
                context=None,  # type: ignore[arg-type]
                result_callback=cb,
            )
        )
    )
    assert len(pushed) == 1 and isinstance(pushed[0], TTSSpeakFrame)
    assert pushed[0].text == "Ein Hund!"
    result, props = results[0]
    assert result == {"ok": True, "saw": "Ein Hund!"} and props.run_llm is False
