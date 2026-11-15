"""Interactive Pygame eyes demo: `uv run eyes-demo`.

Keys 1–9 / 0 / - / = emotions, mouse = gaze, space = blink, W = wink, R = react.
"""

from __future__ import annotations

import sys

from .controller import EyesEngine
from .emotion import EMOTION_NAMES
from .render.pygame_renderer import PygameRenderer
from .state import EyesConfig


def main(argv: list[str] | None = None) -> int:
    try:
        import pygame
    except ImportError:
        print("pygame is required for eyes-demo: uv add pygame", file=sys.stderr)
        return 1

    _ = argv
    pygame.init()
    size = 480
    screen = pygame.display.set_mode((size, size))
    pygame.display.set_caption("shikigami eyes — 1-9 emotions, mouse gaze, space blink")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("menlo", 16)

    engine = EyesEngine(EyesConfig(personality="curious", seed=None))
    renderer = PygameRenderer(size, size)
    running = True
    info = "neutral"

    while running:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    engine.blink()
                elif event.key == pygame.K_w:
                    engine.wink("left")
                elif event.key == pygame.K_r:
                    engine.react("startle")
                elif event.key == pygame.K_h:
                    engine.react("heart")
                elif event.key == pygame.K_z:
                    engine.react("zzz")
                elif event.key == pygame.K_d:
                    engine.react("dizzy")
                elif event.key == pygame.K_l:
                    engine.react("laugh")
                else:
                    emo = _emotion_from_key(event.key)
                    if emo is not None:
                        engine.set_emotion(emo, intensity=1.0)
                        info = emo
            elif event.type == pygame.MOUSEMOTION:
                mx, my = event.pos
                gx = (mx / size) * 2 - 1
                gy = (my / size) * 2 - 1
                engine.look_at(gx, gy)

        frame = engine.update(dt)
        surf = renderer.draw(frame)
        screen.blit(surf, (0, 0))
        label = font.render(
            f"{info}  fps={clock.get_fps():.0f}  [1-9,0,-,=] emo  mouse gaze  space blink",
            True,
            (200, 200, 200),
        )
        screen.blit(label, (8, 8))
        pygame.display.flip()

    pygame.quit()
    return 0


def _emotion_from_key(key: int) -> str | None:
    # pygame key constants: K_1..K_9, K_0, K_MINUS, K_EQUALS
    import pygame

    mapping = [
        pygame.K_1,
        pygame.K_2,
        pygame.K_3,
        pygame.K_4,
        pygame.K_5,
        pygame.K_6,
        pygame.K_7,
        pygame.K_8,
        pygame.K_9,
        pygame.K_0,
        pygame.K_MINUS,
        pygame.K_EQUALS,
    ]
    if key in mapping:
        idx = mapping.index(key)
        if idx < len(EMOTION_NAMES):
            return EMOTION_NAMES[idx]
    return None


if __name__ == "__main__":
    raise SystemExit(main())
