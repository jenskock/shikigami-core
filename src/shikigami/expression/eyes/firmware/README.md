# ESP32 eyes firmware (stub)

Parses the host serial protocol (`EMO`, `GAZE`, `BLINK`, …) and ACKs with `OK`.

Wire a GC9A01 (or similar 240×240 round SPI panel), then either:

1. Port `EyesEngine` springs/life onto the ESP32 and render locally, or
2. Stream RGB565 frames from the Pi (heavier UART/SPI use).

Host side: `shikigami.expression.eyes.protocol.SerialEyesClient`.
