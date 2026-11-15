/*
 * shikigami eyes — ESP32 firmware STUB
 *
 * Receives compact ASCII commands over Serial (115200) and would drive a
 * GC9A01 240x240 round display. Animation math can either:
 *   A) run on-device (port the Python EyesEngine), or
 *   B) receive pre-rendered RGB565 frames (not implemented here).
 *
 * Protocol: see shikigami.expression.eyes.protocol
 *
 * This stub only parses commands and ACKs them — replace handle*() bodies
 * with real GC9A01 draws / local springs.
 */

#include <Arduino.h>

String lineBuf;

void replyOk() { Serial.println(F("OK")); }
void replyErr(const char* msg) { Serial.print(F("ERR ")); Serial.println(msg); }

void handleEmo(const String& args) {
  // args: <name> [intensity] [duration]
  replyOk();
}
void handleGaze(const String& args) { replyOk(); }
void handleBlink() { replyOk(); }
void handleWink(const String& args) { replyOk(); }
void handleReact(const String& args) { replyOk(); }
void handleSpeak(const String& args) { replyOk(); }

void dispatch(String line) {
  line.trim();
  if (line.length() == 0) return;
  int sp = line.indexOf(' ');
  String cmd = (sp < 0) ? line : line.substring(0, sp);
  String args = (sp < 0) ? "" : line.substring(sp + 1);
  cmd.toUpperCase();

  if (cmd == "PING") { Serial.println(F("PONG")); return; }
  if (cmd == "EMO") { handleEmo(args); return; }
  if (cmd == "GAZE" || cmd == "TRACK") { handleGaze(args); return; }
  if (cmd == "BLINK") { handleBlink(); return; }
  if (cmd == "WINK") { handleWink(args); return; }
  if (cmd == "REACT") { handleReact(args); return; }
  if (cmd == "SPEAK") { handleSpeak(args); return; }
  if (cmd == "SYNC") {
    Serial.println(F("STATE emotion=neutral gaze=0,0"));
    return;
  }
  replyErr("unknown");
}

void setup() {
  Serial.begin(115200);
  // TODO: init GC9A01 SPI, clear to black, draw neutral eyes
  Serial.println(F("OK eyes-stub"));
}

void loop() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (lineBuf.length()) {
        dispatch(lineBuf);
        lineBuf = "";
      }
    } else if (lineBuf.length() < 200) {
      lineBuf += c;
    }
  }
  // TODO: EyesEngine::update(dt); blit RGB565 dirty rects
}
