"""Energy VAD state machine."""

from shikigami.perception.vad import EnergyVAD


def test_vad_speech_then_end():
    vad = EnergyVAD(threshold=300.0, silence_s=0.3, min_speech_s=0.2)
    assert vad.feed(0.0) == "silence"
    assert vad.feed(1000.0) == "speech_start"
    assert vad.feed(900.0, dt_s=0.2) == "speech"
    assert vad.feed(10.0, dt_s=0.1) == "speech"  # short silence, still speaking
    assert vad.feed(10.0, dt_s=0.2) == "speech_end"
    assert vad.feed(10.0) == "silence"
    assert not vad.speaking


def test_vad_short_blip_is_noise():
    vad = EnergyVAD(threshold=300.0, silence_s=0.2, min_speech_s=0.3)
    assert vad.feed(1000.0) == "speech_start"
    assert vad.feed(0.0, dt_s=0.3) == "silence"  # burst too short -> noise, not speech_end
    assert not vad.speaking
