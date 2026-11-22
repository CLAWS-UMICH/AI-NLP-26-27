"""
Optional: microphone -> Whisper -> intent model -> spoken reply.

1. Push-to-talk: Enter starts recording, Enter again stops (capped at MAX_SECONDS)
2. faster-whisper on CPU, with a domain-word prompt so it writes "CO2", "UIA", "scrubber" like our data
   and a VAD filter so silence gives no text instead of a made-up sentence
3. predict() picks the intent (+ slot for the five name intents)
4. Speaks a short reply (no telemetry here, so it says what it would show, not the value)

Record mode (--record) runs a recording session instead: it shows each prompt line, records it, saves the
wav to data/audio/<session>/<speaker>/, transcribes it with exactly the settings above and appends
{text: transcript, intent, slot, prompt, audio, speaker, source: "rec:<session>"} to dev_audio.jsonl
(session a) or test.jsonl (session b). The slot is stored as Whisper spelled it (found with rapidfuzz).
Prompts are a JSONL of {"text", "intent", "slot"}; ones this speaker already recorded are skipped, so an
interrupted session picks up where it stopped.

Usage (pip install faster-whisper sounddevice, plus pyttsx3 on Windows/Linux, plus rapidfuzz for --record):
    python code/voice.py                  talk into the mic, Ctrl+C quits
    python code/voice.py clip.wav ...     transcribe 16-bit .wav files instead
    python code/voice.py --record data/prompts_a.jsonl --speaker raiana --session a
"""

import argparse
import json
import random
import re
import subprocess
import sys
import threading
import wave

import numpy as np
from faster_whisper import WhisperModel

from config import AUDIO_DIR, DEV_AUDIO_FILE, ROOT, TEST_FILE
from data import read_jsonl

RATE = 16000 # Whisper expects 16 kHz mono
MAX_SECONDS = 10
WHISPER_SIZE = "base.en" # "small.en" is more accurate but slower apparently
HINT = "EVA, UIA, HUD, O2, CO2, scrubber A, scrubber B, coolant, waypoint, RPM, airlock, primary, secondary"

# What to say back for each intent; {slot} is filled in for the five name intents
REPLIES = {
    "get_oxy_pri_storage": "Primary oxygen storage.",
    "get_oxy_sec_storage": "Secondary oxygen storage.",
    "get_oxy_pri_pressure": "Primary oxygen tank pressure.",
    "get_oxy_sec_pressure": "Secondary oxygen tank pressure.",
    "get_suit_pressure_oxy": "Suit oxygen pressure.",
    "get_suit_pressure_co2": "Suit C O 2 pressure.",
    "get_suit_pressure_other": "Suit trace gas pressure.",
    "get_suit_pressure_total": "Total suit pressure.",
    "get_helmet_pressure_co2": "Helmet C O 2 pressure.",
    "get_fan_pri_rpm": "Primary fan speed.",
    "get_fan_sec_rpm": "Secondary fan speed.",
    "get_scrubber_a_co2_storage": "Scrubber A storage.",
    "get_scrubber_b_co2_storage": "Scrubber B storage.",
    "get_temperature": "Suit temperature.",
    "get_coolant_storage": "Coolant storage.",
    "get_coolant_gas_pressure": "Coolant gas pressure.",
    "get_coolant_liquid_pressure": "Coolant liquid pressure.",
    "get_heart_rate": "Heart rate.",
    "get_oxy_consumption": "Oxygen consumption rate.",
    "get_co2_production": "C O 2 production rate.",
    "get_eva_elapsed_time": "Elapsed EVA time.",
    "get_primary_battery_level": "Primary battery level.",
    "get_secondary_battery_level": "Secondary battery level.",
    "get_warnings": "Showing active warnings.",
    "open_menu_vitals": "Opening vitals.",
    "open_menu_navigation": "Opening navigation.",
    "open_menu_tasks": "Opening tasks.",
    "open_menu_uia": "Opening U I A panel.",
    "close_menu": "Closing panel.",
    "Get_coordinates": "Coordinates for {slot}.",
    "Set_navigation_target": "Navigating to {slot}.",
    "reroute_navigation": "Calculating a new route.",
    "Add_waypoint": "Adding waypoint {slot}.",
    "Delete_waypoint": "Deleting waypoint {slot}.",
    "Complete_task": "Marking {slot} complete.",
    "unhandled": "Sorry, say that again.",
}


# Reply sentence for a predict()
def reply_for(result):
    template = REPLIES.get(result["selected_intent"], "Sorry, say that again.")
    if "{slot}" in template and not result.get("slot"):
        return "Which one? I didn't get the name"
    return template.format(slot=result.get("slot"))


# Say text out loud: built-in `say` on macOS, pyttsx3 elsewhere
def speak(text):
    if sys.platform == "darwin":
        subprocess.run(["say", text]) # pyttsx3 can hang on repeat calls on macOS
    else:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()


# Record from the default mic until Enter is pressed (or MAX_SECONDS hit), return float32 samples
def record():
    import sounddevice as sd # imported here so file mode works without importing mic library

    chunks = []
    stop = threading.Event()

    def on_audio(data, frames, time, status):
        chunks.append(data.copy())
        if sum(len(c) for c in chunks) >= RATE * MAX_SECONDS:
            stop.set()

    with sd.InputStream(samplerate=RATE, channels=1, dtype="float32", callback=on_audio):
        threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()
        stop.wait()
    return np.concatenate(chunks).flatten() if chunks else np.zeros(0, dtype=np.float32)


# Read a .wav into float32 mono samples at RATE (stdlib only, avoids faster-whisper's own decoder)
def load_wav(path):
    with wave.open(str(path), "rb") as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        sys.exit(f"{path}: only 16-bit WAV is supported")
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768
    audio = audio.reshape(-1, channels).mean(axis=1) # stereo -> mono
    if rate != RATE: # simple linear resample to 16 kHz
        audio = np.interp(np.arange(0, len(audio), rate / RATE), np.arange(len(audio)), audio)
    return audio.astype(np.float32)


# Write float32 samples at RATE as a 16-bit mono .wav
def save_wav(path, audio):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


# Speech -> text. audio is a float32 array at RATE
def transcribe(stt, audio):
    segments, _ = stt.transcribe(audio, language="en", initial_prompt=HINT, vad_filter=True)
    return " ".join(s.text for s in segments).strip()


# Transcribe, classify, print and speak one utterance
def handle(stt, audio):
    from predict import predict

    text = transcribe(stt, audio)
    if not text:
        print("(heard nothing)")
        return
    result = predict(text)
    reply = reply_for(result)
    print(result)
    print(f"-> {reply}")
    speak(reply)


# The slot as Whisper spelled it: the transcript span that best matches the prompt's slot, or None
def find_slot(transcript, slot):
    from rapidfuzz import fuzz

    match = fuzz.partial_ratio_alignment(slot.lower(), transcript.lower())
    if match is None or match.score < 70:
        return None
    start, end = match.dest_start, match.dest_end
    while start > 0 and transcript[start - 1].isalnum():  # widen to whole words
        start -= 1
    while end < len(transcript) and transcript[end].isalnum():
        end += 1
    return re.sub(r"^\W+|\W+$", "", transcript[start:end]) or None


# Recording session: prompt, record, transcribe, keep / redo / skip, append one row per kept clip
def record_session(stt, prompts_path, speaker, session):
    out_file = DEV_AUDIO_FILE if session == "a" else TEST_FILE
    prompts = read_jsonl(prompts_path)
    done = {r.get("prompt") for r in read_jsonl(out_file) if r.get("speaker") == speaker} if out_file.exists() else set()
    todo = [p for p in prompts if p["text"] not in done]
    random.Random(speaker).shuffle(todo)  # every speaker gets their own order
    folder = AUDIO_DIR / session / speaker
    print(f"{len(prompts) - len(todo)} of {len(prompts)} prompts already recorded, {len(todo)} to go -> {out_file}")

    try:
        for n, prompt in enumerate(todo, 1):
            while True:
                print(f"\n[{n}/{len(todo)}]  say:  {prompt['text']}")
                input("Enter to start, Enter again to stop...")
                print("listening...")
                audio = record()
                text = transcribe(stt, audio)
                print(f"heard: {text or '(nothing)'}")
                choice = input("Enter keeps it, r redoes, s skips: ").strip().lower()
                if choice != "r":
                    break
            if choice == "s" or not text:
                print("skipped")
                continue
            path = folder / f"{len(list(folder.glob('*.wav'))) + 1:03d}.wav"
            save_wav(path, audio)
            row = {"text": text, "intent": prompt["intent"], "prompt": prompt["text"], "audio": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
                   "speaker": speaker, "source": f"rec:{session}"}
            if prompt.get("slot"):
                heard = find_slot(text, prompt["slot"])
                row["slot"] = heard or prompt["slot"]
                if heard is None:
                    row["slot_not_heard"] = True
                    print(f"(slot {prompt['slot']!r} not found in the transcript, flagged for review)")
            with open(out_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except (KeyboardInterrupt, EOFError):
        print("\nstopped, run the same command again to pick up where you left off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wavs", nargs="*", help="transcribe these 16-bit .wav files instead of the mic")
    parser.add_argument("--record", metavar="PROMPTS", help="recording session: JSONL of prompt lines")
    parser.add_argument("--speaker", help="who is recording (--record)")
    parser.add_argument("--session", choices=["a", "b"], help="a -> dev_audio.jsonl, b -> test.jsonl (--record)")
    args = parser.parse_args()
    if args.record and not (args.speaker and args.session):
        parser.error("--record needs --speaker and --session")

    print(f"loading Whisper {WHISPER_SIZE}...")
    stt = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")

    if args.record:
        record_session(stt, args.record, args.speaker.lower(), args.session)
        return

    from predict import predict

    predict("warm up") # load the intent model now, not on the first command

    if args.wavs: # file mode
        for path in args.wavs:
            print(f"\n{path}")
            handle(stt, load_wav(path))
        return

    try:
        while True:
            input("\nPress Enter and speak, Enter again to stop (Ctrl+C quits)...")
            print("listening...")
            handle(stt, record())
    except (KeyboardInterrupt, EOFError):
        print("\nbye")


if __name__ == "__main__":
    main()
