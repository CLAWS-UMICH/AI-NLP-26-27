"""ASR pass (DATA_GENERATION.md §4): text -> TTS (macOS say, 10 voices) -> radio filter (300-3400 Hz,
pink noise + fan hum at 5-20 dB SNR, light clipping) -> half through Opus 24 kbps with 0-5% packet loss (WiFi path)
-> faster-whisper base.en with the shipped settings. Runs in .venv-runtime.
Usage: python datagen/asr.py rows.jsonl out.jsonl [--workers 3]
       python datagen/asr.py --silence 120 out.jsonl            (Whisper on noise-only clips)"""
import argparse, hashlib, json, os, subprocess, sys, tempfile
from multiprocessing import Pool
import numpy as np, soundfile as sf
from scipy.signal import butter, sosfilt

SR = 16000
VOICES = ["Samantha", "Daniel", "Karen", "Moira", "Rishi", "Tessa", "Fred", "Ralph", "Kathy", "Junior"]
HINT = "Suit telemetry: O2, CO2, scrubber, coolant, UIA, waypoint, primary, secondary."
_model = None

def rng_for(key):
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:12], 16))

def tts(text, voice, rate, path):
    subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", path, "--file-format=WAVE", "--data-format=LEI16@16000", text],
                   check=True, capture_output=True)
    a, sr = sf.read(path, dtype="float32")
    return a if a.ndim == 1 else a.mean(1)

def pink(n, rng):
    f = np.fft.rfftfreq(n); f[0] = f[1]
    x = np.fft.irfft((rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))) / np.sqrt(f), n)
    return x / (np.abs(x).max() + 1e-9)

def noise(n, rng):
    t = np.arange(n) / SR
    hum = sum(np.sin(2 * np.pi * f0 * t + rng.uniform(0, 6.28)) / k for k, f0 in enumerate((120, 240, 360, 480), 1))
    return 0.7 * pink(n, rng) + 0.3 * hum / np.abs(hum).max()

def radio(a, rng, snr_db):
    a = np.concatenate([np.zeros(int(SR * rng.uniform(.1, .4))), a, np.zeros(int(SR * rng.uniform(.2, .5)))])
    nz = noise(len(a), rng)
    p_sig = np.mean(a ** 2) + 1e-9
    nz *= np.sqrt(p_sig / (10 ** (snr_db / 10)) / (np.mean(nz ** 2) + 1e-12))
    y = sosfilt(butter(4, [300, 3400], btype="band", fs=SR, output="sos"), a + nz)
    y = y / (np.abs(y).max() + 1e-9) * rng.uniform(1.0, 1.3)   # light clipping
    return np.clip(y, -1, 1).astype(np.float32)

def opus_roundtrip(y, rng, tmp):
    src, enc, dec = (os.path.join(tmp, n) for n in ("o_in.wav", "o.opus", "o_out.wav"))
    sf.write(src, y, SR)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-c:a", "libopus", "-b:a", "24k", enc], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", enc, "-ar", str(SR), "-ac", "1", dec], check=True)
    z, _ = sf.read(dec, dtype="float32")
    loss = rng.uniform(0, 0.05); fr = int(0.02 * SR)
    for i in range(0, len(z) - fr, fr):
        if rng.random() < loss: z[i:i + fr] = 0
    return z

def model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel("base.en", device="cpu", compute_type="float32", cpu_threads=3)
    return _model

def transcribe(y):
    segs, info = model().transcribe(y, beam_size=5, language="en", vad_filter=False, without_timestamps=True,
                                    condition_on_previous_text=False, initial_prompt=HINT)
    segs = list(segs)
    text = " ".join(s.text.strip() for s in segs).strip()
    alp = float(np.mean([s.avg_logprob for s in segs])) if segs else -5.0
    nsp = float(np.mean([s.no_speech_prob for s in segs])) if segs else 1.0
    return text, alp, nsp

def work(r):
    rng = rng_for(r["id"])
    with tempfile.TemporaryDirectory() as tmp:
        voice = VOICES[rng.integers(len(VOICES))]; rate = int(rng.integers(160, 235)); snr = float(rng.uniform(5, 20))
        link = "wifi" if rng.random() < 0.5 else "cable"
        try:
            y = radio(tts(r["text"], voice, rate, os.path.join(tmp, "t.wav")), rng, snr)
            if link == "wifi": y = opus_roundtrip(y, rng, tmp)
            text, alp, nsp = transcribe(y)
        except Exception as e:
            return {"id": r["id"], "error": repr(e)}
    return {"id": r["id"], "asr": text, "voice": voice, "rate": rate, "snr": round(snr, 1), "link": link,
            "avg_logprob": round(alp, 3), "no_speech_prob": round(nsp, 3)}

def silence_work(i):
    rng = rng_for(f"silence-{i}")
    n = int(SR * rng.uniform(0.8, 3.0))
    level = 10 ** rng.uniform(-3, -0.7)
    y = (noise(n, rng) * level).astype(np.float32)
    if rng.random() < 0.3:  # radio click / squelch burst
        k = int(rng.integers(0, n - 800)); y[k:k + 800] += rng.standard_normal(800).astype(np.float32) * 0.3
    y = sosfilt(butter(4, [300, 3400], btype="band", fs=SR, output="sos"), y).astype(np.float32)
    text, alp, nsp = transcribe(y)
    return {"id": f"silence-{i}", "asr": text, "avg_logprob": round(alp, 3), "no_speech_prob": round(nsp, 3)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", nargs="?"); ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=3); ap.add_argument("--silence", type=int, default=0)
    a = ap.parse_args()
    with Pool(a.workers) as pool:
        if a.silence:
            res = pool.map(silence_work, range(a.silence), chunksize=4)
        else:
            rows = [json.loads(l) for l in open(a.inp)]
            done = {json.loads(l)["id"] for l in open(a.out)} if os.path.exists(a.out) else set()
            rows = [r for r in rows if r["id"] not in done]
            res = []
            with open(a.out, "a") as f:
                for i, o in enumerate(pool.imap_unordered(work, rows, chunksize=4)):
                    f.write(json.dumps(o) + "\n"); f.flush()
                    if i % 200 == 0: print(f"{i}/{len(rows)}", flush=True)
            sys.exit(0)
    with open(a.out, "w") as f:
        for o in res: f.write(json.dumps(o) + "\n")
    print(f"wrote {len(res)}")
