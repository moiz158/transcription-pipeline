# transcribe_segments.py
# Q2: Return the transcription with timestamps per segment.
# Usage: python transcribe_segments.py audio.mp3

import json
import os
import subprocess
import sys
import tempfile

from faster_whisper import WhisperModel

SUPPORTED = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4", ".aac"}

model = WhisperModel("base", device="cpu", compute_type="int8")


def to_wav(path: str) -> str:
    """Normalize any supported input to 16 kHz, mono, 16-bit PCM WAV."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported format: {ext}")

    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "pcm_s16le", out],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError:
        os.remove(out)
        raise ValueError("Corrupt or unreadable audio file")
    return out


def fmt(t: float) -> str:
    """Convert seconds to HH:MM:SS.mmm."""
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02}:{int(m):02}:{s:06.3f}"


def transcribe_with_segments(path: str) -> dict:
    """Return language, duration, full text and timestamped segments."""
    wav = to_wav(path)
    try:
        segments, info = model.transcribe(wav, vad_filter=True, beam_size=5)
        # segments is a generator: decoding happens here, one segment at a time
        segs = [
            {
                "id": i,
                "start": round(s.start, 2),
                "end": round(s.end, 2),
                "start_ts": fmt(s.start),
                "end_ts": fmt(s.end),
                "text": s.text.strip(),
            }
            for i, s in enumerate(segments)
        ]
    finally:
        os.remove(wav)

    return {
        "language": info.language,
        "duration": round(info.duration, 2),
        "text": " ".join(s["text"] for s in segs),
        "segments": segs,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python transcribe_segments.py <audio_file>")

    result = transcribe_with_segments(sys.argv[1])

    for s in result["segments"]:
        print(f"[{s['start_ts']} --> {s['end_ts']}] {s['text']}")

    with open("transcript.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("\nSaved structured output to transcript.json")
