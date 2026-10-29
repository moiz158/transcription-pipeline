# transcribe.py
# Q1: Transcribe spoken language into text.
# Usage: python transcribe.py audio.mp3

import os
import subprocess
import sys
import tempfile

from faster_whisper import WhisperModel

SUPPORTED = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4", ".aac"}

# Model size is configurable: tiny / base / small / medium (speed vs accuracy)
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


def transcribe(path: str) -> str:
    """Return the full transcript of an audio file as plain text."""
    wav = to_wav(path)
    try:
        segments, info = model.transcribe(wav, vad_filter=True, beam_size=5)
        text = " ".join(s.text.strip() for s in segments)
    finally:
        os.remove(wav)
    return text


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python transcribe.py <audio_file>")
    print(transcribe(sys.argv[1]))
