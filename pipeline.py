"""
pipeline.py - Audio transcription pipeline in a single file.

Stages:
  1. Accept & validate   -> validate_audio()
  2. Normalize           -> to_wav()        (ffmpeg: 16 kHz, mono, 16-bit PCM WAV)
  3. Transcribe          -> transcribe()    (faster-whisper + VAD)
  4. Segments & output   -> timestamps per segment, JSON
  5. Serve               -> FastAPI app

CLI:
  python pipeline.py audio.mp3                 # plain text transcript
  python pipeline.py audio.mp3 --timestamps    # segments with timestamps (+ transcript.json)
  python pipeline.py audio.mp3 --validate      # only validate the file and show metadata

API:
  uvicorn pipeline:app --reload
  POST /validate    -> file metadata
  POST /transcribe  -> transcript (+ segments with timestamps)
  GET  /health      -> health check
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache

from fastapi import FastAPI, HTTPException, UploadFile

# ---------------------------------------------------------------------------
# Configuration (overridable with environment variables)
# ---------------------------------------------------------------------------
SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4", ".aac"}
MAX_SIZE_MB = int(os.getenv("MAX_SIZE_MB", "200"))
MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")             # tiny / base / small / medium
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")                 # cpu / cuda
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")    # int8 for CPU, float16 for GPU


def _run(cmd: list) -> subprocess.CompletedProcess:
    """Run an ffmpeg/ffprobe command, with a clear error if ffmpeg is not installed."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg/ffprobe is not installed or not on PATH")


# ---------------------------------------------------------------------------
# Stage 1: Accept & validate
# ---------------------------------------------------------------------------
def validate_audio(path: str) -> dict:
    """Check the file is a real, readable audio file and return its metadata."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {ext or 'no extension'}")

    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb == 0:
        raise ValueError("File is empty")
    if size_mb > MAX_SIZE_MB:
        raise ValueError(f"File too large: {size_mb:.1f} MB (max {MAX_SIZE_MB} MB)")

    # ffprobe reads the actual file content, not just the extension
    probe = _run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                  "-show_entries", "stream=codec_name,sample_rate,channels:format=duration",
                  "-of", "json", path])
    if probe.returncode != 0:
        raise ValueError("Corrupt or unreadable audio file")

    info = json.loads(probe.stdout or "{}")
    if not info.get("streams"):
        raise ValueError("No audio stream found in file")

    stream = info["streams"][0]
    return {
        "file": os.path.basename(path),
        "format": ext[1:],
        "codec": stream.get("codec_name"),
        "sample_rate": int(stream.get("sample_rate", 0)),
        "channels": stream.get("channels"),
        "duration_sec": round(float(info.get("format", {}).get("duration", 0)), 2),
        "size_mb": round(size_mb, 2),
    }


# ---------------------------------------------------------------------------
# Stage 2: Normalize to one canonical format
# ---------------------------------------------------------------------------
def to_wav(path: str) -> str:
    """Convert any supported input to 16 kHz, mono, 16-bit PCM WAV (what Whisper expects)."""
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    result = _run(["ffmpeg", "-y", "-i", path, "-vn", "-ac", "1", "-ar", "16000",
                   "-c:a", "pcm_s16le", out])
    if result.returncode != 0:
        os.remove(out)
        raise ValueError("Could not convert audio file")
    return out


# ---------------------------------------------------------------------------
# Stage 3 & 4: Transcribe and build timestamped segments
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_model():
    """Load the Whisper model once and reuse it (loading is slow)."""
    from faster_whisper import WhisperModel
    return WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)


def fmt(t: float) -> str:
    """Convert seconds to HH:MM:SS.mmm."""
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02}:{int(m):02}:{s:06.3f}"


def transcribe(path: str, timestamps: bool = True) -> dict:
    """Validate, normalize and transcribe an audio file.

    Returns the full text, and the timestamped segments when timestamps=True.
    """
    metadata = validate_audio(path)
    wav = to_wav(path)
    try:
        segments, info = get_model().transcribe(wav, vad_filter=True, beam_size=5)
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

    result = {
        "file": metadata["file"],
        "language": info.language,
        "duration": round(info.duration, 2),
        "text": " ".join(s["text"] for s in segs),
    }
    if timestamps:
        result["segments"] = segs
    return result


# ---------------------------------------------------------------------------
# Stage 5: API
# ---------------------------------------------------------------------------
app = FastAPI(title="Transcription Pipeline")


def _save_upload(file: UploadFile) -> str:
    """Save an uploaded file to a temp file, streaming in chunks (not all into memory)."""
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=415, detail=f"Unsupported format: {suffix or 'no extension'}")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
    return tmp.name


@app.post("/validate")
def validate_endpoint(file: UploadFile):
    """Accept an audio file and return its metadata, without transcribing."""
    path = _save_upload(file)
    try:
        result = validate_audio(path)
        result["file"] = file.filename
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.remove(path)


@app.post("/transcribe")
def transcribe_endpoint(file: UploadFile, timestamps: bool = True):
    """Accept an audio file and return the transcript, with segment timestamps by default.

    Plain `def` (not async): transcription is CPU-heavy and blocking, so FastAPI
    runs it in a threadpool instead of blocking the event loop.
    """
    path = _save_upload(file)
    try:
        result = transcribe(path, timestamps)
        result["file"] = file.filename
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.remove(path)


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Transcribe an audio file.")
    parser.add_argument("audio", help="Path to an audio file (wav, mp3, m4a, ...)")
    parser.add_argument("--timestamps", action="store_true",
                        help="Print timestamps per segment and save transcript.json")
    parser.add_argument("--validate", action="store_true",
                        help="Only validate the file and show its metadata")
    args = parser.parse_args()

    try:
        if args.validate:
            print(json.dumps(validate_audio(args.audio), indent=2))
            return

        result = transcribe(args.audio, timestamps=args.timestamps)
        if args.timestamps:
            for s in result["segments"]:
                print(f"[{s['start_ts']} --> {s['end_ts']}] {s['text']}")
            with open("transcript.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print("\nSaved structured output to transcript.json")
        else:
            print(result["text"])
    except (ValueError, RuntimeError) as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
