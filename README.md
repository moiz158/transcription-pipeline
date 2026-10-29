# Transcription Pipeline

A simple audio transcription pipeline: accept an audio file, convert speech to text, and return the transcript with timestamps per segment, as scripts or as a REST API.

Built with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (an open-source Whisper implementation), ffmpeg and FastAPI.

## Pipeline

```
Audio file (WAV, MP3, M4A, ...)
   │
   ▼
1. Accept & validate      validate_audio()      extension, size, ffprobe check, metadata
   │
   ▼
2. Normalize              to_wav()              ffmpeg → 16 kHz, mono, 16-bit PCM WAV
   │
   ▼
3. Transcribe             transcribe()          faster-whisper + VAD: speech → text
   │
   ▼
4. Segments & output      transcribe()          timestamps per segment, JSON
   │
   ▼
5. Serve                  FastAPI app           POST /validate, POST /transcribe
```

All stages live in one file, `pipeline.py`, usable both as a command-line script and as a REST API.

## Files

| File | Purpose |
|---|---|
| `pipeline.py` | The whole pipeline: validation, normalization, transcription with timestamps, CLI and FastAPI service. |
| `requirements.txt` | Python dependencies. |

### Main functions in `pipeline.py`

| Function | What it does |
|---|---|
| `validate_audio(path)` | Accepts an audio file: checks extension, rejects empty/oversized files, uses ffprobe to confirm a real audio stream, returns metadata. |
| `to_wav(path)` | Converts any supported input to 16 kHz mono 16-bit PCM WAV with ffmpeg. |
| `get_model()` | Loads the Whisper model once and reuses it (cached). |
| `transcribe(path, timestamps)` | Validates, normalizes and transcribes; returns full text and, optionally, timestamped segments. |
| `fmt(t)` | Formats seconds as `HH:MM:SS.mmm`. |
| `/validate`, `/transcribe`, `/health` | API endpoints. |
| `main()` | Command-line interface. |

## Setup

Requires Python 3.9+ and **ffmpeg** installed on the system:

```bash
# Ubuntu/Debian
sudo apt install ffmpeg
# macOS
brew install ffmpeg
# Windows
choco install ffmpeg

pip install -r requirements.txt
```

The first run downloads the Whisper `base` model (~150 MB).

## Usage

```bash
# 1. Validate an audio file and show its metadata
python pipeline.py sample.mp3 --validate

# 2. Transcribe to plain text
python pipeline.py sample.mp3

# 3. Transcribe with timestamps per segment (also writes transcript.json)
python pipeline.py sample.mp3 --timestamps
```

Settings can be changed with environment variables, e.g. `WHISPER_MODEL=small python pipeline.py sample.mp3` (also `WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE`, `MAX_SIZE_MB`).

Example output:

```
[00:00:00.000 --> 00:00:03.200] Hello, thanks for joining.
[00:00:03.400 --> 00:00:06.100] Let's review the project.
```

### API

```bash
uvicorn pipeline:app --reload

# Validate only (returns metadata)
curl -F "file=@sample.mp3" http://localhost:8000/validate

# Transcribe with timestamps
curl -F "file=@sample.mp3" "http://localhost:8000/transcribe?timestamps=true"
```

Error responses: `415` unsupported format, `400` empty, too large or corrupt file.

```json
{
  "file": "sample.mp3",
  "language": "en",
  "duration": 12.4,
  "text": "Hello, thanks for joining. Let's review the project.",
  "segments": [
    {"id": 0, "start": 0.0, "end": 3.2, "start_ts": "00:00:00.000",
     "end_ts": "00:00:03.200", "text": "Hello, thanks for joining."}
  ]
}
```

Interactive API docs are available at `http://localhost:8000/docs`.

## Design decisions

- **faster-whisper over a cloud API:** open source, runs locally on CPU (int8), no per-minute cost, and audio never leaves the machine. Model size (`tiny` → `medium`) is configurable to trade speed for accuracy.
- **One canonical audio format:** every input is converted with ffmpeg to 16 kHz mono 16-bit PCM WAV, which is what Whisper expects. The model always sees the same input, and there is one code path to test.
- **Validation before processing:** ffprobe checks the actual file content, so renamed or corrupt files fail early with a clear error.
- **VAD filtering:** removes silence, reducing processing time and preventing Whisper from hallucinating text on silent parts.
- **Structured JSON output:** segments include seconds (for code) and HH:MM:SS.mmm (for humans), ready for search, subtitles or summarization.
- **Blocking endpoints use `def`, not `async def`:** the model call is CPU-bound, so FastAPI runs it in a threadpool instead of blocking the event loop.
- **One file, one pipeline:** each stage is a small function, and both the CLI and the API call the same functions, so there is no duplicated logic.
- **Model loaded once:** `get_model()` is cached, so the model is loaded on first use and reused across requests; validation-only calls never load it.
- **Uploads streamed to disk:** uploaded files are copied to a temp file in chunks rather than read fully into memory, and temp files are always removed in `finally` blocks.
- **Configurable via environment variables:** model size, device, compute type and size limit can be changed without code edits.

## Production considerations

- **Long files:** chunk audio at silences (~10 min with overlap), transcribe chunks in parallel, merge and offset timestamps, and checkpoint per chunk.
- **Concurrent uploads:** the API stores the file in S3 and returns a job ID; workers pull jobs from a queue (Celery + Redis/RabbitMQ) and scale horizontally.
- **Storage:** audio and transcript JSON in object storage (S3); job metadata in PostgreSQL; raw audio deleted after a retention period.
- **Reliability:** outbox pattern for reliable job publishing, retries with exponential backoff for temporary errors, idempotent workers, and a dead-letter queue.
- **Real-time:** a WebSocket or gRPC streaming endpoint with VAD-based buffering and partial/final results.

## Limitations & next steps

- No speaker diarization (could add `pyannote.audio`).
- Accuracy drops on heavy accents, code-switched speech (e.g., Urdu-English) and overlapping speakers; a larger model or fine-tuning would help.
- Add automated tests with a mock transcriber, WER evaluation with `jiwer`, and a Dockerfile.
