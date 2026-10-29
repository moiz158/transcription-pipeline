# Transcription Pipeline

A simple audio transcription pipeline: accept an audio file, convert speech to text, and return the transcript with timestamps per segment, as scripts or as a REST API.

Built with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (an open-source Whisper implementation), ffmpeg and FastAPI.

## Pipeline

```
Audio file (WAV, MP3, M4A, ...)
   │
   ▼
1. Accept & validate      audio_input.py        extension, size, ffprobe check, metadata
   │
   ▼
2. Normalize              ffmpeg                16 kHz, mono, 16-bit PCM WAV
   │
   ▼
3. Transcribe             faster-whisper + VAD  speech → text
   │
   ▼
4. Segments & output      transcribe_segments.py  timestamps per segment, JSON
   │
   ▼
5. Serve                  api.py                FastAPI endpoint
```

## Files

| File | Purpose |
|---|---|
| `audio_input.py` | Accepts an audio file and validates it (format, size, real audio stream via ffprobe). Script + `POST /upload` endpoint. |
| `transcribe.py` | Transcribes speech into plain text. |
| `transcribe_segments.py` | Returns the transcript with timestamps per segment; prints them and saves `transcript.json`. |
| `api.py` | FastAPI service: `POST /transcribe` returns the transcript and segments as JSON. |
| `requirements.txt` | Python dependencies. |

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
python audio_input.py sample.mp3

# 2. Transcribe to plain text
python transcribe.py sample.mp3

# 3. Transcribe with timestamps per segment (also writes transcript.json)
python transcribe_segments.py sample.mp3
```

Example output:

```
[00:00:00.000 --> 00:00:03.200] Hello, thanks for joining.
[00:00:03.400 --> 00:00:06.100] Let's review the project.
```

### API

```bash
uvicorn api:app --reload
curl -F "file=@sample.mp3" "http://localhost:8000/transcribe?timestamps=true"
```

```json
{
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
