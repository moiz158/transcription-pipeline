# api.py
# Service version: HTTP endpoint returning transcription with segment timestamps.
# Run:  uvicorn api:app --reload
# Test: curl -F "file=@audio.mp3" "http://localhost:8000/transcribe?timestamps=true"

import os
import tempfile

from fastapi import FastAPI, HTTPException, UploadFile

from transcribe_segments import transcribe_with_segments

app = FastAPI(title="Transcription Service")


@app.post("/transcribe")
def transcribe_endpoint(file: UploadFile, timestamps: bool = True):
    # Plain `def` (not async): the model call is blocking, so FastAPI
    # runs it in a threadpool instead of blocking the event loop.
    suffix = os.path.splitext(file.filename or "")[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file.file.read())

    try:
        result = transcribe_with_segments(tmp.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.remove(tmp.name)

    if not timestamps:
        result.pop("segments")
    return result


@app.get("/health")
def health():
    return {"status": "ok"}
