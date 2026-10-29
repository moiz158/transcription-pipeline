# audio_input.py
# Script:  python audio_input.py audio.mp3
# Service: uvicorn audio_input:app   ->   POST /upload
import os, sys, json, subprocess, tempfile
from fastapi import FastAPI, UploadFile, HTTPException

SUPPORTED = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4", ".aac"}
MAX_SIZE_MB = 200

def validate_audio(path: str) -> dict:
    """Check the file is a real, readable audio file and return its metadata."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported format: {ext}")

    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb == 0:
        raise ValueError("File is empty")
    if size_mb > MAX_SIZE_MB:
        raise ValueError(f"File too large: {size_mb:.1f} MB (max {MAX_SIZE_MB} MB)")

    # ffprobe reads the actual file content, not just the extension
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_name,sample_rate,channels:format=duration",
         "-of", "json", path],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        raise ValueError("Corrupt or unreadable audio file")

    info = json.loads(probe.stdout)
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

app = FastAPI(title="Audio Upload Service")

@app.post("/upload")
def upload(file: UploadFile):
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in SUPPORTED:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {suffix}")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file.file.read())
    try:
        return validate_audio(tmp.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.remove(tmp.name)

if __name__ == "__main__":
    try:
        print(json.dumps(validate_audio(sys.argv[1]), indent=2))
    except ValueError as e:
        sys.exit(f"Error: {e}")
