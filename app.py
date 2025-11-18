from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.concurrency import run_in_threadpool

import tempfile
import os
import logging
import re

import whisper

app = FastAPI()
logger = logging.getLogger("ubematcha")

model = None
MAX_UPLOAD_SIZE = 75 * 1024 * 1024  # ~75 MB
CHUNK_SIZE = 1024 * 1024            # 1 MB


@app.on_event("startup")
async def load_model():
    global model
    print("Loading Whisper model...")
    # You can switch "tiny" -> "base" later if performance is OK
    model = whisper.load_model("tiny", device="cpu")
    print("Model loaded successfully!")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the landing page."""
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """
    Streams the uploaded file to a temp file, then runs Whisper with
    better decoding settings and returns a formatted transcript.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet, please try again in a moment.",
        )

    tmp_path = None

    try:
        # ---- stream upload to a temp file on disk ----
        suffix = os.path.splitext(file.filename or "")[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            total = 0

            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break

                total += len(chunk)
                if total > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="File too large (over ~75 MB). Please upload a shorter/smaller file.",
                    )

                tmp.write(chunk)

        # ---- run Whisper in a thread with better decoding settings ----
        result = await run_in_threadpool(
            model.transcribe,
            tmp_path,
            language="en",      # assume English; remove if you want auto
            temperature=0.0,    # deterministic, fewer silly mistakes
            best_of=3,          # try several candidates
            beam_size=5,        # beam search
        )

        text = (result.get("text") or "").strip()
        if not text:
            raise HTTPException(status_code=500, detail="Transcription produced empty text.")

        # ---- optional: basic paragraphing for readability ----
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = []

        for s in sentences:
            if not s:
                continue
            current.append(s)
            if len(current) >= 2:  # 2 sentences per paragraph
                chunks.append(" ".join(current))
                current = []

        if current:
            chunks.append(" ".join(current))

        formatted = "\n\n".join(chunks) if chunks else text

        return {"text": formatted}

    except HTTPException:
        # re-raise clean HTTP errors so the client can show them
        raise
    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        # always clean up the temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
