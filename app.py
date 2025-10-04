import os
import whisper
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import tempfile

# Load Whisper model once at startup
model = whisper.load_model("base")

app = FastAPI()

# Serve static files (like index.html)
app.mount("/static", StaticFiles(directory="."), name="static")

# Root route serves index.html
@app.get("/", response_class=FileResponse)
async def root():
    return "index.html"

# Transcription endpoint
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    global model
    if model is None:
        import whisper
        model = whisper.load_model("small")

    # Run Whisper transcription
    result = model.transcribe(tmp_path, language=None, beam_size=5)

    # Clean up temp file
    os.remove(tmp_path)

    return {
        "language": result.get("language"),
        "text": result["text"],
        "segments": [
            {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
            for seg in result["segments"]
        ],
    }

# ---- LOCAL TESTING ONLY ----
# When running locally, you can use `python app.py` to test.
# On Render, remove this part / comment it out.
#if __name__ == "__main__" and os.environ.get("RENDER") is None:
 #   import uvicorn
  #  uvicorn.run(app, host="0.0.0.0", port=8000)
