import os
import tempfile
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Lazy-load whisper
model = None

app = FastAPI()

# Serve static files (index.html)
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/", response_class=FileResponse)
async def root():
    return "index.html"

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    global model
    if model is None:
        import whisper
        model = whisper.load_model("tiny.en")  # small = faster for Render free tier

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Transcribe
    result = model.transcribe(tmp_path, language=None, beam_size=5)

    # Delete temp file
    os.remove(tmp_path)

    # Return transcription
    return {
        "language": result.get("language"),
        "text": result["text"],
        "segments": [
            {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
            for seg in result["segments"]
        ],
    }

# --- LOCAL TESTING ONLY ---
# Uncomment the lines below to run locally with `python app.py`
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
