import os
import whisper
from fastapi import FastAPI, File, UploadFile

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import uvicorn
import tempfile

#load model once when server starts so it doesn't reload every request
model = whisper.load_model("base")

app = FastAPI()

app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/", response_class=FileResponse)
async def root():
    return "index.html"

@app.post("/transcribe")

async def transcribe(file: UploadFile = File(...)):
    #save upload file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    #run whisper transcription
    result = model.transcribe(tmp_path, language=None, beam_size=5)

    return {
        "language": result.get("language"),
        "text": result["text"],
        "segments": [
            {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
            for seg in result["segments"]
        ],
    }

"""
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
"""