import whisper
from fastapi import FastAPI, File, UploadFile, HTMLResponse

import uvicorn
import tempfile

#load model once when server starts so it doesn't reload every request
model = whisper.load_model("base")

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def main():
    return """
    <html>
        <body>
            <h2>Upload an audio/video file</h2>
            <form action="/transcribe" enctype="multipart/form-data" method="post">
                <input name="file" type="file">
                <input type="submit" value="Transcribe">
            </form>
        </body>
    </html>
    """

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))