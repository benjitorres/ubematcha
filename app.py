from fastapi import FastAPI, UploadFile, File
import whisper
import tempfile

app = FastAPI()

model = None

@app.on_event("startup")
async def load_model():
    global model
    print("Loading Whisper model...")
    model = whisper.load_model("tiny")
    print("Model loaded successfully!")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    global model
    if model is None:
        model = whisper.load_model("tiny")

    # Save uploaded audio to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    result = model.transcribe(tmp_path)
    return {"text": result["text"]}


# --- LOCAL TESTING ONLY ---
# Uncomment the lines below to run locally with `python app.py`
#if __name__ == "__main__":
    #import uvicorn
    #uvicorn.run(app, host="0.0.0.0", port=8000)
