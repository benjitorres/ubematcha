from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import whisper
import tempfile

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as f:
        return f.read()

model = None

@app.on_event("startup")
async def load_model():
    global model
    print("Loading Whisper model...")
    model = whisper.load_model("base", device="cpu")
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

    result = await run_in_threadpool(
    model.transcribe,
    tmp_path,
    language="en",     # assume English; remove if mixed-language
    temperature=0.0,   # more stable, less random
    best_of=3,         # try several decoding samples (improves accuracy)
    beam_size=5,       # beam search width (improves accuracy)
)

    return {"text": result["text"]}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Railway sets PORT, default 8000 locally
    uvicorn.run(app, host="0.0.0.0", port=port)

# --- LOCAL TESTING ONLY ---
# Uncomment the lines below to run locally with `python app.py`
#if __name__ == "__main__":
    #import uvicorn
    #uvicorn.run(app, host="0.0.0.0", port=8000)
