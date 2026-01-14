from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from faster_whisper import WhisperModel

app = FastAPI()

model = WhisperModel(
    "/models/faster-whisper-large-v3",
    device="cuda",
    compute_type="float16"
)

@app.post("/asr")
async def asr(file: UploadFile = File(...)):
    audio = await file.read()

    segments, _ = model.transcribe(audio, beam_size=5)

    def stream():
        for seg in segments:
            yield seg.text + "\n"

    return StreamingResponse(stream(), media_type="text/plain")
