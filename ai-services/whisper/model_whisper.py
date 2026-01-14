from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from faster_whisper import WhisperModel
import uvicorn

app = FastAPI()

# 关键：使用本地模型路径的正确方式
MODEL_PATH = "/app/models/faster-whisper-large-v3"

# 使用模型路径初始化
model = WhisperModel(
    MODEL_PATH,  # 直接传递本地路径
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


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)