from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
from io import BytesIO
import uvicorn

app = FastAPI()

# 模型路径（根据你的部署环境调整）
MODEL_PATH = "/app/models/faster-whisper-large-v3"

# 加载模型（建议只加载一次）
model = WhisperModel(
    MODEL_PATH,
    device="cuda",
    compute_type="float16"
)


@app.post("/asr")
async def asr(file: UploadFile = File(...)):
    try:
        # 读取上传的音频文件内容
        audio_bytes = await file.read()

        # 包装成 BytesIO（faster-whisper 支持 file-like 对象）
        audio = BytesIO(audio_bytes)

        # 转写：指定中文，开启 VAD 过滤静音，beam_size 适中
        segments, info = model.transcribe(
            audio,
            language="zh",  # 强制中文，关键优化！
            beam_size=5,
            vad_filter=True,  # 过滤掉无声部分，提高效率
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        # 收集所有段落文本
        full_text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())

        # 返回结果
        return {
            "text": full_text,
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration  # 音频时长（秒）
        }

    except Exception as e:
        # 捕获错误，返回友好提示（避免 500 裸崩）
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "detail": "音频处理失败，请检查文件格式或服务器日志"}
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)