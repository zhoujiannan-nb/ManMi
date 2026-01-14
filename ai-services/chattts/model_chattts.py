import io
import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import ChatTTS
import torch
import uvicorn
import os

app = FastAPI()

# 初始化ChatTTS - 使用本地模型
chat = ChatTTS.Chat()

# 模型路径
model_path = "/app/models/ChatTTS"

# 检查模型是否存在
print(f"检查模型路径: {model_path}")
if os.path.exists(model_path):
    print("模型目录存在")
    # 列出目录内容
    for item in os.listdir(model_path):
        print(f"  - {item}")
else:
    print("警告: 模型目录不存在")

# 直接从本地加载模型（不下载）
chat.load_models(compile=False)  # compile=False 可以加快加载速度

print("ChatTTS模型加载完成")


@app.post("/tts")
async def tts(req: dict):
    text = req.get("text", "你好，这是一个语音测试")
    print(f"收到TTS请求: {text}")

    try:
        # 生成语音
        wavs = chat.infer([text], skip_refine_text=False)

        # 获取音频数据
        if isinstance(wavs, list) and len(wavs) > 0:
            wav = wavs[0]
        else:
            wav = wavs

        # 转换为numpy数组（如果需要）
        if torch.is_tensor(wav):
            wav = wav.cpu().numpy()

        # 创建音频流
        buf = io.BytesIO()
        sf.write(buf, wav, 24000, format="WAV", subtype='PCM_16')
        buf.seek(0)

        print(f"语音生成成功，长度: {len(wav)} 采样点")
        return StreamingResponse(buf, media_type="audio/wav")

    except Exception as e:
        print(f"生成语音时出错: {str(e)}")
        return {"error": str(e), "message": "语音生成失败"}


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": True}


if __name__ == "__main__":
    print("启动ChatTTS服务，端口: 8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)