from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import io
import numpy as np
import torch
import torchaudio

import ChatTTS

app = FastAPI(title="ChatTTS API")

# 全局加载（容器启动时只加载一次，节省时间）
chat = None


@app.on_event("startup")
async def startup_event():
    global chat
    torch.set_float32_matmul_precision('high')

    chat = ChatTTS.Chat()
    # 重要：使用你挂载的路径
    chat.load(
        source='local',
        local_path='/models/ChatTTS/asset',  # 容器内路径，见 docker-compose volumes
        compile=False,  # 先 False，内存不够再 True
    )
    print("ChatTTS loaded successfully")


class TTSRequest(BaseModel):
    text: str
    temperature: float = 0.3
    top_P: float = 0.7
    top_K: int = 20
    refine_text_prompt: str = '[oral_2][laugh_0][break_6]'  # 默认自然一点


@app.post("/tts")
async def tts(request: TTSRequest):
    if not chat:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        params_infer_code = {
            'spk_emb': chat.sample_random_speaker(),  # 随机音色，或你固定一个
            'temperature': request.temperature,
            'top_P': request.top_P,
            'top_K': request.top_K,
        }

        params_refine_text = {'prompt': request.refine_text_prompt}

        wavs = chat.infer(
            [request.text],
            params_infer_code=params_infer_code,
            params_refine_text=params_refine_text,
            use_decoder=True,
        )

        if not wavs or len(wavs) == 0:
            raise ValueError("No audio generated")

        audio_np = np.array(wavs[0], dtype=np.float32)
        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)  # [1, samples]

        # 转成 wav bytes 流式返回（或直接返回文件）
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_tensor, 24000, format="wav")
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=output.wav"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": chat is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        # workers=1   # 單進程就不要開 workers
    )