from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import io
import numpy as np
import torch
import torchaudio
import os
import shutil
import stat
from pathlib import Path

import ChatTTS
from ChatTTS.core import Chat

app = FastAPI(title="ChatTTS API")

# 全局变量，保存加载后的模型实例
chat = None


def init_chattts_cache():
    """
    初始化 Hugging Face 缓存目录 + 预置 rvcmd 可执行文件，防止 load() 去联网下载
    """
    # Hugging Face 缓存路径
    HF_CACHE_BASE = Path.home() / ".cache" / "huggingface" / "hub"
    REPO_SLUG = "models--2Noise--ChatTTS"
    SNAPSHOT_NAME = "local-snapshot"

    target_dir = HF_CACHE_BASE / REPO_SLUG / "snapshots" / SNAPSHOT_NAME
    source_dir = Path("/models/ChatTTS/asset")

    # 复制模型权重（如果存在）
    if source_dir.exists() and source_dir.is_dir():
        key_subdirs = ["DVAE", "GPT", "Tokenizer", "Vocos"]
        if target_dir.exists() and all((target_dir / sub).exists() for sub in key_subdirs):
            print(f"HF 缓存已完整：{target_dir}，跳过模型复制")
        else:
            print(f"正在复制模型到 HF 缓存：{source_dir} → {target_dir}")
            shutil.rmtree(target_dir, ignore_errors=True)
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
            print("模型复制完成")

        # 伪造 refs/main 和 .gitattributes
        refs_dir = HF_CACHE_BASE / REPO_SLUG / "refs"
        refs_dir.mkdir(parents=True, exist_ok=True)
        with open(refs_dir / "main", "w") as f:
            f.write(SNAPSHOT_NAME)
        print("已伪造 refs/main")

        with open(target_dir / ".gitattributes", "w") as f:
            f.write("# fake to skip remote check\n")
        print("已伪造 .gitattributes")

    # 强制预置 rvcmd 可执行文件，防止 download_all_assets 去下载
    rvcmd_source = Path("/app/ai-services/chattts/rvcmd")  # 你放 rvcmd 的位置
    if rvcmd_source.exists() and rvcmd_source.is_file():
        tmpdir = Path("/tmp/rvcmd_dir")
        tmpdir.mkdir(parents=True, exist_ok=True)
        cmdfile = tmpdir / "rvcmd"

        # 复制并加执行权限
        shutil.copy2(rvcmd_source, cmdfile)
        cmdfile.chmod(cmdfile.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        print(f"已预置 rvcmd 到 {cmdfile}，跳过下载步骤")

        # 可选：patch download_and_extract_tar_gz 函数，让它直接返回（更保险）
        import ChatTTS.utils.dl as dl_utils
        def fake_download_and_extract(url, dest):
            print(f"强制跳过下载 rvcmd: {url}")
            # 假设已经存在，不做任何事
            return
        dl_utils.download_and_extract_tar_gz = fake_download_and_extract

    else:
        print("未找到 rvcmd 可执行文件，将可能触发下载（请确保 /app/ai-services/chattts/rvcmd 存在）")

    # 强制离线模式（双保险）
    os.environ["HF_HUB_OFFLINE"] = "1"
    print("已启用 HF_HUB_OFFLINE=1")


@app.on_event("startup")
async def startup_event():
    global chat

    init_chattts_cache()

    torch.set_float32_matmul_precision('high')

    chat = Chat()
    chat.load(compile=False)

    print("ChatTTS 模型加载成功")


# 以下部分保持不变（TTSRequest、/tts、/health、if __name__ ...）
class TTSRequest(BaseModel):
    text: str
    temperature: float = 0.3
    top_P: float = 0.7
    top_K: int = 20
    refine_text_prompt: str = '[oral_2][laugh_0][break_6]'


@app.post("/tts")
async def tts(request: TTSRequest):
    if chat is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")

    try:
        params_infer_code = {
            'spk_emb': chat.sample_random_speaker(),
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
            raise ValueError("未生成任何音频")

        audio_np = np.array(wavs[0], dtype=np.float32)
        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)

        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_tensor, 24000, format="wav")
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=output.wav"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失败：{str(e)}")


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": chat is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003,
        log_level="info",
    )