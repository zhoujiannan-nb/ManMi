import io
import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import ChatTTS
import torch

app = FastAPI()

chattts = ChatTTS.ChatTTS()
chattts.load_models(source="local", local_path="/models/ChatTTS")

@app.post("/tts")
def tts(req: dict):
    text = req.get("text", "")

    wavs = chattts.infer(
        [text],
        stream=True,
        use_decoder=True
    )

    def audio_stream():
        for wav in wavs:
            buf = io.BytesIO()
            sf.write(buf, wav, 24000, format="WAV")
            buf.seek(0)
            yield buf.read()

    return StreamingResponse(audio_stream(), media_type="audio/wav")
