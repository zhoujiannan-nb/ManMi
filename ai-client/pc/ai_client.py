# ai_client.py
from dataclasses import dataclass
from typing import Optional, List, Dict, Union, Generator
import os
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # 自动加载 .env 文件

@dataclass
class AIConfig:
    base_url: str
    timeout: float = 30.0

    @classmethod
    def from_env(cls) -> "AIConfig":
        base_url = os.getenv("VOICE_AI_BASE_URL")
        if not base_url:
            raise ValueError("VOICE_AI_BASE_URL 未在 .env 中设置")

        timeout = float(os.getenv("REQUEST_TIMEOUT", "30.0"))
        return cls(base_url=base_url.rstrip("/"), timeout=timeout)


class AIClient:
    def __init__(self, config: Optional[AIConfig] = None):
        if config is None:
            config = AIConfig.from_env()

        self.config = config
        self.session = requests.Session()
        self.session.timeout = self.config.timeout
        # 可选：统一添加 headers
        # self.session.headers.update({"User-Agent": "AI-Client/1.0"})

        # 服务端点（相对路径）
        self._qwen_endpoint = "/qwen/v1/chat/completions"
        self._chattts_endpoint = "/chattts/tts"
        self._chattts_base = "/chattts/"
        self._whisper_endpoint = "/whisper/asr"

    @property
    def qwen_url(self) -> str:
        return f"{self.config.base_url}{self._qwen_endpoint}"

    @property
    def chattts_url(self) -> str:
        return f"{self.config.base_url}{self._chattts_endpoint}"

    @property
    def whisper_url(self) -> str:
        return f"{self.config.base_url}{self._whisper_endpoint}"

    def qwen_chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "/models/Qwen2.5-7B-Instruct",
        stream: bool = False,
        **extra_params
    ) -> Union[str, Generator[str, None, None]]:
        """
        调用 Qwen 对话接口
        :return: 非流式返回完整文本，流式返回生成器（每行 bytes）
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **extra_params
        }

        resp = self.session.post(self.qwen_url, json=payload, stream=stream)
        resp.raise_for_status()

        if stream:
            def line_generator():
                for line in resp.iter_lines():
                    if line:
                        yield line.decode("utf-8", errors="replace")
            return line_generator()
        else:
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    def chattts_synthesize(
        self,
        text: str,
        voice: str = "4785.pt",
        speed: str = "5",
        temperature: str = "0.3",
        top_p: str = "0.7",
        top_k: str = "20",
        prompt: str = "[break_6]",
        skip_refine: str = "1",
        output_path: Optional[str | Path] = "output.wav",
        return_bytes: bool = False
    ) -> Union[bool, bytes]:
        """
        调用 ChatTTS 语音合成，返回是否成功或音频字节
        """
        form_data = {
            "text": text,
            "prompt": prompt,
            "voice": voice,
            "speed": speed,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "skip_refine": skip_refine,
        }

        resp = self.session.post(self.chattts_url, data=form_data)
        resp.raise_for_status()

        result = resp.json()
        if result.get("code") != 0:
            raise RuntimeError(f"ChatTTS 失败: {result.get('msg', '未知错误')}")

        audio_rel_path = result["filename"].replace("/app", "", 1).lstrip("/")
        audio_url = f"{self.config.base_url}{self._chattts_base}{audio_rel_path}"

        audio_resp = self.session.get(audio_url)
        audio_resp.raise_for_status()

        audio_bytes = audio_resp.content

        if return_bytes:
            return audio_bytes

        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(audio_bytes)
            print(f"音频已保存至: {path} ({len(audio_bytes)/1024:.1f} KB)")

        return True

    def whisper_asr(
        self,
        audio_file: str | Path | bytes,
        filename: str = "audio.wav"
    ) -> str:
        """
        调用 Whisper 语音转文字
        """
        if isinstance(audio_file, (str, Path)):
            path = Path(audio_file)
            if not path.is_file():
                raise FileNotFoundError(f"音频文件不存在: {path}")
            with open(path, "rb") as f:
                files = {"file": (filename, f, "audio/wav")}
                resp = self.session.post(self.whisper_url, files=files)
        elif isinstance(audio_file, bytes):
            files = {"file": (filename, audio_file, "audio/wav")}
            resp = self.session.post(self.whisper_url, files=files)
        else:
            raise TypeError("audio_file 必须是路径(str/Path)或bytes")

        resp.raise_for_status()
        return resp.text.strip()


# 方便直接导入使用
def get_ai_client() -> AIClient:
    return AIClient()