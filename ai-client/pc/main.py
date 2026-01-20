# main.py
import re
import tkinter as tk
from tkinter import messagebox
from ai_client import get_ai_client  # 从之前的 ai_client.py 导入
from recorder import AudioRecorder
from player import play_audio
import os
import time

# 初始化
client = get_ai_client()
recorder = AudioRecorder(sample_rate=16000)  # Whisper 推荐 16kHz

# 临时文件
INPUT_WAV = "input.wav"
OUTPUT_WAV = "output.wav"

# GUI
root = tk.Tk()
root.title("语音交互演示")
root.geometry("300x200")

status_label = tk.Label(root, text="就绪", font=("Arial", 12))
status_label.pack(pady=20)


def clean_text_simple(text, custom_invalid_chars=None):
    """
    简单直接地删除无效字符

    Args:
        text: 需要清理的文本
        custom_invalid_chars: 自定义的无效字符集合

    Returns:
        清理后的文本
    """
    default_invalid_chars = {'·'}
    invalid_chars = default_invalid_chars
    if custom_invalid_chars:
        invalid_chars = invalid_chars.union(custom_invalid_chars)
    cleaned_text = text
    for char in invalid_chars:
        cleaned_text = cleaned_text.replace(char, '')
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def start_recording():
    status_label.config(text="录音中...")
    recorder.start_recording()
    stop_button.config(state=tk.NORMAL)
    start_button.config(state=tk.DISABLED)

def stop_recording_and_process():
    status_label.config(text="处理中...")
    stop_button.config(state=tk.DISABLED)
    input_file = recorder.stop_recording(INPUT_WAV)
    if not input_file:
        messagebox.showerror("错误", "录音失败")
        reset_buttons()
        return

    try:
        # 2. Whisper ASR
        asr_text = client.whisper_asr(INPUT_WAV)
        print(f"识别文字: {asr_text}")
        if not asr_text.strip():
            raise ValueError("未识别到文字")

        # 3. Qwen Chat
        messages = [{"role": "user", "content": asr_text}]
        qwen_reply = client.qwen_chat(messages)
        print(f"Qwen 回复: {qwen_reply}")

        # 4. ChatTTS 合成
        success = client.chattts_synthesize(
            text=clean_text_simple(qwen_reply),
            output_path=OUTPUT_WAV,
            return_bytes=False
        )
        if not success:
            raise ValueError("TTS 合成失败")

        # 5. 播放
        status_label.config(text="播放中...")
        play_audio(OUTPUT_WAV)

        # 清理
        os.remove(INPUT_WAV) if os.path.exists(INPUT_WAV) else None
        os.remove(OUTPUT_WAV) if os.path.exists(OUTPUT_WAV) else None

        messagebox.showinfo("完成", "交互流程结束")
    except Exception as e:
        messagebox.showerror("错误", str(e))
    finally:
        reset_buttons()

def reset_buttons():
    status_label.config(text="就绪")
    start_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)

start_button = tk.Button(root, text="开始录音", command=start_recording, width=20)
start_button.pack(pady=10)

stop_button = tk.Button(root, text="停止录音并处理", command=stop_recording_and_process, width=20, state=tk.DISABLED)
stop_button.pack(pady=10)

if __name__ == "__main__":
    root.mainloop()