# main.py
import json
import re
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os
import time
from pathlib import Path
import sounddevice as sd

# 导入自定义模块
from ai_client import get_ai_client
from recorder import AudioRecorder
from player import play_audio
from player import play_audio, cleanup_player  # 更新导入
# 初始化
client = get_ai_client()

# 配置录音参数
SAMPLE_RATE = 16000
VAD_THRESHOLD = 0.015  # 人声检测阈值，可以根据环境调整
SILENCE_DURATION = 1.5  # 静音持续时间（秒）
MAX_DURATION = 20.0  # 最长录音时间（秒）

# 创建录音器
recorder = AudioRecorder(
    sample_rate=SAMPLE_RATE,
    vad_threshold=VAD_THRESHOLD,
    silence_duration=SILENCE_DURATION,
    max_duration=MAX_DURATION
)

# 临时文件
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)
INPUT_WAV = TEMP_DIR / "input.wav"
OUTPUT_WAV = TEMP_DIR / "output.wav"


class VoiceAssistantGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("智能语音助手")
        self.root.geometry("450x550")
        self.root.resizable(False, False)

        # 设置窗口图标（可选）
        # try:
        #     self.root.iconbitmap('icon.ico')
        # except:
        #     pass

        # 配置样式
        self.setup_styles()

        # 创建界面
        self.create_widgets()

        # 更新UI状态
        self.update_status("就绪", "green")

        # 绑定窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_styles(self):
        """配置界面样式"""
        style = ttk.Style()
        style.theme_use('clam')

        # 自定义颜色
        self.colors = {
            'primary': '#4A90E2',
            'secondary': '#F5F7FA',
            'success': '#50C878',
            'warning': '#FFA500',
            'error': '#FF5252',
            'text': '#333333'
        }

        # 配置标签样式
        style.configure('Title.TLabel',
                        font=('微软雅黑', 16, 'bold'),
                        foreground=self.colors['primary'])
        style.configure('Status.TLabel',
                        font=('微软雅黑', 12),
                        padding=5)

        # 配置按钮样式
        style.configure('Primary.TButton',
                        font=('微软雅黑', 11, 'bold'),
                        padding=10,
                        background=self.colors['primary'],
                        foreground='white')
        style.map('Primary.TButton',
                  background=[('active', '#3A7BC8')])

        style.configure('Secondary.TButton',
                        font=('微软雅黑', 11),
                        padding=8,
                        background=self.colors['secondary'],
                        foreground=self.colors['text'])

    def create_widgets(self):
        """创建界面组件"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 标题
        title_label = ttk.Label(main_frame,
                                text="🎤 智能语音助手",
                                style='Title.TLabel')
        title_label.pack(pady=(0, 20))

        # 状态显示区域
        status_frame = ttk.Frame(main_frame, relief=tk.RIDGE, borderwidth=2)
        status_frame.pack(fill=tk.X, pady=(0, 20))

        self.status_label = ttk.Label(status_frame,
                                      text="状态: 就绪",
                                      style='Status.TLabel')
        self.status_label.pack(pady=10, padx=10)

        # 录音控制区域
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(pady=10)

        # 录音按钮
        self.record_button = ttk.Button(control_frame,
                                        text="🎤",
                                        style='Primary.TButton',
                                        command=self.start_recording,
                                        width=20)
        self.record_button.pack(pady=10)

        # 停止按钮（初始禁用）
        self.stop_button = ttk.Button(control_frame,
                                      text="⏹️",
                                      style='Secondary.TButton',
                                      command=self.stop_recording,
                                      state=tk.DISABLED,
                                      width=20)
        self.stop_button.pack(pady=5)

        # 设置区域
        settings_frame = ttk.LabelFrame(main_frame, text="录音设置", padding=10)
        settings_frame.pack(fill=tk.X, pady=20)

        # 灵敏度设置
        sensitivity_frame = ttk.Frame(settings_frame)
        sensitivity_frame.pack(fill=tk.X, pady=5)

        ttk.Label(sensitivity_frame, text="人声灵敏度:").pack(side=tk.LEFT)

        self.sensitivity_var = tk.DoubleVar(value=VAD_THRESHOLD)
        sensitivity_scale = ttk.Scale(sensitivity_frame,
                                      from_=0.005,
                                      to=0.05,
                                      variable=self.sensitivity_var,
                                      orient=tk.HORIZONTAL,
                                      length=200)
        sensitivity_scale.pack(side=tk.RIGHT, padx=10)

        self.sensitivity_label = ttk.Label(sensitivity_frame,
                                           text=f"{VAD_THRESHOLD:.3f}")
        self.sensitivity_label.pack(side=tk.RIGHT)

        # 静音检测设置
        silence_frame = ttk.Frame(settings_frame)
        silence_frame.pack(fill=tk.X, pady=5)

        ttk.Label(silence_frame, text="静音检测时间:").pack(side=tk.LEFT)

        self.silence_var = tk.DoubleVar(value=SILENCE_DURATION)
        silence_scale = ttk.Scale(silence_frame,
                                  from_=0.5,
                                  to=3.0,
                                  variable=self.silence_var,
                                  orient=tk.HORIZONTAL,
                                  length=200)
        silence_scale.pack(side=tk.RIGHT, padx=10)

        self.silence_label = ttk.Label(silence_frame,
                                       text=f"{SILENCE_DURATION:.1f}s")
        self.silence_label.pack(side=tk.RIGHT)

        # 信息显示区域
        info_frame = ttk.LabelFrame(main_frame, text="信息", padding=10)
        info_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # 创建文本显示区域
        text_frame = ttk.Frame(info_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        # 添加滚动条
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.text_display = tk.Text(text_frame,
                                    height=8,
                                    wrap=tk.WORD,
                                    font=('微软雅黑', 10),
                                    yscrollcommand=scrollbar.set)
        self.text_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.text_display.yview)

        # 绑定滑块变化事件
        sensitivity_scale.configure(command=self.update_sensitivity_label)
        silence_scale.configure(command=self.update_silence_label)

    def update_sensitivity_label(self, value):
        """更新灵敏度标签"""
        self.sensitivity_label.config(text=f"{float(value):.3f}")

    def update_silence_label(self, value):
        """更新静音检测标签"""
        self.silence_label.config(text=f"{float(value):.1f}s")

    def update_status(self, message, color="black"):
        """更新状态显示"""
        status_text = f"状态: {message}"
        self.status_label.config(text=status_text)

        # 设置颜色
        color_map = {
            'green': self.colors['success'],
            'red': self.colors['error'],
            'orange': self.colors['warning'],
            'blue': self.colors['primary'],
            'black': self.colors['text']
        }

        self.status_label.config(foreground=color_map.get(color, self.colors['text']))

        # 添加到信息显示区域
        timestamp = time.strftime("%H:%M:%S")
        self.text_display.insert(tk.END, f"[{timestamp}] {message}\n")
        self.text_display.see(tk.END)
        self.root.update()

    def start_recording(self):
        """开始录音"""
        self.update_status("检测人声中...", "blue")

        # 更新录音器参数
        global recorder
        recorder = AudioRecorder(
            sample_rate=SAMPLE_RATE,
            vad_threshold=self.sensitivity_var.get(),
            silence_duration=self.silence_var.get(),
            max_duration=MAX_DURATION
        )

        # 禁用开始按钮，启用停止按钮
        self.record_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        # 开始录音
        if recorder.start_recording_async(callback=self.on_recording_finished):
            self.update_status("正在录音...", "orange")

            # 启动定时器检查录音状态
            self.check_recording_status()
        else:
            self.update_status("录音启动失败", "red")
            self.reset_buttons()

    def check_recording_status(self):
        """检查录音状态"""
        if recorder.is_recording():
            # 如果还在录音，继续检查
            self.root.after(100, self.check_recording_status)
        else:
            # 录音已停止
            self.on_recording_finished()

    def on_recording_finished(self):
        """录音完成后的回调"""
        self.update_status("录音完成，处理中...", "blue")

        # 保存录音文件
        audio_file = recorder.save_recording(INPUT_WAV)

        if audio_file and os.path.exists(audio_file):
            # 在新线程中处理录音
            processing_thread = threading.Thread(target=self.process_audio, args=(audio_file,))
            processing_thread.daemon = True
            processing_thread.start()
        else:
            self.update_status("录音失败或无声", "red")
            self.reset_buttons()

    def stop_recording(self):
        """手动停止录音"""
        self.update_status("手动停止录音...", "orange")
        recorder.stop_recording()

    def process_audio(self, audio_file):
        """处理音频文件"""
        try:
            # 1. Whisper ASR
            self.update_status("语音识别中...", "blue")
            asr_text = client.whisper_asr(audio_file)
            print(f"识别文字: {asr_text}")

            if not asr_text.strip():
                raise ValueError("未识别到有效文字")

            # 显示识别结果
            self.root.after(0, lambda: self.text_display.insert(tk.END, f"你说: {asr_text}\n"))
            self.root.after(0, lambda: self.text_display.see(tk.END))

            # 2. Qwen Chat
            asr_text = json.loads(asr_text)["text"]
            self.update_status("AI思考中...", "blue")
            prompt = ("你现在的角色是一个助理,你的名字叫哈基米。\n"
                      "你的职责是你的要根据已有的知识回答我的一切问题"
                      "聊天要求:1.聊天输出口语化 2.控制下字数,非必要不要超过日常讲话的30-50个字"
                      "注意事项:1.严禁透露自己是什么模型 2.不要输出书面化的语句"
                      "请根据我的话进行回复。接下来为我说的话:{")
            prompt2 = " 请根据{}内的话回复我"
            f_prompt = prompt + asr_text + "}"
            messages = [{"role": "user", "content": f_prompt}]
            qwen_reply = client.qwen_chat(messages)
            print(f"AI回复: {qwen_reply}")

            # 显示AI回复
            self.root.after(0, lambda: self.text_display.insert(tk.END, f"AI: {qwen_reply}\n"))
            self.root.after(0, lambda: self.text_display.see(tk.END))

            # 3. ChatTTS 合成
            self.update_status("语音合成中...", "blue")

            # 清理文本
            cleaned_text = clean_text_simple(qwen_reply)

            # 注意：这里我们确保返回的是True，这样能继续播放
            success = client.chattts_synthesize(
                text=cleaned_text,
                output_path=OUTPUT_WAV,
                return_bytes=False
            )

            if success is False:
                raise ValueError("语音合成失败")

            # 4. 等待文件完全写入
            time.sleep(0.5)

            # 5. 播放语音
            self.update_status("播放语音...", "blue")
            play_audio(str(OUTPUT_WAV))  # 确保是字符串路径

            # 等待播放完成
            time.sleep(0.5)  # 给播放一点启动时间

            # 6. 清理临时文件
            self.cleanup_temp_files()

            self.update_status("交互完成", "green")

        except Exception as e:
            error_msg = str(e)
            print(f"处理过程中出错: {error_msg}")
            self.root.after(0, lambda: self.update_status(f"错误: {error_msg}", "red"))

        finally:
            self.root.after(0, self.reset_buttons)

    def cleanup_temp_files(self):
        """清理临时文件"""
        try:
            if os.path.exists(INPUT_WAV):
                os.remove(INPUT_WAV)
            if os.path.exists(OUTPUT_WAV):
                os.remove(OUTPUT_WAV)
        except Exception as e:
            print(f"清理临时文件失败: {e}")

    def reset_buttons(self):
        """重置按钮状态"""
        self.record_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def on_closing(self):
        """关闭窗口时的处理"""
        # 停止录音
        if recorder.is_recording():
            recorder.stop_recording()

        # 清理播放器
        cleanup_player()

        # 清理临时文件
        self.cleanup_temp_files()

        # 关闭窗口
        self.root.destroy()


def clean_text_simple(text, custom_invalid_chars=None):
    """
    简单直接地删除无效字符
    """
    default_invalid_chars = {'·', '*', '#', '&', '@', '$', '%', '^', '~', '`'}
    invalid_chars = default_invalid_chars
    if custom_invalid_chars:
        invalid_chars = invalid_chars.union(custom_invalid_chars)

    cleaned_text = text
    for char in invalid_chars:
        cleaned_text = cleaned_text.replace(char, '')

    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text


def list_audio_devices():
    """列出可用的音频设备"""
    try:
        devices = sd.query_devices()
        print("可用的音频设备:")
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"{i}: {device['name']} (输入通道: {device['max_input_channels']})")
    except Exception as e:
        print(f"获取音频设备失败: {e}")


if __name__ == "__main__":
    # 列出音频设备（调试用）
    list_audio_devices()

    # 创建GUI
    root = tk.Tk()
    app = VoiceAssistantGUI(root)

    # 设置默认输入设备（如果需要）
    # sd.default.device = [input_device_id, output_device_id]

    root.mainloop()