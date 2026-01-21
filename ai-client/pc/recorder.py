# recorder.py
import sounddevice as sd
import numpy as np
import threading
import time
from scipy.io.wavfile import write
from pathlib import Path
import wave


class VoiceActivityDetector:
    def __init__(self, threshold=0.02, silence_duration=1.5, max_duration=20.0, sample_rate=16000):
        self.threshold = threshold
        self.silence_duration = silence_duration
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.recording = False
        self.frames = []
        self.recording_start_time = None
        self.last_voice_time = None

    def _calculate_rms(self, data):
        """计算音频数据的RMS（均方根）值"""
        if len(data) == 0:
            return 0
        return np.sqrt(np.mean(np.square(data.astype(np.float32))))

    def _has_voice(self, audio_chunk):
        """检测音频块中是否有人声"""
        rms = self._calculate_rms(audio_chunk)
        return rms > self.threshold

    def record_with_vad(self, callback=None):
        """
        带有人声活动检测的录音
        callback: 录音完成后的回调函数
        """
        self.recording = True
        self.frames = []
        self.recording_start_time = time.time()
        self.last_voice_time = time.time()

        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"录音状态: {status}")

            if not self.recording:
                raise sd.CallbackStop()

            # 检测当前帧是否有人声
            has_voice = self._has_voice(indata[:, 0])

            if has_voice:
                self.last_voice_time = time.time()
                self.frames.append(indata.copy())
            elif len(self.frames) > 0:
                # 如果没有检测到人声，但已经有录音数据，继续录音一段时间
                self.frames.append(indata.copy())

            # 检查停止条件
            current_time = time.time()

            # 条件1: 录音时长超过最大限制
            if current_time - self.recording_start_time >= self.max_duration:
                self.recording = False
                print(f"达到最大录音时长: {self.max_duration}秒")
                return

            # 条件2: 已有录音数据且静音时间超过阈值
            if len(self.frames) > 0:
                silence_time = current_time - self.last_voice_time
                if silence_time >= self.silence_duration:
                    self.recording = False
                    print(f"检测到静音，停止录音。静音时长: {silence_time:.1f}秒")

        try:
            # 开始录音
            print(
                f"开始录音 - 阈值: {self.threshold}, 静音时长: {self.silence_duration}s, 最大时长: {self.max_duration}s")
            with sd.InputStream(samplerate=self.sample_rate,
                                channels=1,
                                callback=audio_callback,
                                blocksize=1024):
                while self.recording:
                    time.sleep(0.1)

                print("录音结束")

                # 保存录音
                if len(self.frames) > 0:
                    audio_data = np.concatenate(self.frames, axis=0)

                    # 移除末尾的静音部分
                    # 找到最后一个人声的位置
                    chunk_size = 1024
                    last_voice_idx = 0
                    for i in range(0, len(audio_data), chunk_size):
                        chunk = audio_data[i:min(i + chunk_size, len(audio_data))]
                        if self._has_voice(chunk):
                            last_voice_idx = i + len(chunk)

                    if last_voice_idx > 0:
                        # 保留一些人声后的缓冲
                        buffer_samples = int(self.sample_rate * 0.2)  # 200ms缓冲
                        audio_data = audio_data[:min(last_voice_idx + buffer_samples, len(audio_data))]

                    return audio_data
                else:
                    print("未检测到人声")
                    return None

        except Exception as e:
            print(f"录音过程中出错: {e}")
            return None
        finally:
            if callback:
                callback()


class AudioRecorder:
    def __init__(self, sample_rate=16000, channels=1,
                 vad_threshold=0.02, silence_duration=1.5, max_duration=20.0):
        self.sample_rate = sample_rate
        self.channels = channels
        self.vad_threshold = vad_threshold
        self.silence_duration = silence_duration
        self.max_duration = max_duration

        self.recorder = None
        self.recording_thread = None
        self.recording_active = False
        self.audio_data = None

    def start_recording_async(self, callback=None):
        """开始录音（异步方式）"""
        if self.recording_active:
            return False

        self.recording_active = True
        self.audio_data = None

        def record_task():
            try:
                self.recorder = VoiceActivityDetector(
                    threshold=self.vad_threshold,
                    silence_duration=self.silence_duration,
                    max_duration=self.max_duration,
                    sample_rate=self.sample_rate
                )

                self.audio_data = self.recorder.record_with_vad(callback)
            except Exception as e:
                print(f"录音线程出错: {e}")
            finally:
                self.recording_active = False

        self.recording_thread = threading.Thread(target=record_task)
        self.recording_thread.daemon = True
        self.recording_thread.start()

        return True

    def stop_recording(self):
        """停止录音"""
        if self.recorder and self.recorder.recording:
            self.recorder.recording = False

        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2.0)

    def is_recording(self):
        """检查是否正在录音"""
        return self.recording_active

    def save_recording(self, output_file="input.wav"):
        """保存录音到文件"""
        if self.audio_data is None or len(self.audio_data) == 0:
            print("没有录音数据可保存")
            return None

        try:
            # 确保目录存在
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)

            # 保存为WAV文件
            write(output_file, self.sample_rate, self.audio_data)

            print(f"录音保存至: {output_file} (时长: {len(self.audio_data) / self.sample_rate:.2f}s)")
            return output_file
        except Exception as e:
            print(f"保存录音失败: {e}")
            return None

    def get_audio_duration(self):
        """获取录音时长（秒）"""
        if self.audio_data is None:
            return 0
        return len(self.audio_data) / self.sample_rate