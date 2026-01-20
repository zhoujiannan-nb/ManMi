# recorder.py
# 这个文件放录音相关的功能，使用 sounddevice 和 scipy
# 安装：pip install sounddevice scipy

import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import threading

class AudioRecorder:
    def __init__(self, sample_rate=16000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.recording = False
        self.frames = []
        self.thread = None

    def start_recording(self):
        if self.recording:
            return
        self.frames = []
        self.recording = True
        self.thread = threading.Thread(target=self._record)
        self.thread.start()

    def _record(self):
        with sd.InputStream(samplerate=self.sample_rate, channels=self.channels) as stream:
            while self.recording:
                data, overflowed = stream.read(1024)
                if overflowed:
                    print("警告: 录音溢出")
                self.frames.append(data)

    def stop_recording(self, output_file="input.wav"):
        if not self.recording:
            return None
        self.recording = False
        if self.thread:
            self.thread.join()
        audio_data = np.concatenate(self.frames, axis=0)
        wav.write(output_file, self.sample_rate, audio_data)
        print(f"录音保存至: {output_file}")
        return output_file