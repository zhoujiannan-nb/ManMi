# player.py
import pyaudio
import wave
import threading
import time


class AudioPlayer:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.is_playing = False
        self.playing_thread = None

    def play_wav(self, file_path):
        """播放WAV文件"""
        try:
            # 打开WAV文件
            wf = wave.open(file_path, 'rb')

            # 打开音频流
            stream = self.p.open(
                format=self.p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True
            )

            # 读取并播放数据
            chunk_size = 1024
            data = wf.readframes(chunk_size)

            self.is_playing = True

            while data and self.is_playing:
                stream.write(data)
                data = wf.readframes(chunk_size)

            # 停止和关闭流
            stream.stop_stream()
            stream.close()
            wf.close()

            self.is_playing = False
            print(f"✓ 播放完成: {file_path}")

        except Exception as e:
            print(f"✗ 播放失败: {str(e)}")
            self.is_playing = False

    def play_async(self, file_path):
        """异步播放音频"""
        if self.is_playing:
            self.stop()

        self.playing_thread = threading.Thread(
            target=self.play_wav,
            args=(file_path,),
            daemon=True
        )
        self.playing_thread.start()

    def stop(self):
        """停止播放"""
        self.is_playing = False
        if self.playing_thread and self.playing_thread.is_alive():
            self.playing_thread.join(timeout=0.5)

    def cleanup(self):
        """清理资源"""
        self.stop()
        self.p.terminate()


# 全局播放器实例
player = AudioPlayer()


def play_audio(file_path):
    """播放音频文件（兼容旧接口）"""
    player.play_async(file_path)


def stop_audio():
    """停止播放"""
    player.stop()


def cleanup_player():
    """清理播放器"""
    player.cleanup()