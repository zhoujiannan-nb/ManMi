# player.py
from pydub import AudioSegment
from pydub.playback import play

def play_audio(file_path):
    try:
        sound = AudioSegment.from_file(file_path)
        play(sound)
        print(f"播放完成: {file_path}")
    except Exception as e:
        print(f"播放失败: {str(e)}")