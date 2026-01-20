import requests

def test_qwen(stream = False):
    r = requests.post(
        "http://103.219.36.196:9007/qwen/v1/chat/completions",
        json={
            "model": "/models/Qwen2.5-7B-Instruct",
            "messages": [{"role": "user", "content": "哈基米南北绿豆"}],
            "stream": stream
        },
        stream=stream
    )
    if stream:
        for line in r.iter_lines():
            if line:
                print(line.decode())
    else:
        print(r.json()['choices'][0]['message']['content'])


def test_chattts():
    """Form Data格式请求"""

    url = "http://103.219.36.196:8003/tts"

    # 使用Form Data格式（application/x-www-form-urlencoded）
    data = {
        "text": "哈基米你要干鸡毛我操你妈妈吗的蛋",
        "prompt": "[break_6]",
        "voice": "4785.pt",
        "speed": "5",
        "temperature": "0.1",
        "top_p": "0.701",
        "top_k": "20",
        "skip_refine": "1"
    }

    try:
        # 使用Form Data格式发送请求
        response = requests.post(url, data=data, timeout=30)

        if response.status_code == 200:
            result = response.json()

            if result.get("code") == 0:
                audio_url = result.get("url")
                audio_response = requests.get(audio_url, timeout=30)

                if audio_response.status_code == 200:
                    with open("test_output.wav", "wb") as f:
                        f.write(audio_response.content)
                    print(f"✅ 音频保存成功: {len(audio_response.content) / 1024:.1f}KB")
                    return True
                else:
                    print(f"❌ 音频下载失败: {audio_response.status_code}")
            else:
                print(f"❌ TTS失败: {result.get('msg')}")

        print(f"❌ 请求失败: {response.status_code}")
        return False

    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        return False

def test_whisper():
    """测试ASR语音转文字服务"""

    # 准备音频文件
    with open("test.wav", "rb") as f:
        files = {"file": ("test.wav", f, "audio/wav")}

        # 发送请求
        r = requests.post(
            "http://103.219.36.196:9007/whisper/asr",
            files=files
        )

    # 打印结果
    print(r.text)



def main():
    # test_qwen()
    test_chattts()
    # test_whisper()

if __name__ == "__main__":
    main()