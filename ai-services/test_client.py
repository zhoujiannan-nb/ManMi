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
    """简化版测试函数"""

    # API 配置
    url = "http://103.219.36.196:9007/chattts/tts"
    data = {
        "text": "你好，这是一个测试音频。",
        "voice": "3333",
        "temperature": 0.3,
        "top_P": 0.7,
        "top_K": 20
    }

    try:
        # 发送请求
        response = requests.post(url, json=data, timeout=30)

        # 检查响应
        if response.status_code == 200:
            print("✅ 请求成功！")
            print(f"音频大小: {len(response.content) / 1024:.1f} KB")

            # 保存音频
            with open("test_output.wav", "wb") as f:
                f.write(response.content)
            print("✅ 音频已保存为 test_output.wav")

        else:
            print(f"❌ 请求失败: {response.status_code}")
            print(f"错误信息: {response.text[:200]}")

    except Exception as e:
        print(f"❌ 测试出错: {str(e)}")


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