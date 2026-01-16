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
    r = requests.post(
        "http://localhost:9007/chattts/tts",
        json={"text": "你好，这是 ChatTTS"},
        stream=True
    )

    with open("out.wav", "wb") as f:
        for chunk in r.iter_content(1024):
            f.write(chunk)

def main():
    test_qwen()


if __name__ == "__main__":
    main()