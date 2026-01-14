import requests

# Qwen
r = requests.post(
    "http://localhost:9007/qwen/v1/chat/completions",
    json={
        "model": "Qwen2.5-7B-Instruct",
        "messages": [{"role": "user", "content": "你好"}],
        "stream": True
    },
    stream=True
)

for line in r.iter_lines():
    if line:
        print(line.decode())

# ChatTTS
r = requests.post(
    "http://localhost:9007/chattts/tts",
    json={"text": "你好，这是 ChatTTS"},
    stream=True
)

with open("out.wav", "wb") as f:
    for chunk in r.iter_content(1024):
        f.write(chunk)
