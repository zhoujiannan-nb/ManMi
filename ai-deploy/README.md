# AI Deploy

## 启动
```bash
docker-compose build
docker-compose up -d
```

## 停止
```bash
docker-compose down
```

## docker-compose 的文件在不同环境系统上的修改
首先最低要求: 30系 16G显卡以上
单卡最低部署：
    qwen：7.5GB
    chatts：4.1GB
    whisper: 4.1GB

多卡可以增加vllm冗余:
    单卡部署 qwen
    单卡部署 chatts+whisper

体现到docker-compose.yaml中自行修改