# AI Deploy

## 说明
### docker-compose 的文件在不同环境系统上的修改
    首先最低要求: 30系 16G显卡以上
    单卡最低部署：
        qwen：7.5GB
        chatts：4.1GB
        whisper: 4.1GB
    
    多卡可以增加vllm冗余:
        单卡部署 qwen
        单卡部署 chatts+whisper
    
    体现到docker-compose.yaml中自行修改

## 基础命令
### 编译
```bash
docker-compose build
```

### 启动 停止 重启
```bash
docker-compose up -d
docker-compose down
docker-compose restart

```

## 具体部署步骤

### 1.准备各种模型基础文件
    1.准备一台可以连接hugface & github.com 电脑或者服务器
    2.执行preprare_models/download.py 下载
    3.准备好三个模型,上传至目标服务器

### 2.打基础镜像
    1.上传本项目到/home/下,保持项目名为ManMi
    2.执行镜像构建命令

```bash
cd /home/ManMi/ai-deploy/dockerfiles/
docker build -f Dockerfile.whisper -t ai-deploy_whisper:latest .
docker build -f Dockerfile.chattts -t ai-deploy_chattts:latest .
```

### 3.装填权重文件
    1.建立models文件夹存放 qwen和whisper模型文件
    2.chatts下载为all-models.7z,解压为 all-models文件夹
    3.将文件夹下all-models/asset/*.pt文件上传到/home/ManMi/ai-services/chattts/asset

### 4.启动
```bash
docker-compose up -d
```

### 5.执行测试
```bash
python test_client.py
```