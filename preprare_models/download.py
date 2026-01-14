import os
from huggingface_hub import snapshot_download, HfApi, get_token
from pathlib import Path
import sys

# ================= 配置区 =================
MODELS_DIR = "/home/models"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 香港/大陆加速（2026 年仍推荐）

# 如果 hf-mirror 今天不稳，临时注释上面一行，用官方：
# os.environ["HF_ENDPOINT"] = "https://huggingface.co"

MODELS = [
    {
        "name": "Whisper (faster-whisper large-v3)",
        "repo_id": "Systran/faster-whisper-large-v3",
        "local_dir": os.path.join(MODELS_DIR, "faster-whisper-large-v3"),
        "note": "CTranslate2 优化版，直接用于 faster-whisper"
    },
    {
        "name": "Qwen2.5-7B-Instruct",
        "repo_id": "Qwen/Qwen2.5-7B-Instruct",
        "local_dir": os.path.join(MODELS_DIR, "Qwen2.5-7B-Instruct"),
        "note": "标准 Transformers 格式"
    },
    {
        "name": "ChatTTS",
        "repo_id": "2noise/ChatTTS",
        "local_dir": os.path.join(MODELS_DIR, "ChatTTS"),
        "note": "包含 asset 等完整文件"
    }
]


# ================= 下载函数 =================
def download_model(repo_id, local_dir):
    print(f"\n=== 开始下载 {repo_id} 到 {local_dir} ===")

    try:
        # 检查是否已完整下载（避免重复）
        if os.path.exists(local_dir) and os.listdir(local_dir):
            api = HfApi()
            # 推荐加 files_metadata=True 以获取更多文件信息（如 size），但不强制
            repo_info = api.repo_info(repo_id=repo_id, files_metadata=True)
            local_files = set(os.listdir(local_dir))
            # 用 rfilename（官方字段），去掉 type 判断（已移除）
            remote_files = {f.rfilename for f in repo_info.siblings}

            missing = remote_files - local_files
            if not missing:
                print(f"模型 {repo_id} 已完整存在，跳过...")
                return
            else:
                print(f"检测到缺少文件，继续续传... ({len(missing)} 个文件)")

        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            ignore_patterns=["*.gitattributes", "*.gitignore", "README.md"],  # 可选忽略非模型文件
            token=get_token(),
            # 可选：减少并发线程，香港网络更稳
            max_workers=4,
        )
        print(f"下载完成: {repo_id}")

    except Exception as e:
        print(f"下载失败: {repo_id} → {str(e)}")
        if "authentication" in str(e).lower() or "token" in str(e).lower():
            print("提示：如果模型需要登录，请先运行：huggingface-cli login （或 export HF_TOKEN=你的token）")
        print(f"或者手动从浏览器 https://hf-mirror.com/{repo_id} 下载缺失文件")
        # 不直接 sys.exit(1)，让它继续下一个模型（更友好）
        print("跳过此模型，继续下一个...\n")
        return


# ================= 主流程 =================
def main():
    print("=== AI 模型批量下载脚本（兼容 huggingface_hub v1.0+ / 2026 更新版） ===")
    print(f"目标目录: {MODELS_DIR}")
    print(f"当前 HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', 'https://huggingface.co (默认)')}")
    print("按顺序下载三个模型...\n")

    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

    for idx, model in enumerate(MODELS, 1):
        print(f"\n[{idx}/{len(MODELS)}] {model['name']}")
        print(f"Repo: {model['repo_id']}")
        print(f"保存到: {model['local_dir']}")
        if "note" in model:
            print(f"备注: {model['note']}")

        download_model(model["repo_id"], model["local_dir"])

    print("\n=== 全部处理完成！ ===")
    print("模型路径总结（已下载/续传的）：")
    for m in MODELS:
        print(f"- {m['name']}: {m['local_dir']}")
    print("\n接下来运行 docker-compose up -d 启动服务（路径已匹配）")
    print("提示：如果 Qwen 仍卡住，可手动下载缺失的 .safetensors 分片后重新运行脚本（会自动校验续传）")


if __name__ == "__main__":
    main()