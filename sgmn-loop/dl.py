from huggingface_hub import snapshot_download
import os
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
for repo in ["Qwen/Qwen2.5-0.5B-Instruct", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"]:
    p = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
    print("OK", repo, "->", p, flush=True)
