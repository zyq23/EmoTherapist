# Load model directly
from transformers import AutoModel
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="CAS-SIAT-XinHai/CPsyCounX", repo_type="model",
                  cache_dir="./CPsyCounX",
                  local_dir_use_symlinks=False, resume_download=True,
                  token='hf_***')

snapshot_download(repo_id="qiuhuachuan/PsyChat", repo_type="model",
                  cache_dir="./PsyChat",
                  local_dir_use_symlinks=False, resume_download=True,
                  token='hf_***')

snapshot_download(repo_id="scutcyr/SoulChat", repo_type="model",
                  cache_dir="./SoulChat",
                  local_dir_use_symlinks=False, resume_download=True,
                  token='hf_***')

snapshot_download(repo_id="YIRONGCHEN/SoulChat2.0-Qwen2-7B", repo_type="model",
                  cache_dir="./SoulChat2",
                  local_dir_use_symlinks=False, resume_download=True,
                  token='hf_***')