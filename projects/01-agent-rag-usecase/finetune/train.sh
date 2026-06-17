#!/usr/bin/env bash
# 一键启动 LoRA 微调
# 前置：pip install -e LLaMA-Factory && pip install flash-attn

set -euo pipefail
cd "$(dirname "$0")/.."

# 检查显存：单卡 24GB 够，否则换 DeepSpeed ZeRO-2
nvidia-smi --query-gpu=memory.total --format=csv,noheader

# 启动训练
llamafactory-cli train finetune/lora_config.yaml

# 训完合并适配器到一个独立目录，vLLM 加载方便
llamafactory-cli export \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --adapter_name_or_path finetune/output/lora \
  --export_dir finetune/output/merged \
  --export_size 4 \
  --export_legacy_format false

echo "Done. Adapter at ./finetune/output/lora; merged model at ./finetune/output/merged"
