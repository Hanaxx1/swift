# Experimental environment: A100
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --ckpt_dir "output/qwen-audio-chat/vx-xxx/checkpoint-xxx" \
    --load_dataset_config true \
    --use_flash_attn true \
