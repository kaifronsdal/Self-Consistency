CUDA_VISIBLE_DEVICES="6,7" PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1028 CUDA_LAUNCH_BLOCKING=1 accelerate launch --config_file=accelerate_config4.yaml --main_process_port=29501 lora2.py

CUDA_VISIBLE_DEVICES="6,7" python vllm2.py