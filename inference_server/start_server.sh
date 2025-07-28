#!/bin/bash
export MSYS_NO_PATHCONV=1 # stop git bash from changing paths
YOUR_PATH="D:/models/Huggingface/" # change this to the folder path in which the hub folder containing your huggingface models is located
CONTAINER_PATH="/sgl-workspace/sglang/models/"
docker run --gpus all \
    --shm-size=32g \
    -p 30000:30000 \
    --mount type=bind,source="$YOUR_PATH",target="$CONTAINER_PATH" \
    --env HF_TOKEN="<your_token>" \
    --env HF_HOME="/sgl-workspace/sglang/models" \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server \
        --model-path /sgl-workspace/sglang/models/hub/models--meta-llama--Llama-3.1-8B-Instruct \
        --host 0.0.0.0 \
        --port 30000 \
        --dtype bfloat16 \
        --context-length 8192 \
        --attention-backend "fa3" \
        --enable-metrics
