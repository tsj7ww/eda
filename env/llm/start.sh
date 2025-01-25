#!/bin/bash
cd /workspace/llama.cpp
./server -m models/llama-2-7b.gguf -c 4096 --host 0.0.0.0 --port 8080 &
jupyter notebook --allow-root --ip 0.0.0.0 --port 8888
