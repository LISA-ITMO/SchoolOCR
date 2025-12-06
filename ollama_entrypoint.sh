#!/usr/bin/sh

ollama serve &

sleep 5

ollama pull qwen2.5vl:7b
ollama run qwen2.5vl:7b

tail -f /dev/null