#!/usr/bin/sh

ollama serve &

sleep 5

ollama pull qwen3-vl:235b-cloud

tail -f /dev/null