# IBM Custom Granite 4.0 + vLLM (Containerized)

**Deploying Granite models using vLLM inside a Docker container**

## Introduction

This guide shows how to run IBM Granite models inside a container using vLLM, exposing an OpenAI-compatible API.
Suitable for inference on GPU machines via Docker (or Podman).

## Prerequisites

* A container runtime (e.g., Docker Desktop, Podman)
* NVIDIA GPU and appropriate drivers installed

## Deployment Steps

### 1. Pull the vLLM container image

```bash
docker pull vllm/vllm-openai:latest
```

> For stability it is recommended to pin a version tag, e.g., `vllm/vllm-openai:v0.10.2`. Granite models require vLLM version 0.10.2 or above.

# make sure packages are update
```bash
sudo apt-get update
sudo apt-get install -y build-essential libc6-dev
```
### 2. Run Granite in the container

```bash
docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
    --model ibm-granite/granite-4.0-micro \
    --max_model_len 65536
```

* Mount `~/.cache/huggingface` so models are cached locally for reuse.
* If the model is not pre-downloaded, vLLM will fetch it automatically from Hugging Face.
* Here I used the model `ibm-granite/granite-4.0-micro`.

### 3. Run a sample request

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "ibm-granite/granite-4.0-micro",
        "messages": [
          {"role": "user", "content": "How are you today?"}
        ]
      }'
```

This will send a chat request via the OpenAI-compatible endpoint.

### 4. Enabling tool-calling and extended capabilities

If you need model capabilities such as tool-calling (e.g., function / API invocation), launch the container with additional flags:

```bash
docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
    --model ibm-granite/granite-4.0-h-small \
    --tool-call-parser hermes \
    --enable-auto-tool-choice
```

* `--tool-call-parser hermes` enables the tool-call parser engine.
* `--enable-auto-tool-choice` allows the model to pick tools automatically based on input context.
  Refer to the official vLLM documentation for further flags and controls.

---

## Notes & Best Practices

* For long-context tasks or heavy inference, ensure the GPU has sufficient memory and the container has proper resource limits.
* Use version-pinning (both for vLLM and Granite) for production consistency.
* Mounting the HF cache directory ensures model downloads are reused across runs.
* The exposed API is OpenAI-compatible, so many existing integrations will work with minimal changes.

---

## Summary

With a single Docker run command, you can deploy an IBM Granite model via vLLM and start sending OpenAI-style requests. Adding tool-calling capabilities only requires a few extra flags. Mounting the cache directory, selecting the correct model and version, and enabling GPU access ensures an efficient production-friendly setup.

---

## References

* vLLM Docker deployment: [vLLM docs](https://docs.vllm.ai/en/latest/serving/docker.html)
* IBM Granite + vLLM guide: IBM Granite docs


