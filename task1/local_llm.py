import argparse
import os
import platform
import shutil
import subprocess
from pathlib import Path

from rag import load_config, load_runtime, read_json


DOWNLOAD = """
import json
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

model, revision = sys.argv[1:]
snapshot_download(repo_id=model, revision=revision, local_dir='/model',
                  allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt', '*.model', '*.tiktoken'])
Path('/model/model_manifest.json').write_text(
    json.dumps({'model': model, 'revision': revision}), encoding='utf-8')
"""


def main():
    parser = argparse.ArgumentParser(description="Скачать Qwen и запустить локальный vLLM через Docker")
    parser.add_argument("action", choices=("download", "serve"))
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or not shutil.which("docker"):
        parser.error("Для локального сервера нужны Linux, Docker и NVIDIA Container Toolkit")
    config = load_config(args.config)
    runtime = load_runtime(config["runtime_path"])
    settings = config["local_llm"]
    directory = (Path(args.config).resolve().parent / settings["model_dir"]).resolve()
    image = settings["image"]
    expected = {"model": runtime["model"], "revision": runtime["revision"]}
    manifest = directory / "model_manifest.json"
    if manifest.exists() and read_json(manifest) != expected:
        parser.error("В model_dir другая ревизия: укажите новую пустую папку")

    if args.action == "download":
        directory.mkdir(parents=True, exist_ok=True)
        subprocess.run(["docker", "pull", image], check=True)
        command = ["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}",
                   "--env", "HF_HOME=/tmp/huggingface", "--env", "HF_HUB_DISABLE_TELEMETRY=1",
                   "--volume", f"{directory}:/model", "--entrypoint", "python3", image,
                   "-c", DOWNLOAD, runtime["model"], runtime["revision"]]
        subprocess.run(command, check=True)
        print(f"Веса и манифест ревизии сохранены в {directory}")
        return

    if not manifest.exists():
        parser.error("Сначала выполните local_llm.py download")
    if not (directory / "config.json").exists() or not list(directory.glob("*.safetensors")):
        parser.error("В model_dir отсутствуют config.json или веса safetensors")
    server = runtime.get("local_server", {})
    if server.get("dtype") not in ("auto", "half", "float16", "bfloat16", "float", "float32"):
        parser.error("Укажите выданный dtype в runtime.local_server")
    if "quantization" not in server or (server["quantization"] is not None and
                                       not isinstance(server["quantization"], str)):
        parser.error("Укажите выданную quantization, null означает отсутствие явного override")
    port = settings["port"]
    if type(port) is not int or not 1024 <= port <= 65535:
        parser.error("local_llm.port должен быть целым в пределах 1024..65535")
    gpu_fraction = settings["gpu_memory_utilization"]
    parallel = settings["tensor_parallel_size"]
    if not isinstance(gpu_fraction, (int, float)) or not 0 < gpu_fraction <= 1:
        parser.error("Нужно 0 < gpu_memory_utilization <= 1")
    if type(parallel) is not int or parallel < 1:
        parser.error("tensor_parallel_size должен быть положительным целым")
    model = os.environ.get("LLM_MODEL", runtime["model"])
    key = os.environ.get("LLM_API_KEY", "local-tfkp")
    command = ["docker", "run", "--rm", "--pull", "never", "--gpus", "all",
               "--network", "host", "--ipc", "host", "--volume", f"{directory}:/model:ro"]
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY",
                 "VLLM_NO_USAGE_STATS", "DO_NOT_TRACK"):
        command.extend(["--env", f"{name}=1"])
    command.extend([
        "--entrypoint", "vllm", image, "serve", "/model", "--tokenizer", "/model",
        "--served-model-name", model, "--host", "127.0.0.1", "--port", str(port),
        "--api-key", key, "--max-model-len", "8192", "--max-num-seqs", "1",
        "--gpu-memory-utilization", str(gpu_fraction), "--tensor-parallel-size", str(parallel),
        "--dtype", server["dtype"], "--kv-cache-dtype", server.get("kv_cache_dtype", "auto"),
        "--generation-config", "vllm", "--reasoning-parser", "qwen3",
        "--language-model-only", "--no-enable-prefix-caching", "--enable-request-id-headers"
    ])
    if server["quantization"] is not None:
        command.extend(["--quantization", server["quantization"]])
    print(f"Локальный endpoint: http://127.0.0.1:{port}/v1; модель: {model}", flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
