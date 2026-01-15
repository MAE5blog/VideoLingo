import gc
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import requests
from rich import print as rprint

from core.utils.config_utils import load_key

_SERVER_PROCESS = None


def _get_local_llm_config():
    try:
        return load_key("local_llm")
    except KeyError:
        return None


def _is_enabled(cfg):
    return bool(cfg and cfg.get("enabled", False))


def _server_base_url(cfg):
    host = cfg.get("server_host", "127.0.0.1")
    port = int(cfg.get("server_port", 8000))
    return f"http://{host}:{port}"


def _server_ready(cfg):
    base_url = _server_base_url(cfg)
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def _download_gguf(repo, filename, dest_path):
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    rprint(f"[cyan]⬇️ Downloading GGUF model from {url}[/cyan]")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _resolve_model_path(cfg):
    model_path = (cfg.get("model_path") or "").strip()
    if model_path:
        return Path(model_path)
    model_dir = cfg.get("model_dir", "./_model_cache/llm")
    model_file = (cfg.get("model_file") or "").strip()
    if not model_file:
        raise ValueError("local_llm.model_file is required")
    return Path(model_dir) / model_file


def _ensure_model(cfg):
    model_path = _resolve_model_path(cfg)
    if model_path.exists():
        return model_path
    repo = (cfg.get("model_repo") or "").strip()
    if not repo:
        raise FileNotFoundError(f"Model not found at {model_path} and model_repo is empty")
    _download_gguf(repo, model_path.name, model_path)
    return model_path


def _clear_cuda_cache():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def _build_server_cmd(cfg, model_path):
    host = cfg.get("server_host", "127.0.0.1")
    port = str(cfg.get("server_port", 8000))
    cmd = [
        sys.executable,
        "-m",
        "llama_cpp.server",
        "--model",
        str(model_path),
        "--host",
        host,
        "--port",
        port,
    ]

    api_key = (cfg.get("api_key") or "").strip()
    if api_key:
        cmd += ["--api_key", api_key]

    model_alias = (cfg.get("model_alias") or "").strip()
    if model_alias:
        cmd += ["--model_alias", model_alias]

    for key, flag in [
        ("n_gpu_layers", "--n_gpu_layers"),
        ("n_ctx", "--n_ctx"),
        ("n_threads", "--n_threads"),
        ("n_batch", "--n_batch"),
    ]:
        value = cfg.get(key)
        if value is not None and value != "":
            cmd += [flag, str(value)]

    chat_format = (cfg.get("chat_format") or "").strip()
    if chat_format:
        cmd += ["--chat_format", chat_format]

    return cmd


def start_local_llm_server():
    global _SERVER_PROCESS
    cfg = _get_local_llm_config()
    if not _is_enabled(cfg):
        return False

    if _server_ready(cfg):
        return False

    try:
        import llama_cpp  # noqa: F401
    except Exception as exc:
        raise RuntimeError("llama-cpp-python is not installed. Please install it first.") from exc

    _clear_cuda_cache()
    model_path = _ensure_model(cfg)
    cmd = _build_server_cmd(cfg, model_path)

    log_path = Path(cfg.get("log_path") or "output/log/local_llm_server.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "ab")

    rprint(f"[cyan]🚀 Starting local LLM server: {' '.join(cmd)}[/cyan]")
    _SERVER_PROCESS = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env={**os.environ},
    )

    for _ in range(180):
        if _server_ready(cfg):
            rprint("[green]✅ Local LLM server is ready[/green]")
            return True
        if _SERVER_PROCESS.poll() is not None:
            raise RuntimeError("Local LLM server exited unexpectedly. Check log for details.")
        time.sleep(1)

    raise TimeoutError("Local LLM server startup timed out. Check log for details.")


def stop_local_llm_server():
    global _SERVER_PROCESS
    if not _SERVER_PROCESS:
        return
    rprint("[cyan]🧹 Stopping local LLM server...[/cyan]")
    try:
        _SERVER_PROCESS.terminate()
        _SERVER_PROCESS.wait(timeout=15)
    except subprocess.TimeoutExpired:
        _SERVER_PROCESS.kill()
    except Exception:
        try:
            os.killpg(os.getpgid(_SERVER_PROCESS.pid), signal.SIGTERM)
        except Exception:
            pass
    finally:
        _SERVER_PROCESS = None
        _clear_cuda_cache()


@contextmanager
def local_llm_server(step_name="step"):
    cfg = _get_local_llm_config()
    if not _is_enabled(cfg):
        yield
        return

    manage_server = bool(cfg.get("manage_server", True))
    started = False
    if manage_server:
        rprint(f"[blue]🔧 Preparing local LLM for {step_name}[/blue]")
        started = start_local_llm_server()
    elif not _server_ready(cfg):
        raise RuntimeError("Local LLM server is not running. Please start it first.")

    try:
        yield
    finally:
        if manage_server and started:
            stop_local_llm_server()
