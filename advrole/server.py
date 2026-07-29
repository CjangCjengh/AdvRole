from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

def wait_for_endpoint(api_base: str, timeout: float = 900.0,
                      api_key: str = "EMPTY") -> bool:
    from openai import OpenAI
    client = OpenAI(base_url=api_base, api_key=api_key, timeout=10.0)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            client.models.list()
            return True
        except Exception:
            time.sleep(5.0)
    return False

class VLLMServer:
    def __init__(self, name: str, model_path: str, port: int,
                 gpu_ids: str = "", tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.85,
                 max_model_len: Optional[int] = None,
                 extra_args: Optional[List[str]] = None,
                 log_dir: str = "logs",
                 python_executable: Optional[str] = None):
        self.name = name
        self.model_path = model_path
        self.port = port
        self.gpu_ids = gpu_ids
        self.tp = tensor_parallel_size
        self.gmu = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.extra_args = extra_args or []
        self.log_dir = log_dir
        self.python = python_executable or "python"
        self.proc: Optional[subprocess.Popen] = None

    @property
    def api_base(self) -> str:
        return f"http://localhost:{self.port}/v1"

    def start(self, wait_timeout: float = 1200.0) -> "VLLMServer":
        os.makedirs(self.log_dir, exist_ok=True)
        cmd = [
            self.python, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--served-model-name", self.name,
            "--port", str(self.port),
            "--tensor-parallel-size", str(self.tp),
            "--gpu-memory-utilization", str(self.gmu),
            "--trust-remote-code",
            "--disable-log-requests",
        ]
        if self.max_model_len:
            cmd += ["--max-model-len", str(self.max_model_len)]
        cmd += self.extra_args

        env = dict(os.environ)
        if self.gpu_ids:
            env["CUDA_VISIBLE_DEVICES"] = self.gpu_ids

        log_path = os.path.join(self.log_dir, f"server_{self.name}.log")
        logger.info(f"Starting vLLM server '{self.name}' (gpus={self.gpu_ids or 'all'}, "
                    f"port={self.port}); log: {log_path}")
        with open(log_path, "w", encoding="utf-8") as logf:
            self.proc = subprocess.Popen(cmd, env=env, stdout=logf,
                                         stderr=subprocess.STDOUT,
                                         start_new_session=True)
        if not wait_for_endpoint(self.api_base, wait_timeout):
            self.stop()
            raise RuntimeError(
                f"vLLM server '{self.name}' did not become ready within "
                f"{wait_timeout}s; see {log_path}")
        logger.info(f"Server '{self.name}' ready at {self.api_base}")
        return self

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        self.proc = None
        logger.info(f"Server '{self.name}' stopped")

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

def server_from_config(name: str, cfg: Dict, log_dir: str = "logs") -> VLLMServer:
    return VLLMServer(
        name=name,
        model_path=cfg["model"],
        port=int(cfg.get("port", 8000)),
        gpu_ids=str(cfg.get("gpu_ids", "")),
        tensor_parallel_size=int(cfg.get("tensor_parallel_size", 1)),
        gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.85)),
        max_model_len=cfg.get("max_model_len"),
        extra_args=[str(a) for a in cfg.get("extra_args", [])],
        log_dir=log_dir,
        python_executable=cfg.get("python"),
    )
