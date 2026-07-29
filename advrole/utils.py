from __future__ import annotations

import glob
import logging
import os
import random
import subprocess
import sys

import numpy as np

logger = logging.getLogger(__name__)

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

def setup_logging(log_path: str | None = None, verbose: bool = True) -> None:
    handlers = [logging.StreamHandler()]
    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=handlers,
        force=True,
    )

def find_hf_checkpoint(ckpt_root: str) -> str | None:

    candidates = sorted(
        glob.glob(os.path.join(ckpt_root, "**", "actor", "huggingface", "config.json"),
                  recursive=True)
        + glob.glob(os.path.join(ckpt_root, "**", "huggingface", "config.json"),
                    recursive=True),
        key=os.path.getmtime,
    )
    if not candidates:
        return None
    return os.path.dirname(candidates[-1])

def merge_verl_fsdp_checkpoint(step_dir: str, target_dir: str | None = None) -> str:

    actor_dir = os.path.join(step_dir, "actor")
    target = target_dir or os.path.join(actor_dir, "huggingface")
    cmd = [sys.executable, "-m", "verl.model_merger", "merge",
           "--backend", "fsdp", "--local_dir", actor_dir,
           "--target_dir", target]
    logger.info(f"Merging verl checkpoint: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    if os.path.exists(os.path.join(target, "config.json")):
        return target
    return os.path.join(actor_dir, "huggingface")

def run_command(cmd: list[str], env: dict | None = None,
                log_file: str | None = None) -> int:

    logger.info(f"$ {' '.join(cmd)}")
    f = open(log_file, "a", encoding="utf-8") if log_file else None
    try:
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            if f:
                f.write(line)
                f.flush()
            logger.info(line.rstrip())
        proc.wait()
        return proc.returncode
    finally:
        if f:
            f.close()
