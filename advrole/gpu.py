from __future__ import annotations

import logging
import re
import shutil
import subprocess
from typing import List

logger = logging.getLogger(__name__)

def query_gpus() -> List[dict]:

    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            text=True, timeout=30,
        )
        gpus = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 4:
                idx, total, used, free = (int(float(p)) for p in parts)
                gpus.append({"index": idx, "total_mb": total,
                             "used_mb": used, "free_mb": free})
        if gpus:
            return gpus
    except (subprocess.SubprocessError, FileNotFoundError, ValueError):
        pass

    if shutil.which("gpustat"):
        try:
            out = subprocess.check_output(
                ["gpustat", "--no-color"], text=True, timeout=30)
            gpus = []
            for line in out.splitlines():
                m = re.search(r"\[(\d+)\].*?(\d+)\s*/\s*(\d+)\s*MiB", line)
                if m:
                    idx, used, total = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    gpus.append({"index": idx, "total_mb": total,
                                 "used_mb": used, "free_mb": total - used})
            if gpus:
                return gpus
        except (subprocess.SubprocessError, ValueError):
            pass
    raise RuntimeError("Could not query GPUs (nvidia-smi / gpustat both failed)")

def select_free_gpus(n: int, min_free_mb: int = 40000,
                     exclude: "set[int] | None" = None) -> List[int]:

    exclude = exclude or set()
    gpus = query_gpus()
    free = [g for g in gpus if g["free_mb"] >= min_free_mb and g["index"] not in exclude]
    free.sort(key=lambda g: (-g["free_mb"], g["index"]))
    chosen = [g["index"] for g in free[:n]]
    if len(chosen) < n:
        logger.warning(
            f"Only {len(chosen)}/{n} GPUs have >= {min_free_mb} MiB free "
            f"(excluding {sorted(exclude) if exclude else 'none'}). Status:\n"
            + "\n".join(f"  gpu{g['index']}: {g['free_mb']} MiB free" for g in gpus))
    else:
        logger.info(f"Selected free GPUs: {chosen} "
                    f"({[g['free_mb'] for g in free[:n]]} MiB free)")
    return chosen

def resolve_gpu_ids(spec: str, n_needed: int, min_free_mb: int = 40000,
                    exclude: "set[int] | None" = None) -> str:

    spec = str(spec or "").strip()
    if spec.startswith("auto"):
        n = n_needed
        if ":" in spec:
            n = int(spec.split(":", 1)[1])
        ids = select_free_gpus(n, min_free_mb, exclude=exclude)
        if not ids:
            raise RuntimeError("No free GPUs available on this shared machine; aborting.")
        return ",".join(map(str, ids))
    return spec

def parse_gpu_ids(spec: str) -> "set[int]":
    spec = str(spec or "").strip()
    if not spec:
        return set()
    return {int(x) for x in spec.split(",") if x.strip()}
