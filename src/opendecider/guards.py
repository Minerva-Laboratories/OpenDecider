"""Resource guards for the shared Jetson.

Disk: the user rule is that free space on / never drops below 4 GB. Filling the eMMC root
can corrupt it into needing a reflash, so every download or checkpoint write calls
`require_free_gb(expected_write_gb)` first.

GPU: two concurrent trainers hard-reset this Orin once. GPU jobs share the flock used by the
qwen-surgery project so they serialise with it.
"""
from __future__ import annotations

import contextlib
import fcntl
import os
import shutil
import sys
import time

DISK_FLOOR_GB = float(os.environ.get("OPENDECIDER_DISK_FLOOR_GB", 4.0))
GPU_LOCK_PATH = os.environ.get("OPENDECIDER_GPU_LOCK", "/tmp/qwen-surgery-gpu.lock")
GPU_OWNER_PATH = "/tmp/qwen-surgery-gpu.owner"


def free_gb(path: str = "/") -> float:
    return shutil.disk_usage(path).free / 2**30


def require_free_gb(expected_write_gb: float = 0.0, path: str = "/", floor: float = DISK_FLOOR_GB) -> float:
    """Refuse if writing `expected_write_gb` would leave less than `floor` GB free."""
    free = free_gb(path)
    if free - expected_write_gb < floor:
        raise RuntimeError(
            f"DISK GUARD: {free:.1f} GB free, write needs {expected_write_gb:.1f} GB, "
            f"floor is {floor:.1f} GB. Refusing.")
    return free


@contextlib.contextmanager
def gpu_lock(tag: str, wait: bool = True):
    """Hold the shared GPU flock for the duration of the block.

    Re-entrant within one process tree: if OPENDECIDER_GPU_LOCK_HELD is set (e.g. the job was
    launched through gpu_lock.sh), this is a no-op.
    """
    if os.environ.get("OPENDECIDER_GPU_LOCK_HELD"):
        yield
        return
    fd = os.open(GPU_LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if not wait:
                raise RuntimeError("GPU busy: " + _owner())
            print(f"[gpu_lock] GPU busy, waiting: {_owner()}", file=sys.stderr)
            fcntl.flock(fd, fcntl.LOCK_EX)
        with open(GPU_OWNER_PATH, "w") as f:
            f.write(f"{tag} pid {os.getpid()} since {time.strftime('%H:%M')}\n")
        os.environ["OPENDECIDER_GPU_LOCK_HELD"] = "1"
        try:
            yield
        finally:
            os.environ.pop("OPENDECIDER_GPU_LOCK_HELD", None)
            with contextlib.suppress(FileNotFoundError):
                os.remove(GPU_OWNER_PATH)
    finally:
        os.close(fd)


def _owner() -> str:
    try:
        return open(GPU_OWNER_PATH).read().strip()
    except OSError:
        return "unknown"


def limit_gpu_memory(max_gb: float) -> None:
    """Hard cap on the CUDA caching allocator. On the Orin, GPU memory IS system RAM (unified),
    so an unbounded run can starve the OS and hang the board; with a cap it raises OOM instead."""
    import torch
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, max_gb / total))
