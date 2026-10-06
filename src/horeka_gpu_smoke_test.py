"""Dependency-free smoke test for a Slurm GPU allocation on HoreKa."""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import sys


def require_environment_variable(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Required environment variable {name} is not set")
    return value


def query_nvidia_smi() -> None:
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi is not available on PATH")

    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout.strip()
    if not output:
        raise RuntimeError("nvidia-smi did not report a GPU")
    print("nvidia-smi GPU report:")
    print(output)


def query_cuda_driver() -> None:
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
    except OSError as error:
        raise RuntimeError("Could not load the NVIDIA CUDA driver") from error

    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuInit.restype = ctypes.c_int
    cuda.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cuda.cuDeviceGetCount.restype = ctypes.c_int
    cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cuda.cuDeviceGet.restype = ctypes.c_int
    cuda.cuDeviceGetName.argtypes = [
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_int,
        ctypes.c_int,
    ]
    cuda.cuDeviceGetName.restype = ctypes.c_int

    result = cuda.cuInit(0)
    if result != 0:
        raise RuntimeError(f"cuInit failed with CUDA error code {result}")

    count = ctypes.c_int()
    result = cuda.cuDeviceGetCount(ctypes.byref(count))
    if result != 0:
        raise RuntimeError(f"cuDeviceGetCount failed with CUDA error code {result}")
    if count.value < 1:
        raise RuntimeError("The CUDA driver reported zero visible GPUs")

    print(f"CUDA driver visible device count: {count.value}")
    for index in range(count.value):
        device = ctypes.c_int()
        result = cuda.cuDeviceGet(ctypes.byref(device), index)
        if result != 0:
            raise RuntimeError(f"cuDeviceGet({index}) failed with CUDA error code {result}")

        name = ctypes.create_string_buffer(256)
        result = cuda.cuDeviceGetName(name, len(name), device.value)
        if result != 0:
            raise RuntimeError(
                f"cuDeviceGetName({index}) failed with CUDA error code {result}"
            )
        print(f"CUDA device {index}: {name.value.decode('utf-8')}")


def test_torch_if_installed() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        print("PyTorch is not installed yet; skipping the optional tensor test.")
        return

    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA build: {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch is installed but torch.cuda.is_available() is false")

    tensor = torch.arange(1024, device="cuda", dtype=torch.float32)
    result = tensor.square().sum().item()
    torch.cuda.synchronize()
    print(f"PyTorch tensor test result: {result}")


def main() -> None:
    job_id = require_environment_variable("SLURM_JOB_ID")
    visible_devices = require_environment_variable("CUDA_VISIBLE_DEVICES")

    print("HoreKa GPU smoke test")
    print(f"Slurm job ID: {job_id}")
    print(f"Hostname: {platform.node()}")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"CUDA_VISIBLE_DEVICES: {visible_devices}")

    query_nvidia_smi()
    query_cuda_driver()
    test_torch_if_installed()
    print("HOREKA GPU SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
