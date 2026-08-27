"""Shared loading for local custom CUDA extensions."""

from __future__ import annotations

import os
import sys
from functools import cache
from pathlib import Path


@cache
def load_cuda_extension(name: str, source: Path):
    python_bin = str(Path(sys.prefix) / "bin")
    if python_bin not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = python_bin + os.pathsep + os.environ.get("PATH", "")

    from torch.utils.cpp_extension import load

    return load(
        name=name,
        sources=[str(source)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=False,
    )
