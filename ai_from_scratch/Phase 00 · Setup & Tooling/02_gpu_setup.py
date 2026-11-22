"""Lección: GPU Setup & Cloud.

1. Comprobar qué GPU hay disponible (CUDA en NVIDIA, MPS en Apple Silicon).
2. Elegir el dispositivo automáticamente (cuda > mps > cpu).
3. Benchmark de multiplicación de matrices CPU vs GPU (ejecutado 2 veces).
4. Estimar el modelo más grande que cabe en memoria usando fp16 (2 bytes/parámetro).
"""

import time

import torch


# ---------- 1. Comprobación de GPU ----------
def gpu_check() -> None:
    print("=== Comprobación de GPU ===")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {props.total_memory / 1e9:.1f} GB")
    print(f"MPS (Apple Silicon) available: {torch.backends.mps.is_available()}")
    print()


# ---------- 2. Selección de dispositivo ----------
def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device: str) -> None:
    """Las GPUs son asíncronas: hay que esperar a que terminen para medir bien."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


# ---------- 3. Benchmark CPU vs GPU ----------
def benchmark(device: str, size: int = 5000, run: int = 1) -> None:
    print(f"=== Benchmark (ejecución {run}) ===")
    a = torch.randn(size, size)
    b = torch.randn(size, size)

    start = time.time()
    _ = a @ b
    cpu_time = time.time() - start
    print(f"CPU: {cpu_time:.3f}s")

    if device != "cpu":
        a_gpu, b_gpu = a.to(device), b.to(device)
        sync(device)
        start = time.time()
        _ = a_gpu @ b_gpu
        sync(device)
        gpu_time = time.time() - start
        print(f"GPU ({device}): {gpu_time:.3f}s")
        print(f"Speedup: {cpu_time / gpu_time:.1f}x")
    print()


# ---------- 4. ¿Qué modelo cabe en memoria? ----------
def max_model_estimate(device: str, margin: float = 0.8) -> None:
    print("=== Estimación de tamaño de modelo (fp16) ===")
    if device == "cuda":
        mem_bytes = torch.cuda.get_device_properties(0).total_memory
        label = "VRAM"
    elif device == "mps":
        # En Apple Silicon la memoria es unificada (RAM compartida con la GPU).
        # PyTorch indica cuánta puede usar la GPU de forma recomendada.
        mem_bytes = torch.mps.recommended_max_memory()
        label = "Memoria recomendada para la GPU (unificada)"
    else:
        print("Sin GPU: no aplica.\n")
        return

    params = mem_bytes / 2  # fp16 = 2 bytes por parámetro
    safe_params = params * margin  # margen para activaciones y contexto de la GPU
    print(f"{label}: {mem_bytes / 1e9:.1f} GB")
    print(f"Máximo teórico: {params / 1e9:.1f}B parámetros")
    print(f"Con margen del {round((1 - margin) * 100)}%: ~{safe_params / 1e9:.1f}B parámetros")
    print()


if __name__ == "__main__":
    gpu_check()
    device = pick_device()
    print(f"Using: {device}\n")
    # Ejercicio 1: dos ejecuciones; la primera incluye el coste de "calentar" la GPU.
    benchmark(device, run=1)
    benchmark(device, run=2)
    max_model_estimate(device)
