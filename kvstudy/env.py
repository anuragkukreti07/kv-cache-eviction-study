import platform
import subprocess


def env_info():
    import torch
    import transformers
    info = {"torch": torch.__version__, "transformers": transformers.__version__,
            "python": platform.python_version(), "os": platform.platform()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, gpu_vram_bytes=p.total_memory,
                    gpu_capability=f"{p.major}.{p.minor}", cuda=torch.version.cuda)
        try:
            info["driver"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True, timeout=10).strip().splitlines()[0]
        except Exception:
            pass
    else:
        info.update(gpu="cpu", gpu_vram_bytes=0)
    return info


def result_env(info):
    """The small subset appended to every result line (guide: log versions + GPU)."""
    return {k: info.get(k) for k in ("gpu", "torch", "transformers")}
