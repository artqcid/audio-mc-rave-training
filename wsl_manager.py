import platform
import subprocess
from typing import Dict, Optional

DEFAULT_DISTRO = "Ubuntu-24.04"


def _run_command(command, capture_output=True, text=True, check=False):
    try:
        return subprocess.run(
            command,
            capture_output=capture_output,
            text=text,
            check=check,
        )
    except FileNotFoundError:
        return None


def is_windows() -> bool:
    return platform.system().lower() == "windows"


def is_wsl() -> bool:
    release = platform.uname().release.lower()
    return "microsoft" in release or "wsl" in release


def check_wsl_running(distro: str = DEFAULT_DISTRO) -> bool:
    if not is_windows():
        return False
    result = _run_command(["wsl", "-l", "-v"])
    if not result or result.returncode != 0:
        return False
    return distro in result.stdout and "Running" in result.stdout


def start_wsl(distro: str = DEFAULT_DISTRO) -> bool:
    if not is_windows():
        return False
    result = _run_command(["wsl", "-d", distro, "--", "bash", "-lc", "echo started"], check=False)
    return bool(result and result.returncode == 0)


def stop_wsl(distro: str = DEFAULT_DISTRO) -> bool:
    if not is_windows():
        return False
    result = _run_command(["wsl", "--terminate", distro], check=False)
    return bool(result and result.returncode == 0)


def get_gpu_vram() -> Optional[int]:
    if not is_windows() and not is_wsl():
        return None
    result = _run_command(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"])
    if not result or result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip().splitlines()[0])
    except (IndexError, ValueError):
        return None


def get_cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def get_environment_info() -> Dict[str, Optional[str]]:
    return {
        "platform": platform.system(),
        "is_wsl": str(is_wsl()),
        "wsl_running": str(check_wsl_running()),
        "cuda_available": str(get_cuda_available()),
        "gpu_vram": str(get_gpu_vram()),
    }
