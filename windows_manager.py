"""
Windows-native environment management for RAVE training app.
Replaces wsl_manager.py with native Windows implementations.
"""
import platform
import subprocess
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Any


def is_windows() -> bool:
    """Check if running on Windows."""
    return platform.system().lower() == "windows"


def get_gpu_vram() -> Optional[int]:
    """Get GPU VRAM in MB using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            # Return first GPU's VRAM in MB
            return int(result.stdout.strip().split('\n')[0])
    except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
        pass
    return None


def get_cuda_available() -> bool:
    """Check if CUDA is available via PyTorch in the conda environment."""
    try:
        # Use the conda environment's Python to check CUDA
        conda_python = Path(r"C:\Users\marku\.conda\envs\rave_env\python.exe")
        if conda_python.exists():
            import subprocess
            result = subprocess.run(
                [str(conda_python), "-c", "import torch; print(torch.cuda.is_available())"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip().lower() == "true"
        # Fallback to current environment
        import torch
        return torch.cuda.is_available()
    except (ImportError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_rave_installed() -> bool:
    """Check if RAVE is available via Python module and CLI executable."""
    try:
        # Check if RAVE can be imported and has the required modules
        import rave
        from scripts.main_cli import main
        
        # Also check if the CLI executable exists
        rave_exe = Path(r"C:\Users\marku\.conda\envs\rave_env\Scripts\rave.exe")
        if not rave_exe.exists():
            return False
            
        return True
    except ImportError:
        return False
    except Exception:
        return False


def get_rave_error_hint() -> str:
    """Return helpful error message when RAVE is not found."""
    return (
        "RAVE nicht gefunden. Bitte installieren Sie RAVE nativ unter Windows:\n"
        "  pip install acids-rave\n"
        "oder:\n"
        "  pip install git+https://github.com/acids-ircam/RAVE.git\n"
        "Stellen Sie sicher, dass die Installation in der aktiven venv erfolgt."
    )


def get_native_path(path: str) -> str:
    """Normalize Windows path, handle spaces with quotes."""
    p = Path(path).resolve()
    return str(p)


def get_venv_python() -> str:
    """Get the Python executable from the conda environment or virtual environment."""
    # First check for conda environment
    conda_env = Path(r"C:\Users\marku\.conda\envs\rave_env\python.exe")
    if conda_env.exists():
        return str(conda_env)
    
    # If running in a venv, sys.executable points to venv python
    venv_python = sys.executable
    # Verify it's actually a venv python
    if "venv" in venv_python.lower() or ".conda" in venv_python.lower():
        return venv_python
    # Fallback: try to find .venv in project root
    project_root = Path(__file__).parent
    venv_path = project_root / ".venv" / "Scripts" / "python.exe"
    if venv_path.exists():
        return str(venv_path)
    # Last resort: return system python
    return sys.executable


def get_environment_info() -> Dict[str, Any]:
    """Get comprehensive environment information."""
    return {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "is_windows": is_windows(),
        "cuda_available": get_cuda_available(),
        "gpu_vram_mb": get_gpu_vram(),
        "rave_available": check_rave_installed(),
        "python_executable": sys.executable,
        "venv_python": get_venv_python(),
    }


def run_native_command(command: str, cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> subprocess.Popen:
    """
    Run a command natively on Windows with proper encoding handling.
    
    Args:
        command: Command string to execute
        cwd: Working directory
        env: Additional environment variables
        
    Returns:
        subprocess.Popen object
    """
    # Prepare environment
    full_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    if env:
        full_env.update(env)
    
    return subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1,
        shell=True,
        env=full_env
    )