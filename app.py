import shlex
import time
import os
import re
import threading
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any


def windows_quote(path: str) -> str:
    """Quote a path for Windows command line.
    
    Windows uses double quotes for paths with spaces.
    This function properly escapes double quotes within the path.
    """
    if not path:
        return '""'
    # Escape any existing double quotes by doubling them
    escaped = path.replace('"', '""')
    return f'"{escaped}"'


from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from data_pipeline import preprocess_dataset, scan_dataset, validate_dataset_folder
from export_neutone import export_to_neutone
from train_gui import TrainingConfig, TrainingManager, RAVE_PRESETS
from windows_manager import (
    check_rave_installed,
    get_cuda_available,
    get_gpu_vram,
    get_native_path,
    get_environment_info,
    get_rave_error_hint,
)

BASE_DIR = Path(__file__).parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))

PREPROCESS_LOGS: List[str] = []
PREPROCESS_JOBS: Dict[str, Dict[str, Any]] = {}

def append_preprocess_log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    PREPROCESS_LOGS.append(f"[{timestamp}] {message}")

def run_preprocess_background(job_id: str, dataset_path: str, target_path: str) -> None:
    """Run preprocessing in background thread."""
    try:
        PREPROCESS_JOBS[job_id]["status"] = "running"
        append_preprocess_log(f"Preprocessing gestartet: dataset={dataset_path}, target={target_path}")
        
        processed_dir = preprocess_dataset(dataset_path, target_path, progress_callback=append_preprocess_log)
        processed_files = len(list(Path(processed_dir).glob("*")))
        
        PREPROCESS_JOBS[job_id]["status"] = "completed"
        PREPROCESS_JOBS[job_id]["result"] = {
            "processed_dir": processed_dir,
            "dataset_path": dataset_path,
            "files": processed_files
        }
        append_preprocess_log(f"Preprocessing abgeschlossen: {processed_files} Dateien verarbeitet. Ziel: {processed_dir}")
    except Exception as exc:
        PREPROCESS_JOBS[job_id]["status"] = "failed"
        PREPROCESS_JOBS[job_id]["error"] = str(exc)
        append_preprocess_log(f"Preprocessing-Fehler: {exc}")

def _pick_folder_tkinter(title: str) -> Optional[str]:
    """Run tkinter folder picker in a separate process and return the selected path."""
    import subprocess
    import sys
    
    # Run tkinter in a separate Python process to avoid blocking the main thread
    # and to ensure proper display context
    script = f'''
import tkinter as tk
from tkinter import filedialog
import sys

root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)
folder = filedialog.askdirectory(title={title!r})
root.destroy()
if folder:
    print(folder)
'''
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0 and result.stdout.strip():
            return str(Path(result.stdout.strip()).resolve())
    except Exception as e:
        print(f"Tkinter subprocess failed: {e}")
    return None


def pick_folder_native(title: str = "Ordner wählen") -> Optional[str]:
    """Open native folder picker dialog.

    Runs tkinter in a separate subprocess to avoid blocking the FastAPI worker.
    Falls back to PowerShell FolderBrowserDialog and zenity for Linux/WSL.
    """
    # Try tkinter in a separate subprocess (most reliable on Windows)
    path = _pick_folder_tkinter(title)
    if path:
        return path

    # Try Windows PowerShell FolderBrowserDialog (works from WSL)
    try:
        import subprocess
        ps_script = f'''
        Add-Type -AssemblyName System.Windows.Forms
        $folderBrowser = New-Object System.Windows.Forms.FolderBrowserDialog
        $folderBrowser.Description = "{title}"
        $folderBrowser.ShowNewFolderButton = $true
        $result = $folderBrowser.ShowDialog()
        if ($result -eq "OK") {{
            Write-Output $folderBrowser.SelectedPath
        }}
        '''
        result = subprocess.run(
            ["powershell.exe", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            path = result.stdout.strip()
            # Convert Windows path (C:\...) to WSL path (/mnt/c/...)
            if re.match(r'^[A-Za-z]:\\\\', path):
                # Windows path detected - convert to WSL path
                drive = path[0].lower()
                rest = path[2:].replace('\\\\', '/')
                wsl_path = f"/mnt/{drive}/{rest}"
                return str(Path(wsl_path).resolve())
            # Already a WSL path or other format
            return str(Path(path).resolve())
    except Exception as e:
        print(f"PowerShell folder picker failed: {e}")

    # Try zenity (Linux/WSL with GUI)
    try:
        import subprocess
        result = subprocess.run(
            ["zenity", "--file-selection", "--directory", f"--title={title}"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            return str(Path(result.stdout.strip()).resolve())
    except Exception as e:
        print(f"Zenity folder picker failed: {e}")

    return None

# Import RAVE_PRESETS from train_gui for model catalog
MODEL_CATALOG = {
    name: {
        "label": preset.get("description", name),
        "min_vram": preset.get("min_vram_gb", 0),
        "config_file": preset.get("config_file", "v1.gin"),
        "latent_size": preset.get("latent_size", 32),
        "n_bands": preset.get("n_bands", 16),
        "capacity": preset.get("capacity", 32),
        "encoder": preset.get("encoder", "VariationalEncoder"),
        "decoder": preset.get("decoder", "Generator"),
        "discriminator": preset.get("discriminator", "MultiScaleDiscriminator"),
    }
    for name, preset in RAVE_PRESETS.items()
}

app = FastAPI(title="RLTA Trainer App")
training_manager = TrainingManager()


def get_available_training_models(vram: Optional[int]) -> List[dict]:
    return [{"name": model_name, "label": info["label"]} for model_name, info in MODEL_CATALOG.items()]


def list_available_model_files() -> List[str]:
    model_files = []
    search_paths = [BASE_DIR, BASE_DIR / "trained_models", BASE_DIR / "uploaded_models"]
    for base in search_paths:
        for pattern in ["*.pt", "*.pth", "*.ts"]:
            for path in sorted(base.glob(pattern)):
                if base == BASE_DIR:
                    model_files.append(str(path.name))
                else:
                    model_files.append(str(path.relative_to(BASE_DIR)))
    return model_files


def save_uploaded_files(files: List[UploadFile], dest_dir: Path) -> str:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for upload in files:
        relative_path = Path(upload.filename)
        target_path = dest_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("wb") as handle:
            handle.write(upload.file.read())
    return str(dest_dir.resolve())


@app.get("/api/models")
def api_models():
    return {
        "training_models": get_available_training_models(get_gpu_vram()),
        "model_files": list_available_model_files(),
        "rave_available": check_rave_installed(),
    }


@app.get("/api/training-logs")
def api_training_logs():
    return {"logs": training_manager.get_logs()}


@app.get("/api/preprocess-logs")
def api_preprocess_logs():
    return {"logs": list(PREPROCESS_LOGS)}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    gpu_vram = get_gpu_vram()
    env_info = get_environment_info()
    training_status = training_manager.get_status()
    return TEMPLATES.TemplateResponse(
        request,
        "index.html",
        {
            "request": request,
            "training_status": training_status.get("status", "idle"),
            "training_running": training_status.get("running", False),
            "cuda_available": get_cuda_available(),
            "gpu_vram": gpu_vram,
            "rave_available": check_rave_installed(),
            "platform": env_info.get("platform", "Windows"),
            "training_models": get_available_training_models(gpu_vram),
            "model_files": list_available_model_files(),
        },
    )


@app.get("/api/status")
def api_status():
    return {
        "cuda_available": get_cuda_available(),
        "gpu_vram": get_gpu_vram(),
        "training_status": training_manager.get_status(),
        "rave_available": check_rave_installed(),
        "environment": get_environment_info(),
    }


@app.get("/api/browse-folder")
def api_browse_folder(title: str = "Ordner wählen"):
    """Open native folder picker dialog and return absolute path."""
    path = pick_folder_native(title)
    if path:
        return {"path": path}
    return JSONResponse(
        status_code=400,
        content={"error": "Kein Ordner ausgewählt oder Dialog fehlgeschlagen."},
    )


@app.post("/api/preprocess")
def api_preprocess(
    dataset_path: str = Form("dataset"),
    target_path: str = Form("processed"),
    dataset_files: Optional[List[UploadFile]] = File(None),
):
    # Use absolute paths directly - no more relative path resolution
    dataset_path_obj = Path(dataset_path)
    if not dataset_path_obj.is_absolute():
        return JSONResponse(
            status_code=400,
            content={"error": "Dataset-Pfad muss absolut sein. Bitte nutzen Sie den 'Ordner wählen' Button."},
        )
    dataset_path = str(dataset_path_obj.resolve())

    target_path_obj = Path(target_path)
    if not target_path_obj.is_absolute():
        return JSONResponse(
            status_code=400,
            content={"error": "Ziel-Pfad muss absolut sein. Bitte nutzen Sie den 'Ordner wählen' Button."},
        )
    target_path = str(target_path_obj.resolve())

    if dataset_files:
        # Save uploaded files to the dataset_path directory
        dataset_path = save_uploaded_files(dataset_files, dataset_path_obj)

    if not validate_dataset_folder(dataset_path):
        append_preprocess_log("Preprocessing fehlgeschlagen: Dataset-Ordner wurde nicht gefunden oder enthält keine unterstützten Audio-Dateien.")
        return JSONResponse(
            status_code=400,
            content={"error": "Dataset folder not found or contains no supported audio files. Bitte wähle einen gültigen Ordner oder lade die Dateien hoch."},
        )

    # Start preprocessing in background thread
    job_id = str(uuid.uuid4())
    PREPROCESS_JOBS[job_id] = {"status": "pending", "result": None, "error": None}
    
    thread = threading.Thread(target=run_preprocess_background, args=(job_id, dataset_path, target_path))
    thread.daemon = True
    thread.start()
    
    return {"job_id": job_id, "status": "started"}


@app.get("/api/preprocess-status/{job_id}")
def api_preprocess_status(job_id: str):
    """Get status of preprocessing job."""
    if job_id not in PREPROCESS_JOBS:
        return JSONResponse(
            status_code=404,
            content={"error": "Job nicht gefunden."},
        )
    job = PREPROCESS_JOBS[job_id]
    if job["status"] == "completed":
        return {"status": "completed", "result": job["result"]}
    elif job["status"] == "failed":
        return JSONResponse(
            status_code=500,
            content={"status": "failed", "error": job["error"]},
        )
    else:
        return {"status": job["status"]}


@app.post("/api/train")
def api_train(
    model_name: str = Form(""),
    preprocessed_path: str = Form("processed"),
    model_output_path: str = Form("trained_models"),
    batch_size: int = Form(4),
    epochs: int = Form(3),
    learning_rate: float = Form(0.0002),
    use_rave: bool = Form(False),
    latent_size: int = Form(16),
    training_mode: str = Form("new"),  # 'new' or 'resume'
    checkpoint_path: str = Form(""),
):
    if not model_name:
        return JSONResponse(
            status_code=400,
            content={"error": "Bitte wählen Sie ein Trainingsmodell aus."},
        )

    if not validate_dataset_folder(preprocessed_path):
        return JSONResponse(
            status_code=400,
            content={"error": "Bitte führen Sie zuerst Preprocessing aus oder geben Sie einen gültigen vorverarbeiteten Ordner an."},
        )

    # Validate checkpoint for resume mode
    if training_mode == "resume" and not checkpoint_path:
        return JSONResponse(
            status_code=400,
            content={"error": "Bitte wählen Sie einen Checkpoint zum Fortsetzen aus."},
        )

    if use_rave and not check_rave_installed():
        return JSONResponse(
            status_code=400,
            content={"error": "RAVE CLI ist nicht verfügbar. Bitte installiere RAVE oder deaktiviere echtes RAVE Training."},
        )

    if not model_output_path.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "Bitte gib einen Speicherort für das trainierte Modell an."},
        )

    if latent_size < 2 or latent_size > 64:
        return JSONResponse(
            status_code=400,
            content={"error": "Latent Size muss zwischen 2 und 64 liegen."},
        )

    # Get RAVE preset configuration
    preset = RAVE_PRESETS.get(model_name, RAVE_PRESETS.get("rave_mini", {}))
    
    config = TrainingConfig(
        model_name=model_name,
        preprocessed_path=preprocessed_path,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        training_type="rave" if use_rave else "simulator",
        use_gpu=use_rave,
        output_path=str(BASE_DIR / model_output_path),
        latent_size=latent_size,
        n_bands=preset.get("n_bands", 16),
        capacity=preset.get("capacity", 32),
        encoder=preset.get("encoder", "VariationalEncoder"),
        decoder=preset.get("decoder", "Generator"),
        discriminator=preset.get("discriminator", "MultiScaleDiscriminator"),
        phase_1_duration=preset.get("phase_1_duration", 200000),
        gan_loss=preset.get("gan_loss", "hinge"),
        valid_signal_crop=preset.get("valid_signal_crop", 16384),
        feature_matching_fun=preset.get("feature_matching_fun", "feature_matching_l1"),
        num_skipped_features=preset.get("num_skipped_features", 6),
        audio_distance=preset.get("audio_distance", "multiband_audio_distance"),
        config_file=preset.get("config_file", "v1.gin"),
        description=preset.get("description", ""),
        min_vram_gb=preset.get("min_vram_gb", 8),
        training_mode=training_mode,
        checkpoint_path=checkpoint_path,
    )

    # Build training command for RAVE
    command = None
    if use_rave:
        rave_data_path = get_native_path(preprocessed_path)
        rave_output_path = get_native_path(config.output_path)
        config_file = preset.get("config_file", "v1.gin")
        checkpoint_arg = ""
        if training_mode == "resume" and checkpoint_path:
            checkpoint_arg = f"--ckpt {windows_quote(get_native_path(checkpoint_path))}"
        
        # Use full path to rave.exe in conda environment
        rave_exe = r"C:\Users\marku\.conda\envs\rave_env\Scripts\rave.exe"
        command = (
            f'{windows_quote(rave_exe)} train '
            f'--name {windows_quote(model_name)} '
            f'--db_path {windows_quote(rave_data_path)} '
            f'--out_path {windows_quote(rave_output_path)} '
            f'--config {config_file} '
            f'{checkpoint_arg}'
        ).strip()

    job = training_manager.start_training(config, command=command)
    return {"job": job, "status": training_manager.get_status()}


@app.post("/api/export")
def api_export(
    model_path: str = Form("model.pt"),
    output_path: str = Form("exported_model.nm"),
    model_file: Optional[UploadFile] = File(None),
    export_type: str = Form("neutone"),  # 'torchscript' or 'neutone'
):
    if model_file:
        saved_dir = save_uploaded_files([model_file], BASE_DIR / "uploaded_models")
        model_path = str(Path(saved_dir) / Path(model_file.filename).name)
    else:
        model_path_obj = Path(model_path)
        if not model_path_obj.is_absolute():
            model_path_obj = BASE_DIR / model_path_obj
        model_path = str(model_path_obj)

    output_path_obj = Path(output_path)
    if not output_path_obj.is_absolute():
        output_path_obj = BASE_DIR / output_path_obj
    output_path = str(output_path_obj)

    # Adjust output extension based on export type
    if export_type == "torchscript":
        if output_path_obj.suffix != ".ts":
            output_path_obj = output_path_obj.with_suffix(".ts")
            output_path = str(output_path_obj)
    else:  # neutone
        if output_path_obj.suffix != ".nm":
            output_path_obj = output_path_obj.with_suffix(".nm")
            output_path = str(output_path_obj)

    output_file = export_to_neutone(model_path, output_path, export_type=export_type)
    return {"output_file": output_file}


@app.get("/api/training-status")
def api_training_status():
    return training_manager.get_status()


@app.post("/api/train/stop")
def api_train_stop():
    stopped = training_manager.stop_training()
    if not stopped:
        return JSONResponse(
            status_code=400,
            content={"error": "Kein aktives Training zum Stoppen gefunden."},
        )
    return {"stopped": True, "status": training_manager.get_status()}
