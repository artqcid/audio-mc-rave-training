import shlex
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from data_pipeline import preprocess_dataset, scan_dataset, validate_dataset_folder
from export_neutone import export_to_neutone
from train_gui import TrainingConfig, TrainingManager
from wsl_manager import (
    check_rave_installed,
    check_wsl_running,
    get_cuda_available,
    get_gpu_vram,
    get_wsl_path,
    is_wsl,
    start_wsl,
    stop_wsl,
)

BASE_DIR = Path(__file__).parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))

PREPROCESS_LOGS: List[str] = []

def append_preprocess_log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    PREPROCESS_LOGS.append(f"[{timestamp}] {message}")

MODEL_CATALOG = {
    "raspberry": {"label": "raspberry (6 GB VRAM oder mehr)", "min_vram": 0},
    "rave_mini": {"label": "rave_mini (8 GB VRAM oder mehr)", "min_vram": 8},
    "rave_small": {"label": "rave_small (12 GB VRAM oder mehr)", "min_vram": 12},
}

app = FastAPI(title="RLTA Trainer App")
training_manager = TrainingManager()


def get_available_training_models(vram: Optional[int]) -> List[dict]:
    return [{"name": model_name, "label": info["label"]} for model_name, info in MODEL_CATALOG.items()]


def list_available_model_files() -> List[str]:
    model_files = []
    search_paths = [BASE_DIR, BASE_DIR / "trained_models", BASE_DIR / "uploaded_models"]
    for base in search_paths:
        for pattern in ["*.pt", "*.pth"]:
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
    return TEMPLATES.TemplateResponse(
        "index.html",
        {
            "request": request,
            "status": training_manager.get_status(),
            "wsl_running": check_wsl_running(),
            "cuda_available": get_cuda_available(),
            "gpu_vram": gpu_vram,
            "training_models": get_available_training_models(gpu_vram),
            "model_files": list_available_model_files(),
        },
    )


@app.get("/api/status")
def api_status():
    return {
        "wsl_running": check_wsl_running(),
        "cuda_available": get_cuda_available(),
        "gpu_vram": get_gpu_vram(),
        "training_status": training_manager.get_status(),
        "rave_available": check_rave_installed(),
    }


@app.post("/api/preprocess")
def api_preprocess(
    dataset_path: str = Form("dataset"),
    target_path: str = Form("processed"),
    dataset_files: Optional[List[UploadFile]] = File(None),
):
    append_preprocess_log(f"Preprocessing gestartet: dataset={dataset_path}, target={target_path}")
    if dataset_files:
        dataset_path = save_uploaded_files(dataset_files, BASE_DIR / dataset_path)

    if not validate_dataset_folder(dataset_path):
        append_preprocess_log("Preprocessing fehlgeschlagen: Dataset-Ordner wurde nicht gefunden oder enthält keine unterstützten Audio-Dateien.")
        return JSONResponse(
            status_code=400,
            content={"error": "Dataset folder not found or contains no supported audio files. Bitte wähle einen gültigen Ordner oder lade die Dateien hoch."},
        )

    try:
        processed_dir = preprocess_dataset(dataset_path, target_path)
        processed_files = len(list(Path(processed_dir).glob("*")))
        append_preprocess_log(f"Preprocessing abgeschlossen: {processed_files} Dateien verarbeitet. Ziel: {processed_dir}")
        return {"processed_dir": processed_dir, "files": processed_files}
    except Exception as exc:
        append_preprocess_log(f"Preprocessing-Fehler: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": str(exc)},
        )


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
    )

    command = None
    if use_rave:
        rave_data_path = get_wsl_path(preprocessed_path)
        rave_output_path = get_wsl_path(config.output_path)
        if is_wsl():
            command = (
                f"bash -lc 'cd {shlex.quote(str(BASE_DIR))} && source .venv/bin/activate && rave train "
                f"--config {shlex.quote(model_name)} --data {shlex.quote(rave_data_path)} "
                f"--batch_size {batch_size} --epochs {epochs} --lr {learning_rate} "
                f"--latent_size {latent_size} "
                f"--output {shlex.quote(rave_output_path)} --device cuda'"
            )
        else:
            command = (
                f"wsl -d Ubuntu-24.04 -- bash -lc 'cd {get_wsl_path(str(BASE_DIR))} && source .venv/bin/activate && rave train "
                f"--config {shlex.quote(model_name)} --data {shlex.quote(rave_data_path)} "
                f"--batch_size {batch_size} --epochs {epochs} --lr {learning_rate} "
                f"--latent_size {latent_size} "
                f"--output {shlex.quote(rave_output_path)} --device cuda'"
            )

    job = training_manager.start_training(config, command=command)
    return {"job": job, "status": training_manager.get_status()}


@app.post("/api/export")
def api_export(
    model_path: str = Form("model.pt"),
    output_path: str = Form("exported_model.nm"),
    model_file: Optional[UploadFile] = File(None),
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

    output_file = export_to_neutone(model_path, output_path)
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


@app.post("/api/wsl/start")
def api_wsl_start():
    started = start_wsl()
    return {"started": started, "wsl_running": check_wsl_running()}


@app.post("/api/wsl/stop")
def api_wsl_stop():
    stopped = stop_wsl()
    return {"stopped": stopped, "wsl_running": check_wsl_running()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
