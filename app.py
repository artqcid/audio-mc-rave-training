from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from data_pipeline import preprocess_dataset, scan_dataset, validate_dataset_folder
from export_neutone import export_to_neutone
from train_gui import TrainingConfig, TrainingManager
from wsl_manager import check_wsl_running, get_cuda_available, get_gpu_vram, start_wsl, stop_wsl

BASE_DIR = Path(__file__).parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="RLTA Trainer App")
training_manager = TrainingManager()


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return TEMPLATES.TemplateResponse(
        "index.html",
        {
            "request": request,
            "status": training_manager.get_status(),
            "wsl_running": check_wsl_running(),
            "cuda_available": get_cuda_available(),
            "gpu_vram": get_gpu_vram(),
        },
    )


@app.get("/api/status")
def api_status():
    return {
        "wsl_running": check_wsl_running(),
        "cuda_available": get_cuda_available(),
        "gpu_vram": get_gpu_vram(),
        "training_status": training_manager.get_status(),
    }


@app.post("/api/preprocess")
def api_preprocess(
    dataset_path: str = Form(...), target_path: str = Form("processed")
):
    if not validate_dataset_folder(dataset_path):
        return JSONResponse(
            status_code=400,
            content={"error": "Dataset folder not found or contains no supported audio files."},
        )

    processed_dir = preprocess_dataset(dataset_path, target_path)
    return {"processed_dir": processed_dir, "files": len(list(Path(processed_dir).glob("*")))}


@app.post("/api/train")
def api_train(
    model_name: str = Form("rave_mini"),
    batch_size: int = Form(4),
    epochs: int = Form(3),
    learning_rate: float = Form(0.0002),
):
    config = TrainingConfig(
        model_name=model_name,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    job = training_manager.start_training(config)
    return {"job": job, "status": training_manager.get_status()}


@app.post("/api/export")
def api_export(
    model_path: str = Form("model.pt"), output_path: str = Form("exported_model.nm")
):
    output_file = export_to_neutone(model_path, output_path)
    return {"output_file": output_file}


@app.get("/api/training-status")
def api_training_status():
    return training_manager.get_status()


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
