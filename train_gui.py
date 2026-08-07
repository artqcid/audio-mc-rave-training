import shlex
import subprocess
import threading
import time
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Callable

# RAVE Model Configuration Presets
# Based on RAVE documentation: https://github.com/acids-ircam/RAVE
# These presets define the architecture parameters for different model sizes
# RAVE uses Gin configuration files - these presets map to the gin configs

RAVE_PRESETS = {
    "raspberry": {
        "config_file": "raspberry.gin",
        "latent_size": 16,
        "n_bands": 16,
        "capacity": 16,
        "input_mode": "pqmf",
        "output_mode": "pqmf",
        "encoder": "VariationalEncoder",
        "decoder": "Generator",
        "discriminator": "MultiScaleDiscriminator",
        "phase_1_duration": 100000,
        "gan_loss": "hinge",
        "valid_signal_crop": 16384,
        "feature_matching_fun": "feature_matching_l1",
        "num_skipped_features": 6,
        "audio_distance": "multiband_audio_distance",
        "description": "Mini-Modell für 6 GB VRAM, optimiert für Sound-Design",
        "min_vram_gb": 6,
    },
    "rave_mini": {
        "config_file": "v1.gin",
        "latent_size": 32,
        "n_bands": 16,
        "capacity": 32,
        "input_mode": "pqmf",
        "output_mode": "pqmf",
        "encoder": "VariationalEncoder",
        "decoder": "Generator",
        "discriminator": "MultiScaleDiscriminator",
        "phase_1_duration": 200000,
        "gan_loss": "hinge",
        "valid_signal_crop": 16384,
        "feature_matching_fun": "feature_matching_l1",
        "num_skipped_features": 6,
        "audio_distance": "multiband_audio_distance",
        "description": "Standard-Modell für 8 GB VRAM",
        "min_vram_gb": 8,
    },
    "rave_small": {
        "config_file": "v2_small.gin",
        "latent_size": 64,
        "n_bands": 16,
        "capacity": 64,
        "input_mode": "pqmf",
        "output_mode": "pqmf",
        "encoder": "EncoderV2",
        "decoder": "GeneratorV2",
        "discriminator": "MultiScaleSpectralDiscriminator",
        "phase_1_duration": 400000,
        "gan_loss": "hinge",
        "valid_signal_crop": 16384,
        "feature_matching_fun": "feature_matching_l1",
        "num_skipped_features": 6,
        "audio_distance": "multiband_audio_distance",
        "description": "Größeres Modell für 12 GB VRAM",
        "min_vram_gb": 12,
    },
    "v2_small": {
        "config_file": "v2_small.gin",
        "latent_size": 64,
        "n_bands": 16,
        "capacity": 64,
        "input_mode": "pqmf",
        "output_mode": "pqmf",
        "encoder": "EncoderV2",
        "decoder": "GeneratorV2",
        "discriminator": "MultiScaleSpectralDiscriminator",
        "phase_1_duration": 400000,
        "gan_loss": "hinge",
        "valid_signal_crop": 16384,
        "feature_matching_fun": "feature_matching_l1",
        "num_skipped_features": 6,
        "audio_distance": "multiband_audio_distance",
        "description": "v2-Modell für höhere Qualität",
        "min_vram_gb": 12,
    },
    "causal": {
        "config_file": "causal.gin",
        "latent_size": 32,
        "n_bands": 16,
        "capacity": 32,
        "input_mode": "pqmf",
        "output_mode": "pqmf",
        "encoder": "EncoderV2",
        "decoder": "GeneratorV2",
        "discriminator": "MultiScaleSpectralDiscriminator",
        "phase_1_duration": 200000,
        "gan_loss": "hinge",
        "valid_signal_crop": 16384,
        "feature_matching_fun": "feature_matching_l1",
        "num_skipped_features": 6,
        "audio_distance": "multiband_audio_distance",
        "description": "Niedrige Latenz für Live-Performance",
        "min_vram_gb": 8,
    },
    "onnx": {
        "config_file": "onnx.gin",
        "latent_size": 16,
        "n_bands": 16,
        "capacity": 32,
        "input_mode": "pqmf",
        "output_mode": "pqmf",
        "encoder": "ConvNet",
        "decoder": "ConvNet",
        "discriminator": "MultiScaleSpectralDiscriminator",
        "phase_1_duration": 100000,
        "gan_loss": "hinge",
        "valid_signal_crop": 16384,
        "feature_matching_fun": "feature_matching_l1",
        "num_skipped_features": 6,
        "audio_distance": "multiband_audio_distance",
        "description": "Leichtgewichtiges ONNX-kompatibles Modell",
        "min_vram_gb": 6,
    },
}


@dataclass
class TrainingConfig:
    model_name: str = "rave_mini"
    preprocessed_path: str = "processed"
    batch_size: int = 4
    epochs: int = 3
    learning_rate: float = 0.0002
    training_type: str = "rave"
    use_gpu: bool = True
    output_path: str = "trained_models/model.pt"
    # RAVE-specific configuration
    latent_size: int = 32
    n_bands: int = 16
    capacity: int = 32
    input_mode: str = "pqmf"
    output_mode: str = "pqmf"
    encoder: str = "VariationalEncoder"
    decoder: str = "Generator"
    discriminator: str = "MultiScaleDiscriminator"
    phase_1_duration: int = 200000
    gan_loss: str = "hinge"
    valid_signal_crop: int = 16384
    feature_matching_fun: str = "feature_matching_l1"
    num_skipped_features: int = 6
    audio_distance: str = "multiband_audio_distance"
    config_file: str = "v1.gin"
    description: str = ""
    min_vram_gb: int = 8
    # Training mode
    training_mode: str = "new"  # 'new' or 'resume'
    checkpoint_path: str = ""


@dataclass
class TrainingJob:
    config: TrainingConfig
    status: str = "queued"
    step: int = 0
    metrics: Dict[str, float] = field(default_factory=lambda: {"loss": 0.0, "accuracy": 0.0})
    logs: List[str] = field(default_factory=list)
    command: Optional[str] = None
    process: Optional[subprocess.Popen] = None
    returncode: Optional[int] = None


class TrainingManager:
    def __init__(self):
        self.current_job: Optional[TrainingJob] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def start_training(self, config: TrainingConfig, command: Optional[str] = None) -> Dict[str, object]:
        if self._thread and self._thread.is_alive():
            return {"error": "Ein Training läuft bereits."}

        self.current_job = TrainingJob(config=config, status="running", step=0, command=command)
        self._stop_event.clear()
        
        if config.training_mode == "resume":
            self._append_log("Training fortgesetzt von Checkpoint: {}".format(config.checkpoint_path))
        else:
            self._append_log("Training gestartet mit Modell: {}".format(config.model_name))

        if command:
            self._thread = threading.Thread(target=self._run_training_command, daemon=True)
        else:
            self._thread = threading.Thread(target=self._run_training_simulation, daemon=True)

        self._thread.start()
        mode = "external" if command else "simulator"
        return {"message": "Training gestartet", "config": config.__dict__, "mode": mode}

    def _append_log(self, message: str) -> None:
        if not self.current_job:
            return
        with self._lock:
            timestamp = time.strftime("%H:%M:%S")
            self.current_job.logs.append(f"[{timestamp}] {message}")

    def _run_training_simulation(self) -> None:
        if not self.current_job:
            return

        total_steps = max(1, self.current_job.config.epochs * 5)
        self._append_log(f"Beginne Training: {self.current_job.config.preprocessed_path}")

        for current_step in range(1, total_steps + 1):
            if self._stop_event.is_set():
                self.current_job.status = "stopped"
                self._append_log("Training gestoppt.")
                return

            time.sleep(0.2)
            self.current_job.step = current_step
            self.current_job.metrics["loss"] = round(1.0 / current_step, 4)
            self.current_job.metrics["accuracy"] = round(min(0.95, current_step / total_steps), 4)
            self._append_log(
                f"Schritt {current_step}/{total_steps}: loss={self.current_job.metrics['loss']}, accuracy={self.current_job.metrics['accuracy']}"
            )

        self.current_job.status = "completed"
        output = Path(self.current_job.config.output_path)
        if output.suffix == "":
            output = output / "simulated_model.pt"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as handle:
            handle.write(b"SIMULATED MODEL\n")
            handle.write(f"model_name={self.current_job.config.model_name}\n".encode("utf-8"))
        self._append_log(f"Simuliertes Modell erstellt: {output}")
        self._append_log("Training abgeschlossen.")

    def _run_training_command(self) -> None:
        if not self.current_job or not self.current_job.command:
            return

        self._append_log(f"Führe externes Training aus: {self.current_job.command}")
        process = None
        try:
            # Use shell=True for Windows command execution with proper encoding
            process = subprocess.Popen(
                self.current_job.command,
                cwd=Path(self.current_job.config.preprocessed_path).parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                shell=True,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"}
            )
            self.current_job.process = process

            for line in iter(process.stdout.readline, ""):
                if line:
                    # Normalize line endings (Windows \r\n -> \n)
                    clean_line = line.rstrip('\r\n')
                    self._append_log(clean_line)
                if self._stop_event.is_set():
                    break

            process.wait()
            self.current_job.returncode = process.returncode
            if self._stop_event.is_set():
                self.current_job.status = "stopped"
                self._append_log("Externes Training gestoppt.")
                return

            if process.returncode == 0:
                self.current_job.status = "completed"
                self._append_log("Externes Training erfolgreich abgeschlossen.")
            else:
                self.current_job.status = "failed"
                self._append_log(f"Externes Training fehlgeschlagen mit Exitcode {process.returncode}.")
        except Exception as exc:
            self.current_job.status = "failed"
            self._append_log(f"Fehler beim externen Training: {exc}")
        finally:
            if process and process.stdout:
                process.stdout.close()

    def stop_training(self) -> bool:
        if not self._thread or not self._thread.is_alive():
            return False

        self._stop_event.set()
        if self.current_job and self.current_job.process:
            try:
                self.current_job.process.terminate()
            except Exception:
                pass

        self._thread.join(timeout=5.0)
        if self.current_job:
            self.current_job.status = "stopped"
            self._append_log("Training gestoppt durch Benutzer.")
        return True

    def get_status(self) -> Dict[str, object]:
        if not self.current_job:
            return {"status": "idle"}

        status = {
            "status": self.current_job.status,
            "model_name": self.current_job.config.model_name,
            "preprocessed_path": self.current_job.config.preprocessed_path,
            "step": self.current_job.step,
            "metrics": self.current_job.metrics,
        }
        status["running"] = bool(self._thread and self._thread.is_alive())
        return status

    def get_logs(self) -> List[str]:
        if not self.current_job:
            return []
        with self._lock:
            return list(self.current_job.logs)
