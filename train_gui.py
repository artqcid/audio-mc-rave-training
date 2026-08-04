import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class TrainingConfig:
    model_name: str = "rave_mini"
    batch_size: int = 4
    epochs: int = 3
    learning_rate: float = 0.0002


@dataclass
class TrainingJob:
    config: TrainingConfig
    status: str = "queued"
    step: int = 0
    metrics: Dict[str, float] = field(default_factory=lambda: {"loss": 0.0, "accuracy": 0.0})


class TrainingManager:
    def __init__(self):
        self.current_job: Optional[TrainingJob] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start_training(self, config: TrainingConfig) -> Dict[str, object]:
        if self._thread and self._thread.is_alive():
            return {"error": "Ein Training läuft bereits."}

        self.current_job = TrainingJob(config=config, status="running", step=0)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_training, daemon=True)
        self._thread.start()
        return {"message": "Training gestartet", "config": config.__dict__}

    def _run_training(self) -> None:
        if not self.current_job:
            return

        total_steps = max(1, self.current_job.config.epochs * 5)
        for current_step in range(1, total_steps + 1):
            if self._stop_event.is_set():
                self.current_job.status = "stopped"
                return

            time.sleep(0.1)
            self.current_job.step = current_step
            self.current_job.metrics["loss"] = round(1.0 / current_step, 4)
            self.current_job.metrics["accuracy"] = round(min(0.95, current_step / total_steps), 4)

        self.current_job.status = "completed"

    def stop_training(self) -> bool:
        if not self._thread or not self._thread.is_alive():
            return False
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        if self.current_job:
            self.current_job.status = "stopped"
        return True

    def get_status(self) -> Dict[str, object]:
        if not self.current_job:
            return {"status": "idle"}

        status = {
            "status": self.current_job.status,
            "model_name": self.current_job.config.model_name,
            "step": self.current_job.step,
            "metrics": self.current_job.metrics,
        }
        if self._thread and self._thread.is_alive():
            status["running"] = True
        else:
            status["running"] = False
        return status
