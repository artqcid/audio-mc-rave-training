import os
import tempfile
from pathlib import Path

from data_pipeline import preprocess_dataset, scan_dataset, validate_dataset_folder
from export_neutone import export_to_neutone, validate_neutone_package
from train_gui import TrainingConfig, TrainingManager
from wsl_manager import check_wsl_running, is_windows


def test_scan_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.wav"
        test_file.write_bytes(b"RIFF....WAVEfmt ")
        result = scan_dataset(tmpdir)
        assert isinstance(result, list)


def test_validate_dataset_folder():
    with tempfile.TemporaryDirectory() as tmpdir:
        assert not validate_dataset_folder(tmpdir)


def test_preprocess_dataset():
    import numpy as np
    import soundfile as sf

    with tempfile.TemporaryDirectory() as tmpdir:
        src = Path(tmpdir) / "test.wav"
        data = np.zeros(44100, dtype=np.float32)
        sf.write(str(src), data, 44100)
        processed_dir = Path(tmpdir) / "processed"
        result = preprocess_dataset(str(tmpdir), str(processed_dir))
        assert Path(result).exists()
        assert len(list(Path(result).glob("*.wav"))) == 1


def test_export_to_neutone():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        model_file = tmp_path / "model.pt"
        model_file.write_text("dummy model")
        output_file = export_to_neutone(str(model_file), str(tmp_path / "output.nm"))
        assert Path(output_file).exists()
        assert validate_neutone_package(output_file)


def test_training_manager():
    manager = TrainingManager()
    config = TrainingConfig(model_name="test", batch_size=1, epochs=1, learning_rate=0.001)
    result = manager.start_training(config)
    assert "message" in result
    status = manager.get_status()
    assert status["status"] in {"running", "completed", "stopped"}
    manager.stop_training()


def test_wsl_running():
    if not is_windows():
        return
    assert isinstance(check_wsl_running(), bool)


if __name__ == "__main__":
    test_scan_dataset()
    test_validate_dataset_folder()
    test_preprocess_dataset()
    test_export_to_neutone()
    test_training_manager()
    test_wsl_running()
    print("All tests completed.")
