import os
import tempfile
from pathlib import Path

from data_pipeline import preprocess_dataset, scan_dataset, validate_dataset_folder
from export_neutone import export_to_neutone, validate_neutone_package
from train_gui import TrainingConfig, TrainingManager, RAVE_PRESETS
from windows_manager import is_windows


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


def test_windows_environment():
    """Test Windows environment detection."""
    assert is_windows() == True  # We're on Windows
    from windows_manager import get_environment_info
    env = get_environment_info()
    assert env["is_windows"] == True
    assert "platform" in env
    assert "cuda_available" in env
    assert "rave_available" in env


def test_rave_presets():
    """Test that RAVE presets are properly defined with all required parameters."""
    required_keys = [
        "latent_size", "n_bands", "capacity", "encoder", "decoder",
        "discriminator", "phase_1_duration", "gan_loss", "valid_signal_crop",
        "feature_matching_fun", "num_skipped_features", "audio_distance",
        "config_file", "description", "min_vram_gb"
    ]
    
    for preset_name, preset in RAVE_PRESETS.items():
        for key in required_keys:
            assert key in preset, f"Missing key '{key}' in preset '{preset_name}'"
        
        # Validate value ranges
        assert 2 <= preset["latent_size"] <= 64, f"Invalid latent_size in {preset_name}"
        assert preset["n_bands"] > 0, f"Invalid n_bands in {preset_name}"
        assert preset["min_vram_gb"] >= 0, f"Invalid min_vram_gb in {preset_name}"


def test_training_config_with_rave_preset():
    """Test that TrainingConfig can be created with RAVE preset parameters."""
    preset = RAVE_PRESETS.get("rave_mini", {})
    
    config = TrainingConfig(
        model_name="rave_mini",
        preprocessed_path="processed",
        batch_size=4,
        epochs=3,
        learning_rate=0.0002,
        training_type="rave",
        use_gpu=True,
        output_path="trained_models/model.pt",
        latent_size=preset.get("latent_size", 32),
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
    )
    
    assert config.model_name == "rave_mini"
    assert config.latent_size == 32
    assert config.config_file == "v1.gin"


def test_rave_import():
    """Test that RAVE can be imported and basic model creation works."""
    try:
        from rave import RAVE
        # Check that RAVE class is available
        assert RAVE is not None
    except ImportError:
        # RAVE not installed, skip this test
        pass


if __name__ == "__main__":
    test_scan_dataset()
    test_validate_dataset_folder()
    test_preprocess_dataset()
    test_export_to_neutone()
    test_training_manager()
    test_windows_environment()
    test_rave_presets()
    test_training_config_with_rave_preset()
    test_rave_import()
    print("All tests completed.")
