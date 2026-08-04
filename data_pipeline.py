import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List

import librosa
import numpy as np
import soundfile as sf

SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3"}


@dataclass
class AudioFileInfo:
    path: str
    duration: float
    samplerate: int
    channels: int


def is_audio_file(path: str) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def scan_dataset(dataset_path: str) -> List[AudioFileInfo]:
    root = Path(dataset_path)
    if not root.exists() or not root.is_dir():
        return []

    items: List[AudioFileInfo] = []
    for file_path in root.rglob("*"):
        if file_path.is_file() and is_audio_file(str(file_path)):
            try:
                info = sf.info(str(file_path))
                items.append(
                    AudioFileInfo(
                        path=str(file_path),
                        duration=float(info.duration),
                        samplerate=int(info.samplerate),
                        channels=int(info.channels),
                    )
                )
            except Exception:
                continue
    return items


def validate_dataset_folder(dataset_path: str) -> bool:
    return len(scan_dataset(dataset_path)) > 0


def preprocess_audio_file(source_path: str, target_dir: str, sample_rate: int = 44100) -> str:
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    source_path_obj = Path(source_path)
    target_path = target_dir_path / f"{source_path_obj.stem}.wav"

    y, sr = librosa.load(str(source_path_obj), sr=sample_rate, mono=True)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    sf.write(str(target_path), y, sample_rate, subtype="PCM_16")
    return str(target_path)


def preprocess_dataset(dataset_path: str, target_path: str = "processed") -> str:
    files = scan_dataset(dataset_path)
    if not files:
        raise ValueError("Dataset folder contains no supported audio files.")

    target_folder = Path(target_path)
    target_folder.mkdir(parents=True, exist_ok=True)
    processed_files = []
    for audio in files:
        try:
            processed_files.append(preprocess_audio_file(audio.path, target_path))
        except Exception:
            continue
    if not processed_files:
        raise RuntimeError("Preprocessing konnte keine Datei verarbeiten.")
    return str(target_folder.resolve())
