# RLTA Trainer App

Eine lokale GUI-Anwendung zur Steuerung von RAVE/RVC-Trainingsworkflows auf WSL2 mit NVIDIA RTX-GPU-Unterstützung.

## Ziel
- Lokales Training auf WSL2 Ubuntu-24.04
- Unterstützung für RTX 3060 / 6 GB VRAM
- Preprocessing, Training, Monitoring, Export in ein Neutone FX-kompatibles Format

## Projektstruktur
- `app.py` — FastAPI-Web-Frontend
- `data_pipeline.py` — Dataset-Scan und Audio-Preprocessing
- `train_gui.py` — Trainingsstatus und Simulation
- `wsl_manager.py` — WSL-Start/Stop, GPU-/CUDA-Abfrage
- `export_neutone.py` — Neutone-Export-Platzhalter
- `templates/index.html` — einfache Web-GUI
- `tests.py` — Basis-Tests
- `requirements.txt` — Python-Abhängigkeiten
- `Doc/` — Entwurfsplan und Checkliste
- `scripts/setup_wsl_env.sh` — WSL-Umgebungssetup

## Voraussetzungen
- Windows 10/11 mit WSL2
- Ubuntu-24.04 als WSL-Distribution
- NVIDIA-Treiber und CUDA in WSL installiert
- VS Code mit der Erweiterung "Remote - WSL"

## Setup in WSL
1. Öffne ein WSL-Terminal mit Ubuntu-24.04.
2. Wechsle ins Projektverzeichnis:
   ```bash
   cd /mnt/c/Users/marku/Documents/GitHub/artqcid/ai-projects/audio-mc-rave-training
   ```
3. Führe das Setup-Skript aus:
   ```bash
   bash scripts/setup_wsl_env.sh
   ```
4. Installiere anschließend PyTorch separat für CUDA 12.4:
   ```bash
   .venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
   ```

## App starten
### Dev-Modus
In WSL:
```bash
cd /mnt/c/Users/marku/Documents/GitHub/artqcid/ai-projects/audio-mc-rave-training
source .venv/bin/activate
python app.py
```
Dann `http://localhost:8000` im Browser öffnen.

### VS Code Tasks
- `WSL: Start Ubuntu-24.04`
- `App: Start Dev`
- `App: Stop`
- `Test: Run project tests`

## Testen
```bash
python tests.py
```

## Hinweise
- `torch` wird nicht automatisch in `requirements.txt` installiert, da das korrekte CUDA-Wheel von der Systeminstallation abhängt.
- Die App enthält einen Trainings-Simulator; echtes RAVE/RVC-Training muss noch an ein WSL-Training-Skript gekoppelt werden.
- Export nach Neutone FX ist derzeit als Platzhalter implementiert.

## Weitere Dokumentation
- `Doc/Entwurfsplan_RLTA.md`
- `Doc/Implementierungscheckliste.md`
- `.github/instructions/AudioMC_RAVE_Training.instructions.md`
- `.github/prompts/AudioMC_RAVE_Training.prompt.md`
