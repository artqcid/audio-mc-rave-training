# RLTA Trainer App

Lokale Web-GUI zur Steuerung von RAVE-Trainingsworkflows **nativ unter Windows** mit NVIDIA-GPU-Unterstützung. Die App führt Preprocessing, Training und Monitoring zusammen und kann echtes RAVE-Training direkt über die RAVE-CLI starten.

## Aktueller Status
- FastAPI-Backend mit Jinja2-Template und Vue-Frontend
- Live-Systemstatus: CUDA, VRAM, RAVE CLI-Verfügbarkeit
- Preprocessing über `data_pipeline.py`
- Trainingsmanagement über `train_gui.py` mit Simulator- und nativer RAVE-Ausführung
- Windows-Umgebungsmanagement über `windows_manager.py`
- Export: TorchScript und Neutone (.nm) Export implementiert

## Projektstruktur
- `app.py` — FastAPI-Anwendung und API-Endpunkte
- `data_pipeline.py` — Dataset-Scan und Audio-Preprocessing
- `train_gui.py` — Trainingsstatus, Log-Streaming und externe Kommandoausführung
- `windows_manager.py` — Windows-Status, GPU/CUDA-Abfrage, RAVE-CLI-Erkennung, Pfad-Management
- `export_neutone.py` — Neutone-Export (TorchScript + .nm)
- `templates/index.html` — Vue-basierte Web-GUI
- `tests.py` — Basis-Tests
- `requirements.txt` — Python-Abhängigkeiten
- `Doc/` — Projektdokumentation

## Voraussetzungen
- Windows 10/11
- NVIDIA-Treiber und CUDA installiert
- Python 3.10+ (empfohlen 3.12)
- Miniconda/Anaconda oder Standard Python
- `ffmpeg` und `ffprobe` im PATH (für RAVE Preprocessing)

## Setup (Native Windows mit venv)
1. Öffne PowerShell oder CMD im Projektverzeichnis:
   ```powershell
   cd C:\Users\marku\Documents\GitHub\artqcid\ai-projects\audio-mc-rave-training
   ```
2. Lege die virtuelle Umgebung an und installiere Anforderungen:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Installiere PyTorch mit CUDA-Unterstützung:
   ```powershell
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```
   (Passe `cu124` an deine CUDA-Version an: `cu118`, `cu121`, `cu124`, `cu126`)
4. Installiere RAVE:
   ```powershell
   pip install acids-rave
   ```
   Oder direkt von GitHub:
   ```powershell
   pip install git+https://github.com/acids-ircam/RAVE.git
   ```

## App starten
In aktivierter venv:
```powershell
cd C:\Users\marku\Documents\GitHub\artqcid\ai-projects\audio-mc-rave-training
.venv\Scripts\activate
python app.py
```
Öffne dann `http://localhost:8000` im Browser.

**Hinweis**: Die venv muss für jede Terminal-Session neu aktiviert werden (`.venv\Scripts\activate`).

## Funktionen
- Systemstatusanzeige: CUDA, GPU VRAM, RAVE CLI
- Trainingsstart mit Live-Logs
- Echtes Training über `rave train` nativ unter Windows
- Preprocessing und Export-Schnittstelle
- Training fortsetzen von Checkpoints
- Export zu TorchScript (.ts) und Neutone (.nm)

## Hinweis zur RAVE-Integration
- Echtes Training wird nur gestartet, wenn die RAVE-CLI im Windows PATH gefunden wird (`rave --version`).
- Wenn `RAVE CLI` nicht verfügbar ist, läuft die App im Simulator-Modus.
- Für Preprocessing mit `rave preprocess` werden `ffmpeg` und `ffprobe` benötigt.

## Testen
```powershell
.venv\Scripts\activate
python tests.py
```

## Weitere Dokumentation
- `Doc/Entwurfsplan_RLTA.md`
- `Doc/Implementierungscheckliste.md`
- `Doc/Implementierungsplan_audio-mc-rave-training-v2.md`
- `.github/instructions/AudioMC_RAVE_Training.instructions.md`
- `.github/prompts/AudioMC_RAVE_Training.prompt.md`
