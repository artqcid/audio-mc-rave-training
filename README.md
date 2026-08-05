# RLTA Trainer App

Lokale Web-GUI zur Steuerung von RAVE/RVC-Trainingsworkflows auf WSL2 mit NVIDIA-GPU-Unterstützung. Die App führt Preprocessing, Training und Monitoring zusammen und kann echtes RAVE-Training in Ubuntu-WSL starten, wenn die CLI verfügbar ist.

## Aktueller Status
- FastAPI-Backend mit Jinja2-Template und Vue-Frontend
- Live-Systemstatus: WSL, CUDA, VRAM, RAVE CLI-Verfügbarkeit
- Preprocessing über `data_pipeline.py`
- Trainingsmanagement über `train_gui.py` mit Simulator- und optionaler WSL-gestützter RAVE-Ausführung
- WSL-Pfadübersetzung und RAVE-Erkennung über `wsl_manager.py`
- Export: Platzhalter in `export_neutone.py`, keine finale Neutone-Konvertierung

## Projektstruktur
- `app.py` — FastAPI-Anwendung und API-Endpunkte
- `data_pipeline.py` — Dataset-Scan und Audio-Preprocessing
- `train_gui.py` — Trainingsstatus, Log-Streaming und externe Kommandoausführung
- `wsl_manager.py` — WSL-Status, GPU/CUDA-Abfrage, RAVE-CLI-Erkennung, Pfadkonvertierung
- `export_neutone.py` — Platzhalter für Neutone-Export
- `templates/index.html` — Vue-basierte Web-GUI
- `tests.py` — Basis-Tests
- `requirements.txt` — Python-Abhängigkeiten
- `Doc/` — Projektdokumentation
- `scripts/setup_wsl_env.sh` — Setup-Skript für WSL-Umgebung

## Voraussetzungen
- Windows 10/11 mit WSL2
- Ubuntu-24.04 als WSL-Distribution
- NVIDIA-Treiber und CUDA in WSL installiert
- Python 3.12 im Projekt-`.venv`
- `pip` im WSL-Environment verfügbar

## Setup in WSL
1. Öffne ein WSL-Terminal mit Ubuntu-24.04.
2. Wechsle ins Projektverzeichnis:
   ```bash
   cd /mnt/c/Users/marku/Documents/GitHub/artqcid/ai-projects/audio-mc-rave-training
   ```
3. Lege die virtuelle Umgebung an und installiere Anforderungen:
   ```bash
   bash scripts/setup_wsl_env.sh
   ```
4. Installiere optional `torch` für CUDA separat in WSL:
   ```bash
   .venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
   ```

## App starten
In WSL:
```bash
cd /mnt/c/Users/marku/Documents/GitHub/artqcid/ai-projects/audio-mc-rave-training
source .venv/bin/activate
python app.py
```
Öffne dann `http://localhost:8000` im Browser.

## Funktionen
- WSL-Statusanzeige mit Start/Stop-Optionen
- Anzeige von CUDA- und VRAM-Verfügbarkeit
- Erkennung der RAVE-CLI in WSL
- Trainingsstart mit Live-Logs
- Optionales externes Training über `rave train` in WSL
- Preprocessing und Export-Schnittstelle

## Hinweis zur RAVE-Integration
- Echtes Training wird nur gestartet, wenn die RAVE-CLI in WSL gefunden wird.
- Wenn `RAVE CLI` nicht verfügbar ist, läuft die App im Simulator-Modus.
- `export_neutone.py` ist aktuell noch ein Platzhalter; der finale Neutone-Export erfordert zusätzliche Implementierung.

## Testen
```bash
python tests.py
```

## Weitere Dokumentation
- `Doc/Entwurfsplan_RLTA.md`
- `Doc/Implementierungscheckliste.md`
- `.github/instructions/AudioMC_RAVE_Training.instructions.md`
- `.github/prompts/AudioMC_RAVE_Training.prompt.md`
