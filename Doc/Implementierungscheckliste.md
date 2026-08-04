# Implementierungscheckliste für die RLTA App

## 1. Projektbasis und Umgebung
- [x] `requirements.txt` erstellt mit FastAPI, Jinja2, Librosa, SoundFile, Torchaudio
- [x] `.vscode/tasks.json` vorhanden für WSL- und App-Start/Stop
- [x] `.venv` angelegt
- [ ] Optional: `torch` separat in WSL installieren mit korrektem CUDA-Wheel

## 2. Kernmodule implementieren
- [x] `app.py` als FastAPI-Frontend
- [x] `wsl_manager.py` für WSL-Status, Start/Stop und GPU-Abfrage
- [x] `data_pipeline.py` für Dataset-Scan und Audio-Preprocessing
- [x] `train_gui.py` für Trainingsmanagement und Status
- [x] `export_neutone.py` als Platzhalter für Neutone-Export

## 3. Frontend / UI
- [x] `templates/index.html` erstellt für Status, Training, Preprocessing und Export

## 4. Tests und Validierung
- [x] `tests.py` erstellt
- [ ] Syntaxprüfung für alle Python-Module
- [ ] Funktionsprüfung für Dataset-Scan und Export
- [ ] WSL-Statusprüfung

## 5. Manual Tasks und Entwicklerfluss
- [x] VS Code Tasks für Projektstart und Tests anpassen
- [x] Dokumentation für Installations- und Startvorgang hinzufügen

## 6. Weitere Ziele
- [ ] `README.md` mit Setup-Anleitung
- [ ] Dokumentation im `Doc/Entwurfsplan_RLTA.md` auf Agenten-/Prompt-Dateien verweisen
- [ ] Optional: WSL/Conda Setup-Skript oder Launch-Konfiguration
