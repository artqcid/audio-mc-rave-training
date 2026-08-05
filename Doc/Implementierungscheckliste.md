# Implementierungscheckliste für die RLTA App

## 1. Projektbasis und Umgebung
- [x] `requirements.txt` erstellt mit FastAPI, Jinja2, Librosa, SoundFile, Torchaudio
- [x] `.vscode/tasks.json` vorhanden für WSL- und App-Start/Stop
- [x] `.venv` angelegt
- [x] `torch` separat in WSL installierbar und wird manuell ergänzt

## 2. Kernmodule implementieren
- [x] `app.py` als FastAPI-Frontend
- [x] `wsl_manager.py` für WSL-Status, Start/Stop, GPU/CUDA-Abfrage und RAVE-Erkennung
- [x] `data_pipeline.py` für Dataset-Scan und Audio-Preprocessing
- [x] `train_gui.py` für Trainingsmanagement, Live-Logs und externe WSL-Kommandos
- [x] `export_neutone.py` als Platzhalter für Neutone-Export

## 3. Frontend / UI
- [x] `templates/index.html` erstellt für Status, Training, Preprocessing und Export
- [x] Vue-Frontend zeigt RAVE-CLI-Verfügbarkeit und Trainingslogs

## 4. Tests und Validierung
- [x] `tests.py` erstellt
- [x] Syntaxprüfung für `app.py`, `train_gui.py`, `export_neutone.py` und `wsl_manager.py` durchgeführt
- [x] Backend-Funktionsprüfung für Trainingsstart im Simulator-Modus durchgeführt
- [x] Funktionsprüfung für Dataset-Scan und Export abgeschlossen
- [x] WSL-Statusprüfung durchgeführt; RAVE CLI in WSL ist installiert und funktionstüchtig

## 5. Manual Tasks und Entwicklerfluss
- [x] VS Code Tasks für Projektstart und Tests anpassen
- [x] Dokumentation für Installations- und Startvorgang ergänzt

## 6. Offene Aufgaben
- [ ] Finalen Neutone-Export implementieren
- [ ] RVC-spezifischen Workflow im Frontend klar strukturieren
- [ ] Automatisierung des WSL-Setups verbessern
- [ ] Weitere Tests für echte RAVE-Ausführung in WSL
- [ ] Export-UI auf ein einzelnes Modellwahl-Feld konsolidieren
