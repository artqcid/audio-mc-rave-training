# Implementierungscheckliste für die RLTA App

## 1. Projektbasis und Umgebung
- [x] `requirements.txt` erstellt mit FastAPI, Jinja2, Librosa, SoundFile, Torchaudio
- [x] `.vscode/tasks.json` vorhanden für App-Start/Stop
- [x] `.venv` angelegt

## 2. Kernmodule implementieren
- [x] `app.py` als FastAPI-Frontend
- [x] `windows_manager.py` für GPU/CUDA-Abfrage und RAVE-Erkennung
- [x] `data_pipeline.py` für Dataset-Scan und Audio-Preprocessing
- [x] `train_gui.py` für Trainingsmanagement, Live-Logs
- [x] `export_neutone.py` als Platzhalter für Neutone-Export

## 3. Frontend / UI
- [x] `templates/index.html` erstellt für Status, Training, Preprocessing und Export
- [x] Vue-Frontend zeigt RAVE-CLI-Verfügbarkeit und Trainingslogs

## 4. Tests und Validierung
- [x] `tests.py` erstellt
- [x] Syntaxprüfung für `app.py`, `train_gui.py`, `export_neutone.py` und `windows_manager.py` durchgeführt
- [x] Backend-Funktionsprüfung für Trainingsstart im Simulator-Modus durchgeführt
- [x] Funktionsprüfung für Dataset-Scan und Export abgeschlossen
- [x] RAVE-Statusprüfung durchgeführt; RAVE CLI ist installiert und funktionstüchtig

## 5. Manual Tasks und Entwicklerfluss
- [x] VS Code Tasks für Projektstart und Tests anpassen
- [x] Dokumentation für Installations- und Startvorgang ergänzt

## 7. RAVE-Konfiguration implementiert
- [x] RAVE_PRESETS mit allen erforderlichen Parametern definiert
- [x] TrainingConfig erweitert um RAVE-spezifische Parameter
- [x] app.py aktualisiert um RAVE-Konfigurationen korrekt zu übergeben
- [x] Tests für RAVE-Konfiguration hinzugefügt
- [x] Model-Katalog mit VRAM-Anforderungen aktualisiert

## 8. Neutone-Export implementiert
- [x] Vollständige RAVE-Modellexportfunktionalität
- [x] TorchScript-Export mit Streaming-Unterstützung
- [x] Neutone-Metadaten-Extraktion
- [x] Validierung von Neutone-Paketen

## 9. Export-UI konsolidiert
- [x] Export-Modus-Wahl (TorchScript vs Neutone)
- [x] Unterstützung für .ts und .nm Dateien
- [x] Dynamische Platzhalter-Texte basierend auf Export-Typ
- [x] Workflow: Training erzeugt .ts, Export konvertiert .ts zu .nm
- [x] .ts Dateien werden in Modell-Liste angezeigt

## 10. Training-Fortsetzen implementiert
- [x] Trainingsmodus-Auswahl (Neu vs Fortsetzen)
- [x] Checkpoint-Auswahl für Fortsetzen
- [x] .pt/.pth Dateien in Modell-Liste angezeigt
- [x] TrainingConfig erweitert um training_mode und checkpoint_path
- [x] Backend-Logik für Checkpoint-Ladung implementiert

## Offene Aufgaben (Implementation Plans vorhanden)
- [x] **audio-mc-rave-training-v2 Migration** (Native Windows) → `Doc/Implementierungsplan_audio-mc-rave-training-v2.md`
- [ ] **RVC-Implementierung** (Voice Conversion Workflow) → `Doc/RVC-Implementierung.md`
- [ ] **RVC-Workflow-Analyse** (Applio-ähnlicher Workflow) → `Doc/RVC-Workflow-Analyse.md`
- [ ] **Portable Setup & Installer** (Desktop App + Setup.exe) → `Doc/Portable-Setup-Implementation.md`