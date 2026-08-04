# Audio MC RAVE Training — Prompt-Vorlagen

## Prompt: Projektüberblick und Zielsetzung
Du arbeitest an einer lokalen GUI-App für RAVE/RVC-Training unter WSL2 (Ubuntu-24.04) mit Fokus auf NVIDIA RTX 3060 und 6 GB VRAM. Die App muss:
- Lokales Training sicher starten und stoppen können
- Dev- vs. Prod-Modi logisch trennen
- WSL-Konfiguration automatisch erkennen und nicht bestehende WSL-Installationen beschädigen
- Modell-Assets in ein Neutone FX-kompatibles Format exportieren

Erkläre dein Vorgehen in klaren Schritten, nenne relevante Dateien (`Doc/Entwurfsplan_RLTA.md`, `.vscode/tasks.json`, `requirements.txt`) und fasse die wichtigsten Designentscheidungen zusammen.

## Prompt: WSL/VS Code Setup überprüfen
Prüfe das Projekt auf vorhandene WSL- und VS Code-Konfigurationen. Wenn du Änderungen vorschlägst, beantworte:
1. Welche Task in `.vscode/tasks.json` ist betroffen?
2. Wie sorgt die Änderung dafür, dass Ubuntu-24.04 sauber gestartet und gestoppt wird?
3. Welche Datei dokumentiert die Änderung im Projektentwurf?

## Prompt: GUI-Workflow entwerfen
Entwerfe einen klaren GUI-Workflow für:
- Datenauswahl und -vorverarbeitung
- Trainingsparameter (Modell, Batchsize, VRAM-Anpassung)
- Trainingsstart/Stop
- Neutone FX Export

Nutze die Projektsprache Deutsch, aber verwende technische Begriffe präzise.

## Prompt: Modularisieren und dokumentieren
Schlage einen Modularisierungsplan vor, der das Projekt in mindestens folgende Teile trennt:
- `wsl_manager` (WSL-Start/Stopp, GPU-Abfrage)
- `train_gui` (Benutzeroberfläche, Workflows)
- `data_pipeline` (Audio-Import, Preprocessing)
- `export_neutone` (Neutone FX Export)

Beschreibe dazu kurz, welche Tests oder Verifikation in jedem Modul sinnvoll wären.
