# Entwurfsplan: RAVE Local Trainer App (RLTA)

Dieser Entwurfsplan beschreibt die Architektur und die Module eines Coding Agent, der eine lokale GUI-Anwendung für das Training von Modellen erstellt. Die App bietet auf oberster Ebene eine Modellwahl zwischen:

- RAVE-Modellen für Sound-Design und Audio-Transformation
- RVC-Sprachmodellen (ähnlich Applio) für eigene Stimmen

Unabhängig von der Wahl soll am Ende ein Modell entstehen, das in einem Neutone FX-kompatiblen Exportprozess bereitgestellt werden kann. Die App wird eine visuelle Oberfläche bieten, um Modellwahl, Preprocessing, Training, Monitoring und Export zu steuern. Der Fokus liegt auf einem 6 GB VRAM Setup (RTX 3060 Laptop) und dem Neutone FX VST-Export.

## 0. Hauptentscheidungsebene: RAVE vs. Sprachmodell

- Oberflächen-Logik:
  - Der Nutzer wählt beim Start der App entweder den RAVE-Workflow oder den RVC-Sprachmodell-Workflow.
  - Die App zeigt jeweils eine angepasste GUI mit passenden Feldern, Tipps und Warnungen.
- RAVE-Workflow:
  - Orientierung auf Instrumental- und Sound-Design-Modelle.
  - Beschleunigte Auswahl von `raspberry`, `onnx` und klassischen RAVE-Konfigurationen.
- RVC-Sprachmodell-Workflow:
  - Orientierung auf Voice-Datasets und sprachspezifische Feature-Extraktion.
  - Nachrüsten eines Neutone-kompatiblen Exportpfads, damit das Ergebnis weiterhin in Neutone FX verwendet werden kann.

## 1. Modul: Hardware- & Systemanalyse

- Betriebssystem-Check:
  - Identifiziere Windows (WSL2 empfohlen für volle CUDA-Unterstützung), macOS (M1/M2/M3 Optimierung via MPS) oder Linux.
- GPU-Diagnose:
  - Abfrage der NVIDIA-Treiber, CUDA-Verfügbarkeit und des verfügbaren VRAMs.
- Agent-Logik:
  - Wenn VRAM < 8 GB, Ausgabe einer klaren Warnung.
  - Für komplexe v3-Modelle sind mindestens 32 GB VRAM erforderlich.
- Speicherplatz-Monitor:
  - Sicherstellen, dass genügend SSD-Kapazität für die LMDB-Datenbanken vorhanden ist (oft 10–70 GB je nach Dataset).

## 2. Modul: Automatisierte Umgebungseinrichtung

- Miniconda/Conda-Integration:
  - Automatischer Download und Installation von Miniconda, falls nicht vorhanden.
- Environment Creator:
  - Ausführung von `conda create -n rave_env python=3.9 -y`.
- PyTorch-Selektor:
  - Für Windows/Linux: Installation mit CUDA-Unterstützung (z.B. cu118 oder cu124).
  - Für Mac: Installation der Standard-Version mit MPS-Support.
- RAVE-Kern-Installation:
  - Klonen des offiziellen Repositories und Installation via `pip install -e .` oder `pip install acids-rave`.

## 3. Modul: Daten-Engineering (Preprocessing)

- Dataset-Validierung:
  - Prüfen, ob Audio-Dateien mindestens 2,5 Sekunden lang sind, um Preprocessing-Fehler zu vermeiden.
- Normalisierungs-Workflow:
  - Optionale Peak-Normalisierung oder Dynamikkompression, da RAVE lautere Komponenten beim Training priorisiert.
- Preprocessing-Kommando:
  - Automatisierte Ausführung von `rave preprocess` mit Parametern für Kanäle (Mono für VST-Kompatibilität) und Sampling-Rate (Standard 44,1 kHz).

## 4. Modul: Trainings-Management

- GUI-Trainings-Dashboard:
  - Eine zentrale Oberfläche zur Auswahl von Dataset, Modellkonfigurationen, Batch-Größe und Trainingsparametern.
  - Visuelle Darstellung des aktuellen Trainingsstatus mit Fortschrittsbalken, Metriken und OOM-Warnungen.
- Konfigurations-Interface:
  - `v1`: Basismodell (8 GB VRAM).
  - `v2`: Höhere Qualität, mehr Regularisierungsoptionen (16 GB VRAM).
  - `causal`: Reduziert Latenz für Live-Performance, mindert aber die Rekonstruktionsqualität.
  - `raspberry` / `onnx`: Leichtgewichtige Mini-Modelle für 6 GB VRAM.
- Phasen-Monitoring:
  - Phase 1 (Repräsentationslernen): Fokus auf spektrale Distanz (ca. 1 Mio. Schritte).
  - Phase 2 (Adversarial Fine-Tuning): Aktivierung des Diskriminators für Realismus und Transienten-Schärfe.
  - Zeitleistenansicht mit Status, Dauer und verbleibenden Schritten.
- Tensorboard-Integration:
  - Einbetten eines Viewers zur Überwachung der Verlustkurven und Audio-Samples in Echtzeit.
  - Alternativ: Stream von TensorBoard-Daten an ein eingebettetes Browser-Panel.
- Checkpoint-Steuerung:
  - Automatisches Speichern (`--save_every`) und Wiederaufnahme bei Abbruch (`--ckpt`).
  - GUI-Schaltflächen: "Pause", "Fortsetzen", "Neustart" und "Letzten Checkpoint laden".

## 5. Modul: Modell-Export & Validierung

- TorchScript-Konvertierung:
  - Ausführung von `rave export` mit dem essenziellen `--streaming` Flag, um Klickgeräusche in Echtzeit zu vermeiden.
- Fidelity-Tuning:
  - Analyse der latenten Dimensionen mittels SVD, um das Modell auf die informativsten Kanäle zu komprimieren.
- Plugin-Deployment:
  - Verschieben der `.ts`-Datei in den Zielordner für das `nn~` External oder den RAVE VST.

## Technische Empfehlungen für den Agenten

- GUI-Interaktion:
  - Der Benutzer soll die Umgebungseinrichtung, Dataset-Zuweisung und das Training über intuitive Steuerelemente starten können.
  - Warnmeldungen, Tooltipps und Status-Benachrichtigungen sollen das Training in Echtzeit begleiten.
- VRAM-Schutz:
  - Der Agent sollte die Batch-Größe (Standard 8) automatisch auf 4 oder 2 reduzieren, falls ein "Out of Memory"-Error auftritt.
  - Die GUI soll beim Erkennen limitierter VRAMs direkte Handlungsempfehlungen geben (z.B. Wechsel auf `raspberry` oder `onnx`).
- Dataset-Tipp:
  - Für optimale Ergebnisse sollte der Agent den Nutzer auffordern, mindestens 1 bis 3 Stunden hochwertiges, trockenes Audio-Material (z.B. ein einzelnes Instrument) zu verwenden.
  - Der GUI-Workflow kann eine Empfehlungskarte für Dataset-Qualität und Dateilänge anzeigen.
- macOS-Hinweis:
  - Da Apple Silicon GPUs nicht für das Training großer Faltungsnetzwerke optimiert sind, sollte der Agent bei Mac-Nutzern auf längere Trainingszeiten hinweisen oder das Training auf einem Cloud-Backend (wie RunPod oder Vast.ai) vorschlagen.

## 6. Spezielle Richtlinie für lokale 6 GB VRAM Modelle

- Auswahl der Mini-Architektur (VRAM-optimiert):
  - `raspberry`: Am stärksten optimierte Mini-Konfiguration, benötigt ca. 5 GB VRAM.
  - `onnx`: Noiseless Baseline, ebenfalls für 6 GB VRAM geeignet.
- GUI-Modell-Auswahl:
  - Das Interface zeigt empfohlenen Modus, VRAM-Anforderungen und eine kurze Einschätzung der Qualität.
  - Bei 6 GB VRAM wird automatisch ein Vorschlag für `raspberry` oder `onnx` angezeigt.
- Agent-Logik:
  - Setzt automatisch die Batch-Größe auf 4 oder 2, um OOM-Abstürze zu vermeiden.
  - Bietet in der App die Möglichkeit, das Batch-Size-Level manuell anzupassen.

## 7. Applio-ähnlicher RVC-Workflow für Sprachmodelle

- Ziel:
  - Die App soll den RVC-Trainingsprozess ähnlich Applio abbilden und eine einfache, schrittweise GUI für Sprachmodelle anbieten.
- Schritt 1: Das Dataset vorbereiten
  - Länge: 10 bis 15 Minuten sauberes Sprachmaterial reichen aus.
  - Qualität: Trocken aufgenommen, kein Hall, kein Raum-Echo, keine Hintergrundgeräusche, keine Musik.
  - Format: Unkomprimierte WAV-Dateien (44.1 kHz oder 48 kHz, 16/24 Bit) in einem Ordner auf der Festplatte (z. B. `C:/RVC_Datasets/MeineStimme/`).
  - GUI: Dateiauswahl, Qualitäts-Checklist und Preview-Analyse der Audioqualität.
- Schritt 2: Preprocessing in der App
  - Reiter: `Train -> 1. Preprocess Dataset`.
  - Feld: `Model Name`, `Dataset Path`, `Sample Rate`.
  - Sample Rate: Empfehlung `40k` oder `40k-v2`.
  - Aktion: `Preprocess Dataset`.
  - Ergebnis: Automatisches Schneiden in 3–5 Sekunden Clips, Entfernen von Stille und Normalisierung.
- Schritt 3: Pitch Extraction (Feature Extraction)
  - Reiter: `Train -> 2. Extract Features`.
  - Auswahl: `Pitch Extraction Algorithm` mit `rmvpe` als Standard.
  - Aktion: `Extract Features`.
- Schritt 4: Model Training starten
  - Reiter: `Train -> 3. Train Model`.
  - Total Epochs: 200–300 für 10–15 Minuten Sprachmaterial.
  - Save Frequency: 25 oder 50.
  - Batch Size: 8 oder 16 bei 8 GB VRAM, 4 bei 4–6 GB VRAM.
  - Option: `Save Only Latest` für Speicherplatzersparnis.
  - Aktion: `Train Model`.
- Schritt 5: Index-Datei erstellen
  - Reiter: `Train -> Train Index`.
  - Aktion: `Train Index`.
  - Ergebnis: Erzeugung einer `.index`-Datei für feinere Klangnuancen und stabilere Inferenz.
- Wo das fertige Modell liegt
  - Applio-ähnlicher Speicherpfad: `logs/DeinModellName/DeinModellName.pth` und `logs/DeinModellName/added_DeinModellName.index`.
  - GUI: Anzeige des Ausgabeordners und Option zum Laden der Modell- und Index-Dateien in den Inference-Reiter.
- Neutone-Export für Sprachmodelle
  - Der Workflow endet mit einem Neutone-kompatiblen Export.
  - Die Anwendung konvertiert das Modell aus seinem Trainingsformat in `.nm` über ein SDK-Wrapping oder ein Export-Skript.
  - Ergebnis: Ein Modell, das in Neutone FX geladen werden kann, unabhängig davon, ob es ursprünglich ein RAVE- oder ein Sprachmodell war.

## 8. Neutone-spezifischer Export-Workflow

- Schritt A: TorchScript Export (Streaming-Modus)
  - Befehl: `rave export --run /pfad/zum/modell --streaming True`.
- Schritt B: Neutone SDK Wrapping
  - Erstellung eines Python-Skripts, das die `.ts`-Datei in eine `.nm`-Datei packt.
  - Definition von Metadaten (Name, Autor, Beschreibung).
  - Zuweisung von Echtzeit-Parametern (A, B, C, D), z.B. Chaos oder Z-Scale.
- Sprachmodell-Export:
  - Für RVC-Sprachmodelle muss die App den Exportpfad so gestalten, dass die erzeugten `.pth` und `.index` Dateien in ein Neutone-kompatibles TSB/TS-Format überführt werden können.
  - Der Benutzer wählt im GUI, ob er ein Sound-Design-Modell oder ein Sprachmodell exportieren möchte. Die App schlägt dann den passenden Exportpfad vor.

## 9. Integration in Neutone FX

- Öffne Neutone FX in der DAW.
- Nutze den Button "load your own", um die erstellte `.nm`-Datei zu laden.
- Nutze Mono-Signale für maximale Stabilität in der Standard-RAVE-Konfiguration.

## 9. Zusammenfassung der empfohlenen Trainingsparameter

- Training: `--config raspberry` (minimale Hardwarelast).
- Stabilität: `batch_size: 2` (verhindert Abstürze auf RTX 3060).
- Export 1: `rave export --streaming True` (verhindert Klickartefakte).
- Export 2: `neutone_sdk.save_neutone_model` (erzeugt das `.nm` Format).

## 10. Cloud-Alternative bei lokalem Limit

- Falls das lokale Training zu langsam oder instabil ist, sollte der Agent auf Google Colab oder Kaggle ausweichen.
- Dort kann ein etwas komplexeres Modell (`v2_small`) trainiert und später als Mini-Modell für Neutone exportiert werden.

## 11. WSL-Setup und Testworkflow in VS Code

- Ziel:
  - Eine einzelne WSL-Distribution verwenden, um HD-Platz zu sparen und alle Trainings-Workflows zentral zu testen.
  - Keine vielen verschiedenen WSL-Umgebungen, sondern eine einzige Ubuntu/WSL2-Installation als Standardplattform.

- Warum eine einzelne WSL-Distribution?
  - Jede WSL-Distribution ist ein eigener Linux-Speicherbereich und kann mehrere Gigabyte belegen.
  - Mit nur einer Distribution vermeidest du mehrfachen Overhead und nutzt denselben Linux-Container für RAVE- und RVC-Workflows.

- Empfohlener Ablauf:
  1. Prüfen, ob WSL bereits installiert ist:
     - `wsl -l -v`
     - Wenn keine Distribution gelistet ist, installiere Ubuntu als einzige Distribution:
       - `wsl --install -d Ubuntu`
     - Wenn mehrere Distro-Images vorhanden sind, entferne alle bis auf eine mit:
       - `wsl --unregister <DistroName>`
  2. Sicherstellen, dass die Standardversion WSL2 ist:
     - `wsl --set-default-version 2`
  3. VS Code öffnen und die Erweiterung "Remote - WSL" installieren.
  4. Im Projektordner in VS Code die Befehls-Palette öffnen (`Ctrl+Shift+P`) und `Remote-WSL: Reopen Folder in WSL` wählen.
  5. App-seitig: Die GUI prüft beim Start automatisch die WSL-Konfiguration und startet `Ubuntu-24.04`, falls sie gestoppt ist.
     - Die App sollte ein WSL-Startskript integrieren, das `wsl -d Ubuntu-24.04 --state Running` prüft und gegebenenfalls `wsl -d Ubuntu-24.04` initiiert.
     - Danach wird sichergestellt, dass das Projektverzeichnis korrekt unter `/mnt/c/...` gemountet ist und Python sowie CUDA verfügbar sind.

- Single-Environment-Strategie:
  - Verwende in der WSL-Distribution nur ein Python-Environment für das Projekt, z. B. `rave_env`.
  - Optional: Ein zweites leichtgewichtes Environment für RVC-Sprachmodelle, falls nötig, aber nur innerhalb derselben Distro.
  - Vermeide separate Windows- und WSL-Installationen für dasselbe Projekt.
  - Die App kann im Setup einen WSL-Check durchführen und nur dann neue Umgebungen anlegen, wenn keine vorhandene Distribution nutzbar ist.

- WSL-Umgebung einrichten:
  1. Öffne das WSL-Terminal in VS Code.
  2. Installiere Miniconda oder Mambaforge einmalig in der Distribution.
  3. Erstelle das Projekt-Environment:
     - `conda create -n rave_env python=3.9 -y`
     - `conda activate rave_env`
  4. Installiere benötigte Pakete und Tools innerhalb von WSL statt in Windows.

- Testen in VS Code:
  - Starte die App direkt aus dem WSL-Terminal oder über VS Code Run-Konfigurationen.
  - Nutze `code .` aus dem WSL-Terminal, um die aktuelle Distro-Instanz mit VS Code zu verbinden.
  - Überprüfe, dass GPU/CUDA in WSL funktioniert mit `nvidia-smi` und `python -c "import torch; print(torch.cuda.is_available())"`.
  - Nutze die VS Code Tasks, um WSL und App unabhängig zu starten und zu stoppen.
    - `WSL: Start Ubuntu-24.04`
    - `WSL: Stop Ubuntu-24.04`
    - `App: Start Dev`
    - `App: Start Prod`
    - `App: Stop`
  - Dev-Modus:
    - Erfordert eine bereits laufende WSL-Distro.
    - Startet die App nicht selbst.
  - Prod-Modus:
    - Startet die App unabhängig von einem laufenden WSL-Prozess.
    - Die App kann die vorhandene WSL-Distro automatisch nutzen und sie gegebenenfalls booten.

- Platzsparende Tipps:
  - Nutze `docker` in WSL nur wenn nötig; für eine einfache App reicht eine native WSL-Python-Installation.
  - Verwende in WSL keine großen separaten Basis-Images für jedes Projekt.
  - Halte den Installationspfad schlank: `~/miniconda3`, `~/projects/audio-mc-rave-training`.

- Ergebnis für die App:
  - Deine GUI-App soll in VS Code innerhalb einer einzigen WSL2-Umgebung getestet werden.
  - So kannst du sowohl RAVE- als auch RVC-Workflows aus einer zentralen Linux-Umgebung steuern und musst nicht mehrere WSL-Distributionen pflegen.
## 12. Projekt-Agenten und Prompt-Vorlagen
- Verwende die projektinternen Agenten-Anweisungen in
  `.github/instructions/AudioMC_RAVE_Training.instructions.md`.
- Nutze die Projekt-Prompt-Vorlagen in
  `.github/prompts/AudioMC_RAVE_Training.prompt.md`.
- Diese Dateien enthalten:
  - Projektziele, Stil- und Prüfregeln
  - WSL-/VS Code-Setup-Checks
  - GUI-Workflow- und Modularisierungs-Vorlagen

> Hinweis: Diese Projektdateien sollten als Anleitung für automatische Agenten und für die manuelle Entwicklung gleichermaßen dienen.