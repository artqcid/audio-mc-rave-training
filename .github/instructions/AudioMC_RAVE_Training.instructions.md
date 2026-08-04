# Audio MC RAVE Training — Projektinstruktionen

Diese Datei enthält projektbezogene Regeln und Erwartungen für alle Agenten oder Teammitglieder, die am RAVE/RVC-Trainingstool arbeiten.

## Projektziel
- Erstelle eine plattformübergreifende GUI-App für lokales RAVE/RVC-Training auf WSL2.
- Zielhardware ist NVIDIA RTX 3060 mit 6 GB VRAM.
- Die App muss WSL automatisch starten und konfigurieren können, ohne bestehende WSL-Setups zu zerstören.
- Export muss Neutone FX-kompatible Modell-Assets ermöglichen.

## Kernanforderungen
- Trenne Dev-/Prod-Modi: Dev darf WSL nicht unbeaufsichtigt starten, Prod kann WSL und den Trainingsprozess orchestrieren.
- Die App muss lokale Projektumgebung, `.venv` und Python-Pfade sauber verwalten.
- Dokumentation und Konfigurationsdateien müssen in `Doc/` und `.vscode/` gepflegt werden.

## Stil und Verhalten
- Schreibe klare, kurze Commit- und Dokumentationsbeschreibungen.
- Verwende deutsche Anweisungen dort, wo es für den Projektinhaber besser lesbar ist.
- Verwende technische Begriffe präzise: WSL2, Ubuntu-24.04, RAVE, RVC, Neutone FX, RTX 3060.
- Änderungen dürfen bestehende WSL-Installationen nicht beschädigen.

## Vorgehen beim Entwickeln
1. Prüfe zuerst vorhandene Konfigurationen in `.vscode/tasks.json`, `Doc/Entwurfsplan_RLTA.md` und `requirements.txt`.
2. Ergänze neue Funktionalität immer mit passender Dokumentation im Entwurfsplan.
3. Teste WSL-bezogene Befehle zuerst im Terminal mit `wsl -l -v` und `wsl -d Ubuntu-24.04 -- bash -lc ...`.
4. Bei GUI-Entwicklung nutze bevorzugt Python-Frameworks, die in der Projektumgebung sauber installierbar sind.

## Prüfregeln für Agenten
- Prüfe, ob ein Vorschlag bestehende globale VS Code oder WSL-Konfigurationen beeinflusst.
- Achte auf Kompatibilität mit 6 GB VRAM; vermeide Modelle oder Batchgrößen, die das Limit überschreiten.
- Vermeide Änderungen an globalen Benutzereinstellungen in `C:\Users\marku\AppData\Roaming\Code\User` ohne ausdrückliche Anweisung.

## Spezifische Fokusbereiche
- WSL-Start/Stop-Logik (`.vscode/tasks.json`)
- Lokales Python-Setup und Paketabhängigkeiten
- GUI-Workflow: Datenauswahl, Preprocessing, Training, Export
- Neutone FX Exportformat und kompatible Modellstruktur
- Dokumentation und Benutzerführung
