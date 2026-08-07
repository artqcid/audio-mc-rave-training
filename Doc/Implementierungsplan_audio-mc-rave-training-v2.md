# Implementierungsplan: audio-mc-rave-training-v2 (Native Windows Migration)

## Ziel
Migration der bestehenden RAVE-Frontend-App auf **native Windows-Umgebung**. Vollständige Entfernung der WSL-Abhängigkeit. Alle RAVE-CLI-Befehle werden direkt über Windows Subprozesse (`rave.exe` / Python-Interpreter) ausgeführt.

---

## Ausgangslage (Bestandsaufnahme)

### WSL-abhängige Dateien (müssen refactored/entfernt werden)
| Datei | WSL-Bezug | Aktion |
|-------|-----------|--------|
| `wsl_manager.py` | **Vollständig WSL-spezifisch** (Pfadkonvertierung, WSL Start/Stop, RAVE-Detection via WSL) | **Ersetzen** durch `windows_manager.py` |
| `app.py` | Importiert & nutzt `wsl_manager` Funktionen; baut WSL-Befehle | **Stark refactoren** |
| `train_gui.py` | Führt WSL-Kommandos via `subprocess` aus | **Refactoren** für native Windows |
| `README.md` | Dokumentiert WSL-Setup | **Aktualisieren** |
| `requirements.txt` | Hinweise auf WSL-spezifische PyTorch-Installation | **Aktualisieren** |
| `templates/index.html` | Zeigt WSL-Status, RAVE via WSL | **Anpassen** (WSL-Status entfernen) |
| `scripts/setup_wsl_env.sh` | WSL-Setup-Skript | **Entfernen/Archivieren** |

### Bereits Windows-kompatibel (keine Änderungen nötig)
| Datei | Grund |
|-------|-------|
| `data_pipeline.py` | Reines Python (librosa, soundfile), keine WSL-Abhängigkeit |
| `export_neutone.py` | Nutzt PyTorch direkt, keine WSL-Abhängigkeit |
| `tests.py` | Basis-Tests, anpassbar |

---

## Schritt-für-Schritt Implementierungsplan

---

### Step 1: Environment & Dependency Check

**Ziel**: Abhängigkeiten für native Windows sicherstellen. **Alle Python-Pakete werden in einer virtuellen Umgebung (venv) installiert.**

#### 1.1 requirements.txt aktualisieren
```txt
# Core GUI and audio dependencies for the RAVE training app (Native Windows)
# Installation in venv:
# python -m venv .venv
# .venv\Scripts\activate
# pip install -r requirements.txt
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# pip install acids-rave
fastapi==0.114.0
uvicorn[standard]==0.23.2
jinja2==3.1.2
aiofiles==23.1.0
python-multipart==0.0.6
pydantic==2.9.0
numpy==1.26.4
scipy==1.11.4
soundfile==0.14.0
librosa==0.11.0
pandas==2.2.3
python-dotenv==1.0.0
torchaudio==2.2.1
httpx>=0.23.0
# RAVE (IRCAM) - install via: pip install acids-rave
# oder: pip install git+https://github.com/acids-ircam/RAVE.git
```

#### 1.2 PyTorch CUDA Installation dokumentieren (in venv)
- In aktivierter venv: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124` (oder cu118/cu121 je nach NVIDIA-Treiber)
- Hinweis: `torch`, `torchvision`, `torchaudio` müssen **zusammen** mit gleichem CUDA-Index in der venv installiert werden

#### 1.3 RAVE-Paket prüfen (in venv)
- `acids-rave` auf PyPI verfügbar? Falls nicht: GitHub-Installation dokumentieren
- CLI `rave` muss im venv-PATH sein (nach `pip install acids-rave` in der aktivierten venv automatisch)

---

### Step 2: Pfad-Management Refactoring

**Ziel**: Alle WSL-Pfadkonvertierungen (`/mnt/c/...`) entfernen, native Windows-Pfade (`C:\...`) verwenden.

#### 2.1 `wsl_manager.py` → `windows_manager.py` neu erstellen
**Entfernen**:
- `get_wsl_path()` (Konvertierung `C:\` → `/mnt/c/`)
- `check_wsl_running()`, `start_wsl()`, `stop_wsl()`
- `is_wsl()`
- WSL-spezifische `check_rave_installed()` (nutzt `wsl -d Ubuntu...`)

**Neu implementieren in `windows_manager.py`**:
```python
# windows_manager.py
import platform
import subprocess
from pathlib import Path
from typing import Dict, Optional

def is_windows() -> bool:
    return platform.system().lower() == "windows"

def get_gpu_vram() -> Optional[int]:
    # Unverändert: nvidia-smi auf Windows verfügbar
    ...

def get_cuda_available() -> bool:
    # Unverändert: torch.cuda.is_available()
    ...

def check_rave_installed() -> bool:
    # NEU: Direkter Aufruf von `rave --version` oder `python -m rave --version`
    result = subprocess.run(["rave", "--version"], capture_output=True, text=True)
    return result.returncode == 0

def get_native_path(path: str) -> str:
    # NEU: Normiert Windows-Pfade, setzt Anführungszeichen bei Leerzeichen
    p = Path(path).resolve()
    return str(p)  # Bereits nativer Windows-Pfad

def get_environment_info() -> Dict[str, Optional[str]]:
    return {
        "platform": platform.system(),
        "is_windows": str(is_windows()),
        "cuda_available": str(get_cuda_available()),
        "gpu_vram": str(get_gpu_vram()),
        "rave_available": str(check_rave_installed()),
    }
```

#### 2.2 `app.py` Imports anpassen
```python
# ALT
from wsl_manager import (
    check_rave_installed, check_wsl_running, get_cuda_available,
    get_gpu_vram, get_wsl_path, is_wsl, start_wsl, stop_wsl
)

# NEU
from windows_manager import (
    check_rave_installed, get_cuda_available, get_gpu_vram,
    get_native_path, get_environment_info
)
```

#### 2.3 Alle Pfad-Verwendungen prüfen
- In `app.py`: `rave_data_path = get_wsl_path(preprocessed_path)` → `get_native_path(preprocessed_path)`
- In `train_gui.py`: Working Directory für subprocess auf natives `Path` setzen

---

### Step 3: Subprocess / CLI Execution Refactoring

**Ziel**: RAVE-Befehle nativ auf Windows ausführen (kein `wsl`, kein `bash -lc`).

#### 3.1 `app.py` - Training Command Building (Zeilen 239-269)

**NEU (Native Windows)**:
```python
# Direkter Aufruf über rave CLI oder python -m rave
config_file = preset.get("config_file", "v1.gin")
checkpoint_arg = f"--ckpt {shlex.quote(get_native_path(checkpoint_path))}" if training_mode == "resume" and checkpoint_path else ""

# Option A: rave CLI direkt (bevorzugt, wenn im PATH)
command = (
    f"rave train --name {shlex.quote(model_name)} "
    f"--db_path {shlex.quote(get_native_path(preprocessed_path))} "
    f"--out_path {shlex.quote(config.output_path)} "
    f"--config {config_file} "
    f"{checkpoint_arg}"
)

# Option B: Falls rave nicht im PATH, python -m rave
# command = f"python -m rave train ..."
```

#### 3.2 `train_gui.py` - Subprocess Execution (Zeilen 244-286)

**WICHTIG**: Subprocess muss den venv-Python-Interpreter nutzen. Zwei Optionen:

**Option A**: App wird in aktivierter venv gestartet (empfohlen) → `rave` CLI direkt im PATH verfügbar
**Option B**: Expliziter venv-Pfad im Subprocess (falls App außerhalb venv läuft)

```python
# In _run_training_command() - Venv-Python ermitteln
import sys
venv_python = sys.executable  # Wenn App in venv gestartet: .venv\Scripts\python.exe
# Oder explizit: venv_python = str(BASE_DIR / ".venv" / "Scripts" / "python.exe")

# Option A: rave CLI über venv-Python (sicherste Methode)
command = f'"{venv_python}" -m rave train --name {shlex.quote(model_name)} ...'

# Option B: Wenn rave im venv-PATH (App in venv gestartet)
# command = f'rave train --name {shlex.quote(model_name)} ...'

process = subprocess.Popen(
    command,
    cwd=str(Path(config.preprocessed_path).parent),
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding='utf-8',
    errors='replace',
    bufsize=1,
    shell=True,
    env={**os.environ, "PYTHONIOENCODING": "utf-8"}  # Encoding erzwingen
)
```

#### 3.3 Preprocessing Command (falls `rave preprocess` genutzt wird)
- In `data_pipeline.py` aktuell reines Python → **keine Änderung nötig**
- Falls zukünftig `rave preprocess` CLI genutzt wird: analog zu Training anpassen

---

### Step 4: Stream-Handling & Encoding (Windows Specifics)

**Ziel**: Korrekte Ausgabe-Lesung auf Windows (Encoding, Zeilenenden).

#### 4.1 Subprocess Encoding
```python
# In train_gui.py _run_training_command()
process = subprocess.Popen(
    ...,
    text=True,
    encoding='utf-8',      # oder 'cp1252' falls rave deutsches Windows nutzt
    errors='replace',      # Verhindert UnicodeDecodeError
    bufsize=1,             # Zeilenweise buffering
)
```

#### 4.2 Zeilenenden normalisieren
```python
for line in iter(process.stdout.readline, ""):
    if line:
        # Windows \r\n → \n normalisieren
        clean_line = line.rstrip('\r\n')
        self._append_log(clean_line)
```

#### 4.3 Environment Variables
- `PYTHONIOENCODING=utf-8` setzen für konsistente Ausgabe
- Keine `WSLENV`, `DISPLAY`, `WAYLAND_DISPLAY` mehr nötig

---

### Step 5: Fehlerbehandlung anpassen

**Ziel**: User-freundliche Fehlermeldungen für native Windows.

#### 5.1 `windows_manager.py` - RAVE Check
```python
def check_rave_installed() -> bool:
    try:
        result = subprocess.run(
            ["rave", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False
    except Exception:
        return False

def get_rave_error_hint() -> str:
    """Gibt hilfreichen Hinweis zurück, wenn RAVE nicht gefunden wird."""
    return (
        "RAVE CLI nicht gefunden. Bitte installieren Sie RAVE nativ unter Windows:\n"
        "  pip install acids-rave\n"
        "oder:\n"
        "  pip install git+https://github.com/acids-ircam/RAVE.git\n"
        "Stellen Sie sicher, dass der Installationspfad im Windows PATH ist."
    )
```

#### 5.2 `app.py` - Startup Check (neu)
```python
@app.on_event("startup")
async def startup_check():
    if not check_rave_installed():
        logger.warning("RAVE CLI nicht im PATH gefunden. Echtes Training deaktiviert.")
        # Optional: In App-State speichern für UI-Hinweis
```

#### 5.3 API-Fehlermeldungen anpassen (app.py Zeilen 190-194)
```python
# NEU
content={"error": get_rave_error_hint()}
```

---

### Step 6: Code Cleanup & Review

**Ziel**: Tote WSL-Code-Pfade entfernen, Frontend bereinigen, Testlauf.

#### 6.1 Dateien löschen/archivieren
- `wsl_manager.py` → **Löschen** (ersetzt durch `windows_manager.py`)
- `scripts/setup_wsl_env.sh` → **Löschen** oder in `scripts/archive/` verschieben

#### 6.2 `app.py` - WSL-API-Endpunkte entfernen
```python
# ENTFERNEN (Zeilen 326-335):
@app.post("/api/wsl/start")
def api_wsl_start(): ...

@app.post("/api/wsl/stop")
def api_wsl_stop(): ...
```

#### 6.3 `app.py` - Status-API bereinigen (Zeilen 117-125)
```python
# NEU
return {
    "cuda_available": get_cuda_available(),
    "gpu_vram": get_gpu_vram(),
    "training_status": training_manager.get_status(),
    "rave_available": check_rave_installed(),
    "environment": get_environment_info(),  # Neu: platform, is_windows, etc.
}
```

#### 6.4 `templates/index.html` - Frontend bereinigen
- **Entfernen**: WSL-Status Badge (Zeile 54), WSL Start/Stop Buttons
- **Behalten**: CUDA, GPU VRAM, RAVE CLI, Training Status
- **Neu**: "Windows Native" Badge, PyTorch CUDA Version Anzeige

#### 6.5 `README.md` - Komplett überarbeiten
- WSL-Setup entfernen
- Native Windows Installation dokumentieren (mit venv):
  1. Python 3.10+ installieren
  2. `python -m venv .venv`
  3. `.venv\Scripts\activate`
  4. `pip install -r requirements.txt`
  5. `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`
  6. `pip install acids-rave` (oder GitHub)
  7. `python app.py` → `http://localhost:8000`
  - Hinweis: Die venv muss für jeden Terminal-Session neu aktiviert werden (`.venv\Scripts\activate`)

#### 6.6 Dry-Run Test (Validierung)

**Preprocessing Test**:
```bash
python -c "from data_pipeline import preprocess_dataset; preprocess_dataset('dataset', 'processed')"
```

**RAVE Train Command Dry-Run** (ohne echtes Training):
```bash
# Erwarteter Command String:
rave train --name "test_model" --db_path "C:\path\to\processed" --out_path "C:\path\to\trained_models" --config v1.gin
```

**RAVE Export Dry-Run**:
```bash
rave export --run "C:\path\to\model.pt" --streaming True
```

#### 6.7 End-to-End Test Checklist
- [ ] App startet ohne Fehler (`python app.py`)
- [ ] Systemstatus zeigt: CUDA, VRAM, RAVE Available (true/false)
- [ ] Preprocessing funktioniert (Dataset → Processed)
- [ ] Training im Simulator-Modus läuft
- [ ] Training mit `use_rave=true` startet `rave train` nativ (falls RAVE installiert)
- [ ] Logs werden live im Frontend angezeigt (Encoding korrekt)
- [ ] Export funktioniert (TorchScript → .nm)
- [ ] Stop-Training funktioniert (Process terminate)

---

## Zusammenfassung: Zu ändernde Dateien

| Datei | Änderungstyp | Aufwand |
|-------|--------------|---------|
| `wsl_manager.py` | **Löschen** | - |
| `windows_manager.py` | **Neu erstellen** | Mittel |
| `app.py` | **Stark refactoren** | Hoch |
| `train_gui.py` | **Refactoren** (Subprocess/Encoding) | Mittel |
| `requirements.txt` | **Aktualisieren** | Gering |
| `README.md` | **Neu schreiben** | Mittel |
| `templates/index.html` | **Anpassen** (WSL entfernen) | Gering |
| `scripts/setup_wsl_env.sh` | **Löschen/Archivieren** | - |

---

## Risiken & Mitigation

| Risiko | Wahrscheinlichkeit | Mitigation |
|--------|-------------------|------------|
| `rave` CLI nicht im Windows PATH nach Installation | Mittel | `get_rave_error_hint()` mit klarer Anleitung; `where rave` in Check einbauen |
| PyTorch CUDA Version Mismatch | Hoch | requirements.txt & README mit exaktem `--index-url` dokumentieren |
| Encoding-Probleme bei subprocess Output | Mittel | `encoding='utf-8', errors='replace'` explizit setzen |
| RAVE Gin-Configs nicht gefunden | Gering | Config-Pfade relativ zu `BASE_DIR` oder absoluten Pfad nutzen |
| GPU OOM auf 6GB VRAM | Hoch | Batch-Size Default auf 2 setzen; `raspberry`/`onnx` Presets empfehlen |

---

## Nächste Schritte (nach Plan-Bestätigung)

1. **Step 1-2**: `windows_manager.py` erstellen, `requirements.txt` & `README.md` aktualisieren
2. **Step 3**: `app.py` Command-Building refactoren, `train_gui.py` Subprocess anpassen
3. **Step 4-5**: Encoding & Error Handling verfeinern
4. **Step 6**: Cleanup, Frontend bereinigen, Dry-Run & E2E-Test

---

**Erstellt**: 2026-08-06  
**Basierend auf**: User-provided Execution Plan + Codebase-Analyse  
**Status**: Plan-Phase (bereit für Act Mode)