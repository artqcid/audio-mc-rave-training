# Implementierungsplan: Portable-Setup (Windows Desktop App & Setup.exe)

## Ziel
Die Anwendung (Vue.js Frontend + RAVE/PyTorch Backend) in eine **autarke Windows-Desktop-App** verpacken und als `Setup.exe` ausliefern. Der Endnutzer soll **nichts installieren müssen** (kein Python, Node, CUDA, ffmpeg).

---

## Architektur
- **Frontend**: Vue.js → kompiliert zu statischen Assets (`dist/`)
- **Backend**: Python (FastAPI + RAVE/PyTorch) → serviert Assets, führt Training aus
- **Desktop-Wrapper**: `pywebview` (oder `Eel`) öffnet natives Fenster, lädt `http://localhost:PORT`
- **Bundling**: PyInstaller → `.exe` mit allen Dependencies (Python, PyTorch, CUDA-DLLs, RAVE, ffmpeg)
- **Installer**: Inno Setup → `Setup.exe` aus PyInstaller-Bundle

---

## Schritt-für-Schritt Implementierungsplan

---

### Step 1: Frontend-Backend Bridge (Webview Integration)

**Ziel**: Python-App öffnet natives Fenster, Vue-Frontend läuft darin.

#### 1.1 Aktuelle Kommunikation analysieren
- Aktuell: FastAPI REST API (`/api/*`) + Polling (2s Intervall) für Logs/Status
- WebSocket für Live-Logs? → Prüfen, ob Upgrade nötig
- Für Webview: REST reicht, aber WebSocket eleganter für Live-Updates

#### 1.2 `pywebview` Integration (`main.py` neu erstellen)
```python
# main.py
import webview
import threading
import uvicorn
import sys
from pathlib import Path
from app import app  # FastAPI App importieren

# Pfad-Handling für PyInstaller
def get_resource_path(relative_path):
    """Findet Ressourcen im PyInstaller-Temp-Ordner (_MEIPASS) oder dev."""
    if hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS) / relative_path
    return Path(__file__).parent / relative_path

def run_fastapi():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

if __name__ == "__main__":
    # 1. FastAPI in separatem Thread starten
    api_thread = threading.Thread(target=run_fastapi, daemon=True)
    api_thread.start()

    # 2. Warten bis Server bereit (optional: Health-Check)
    import time
    time.sleep(1.5)

    # 3. Frontend-Pfad ermitteln (PyInstaller: resources/frontend)
    frontend_dist = get_resource_path("resources/frontend")
    if not frontend_dist.exists():
        # Fallback: Dev-Modus
        frontend_dist = Path(__file__).parent / "templates"

    # 4. Webview-Fenster erstellen
    window = webview.create_window(
        title="RLTA - RAVE Local Trainer App",
        url="http://127.0.0.1:8000",  # oder file:// bei rein statischem Serving
        width=1200,
        height=800,
        min_size=(1000, 700),
        frameless=False,
        easy_drag=False,
        on_top=False,
    )

    # 5. Fenster starten (blockiert bis Schließen)
    webview.start(debug=False)
    
    # Cleanup beim Schließen
    sys.exit(0)
```

#### 1.3 Alternative: Statisches File-Serving statt `http://localhost`
```python
# In app.py: Statische Dateien servieren
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
# Dann in webview: url=str(frontend_dist / "index.html") (file:// Protokoll)
```

---

### Step 2: Frontend Build-Pipeline

**Ziel**: Vue `dist/` automatisch ins Python-Backend kopieren.

#### 2.1 Vue-Projektstruktur prüfen
```
project-root/
├── frontend/          # Vue-Projekt (package.json, src/, etc.)
│   └── dist/          # nach npm run build
├── resources/
│   └── frontend/      # Ziel für PyInstaller (wird von PyInstaller eingebunden)
├── app.py
├── main.py
└── ...
```

#### 2.2 Build-Skript (`scripts/build_frontend.py` oder `.ps1`)
```python
# scripts/build_frontend.py
import subprocess
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
FRONTEND_DIR = ROOT / "frontend"
DIST_DIR = FRONTEND_DIR / "dist"
TARGET_DIR = ROOT / "resources" / "frontend"

def build_frontend():
    print("📦 Building Vue Frontend...")
    
    # 1. npm install (falls node_modules fehlt)
    if not (FRONTEND_DIR / "node_modules").exists():
        subprocess.run(["npm", "install"], cwd=FRONTEND_DIR, check=True)
    
    # 2. npm run build
    subprocess.run(["npm", "run", "build"], cwd=FRONTEND_DIR, check=True)
    
    # 3. dist/ nach resources/frontend/ kopieren
    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)
    shutil.copytree(DIST_DIR, TARGET_DIR)
    
    print(f"✅ Frontend gebaut und kopiert nach {TARGET_DIR}")

if __name__ == "__main__":
    build_frontend()
```

---

### Step 3: PyInstaller Spec-File Konfiguration

**Ziel**: `.exe` erstellen mit allen Assets, Python, PyTorch, CUDA, RAVE, ffmpeg.

#### 3.1 `app.spec` erstellen
```python
# app.spec
import sys
from pathlib import Path

# Pfade
ROOT = Path(__file__).parent
RESOURCES = ROOT / "resources"

# Hidden Imports für PyTorch/RAVE/Torchaudio
hiddenimports = [
    # PyTorch
    "torch", "torch.nn", "torch.optim", "torch.autograd",
    "torchvision", "torchaudio",
    # RAVE
    "rave", "rave.core", "rave.blocks", "rave.dataset",
    "rave.discriminator", "rave.pqmf", "rave.gin",
    # FastAPI/UVicorn
    "uvicorn", "uvicorn.logging", "uvicorn.loops",
    "uvicorn.protocols", "uvicorn.lifespan",
    # PyWebView
    "webview", "webview.platforms.winforms",
    # Standard Libs
    "json", "yaml", "jinja2", "aiofiles",
    # Audio
    "librosa", "soundfile", "scipy",
]

# Binaries: CUDA DLLs + ffmpeg
binaries = []
# CUDA DLLs (Pfade anpassen je nach PyTorch-Installation)
cuda_path = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
if cuda_path.exists():
    for dll in cuda_path.glob("*.dll"):
        binaries.append((str(dll), "."))

# ffmpeg (falls lokal vorhanden)
ffmpeg_path = ROOT / "ffmpeg" / "bin" / "ffmpeg.exe"
if ffmpeg_path.exists():
    binaries.append((str(ffmpeg_path), "."))

# Datas: Frontend Assets + Configs + Model-Presets
datas = [
    (str(RESOURCES / "frontend"), "resources/frontend"),
    (str(ROOT / "templates"), "templates"),  # Fallback
    (str(ROOT / "train_gui.py"), "."),
    (str(ROOT / "data_pipeline.py"), "."),
    (str(ROOT / "export_neutone.py"), "."),
    (str(ROOT / "windows_manager.py"), "."),
    (str(ROOT / "requirements.txt"), "."),
]

# Excludes (Größe reduzieren)
excludes = [
    "tkinter", "matplotlib", "PIL", "pandas", "notebook",
    "IPython", "jupyter", "pytest", "sphinx", "docutils",
]

a = Analysis(
    ["main.py"],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="RLTA",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # KEIN CMD-Fenster!
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "assets" / "icon.ico") if (ROOT / "assets" / "icon.ico").exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="RLTA",
)
```

#### 3.2 WICHTIG: PyTorch CUDA DLLs finden
```bash
# In venv aktiviert:
python -c "import torch; import os; print(os.path.dirname(torch.__file__))"
# → .../site-packages/torch/lib/  (hier liegen cublas64_11.dll, cudnn64_8.dll, etc.)
```

Diese DLLs **müssen** in `binaries` oder via `--add-binary` eingebunden werden.

---

### Step 4: System-Checks & Logging (Headless-Mode)

**Ziel**: Da `console=False`, Logging in Datei + Webview-UI.

#### 4.1 Logging-Setup (`logging_config.py`)
```python
# logging_config.py
import logging
import sys
from pathlib import Path

def setup_logging(app_name="RLTA"):
    # Log-Ordner in AppData (persistent, benutzer-spezifisch)
    log_dir = Path.home() / "AppData" / "Local" / app_name / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{app_name}.log"
    
    # File Handler (detailliert)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_format)
    
    # Console Handler (nur für Dev, in PyInstaller deaktiviert)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_format)
    
    # Root Logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers = [file_handler]
    
    # Nur in Dev-Mode Console-Handler hinzufügen
    if not getattr(sys, 'frozen', False):
        root_logger.addHandler(console_handler)
    
    return log_file

# In main.py ganz am Anfang aufrufen:
# log_file = setup_logging()
# logging.info(f"Log-Datei: {log_file}")
```

#### 4.2 Logs an Vue-Frontend streamen (API-Endpunkt)
```python
# In app.py: Bestehendes /api/training-logs nutzen
# Zusätzlich: WebSocket für echte Live-Logs (optional)
from fastapi import WebSocket
from fastapi.responses import HTMLResponse

@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Logs aus TrainingManager holen und senden
            logs = training_manager.get_logs()
            await websocket.send_json({"logs": logs})
            await asyncio.sleep(0.5)
    except Exception:
        await websocket.close()
```

#### 4.3 Startup System-Checks
```python
# In main.py vor webview.start()
def run_startup_checks():
    checks = []
    
    # 1. ffmpeg prüfen
    ffmpeg_path = get_resource_path("ffmpeg/bin/ffmpeg.exe")
    checks.append(("ffmpeg", ffmpeg_path.exists(), str(ffmpeg_path)))
    
    # 2. RAVE CLI prüfen
    import subprocess
    try:
        result = subprocess.run(["rave", "--version"], capture_output=True, timeout=5)
        checks.append(("RAVE CLI", result.returncode == 0, result.stdout.decode().strip()))
    except Exception as e:
        checks.append(("RAVE CLI", False, str(e)))
    
    # 3. CUDA/GPU prüfen
    try:
        import torch
        checks.append(("CUDA", torch.cuda.is_available(), f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}"))
    except Exception as e:
        checks.append(("CUDA", False, str(e)))
    
    # 4. Schreibrechte in AppData
    test_dir = Path.home() / "AppData" / "Local" / "RLTA"
    try:
        test_dir.mkdir(parents=True, exist_ok=True)
        test_file = test_dir / "write_test.tmp"
        test_file.write_text("ok")
        test_file.unlink()
        checks.append(("AppData Write", True, str(test_dir)))
    except Exception as e:
        checks.append(("AppData Write", False, str(e)))
    
    # Ergebnisse loggen
    for name, ok, detail in checks:
        status = "✅" if ok else "❌"
        logging.info(f"Startup Check {status} {name}: {detail}")
        if not ok:
            logging.warning(f"KRITISCH: {name} Check fehlgeschlagen!")
    
    return all(ok for _, ok, _ in checks)
```

---

### Step 5: Automatisches Build-Skript & Inno Setup

**Ziel**: Ein Kommando → fertige `Setup.exe`.

#### 5.1 `build.ps1` (PowerShell)
```powershell
# build.ps1
param(
    [string]$Version = "1.0.0",
    [switch]$SkipFrontend,
    [switch]$SkipPyInstaller,
    [switch]$SkipInno
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$FrontendDir = Join-Path $Root "frontend"
$ResourcesDir = Join-Path $Root "resources"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  RLTA Build Pipeline v$Version" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 1. Frontend Build
if (-not $SkipFrontend) {
    Write-Host "`n📦 Building Vue Frontend..." -ForegroundColor Yellow
    Set-Location $FrontendDir
    if (-not (Test-Path "node_modules")) {
        npm install
    }
    npm run build
    
    # dist/ nach resources/frontend/ kopieren
    $DistDir = Join-Path $FrontendDir "dist"
    $TargetDir = Join-Path $ResourcesDir "frontend"
    if (Test-Path $TargetDir) { Remove-Item $TargetDir -Recurse -Force }
    Copy-Item $DistDir $TargetDir -Recurse
    Write-Host "✅ Frontend fertig" -ForegroundColor Green
}

# 2. PyInstaller
if (-not $SkipPyInstaller) {
    Write-Host "`n🔨 Running PyInstaller..." -ForegroundColor Yellow
    Set-Location $Root
    pyinstaller app.spec --clean --noconfirm
    Write-Host "✅ PyInstaller fertig" -ForegroundColor Green
}

# 3. Inno Setup Compiler
if (-not $SkipInno) {
    Write-Host "`n📦 Creating Installer with Inno Setup..." -ForegroundColor Yellow
    $InnoScript = Join-Path $Root "installer.iss"
    if (Test-Path $InnoScript) {
        & "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" $InnoScript
        Write-Host "✅ Installer.exe erstellt" -ForegroundColor Green
    } else {
        Write-Warning "installer.iss nicht gefunden, überspringe Inno Setup"
    }
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Build abgeschlossen!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
```

#### 5.2 `installer.iss` (Inno Setup Script)
```iss
; installer.iss
#define AppName "RLTA"
#define AppVersion "1.0.0"
#define AppPublisher "Dein Name/Org"
#define AppURL "https://github.com/dein-repo"
#define AppExeName "RLTA.exe"

[Setup]
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
AppUpdatesURL={#AppURL}
DefaultDirName={autopf}\{#AppName}
DisableDirPage=yes
DefaultGroupName={#AppName}
AllowNoIcons=yes
OutputDir=..
OutputBaseFilename=RLTA-Setup-{#AppVersion}
SetupIconFile=assets\icon.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64

[Files]
; PyInstaller Output (dist/RLTA/)
Source: "dist\RLTA\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{commondesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Desktop-Verknüpfung erstellen"; GroupDescription: "Zusätzliche Aktionen:"

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{#AppName} starten"; Flags: nowait postinstall skipifsilent
```

---

## Zusammenfassung: Dateien & Struktur

```
project-root/
├── frontend/                 # Vue.js Projekt
│   ├── package.json
│   ├── src/
│   └── dist/                 # nach npm run build
├── resources/
│   └── frontend/             # kopiert aus frontend/dist/ (für PyInstaller)
├── scripts/
│   ├── build_frontend.py     # Step 2
│   └── build.ps1             # Step 5 (Master-Build)
├── app.spec                  # Step 3 (PyInstaller Config)
├── installer.iss             # Step 5 (Inno Setup)
├── main.py                   # Step 1 (Entry Point + Webview)
├── logging_config.py         # Step 4
├── app.py                    # FastAPI Backend (bestehend)
├── windows_manager.py        # System-Checks (bestehend, refactored)
├── train_gui.py              # Training Logic (bestehend)
├── data_pipeline.py          # Preprocessing (bestehend)
├── export_neutone.py         # Export (bestehend)
└── requirements.txt          # mit pywebview, pyinstaller, etc.
```

---

## Requirements-Ergänzung (`requirements.txt`)
```txt
# ... bestehende Deps ...
pywebview==5.0+          # Desktop Window Wrapper
pyinstaller==6.0+        # Bundling (dev dependency)
```

---

## Risiken & Mitigation

| Risiko | Wahrscheinlichkeit | Mitigation |
|--------|-------------------|------------|
| PyInstaller findet CUDA-DLLs nicht | Hoch | `binaries` in `.spec` explizit listen; `--add-binary` testen |
| `pywebview` WebEngine (Edge/Chrome) fehlt | Mittel | Windows 10/11 hat WebView2; Fallback: `webview.platforms.winforms` |
| RAVE `gin`-Configs nicht gefunden | Mittel | Pfade via `sys._MEIPASS` auflösen |
| ffmpeg nicht im Bundle | Hoch | `ffmpeg.exe` in `binaries` aufnehmen; Startup-Check |
| Installer zu groß (>500MB) | Mittel | UPX-Kompression; `excludes` in `.spec` prüfen |
| Antivirus False-Positive | Niedrig | Code-Signing-Zertifikat nutzen |

---

## Nächste Schritte

1. **Step 1**: `main.py` + `pywebview` Integration testen (Dev-Mode)
2. **Step 2**: `build_frontend.py` erstellen, `npm run build` → `resources/frontend/` prüfen
3. **Step 3**: `app.spec` iterativ verfeinern (PyInstaller Logs analysieren)
4. **Step 4**: Logging + Startup-Checks in `main.py` einbauen
5. **Step 5**: `build.ps1` + `installer.iss` → finale `Setup.exe` erzeugen

---

**Erstellt**: 2026-08-06  
**Status**: Plan-Phase