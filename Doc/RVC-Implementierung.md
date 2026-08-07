# RVC-Workflow Implementierung

## Überblick

Der RVC (Retrieval-based Voice Conversion) Workflow ist ein Sprachmodell-Trainingsprozess, der ähnlich wie Applio funktioniert. Im Gegensatz zum RAVE-Workflow (Sound Design) fokussiert er sich auf Sprachmodelle für eigene Stimmen.

## Architektur

### Workflow-Toggle
Der Benutzer wählt beim Start zwischen zwei Workflows:
- **RAVE**: Für Sound-Design und Audio-Transformation
- **RVC**: Für Sprachmodelle (ähnlich Applio)

### UI-Struktur
```
Workflow-Auswahl
├── RAVE-Modus
│   ├── Trainingsmodell-Auswahl (raspberry, rave_mini, etc.)
│   ├── Dataset-Pfad
│   ├── Batch-Größe, Epochen, Learning Rate
│   └── Latent Dimension
└── RVC-Modus
    ├── Model Name
    ├── Dataset Path
    ├── Sample Rate (40k, 40k-v2)
    ├── Pitch Extraction Algorithm (rmvpe)
    ├── Total Epochs (200-300)
    ├── Save Frequency (25, 50)
    ├── Batch Size (8, 16, 4)
    ├── Save Only Latest
    └── Index Training
```

## RVC-spezifische Komponenten

### 1. Frontend (`templates/index.html`)

#### Workflow-Toggle
```html
<div class="workflow-selector" style="margin-bottom: 20px;">
  <div style="display: flex; gap: 10px;">
    <button 
      :class="['secondary', workflow === 'rave' ? '' : 'secondary']" 
      @click="setWorkflow('rave')"
      style="flex: 1;"
    >RAVE (Sound Design)</button>
    <button 
      :class="['secondary', workflow === 'rvc' ? '' : 'secondary']" 
      @click="setWorkflow('rvc')"
      style="flex: 1;"
    >RVC (Sprachmodell)</button>
  </div>
</div>
```

#### RVC-spezifische Trainingsfelder
```html
<div v-if="workflow === 'rvc'" class="form-row">
  <label for="sample_rate">Sample Rate</label>
  <select id="sample_rate" v-model="rvcForm.sample_rate">
    <option value="40000">40k</option>
    <option value="40000">40k-v2</option>
  </select>

  <label for="pitch_extractor">Pitch Extraction Algorithm</label>
  <select id="pitch_extractor" v-model="rvcForm.pitch_extractor">
    <option value="rmvpe">rmvpe (Standard)</option>
    <option value="crepe">crepe</option>
    <option value="harvest">harvest</option>
  </select>

  <label for="total_epochs">Total Epochs</label>
  <input id="total_epochs" type="number" v-model.number="rvcForm.total_epochs" min="1" max="500" />
  <p class="small-note">200-300 für 10-15 Minuten Sprachmaterial.</p>

  <label for="save_frequency">Save Frequency</label>
  <input id="save_frequency" type="number" v-model.number="rvcForm.save_frequency" min="1" />
  <p class="small-note">Wie oft ein Checkpoint gespeichert wird (25 oder 50 empfohlen).</p>

  <label for="save_only_latest">Save Only Latest</label>
  <select id="save_only_latest" v-model="rvcForm.save_only_latest">
    <option :value="true">Ja (Speicher sparen)</option>
    <option :value="false">Nein (alle Checkpoints behalten)</option>
  </select>

  <label for="train_index">Index-Datei erstellen</label>
  <select id="train_index" v-model="rvcForm.train_index">
    <option :value="true">Ja</option>
    <option :value="false">Nein</option>
  </select>
</div>
```

#### RVC-spezifische Export-Optionen
```html
<div v-if="workflow === 'rvc'" class="form-row">
  <label for="index_file">Index-Datei (.index)</label>
  <select id="index_file" v-model="exportForm.index_file">
    <option value="">Keine Index-Datei</option>
    <option v-for="file in indexFiles" :key="file" :value="file" v-text="file"></option>
  </select>
  <button class="secondary" type="button" @click="pickFile('index')">Index-Datei wählen</button>
  <input ref="indexFile" type="file" accept=".index" style="display:none" @change="onIndexFilePick" />
</div>
```

### 2. Backend (`app.py`)

#### Neue API-Endpunkte
```python
@app.post("/api/rvc/preprocess")
def api_rvc_preprocess(
    dataset_path: str = Form("dataset"),
    model_name: str = Form("rvc_model"),
    sample_rate: str = Form("40000"),
    dataset_files: Optional[List[UploadFile]] = File(None),
):
    """RVC-spezifisches Preprocessing: Schneiden in 3-5s Clips, Stille entfernen."""
    # Implementation folgt Applio-ähnlichem Workflow

@app.post("/api/rvc/extract-features")
def api_rvc_extract_features(
    model_name: str = Form("rvc_model"),
    pitch_extractor: str = Form("rmvpe"),
    dataset_path: str = Form("dataset"),
):
    """Pitch Extraction mit rmvpe oder alternativen Algorithmen."""
    # Implementation folgt Applio-ähnlichem Workflow

@app.post("/api/rvc/train-index")
def api_rvc_train_index(
    model_name: str = Form("rvc_model"),
    dataset_path: str = Form("dataset"),
):
    """Erstellt eine .index Datei für feinere Klangnuancen."""
    # Implementation folgt Applio-ähnlichem Workflow

@app.get("/api/rvc/models")
def api_rvc_models():
    """Listet verfügbare RVC-Modelle und Index-Dateien auf."""
    return {
        "models": list_rvc_models(),
        "index_files": list_rvc_index_files(),
    }
```

#### Erweiterte `/api/train` Route
```python
@app.post("/api/train")
def api_train(
    # ... existing parameters ...
    workflow: str = Form("rave"),  # 'rave' or 'rvc'
    # RVC-specific parameters
    sample_rate: str = Form("40000"),
    pitch_extractor: str = Form("rmvpe"),
    total_epochs: int = Form(200),
    save_frequency: int = Form(50),
    save_only_latest: bool = Form(True),
    train_index: bool = Form(True),
):
    if workflow == "rvc":
        return _start_rvc_training(...)
    else:
        return _start_rave_training(...)
```

### 3. Trainingsmanager (`train_gui.py`)

#### RVC-spezifische Konfiguration
```python
@dataclass
class RVCTrainingConfig:
    model_name: str = "rvc_model"
    dataset_path: str = "dataset"
    sample_rate: str = "40000"  # 40k or 40k-v2
    pitch_extractor: str = "rmvpe"
    total_epochs: int = 200
    batch_size: int = 8
    save_frequency: int = 50
    save_only_latest: bool = True
    train_index: bool = True
    learning_rate: float = 0.0002
    use_gpu: bool = True
    output_path: str = "logs/rvc_model/rvc_model.pth"
    index_path: str = "logs/rvc_model/added_rvc_model.index"
```

#### RVC-spezifische Trainingslogik
```python
class RVCTrainingManager:
    def start_rvc_training(self, config: RVCTrainingConfig) -> Dict[str, object]:
        """Start RVC training with Applio-like workflow."""
        # 1. Preprocessing
        # 2. Feature Extraction
        # 3. Model Training
        # 4. Index Training
        pass
```

### 4. Daten-Pfade

#### RVC-spezifische Pfade
| Komponente | Pfad |
|------------|------|
| Modell | `logs/{model_name}/{model_name}.pth` |
| Index | `logs/{model_name}/added_{model_name}.index` |
| Dataset | `dataset/{model_name}/` |
| Preprocessing | `dataset/{model_name}/preprocessed/` |

## RVC-Workflow Schritte

### Schritt 1: Dataset vorbereiten
- **Länge**: 10-15 Minuten sauberes Sprachmaterial
- **Qualität**: Trocken, kein Hall, kein Echo, keine Hintergrundgeräusche
- **Format**: WAV (44.1 kHz oder 48 kHz, 16/24 Bit)

### Schritt 2: Preprocessing
- **Aktion**: Schneiden in 3-5 Sekunden Clips
- **Stille entfernen**: Automatische Erkennung und Entfernung
- **Normalisierung**: Peak-Normalisierung

### Schritt 3: Pitch Extraction
- **Standard**: rmvpe
- **Alternativen**: crepe, harvest
- **Ergebnis**: Pitch-Dateien für Training

### Schritt 4: Model Training
- **Epochen**: 200-300
- **Batch Size**: 8-16 (8 GB VRAM), 4 (4-6 GB VRAM)
- **Save Frequency**: 25 oder 50
- **Save Only Latest**: Für Speicherersparnis

### Schritt 5: Index-Datei erstellen
- **Zweck**: Feinere Klangnuancen, stabiler Inferenz
- **Format**: `.index` Datei
- **Pfad**: `logs/{model_name}/added_{model_name}.index`

### Schritt 6: Neutone-Export
- **Eingabe**: `.pth` Modell + `.index` Datei
- **Ausgabe**: `.nm` Datei für Neutone FX
- **Workflow**: Konvertierung über SDK-Wrapping

## Implementierungsschritte

### Phase 1: Grundgerüst
1. Workflow-Toggle im Frontend
2. RVC-spezifische Datenstrukturen im Backend
3. Neue API-Endpunkte

### Phase 2: RVC-Training
1. RVC-Preprocessing-Logik
2. Pitch Extraction Integration
3. RVC-Training-Manager

### Phase 3: Index & Export
1. Index-Datei-Training
2. RVC-Neutone-Export
3. Index-Datei-Verwaltung

## Technische Anforderungen

### Abhängigkeiten
```bash
# RVC-spezifische Pakete
pip install pyworld  # Für Pitch Extraction
pip install librosa  # Für Audio-Processing
pip install scikit-learn  # Für Index-Training
```

### VRAM-Anforderungen
| Batch Size | VRAM |
|------------|------|
| 4 | 4-6 GB |
| 8 | 8 GB |
| 16 | 12+ GB |

## Fazit

Der RVC-Workflow erfordert:
1. **Frontend**: Workflow-Toggle, RVC-spezifische UI-Elemente
2. **Backend**: Neue API-Endpunkte, RVC-spezifische Konfiguration
3. **Trainingsmanager**: RVC-spezifische Trainingslogik
4. **Export**: RVC-Neutone-Export mit Index-Datei-Unterstützung

Dies ist eine umfassende Erweiterung, die den gesamten Workflow erweitert.