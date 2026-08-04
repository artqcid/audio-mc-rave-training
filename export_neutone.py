import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict


@dataclass
class NeutoneMetadata:
    name: str
    author: str
    description: str
    version: str = "1.0"
    parameters: Dict[str, object] = None


def export_to_neutone(model_path: str, output_path: str, author: str = "RLTA") -> str:
    source = Path(model_path)
    if not source.exists():
        raise FileNotFoundError(f"Modellpfad nicht gefunden: {model_path}")

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    metadata = NeutoneMetadata(
        name=target.stem,
        author=author,
        description=f"Neutone FX Export für {source.name}",
        parameters={"model_source": str(source), "exported_by": "RLTA"},
    )

    with target.open("wb") as handle:
        handle.write(b"NEUTONE\n")
        handle.write(json.dumps(asdict(metadata), ensure_ascii=False, indent=2).encode("utf-8"))

    return str(target.resolve())


def validate_neutone_package(path: str) -> bool:
    package = Path(path)
    if not package.exists() or package.stat().st_size == 0:
        return False
    with package.open("rb") as handle:
        content = handle.read().decode("utf-8", errors="ignore")
    return "NEUTONE" in content and "exported_by" in content
