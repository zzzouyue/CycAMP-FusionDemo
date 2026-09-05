from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(r"D:\CycAMP-FusionDemo")
OUT = ROOT / "outputs" / "CycAMP-FusionDemo-demo.zip"
INCLUDE_DIRS = [
    "src", "scripts", "tests", "configs", "handoffs",
    "artifacts/metrics", "artifacts/models", "artifacts/embeddings",
    "artifacts/structures_m2", "data/processed",
]
INCLUDE_FILES = [
    ".gitignore", "AGENTS.md", "DECISIONS.md", "PROJECT_STATE.yaml", "README.md",
    "app.py", "environment.yml", "environment-lock.yml", "environment-lock.txt", "requirements.txt",
    "outputs/CycAMP-FusionDemo-科研基础训练报告.docx",
    "outputs/CycAMP-FusionDemo-科研基础训练报告.pdf",
]

def allowed(path: Path) -> bool:
    return path.is_file() and path.suffix not in {".pyc", ".pyo", ".tmp"} and "__pycache__" not in path.parts

files = []
for rel in INCLUDE_FILES:
    p = ROOT / rel
    if allowed(p):
        files.append(p)
for rel_dir in INCLUDE_DIRS:
    base = ROOT / rel_dir
    if base.exists():
        files.extend(p for p in base.rglob("*") if allowed(p))

with ZipFile(OUT, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
    for path in sorted(set(files)):
        archive.write(path, Path("CycAMP-FusionDemo") / path.relative_to(ROOT))

print(f"{OUT} ({len(files)} files)")
