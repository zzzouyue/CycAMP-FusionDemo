"""Report whether the local runtime is ready for later project stages.

Default mode is suitable for S0 before dependencies are installed. Strict mode
requires Python 3.11, all declared scientific packages, CUDA availability, and
sufficient free space on the final D: project drive.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import shutil
import sys


REQUIRED_IMPORTS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "yaml": "PyYAML",
    "joblib": "joblib",
    "Bio": "biopython",
    "rdkit": "rdkit",
    "torch": "pytorch",
    "transformers": "transformers",
    "peft": "peft",
    "accelerate": "accelerate",
    "safetensors": "safetensors",
    "streamlit": "streamlit",
    "py3Dmol": "py3Dmol",
}

# These upstream reference packages currently pin obsolete NumPy/RDKit builds.
# The project implements the required features locally, so their absence must
# not make the reproducible Demo environment appear incomplete.
OPTIONAL_IMPORTS = {
    "modlamp": "modlamp",
    "cyclicpeptide": "cyclicpeptide",
}


def package_status() -> tuple[dict[str, str], list[str]]:
    versions: dict[str, str] = {}
    missing: list[str] = []
    for module_name, display_name in REQUIRED_IMPORTS.items():
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            missing.append(display_name)
            continue
        versions[display_name] = str(getattr(module, "__version__", "installed"))
    return versions, missing


def optional_package_status() -> tuple[dict[str, str], list[str]]:
    versions: dict[str, str] = {}
    missing: list[str] = []
    for module_name, display_name in OPTIONAL_IMPORTS.items():
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            missing.append(display_name)
            continue
        versions[display_name] = str(getattr(module, "__version__", "installed"))
    return versions, missing


def d_drive_free_gib() -> float | None:
    root = Path("D:/")
    if not root.exists():
        return None
    return round(shutil.disk_usage(root).free / (1024**3), 2)


def cuda_status() -> dict[str, object]:
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return {"torch_installed": False, "cuda_available": False, "device": None}
    available = bool(torch.cuda.is_available())
    device = torch.cuda.get_device_name(0) if available else None
    return {"torch_installed": True, "cuda_available": available, "device": device}


def run(strict: bool = False) -> int:
    versions, missing = package_status()
    optional_versions, optional_missing = optional_package_status()
    free_gib = d_drive_free_gib()
    cuda = cuda_status()
    python_ok = sys.version_info[:2] == (3, 11)
    space_ok = free_gib is not None and free_gib >= 45.0
    report = {
        "mode": "strict" if strict else "report-only",
        "python": sys.version.split()[0],
        "python_3_11": python_ok,
        "executable": sys.executable,
        "d_drive_free_gib": free_gib,
        "d_drive_has_45_gib": space_ok,
        "packages": versions,
        "missing_packages": missing,
        "optional_packages": optional_versions,
        "missing_optional_packages": optional_missing,
        "cuda": cuda,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not strict:
        return 0
    failures = []
    if not python_ok:
        failures.append("Python must be 3.11")
    if not space_ok:
        failures.append("D: must have at least 45 GiB free")
    if missing:
        failures.append("missing packages: " + ", ".join(missing))
    if not cuda["cuda_available"]:
        failures.append("CUDA is unavailable")
    if failures:
        print("STRICT CHECK FAILED: " + "; ".join(failures), file=sys.stderr)
        return 1
    print("STRICT CHECK PASSED")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="fail if the complete ML environment is not ready")
    args = parser.parse_args()
    return run(strict=args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
