#!/usr/bin/env python
"""Import audited HighFold2 relaxed rank-001 models as 3D descriptor records."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cycamp.structure import (DESCRIPTOR_NAMES, StructureResult, assess_m2_eligibility,
                              calculate_3d_descriptors, structure_join_key, write_failure_log)

DISCLAIMER = "HighFold2预测结构模型，仅用于预测结构模型衍生描述符；不代表实验测定结构、真实溶液构象或已验证活性构象。"


def atoms_by_residue(path: Path) -> dict[tuple[str, int], tuple[float, float, float]]:
    atoms = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("ATOM", "HETATM")):
            atoms[(line[12:16].strip(), int(line[22:26]))] = tuple(float(line[x:y]) for x, y in ((30, 38), (38, 46), (46, 54)))
    return atoms


def distance(a, b) -> float:
    return math.dist(a, b)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, default=PROJECT_ROOT / "data" / "processed" / "cyclic_core.csv")
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "structures_highfold")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "structures_highfold")
    args = parser.parse_args()
    from rdkit import Chem
    from rdkit.Chem import AllChem

    with args.core.open(encoding="utf-8-sig", newline="") as h:
        core = {row["peptide_id"]: row for row in csv.DictReader(h)}
    with (args.input_dir / "manifest.csv").open(encoding="utf-8-sig", newline="") as h:
        manifest = list(csv.DictReader(h))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, results, plddts, topology_failures = [], [], [], []
    for item in manifest:
        pid = item["peptide_id"]
        row = core.get(pid)
        result = StructureResult(sequence=(row or item).get("sequence", ""), canonical_sequence=(row or item).get("canonical_sequence", ""), peptide_id=pid, success=False)
        try:
            if row is None:
                raise ValueError("peptide_id is absent from cyclic core")
            directory = args.input_dir / "results" / pid
            pdbs = list(directory.glob("*_relaxed_rank_001_*.pdb"))
            scores = list(directory.glob("*_scores_rank_001_*.json"))
            if len(pdbs) != 1 or len(scores) != 1:
                raise ValueError(f"expected one relaxed rank-001 PDB and score JSON, got {len(pdbs)}/{len(scores)}")
            atoms = atoms_by_residue(pdbs[0])
            pairs = [int(x) for x in item["disulfide_pairs_flat"].split()] if item["disulfide_pairs_flat"].strip() else []
            if len(pairs) % 2:
                raise ValueError("odd disulfide index count")
            for left, right in zip(pairs[::2], pairs[1::2]):
                d = distance(atoms[("SG", left)], atoms[("SG", right)])
                if not 1.8 <= d <= 2.5:
                    raise ValueError(f"SS {left}-{right} distance {d:.2f} A outside [1.8,2.5]")
            if item["flag_cyclic"] == "1":
                residues = sorted({residue for _, residue in atoms})
                d = distance(atoms[("N", residues[0])], atoms[("C", residues[-1])])
                if not 1.2 <= d <= 1.6:
                    raise ValueError(f"N-C closure distance {d:.2f} A outside [1.2,1.6]")
            payload = json.loads(scores[0].read_text(encoding="utf-8"))
            values = payload.get("plddt")
            if not isinstance(values, list) or not values or not all(math.isfinite(float(x)) for x in values):
                raise ValueError("invalid pLDDT payload")
            plddt = sum(map(float, values)) / len(values)
            molecule = Chem.MolFromPDBFile(str(pdbs[0]), removeHs=False, sanitize=True)
            if molecule is None or molecule.GetNumConformers() != 1:
                raise ValueError("RDKit could not parse a single PDB conformer")
            properties = AllChem.MMFFGetMoleculeProperties(molecule, mmffVariant="MMFF94s")
            force_field = AllChem.MMFFGetMoleculeForceField(molecule, properties, confId=0) if properties else None
            if force_field is None:
                raise ValueError("MMFF94s single-point energy unavailable")
            descriptors = calculate_3d_descriptors(molecule, 0, float(force_field.CalcEnergy()))
            topology = row["topology_key"]
            result = StructureResult(sequence=row["sequence"], canonical_sequence=row["canonical_sequence"], peptide_id=pid,
                cyclization_type=row["cyclization_type"], topology=topology,
                join_key=structure_join_key(pid, row["cyclization_type"], topology), success=True,
                structure_path=str(pdbs[0]), builder="HighFold2 alphafold2_multimer_v3", force_field="MMFF94s_single_point_fixed_model_geometry", conformers_requested=1, conformers_generated=1, selected_conformer_id=0, descriptors=descriptors, disclaimer=DISCLAIMER)
            record = result.to_record()
            record.update({"structure_source": "HighFold2", "rank": 1, "mean_plddt": plddt,
                           "energy_semantics": "MMFF94s single-point energy at relaxed HighFold2 model coordinates; no RDKit geometry optimization performed"})
            records.append(record); plddts.append(plddt)
        except Exception as exc:
            result.failure_reason = f"{type(exc).__name__}: {exc}"; topology_failures.append({"peptide_id": pid, "reason": result.failure_reason})
        results.append(result)
    fields = list(StructureResult(sequence="", canonical_sequence="", success=True).to_record()) + list(DESCRIPTOR_NAMES) + ["structure_source", "rank", "mean_plddt", "energy_semantics"]
    with (args.output_dir / "structure_descriptors.csv").open("w", encoding="utf-8-sig", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=list(dict.fromkeys(fields)), extrasaction="ignore"); writer.writeheader(); writer.writerows(records)
    write_failure_log(results, args.output_dir / "structure_failures.json")
    labels = [int(core[item["peptide_id"]]["label"]) for item in manifest]
    qc = assess_m2_eligibility(results, labels)
    qc.update({"structure_source": "HighFold2 alphafold2_multimer_v3", "source_disclaimer": DISCLAIMER,
               "topology_qc": {"ss_distance_window_a": [1.8, 2.5], "nc_distance_window_a": [1.2, 1.6], "failures": topology_failures},
               "plddt": {"count": len(plddts), "mean": sum(plddts)/len(plddts) if plddts else None, "min": min(plddts) if plddts else None, "max": max(plddts) if plddts else None},
               "energy_semantics": "MMFF94s single-point energy at relaxed HighFold2 model coordinates; no RDKit geometry optimization performed"})
    (args.output_dir / "structure_qc.json").write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"imported": len(records), "failed": len(results)-len(records), "formal_m2_allowed": qc["formal_m2_allowed"]}, ensure_ascii=False))
    return 0 if len(records) == len(manifest) else 1


if __name__ == "__main__":
    raise SystemExit(main())
