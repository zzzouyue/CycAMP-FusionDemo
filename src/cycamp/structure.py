"""RDKit three-dimensional candidate conformers for head-to-tail cyclic peptides."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .embeddings import HEAD_TO_TAIL, normalize_cyclization_type
from .sequence import canonical_cyclic_sequence, normalize_sequence


CANDIDATE_CONFORMER_DISCLAIMER = (
    "RDKit快速生成的三维候选构象，仅用于特征计算和可视化，"
    "不代表实验测定结构或高精度结构预测结果。"
)
DESCRIPTOR_NAMES = (
    "tpsa",
    "molecular_volume",
    "radius_of_gyration",
    "eccentricity",
    "asphericity",
    "spherocity_index",
    "pmi1_over_pmi3",
    "pmi2_over_pmi3",
    "max_interatomic_distance",
    "minimum_conformer_energy",
)


def deterministic_conformer_seed(sequence: str) -> int:
    canonical = canonical_cyclic_sequence(sequence)
    return int(hashlib.sha256(canonical.encode("ascii")).hexdigest()[:8], 16) & 0x7FFFFFFF


@dataclass
class StructureResult:
    sequence: str
    canonical_sequence: str
    success: bool
    peptide_id: str = ""
    cyclization_type: str = HEAD_TO_TAIL
    topology: str = ""
    join_key: str = ""
    structure_path: str | None = None
    smiles: str | None = None
    builder: str | None = None
    force_field: str | None = None
    conformers_requested: int = 10
    conformers_generated: int = 0
    selected_conformer_id: int | None = None
    descriptors: dict[str, float] | None = None
    failure_reason: str | None = None
    disclaimer: str = CANDIDATE_CONFORMER_DISCLAIMER

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        descriptors = record.pop("descriptors") or {}
        record.update(descriptors)
        return record


def _load_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors3D, rdMolDescriptors
    except ImportError as exc:
        raise RuntimeError("structure generation requires RDKit") from exc
    return Chem, AllChem, Descriptors3D, rdMolDescriptors


def _from_cyclicpeptide(sequence: str):
    """Use cyclicpeptide's documented essential-amino-acid constructor."""

    try:
        from cyclicpeptide import Sequence2Structure
        smiles, molecule = Sequence2Structure.seq2stru_essentialAA(sequence=sequence, cyclic=True)
    except (ImportError, AttributeError, TypeError):
        return None
    if molecule is None:
        return None
    return molecule, smiles, "cyclicpeptide.Sequence2Structure"


def _from_rdkit_head_to_tail(sequence: str):
    """Auditable fallback: close RDKit's linear peptide C terminus onto N terminus."""

    Chem, _, _, _ = _load_rdkit()
    molecule = Chem.MolFromSequence(sequence)
    if molecule is None:
        raise ValueError("RDKit could not parse peptide sequence")
    n_matches = molecule.GetSubstructMatches(Chem.MolFromSmarts("[N;H1,H2;!$(N-C=O)]"))
    c_matches = molecule.GetSubstructMatches(Chem.MolFromSmarts("[C](=O)[O;H1,-1]"))
    if not n_matches or not c_matches:
        raise ValueError("could not identify peptide termini")
    n_index = n_matches[0][0]
    c_index, _, hydroxyl_index = c_matches[-1]
    editable = Chem.RWMol(molecule)
    editable.RemoveAtom(hydroxyl_index)
    if hydroxyl_index < n_index:
        n_index -= 1
    if hydroxyl_index < c_index:
        c_index -= 1
    editable.AddBond(c_index, n_index, Chem.BondType.SINGLE)
    cyclic = editable.GetMol()
    Chem.SanitizeMol(cyclic)
    return cyclic, Chem.MolToSmiles(cyclic, canonical=True), "rdkit_head_to_tail_fallback"


def build_head_to_tail_molecule(sequence: str):
    sequence = normalize_sequence(sequence)
    built = _from_cyclicpeptide(sequence)
    return built if built is not None else _from_rdkit_head_to_tail(sequence)


def _normalized_bond_pairs(bond_pairs: list[tuple[int, int]] | None) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((min(int(left), int(right)), max(int(left), int(right))) for left, right in (bond_pairs or [])))


def _explicit_molecule(smiles: str):
    Chem, _, _, _ = _load_rdkit()
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("explicit SMILES could not be parsed")
    return molecule, Chem.MolToSmiles(molecule, canonical=True), "explicit_smiles"


def normalize_topology_bonds(topology_bonds: list[list[Any]] | None) -> tuple[tuple[Any, ...], ...]:
    """Validate the DBAASP bond-record shape without interpreting unknown codes."""

    normalized = []
    for bond in topology_bonds or []:
        if not isinstance(bond, (list, tuple)) or len(bond) < 3:
            raise ValueError("topology bonds must contain residue1, residue2 and bond code")
        left, right = int(bond[0]), int(bond[1])
        if left < 0 or right < 1 or left == right:
            raise ValueError("topology bond residue indices are invalid")
        normalized.append((left, right, *(str(item) for item in bond[2:])))
    return tuple(normalized)


def _add_explicit_disulfides(molecule, sequence: str, bonds: tuple[tuple[Any, ...], ...]):
    Chem, _, _, _ = _load_rdkit()
    sulfur_by_residue: dict[int, int] = {}
    for atom in molecule.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if atom.GetSymbol() == "S" and info is not None:
            sulfur_by_residue[int(info.GetResidueNumber())] = atom.GetIdx()
    editable = Chem.RWMol(molecule)
    for bond in bonds:
        left, right, code = int(bond[0]), int(bond[1]), str(bond[2]).upper()
        if code != "DSB":
            continue
        if sequence[left - 1] != "C" or sequence[right - 1] != "C":
            raise ValueError(f"DSB {left}-{right} does not connect two cysteine residues")
        try:
            left_atom, right_atom = sulfur_by_residue[left], sulfur_by_residue[right]
        except KeyError as exc:
            raise ValueError(f"could not map DSB residue {exc.args[0]} to a sulfur atom") from exc
        if editable.GetBondBetweenAtoms(left_atom, right_atom) is not None:
            raise ValueError(f"duplicate DSB bond {left}-{right}")
        editable.AddBond(left_atom, right_atom, Chem.BondType.SINGLE)
    result = editable.GetMol()
    Chem.SanitizeMol(result)
    return result


def topology_label(
    sequence: str,
    cyclization_type: str,
    *,
    smiles: str | None = None,
    bond_pairs: list[tuple[int, int]] | None = None,
    topology_bonds: list[list[Any]] | None = None,
) -> str:
    normalized_type = normalize_cyclization_type(cyclization_type)
    if smiles:
        digest = hashlib.sha256(smiles.strip().encode("utf-8")).hexdigest()[:16]
        bonds = _normalized_bond_pairs(bond_pairs)
        return f"explicit_smiles:{digest}:bonds={bonds}"
    if topology_bonds:
        normalized = normalize_topology_bonds(topology_bonds)
        payload = json.dumps(normalized, separators=(",", ":"), ensure_ascii=True)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return f"dbaasp_topology:{digest}"
    if normalized_type == HEAD_TO_TAIL:
        return f"head_to_tail:1-{len(normalize_sequence(sequence))}"
    if bond_pairs:
        return f"{normalized_type}:residue_bonds={_normalized_bond_pairs(bond_pairs)}"
    return f"{normalized_type}:unspecified"


def structure_join_key(peptide_id: str, cyclization_type: str, topology: str) -> str:
    """M2 join identity; sequence alone is intentionally insufficient."""

    return "|".join((str(peptide_id), normalize_cyclization_type(cyclization_type), topology))


def build_molecule_for_topology(
    sequence: str,
    cyclization_type: str,
    *,
    smiles: str | None = None,
    bond_pairs: list[tuple[int, int]] | None = None,
    topology_bonds: list[list[Any]] | None = None,
):
    """Build only topology that is explicit; never infer non-head-to-tail bonds."""

    normalized_type = normalize_cyclization_type(cyclization_type)
    if smiles and smiles.strip():
        return _explicit_molecule(smiles.strip())
    if topology_bonds:
        bonds = normalize_topology_bonds(topology_bonds)
        supported = {str(bond[2]).upper() for bond in bonds}
        unsupported = supported - {"AMD", "DSB"}
        if unsupported:
            raise ValueError(f"unsupported explicit topology bond codes: {sorted(unsupported)}")
        amide_bonds = [bond for bond in bonds if str(bond[2]).upper() == "AMD"]
        disulfides = [bond for bond in bonds if str(bond[2]).upper() == "DSB"]
        if normalized_type == HEAD_TO_TAIL:
            if len(amide_bonds) != 1 or {int(amide_bonds[0][0]), int(amide_bonds[0][1])} != {1, len(sequence)}:
                raise ValueError("head_to_tail topology requires one explicit terminal AMD bond")
            molecule, _, _ = _from_rdkit_head_to_tail(sequence)
        elif normalized_type == "disulfide":
            if amide_bonds or not disulfides:
                raise ValueError("disulfide topology must contain DSB bonds and no AMD bond")
            Chem, _, _, _ = _load_rdkit()
            molecule = Chem.MolFromSequence(sequence)
            if molecule is None:
                raise ValueError("RDKit could not parse peptide sequence")
        else:
            raise ValueError(f"unsupported cyclization type for explicit topology: {normalized_type}")
        molecule = _add_explicit_disulfides(molecule, sequence, bonds)
        Chem, _, _, _ = _load_rdkit()
        return molecule, Chem.MolToSmiles(molecule, canonical=True), "dbaasp_explicit_amd_dsb"
    if normalized_type == HEAD_TO_TAIL:
        return build_head_to_tail_molecule(sequence)
    if bond_pairs:
        raise ValueError(
            "non-head-to-tail residue bond_pairs require an explicit structure/atom mapping; "
            "automatic atom-level bonds are intentionally not guessed"
        )
    raise ValueError(
        "non-head-to-tail topology requires explicit SMILES or a supported explicit bond mapping"
    )


def _optimize_conformers(molecule, conformer_ids: list[int]):
    _, AllChem, _, _ = _load_rdkit()
    mmff_energies: list[tuple[float, int, str]] = []
    mmff = AllChem.MMFFGetMoleculeProperties(molecule, mmffVariant="MMFF94s")
    if mmff is not None:
        for conformer_id in conformer_ids:
            try:
                field = AllChem.MMFFGetMoleculeForceField(
                    molecule, mmff, confId=conformer_id
                )
                field.Minimize(maxIts=500)
                mmff_energies.append((float(field.CalcEnergy()), conformer_id, "MMFF94s"))
            except Exception:
                continue
    if mmff_energies:
        return min(mmff_energies, key=lambda item: item[0])
    uff_energies: list[tuple[float, int, str]] = []
    for conformer_id in conformer_ids:
        try:
            field = AllChem.UFFGetMoleculeForceField(molecule, confId=conformer_id)
            field.Minimize(maxIts=500)
            uff_energies.append((float(field.CalcEnergy()), conformer_id, "UFF"))
        except Exception:
            continue
    if not uff_energies:
        raise ValueError("all MMFF and UFF conformer optimizations failed")
    return min(uff_energies, key=lambda item: item[0])


def calculate_3d_descriptors(molecule, conformer_id: int, energy: float) -> dict[str, float]:
    _, AllChem, Descriptors3D, rdMolDescriptors = _load_rdkit()
    conformer = molecule.GetConformer(conformer_id)
    maximum_distance = 0.0
    positions = conformer.GetPositions()
    for left in range(len(positions)):
        for right in range(left + 1, len(positions)):
            distance = math.sqrt(float(((positions[left] - positions[right]) ** 2).sum()))
            maximum_distance = max(maximum_distance, distance)
    pmi3 = float(Descriptors3D.PMI3(molecule, confId=conformer_id))
    descriptors = {
        "tpsa": float(rdMolDescriptors.CalcTPSA(molecule)),
        "molecular_volume": float(AllChem.ComputeMolVolume(molecule, confId=conformer_id)),
        "radius_of_gyration": float(Descriptors3D.RadiusOfGyration(molecule, confId=conformer_id)),
        "eccentricity": float(Descriptors3D.Eccentricity(molecule, confId=conformer_id)),
        "asphericity": float(Descriptors3D.Asphericity(molecule, confId=conformer_id)),
        "spherocity_index": float(Descriptors3D.SpherocityIndex(molecule, confId=conformer_id)),
        "pmi1_over_pmi3": float(Descriptors3D.PMI1(molecule, confId=conformer_id) / pmi3) if pmi3 else 0.0,
        "pmi2_over_pmi3": float(Descriptors3D.PMI2(molecule, confId=conformer_id) / pmi3) if pmi3 else 0.0,
        "max_interatomic_distance": maximum_distance,
        "minimum_conformer_energy": float(energy),
    }
    if set(descriptors) != set(DESCRIPTOR_NAMES) or not all(
        math.isfinite(value) for value in descriptors.values()
    ):
        raise ValueError("non-finite or incomplete three-dimensional descriptors")
    return descriptors


def generate_candidate_conformer(
    sequence: str,
    output_path: str | Path,
    *,
    peptide_id: str | None = None,
    cyclization_type: str = HEAD_TO_TAIL,
    smiles: str | None = None,
    bond_pairs: list[tuple[int, int]] | None = None,
    topology_bonds: list[list[Any]] | None = None,
    num_conformers: int = 10,
    retry_conformers: int = 20,
    embedding_timeout_seconds: int = 30,
) -> StructureResult:
    """Build, embed, optimize and save one lowest-energy candidate conformer."""

    if num_conformers < 1 or retry_conformers < num_conformers:
        raise ValueError("conformer counts must satisfy 1 <= num_conformers <= retry_conformers")
    if embedding_timeout_seconds < 1:
        raise ValueError("embedding_timeout_seconds must be positive")
    normalized = normalize_sequence(sequence)
    normalized_type = normalize_cyclization_type(cyclization_type)
    canonical = (
        canonical_cyclic_sequence(normalized)
        if normalized_type == HEAD_TO_TAIL
        else normalized
    )
    topology = topology_label(
        normalized,
        normalized_type,
        smiles=smiles,
        bond_pairs=bond_pairs,
        topology_bonds=topology_bonds,
    )
    identifier = str(peptide_id or canonical)
    result = StructureResult(
        peptide_id=identifier,
        sequence=normalized,
        canonical_sequence=canonical,
        cyclization_type=normalized_type,
        topology=topology,
        join_key=structure_join_key(identifier, normalized_type, topology),
        success=False,
        conformers_requested=num_conformers,
    )
    try:
        molecule, resolved_smiles, builder = build_molecule_for_topology(
            normalized,
            normalized_type,
            smiles=smiles,
            bond_pairs=bond_pairs,
            topology_bonds=topology_bonds,
        )
        Chem, AllChem, _, _ = _load_rdkit()
        molecule = Chem.AddHs(molecule)
        params = AllChem.ETKDGv3()
        params.useMacrocycleTorsions = True
        params.randomSeed = deterministic_conformer_seed(canonical)
        params.pruneRmsThresh = 0.5
        params.timeout = int(embedding_timeout_seconds)
        params.numThreads = 0
        conformer_ids = list(AllChem.EmbedMultipleConfs(molecule, numConfs=num_conformers, params=params))
        if not conformer_ids and retry_conformers > num_conformers:
            params.pruneRmsThresh = -1.0
            conformer_ids = list(
                AllChem.EmbedMultipleConfs(molecule, numConfs=retry_conformers, params=params)
            )
            result.conformers_requested = retry_conformers
        if not conformer_ids:
            raise ValueError("ETKDGv3 generated no conformers")
        energy, selected_id, force_field = _optimize_conformers(molecule, conformer_ids)
        descriptors = calculate_3d_descriptors(molecule, selected_id, energy)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = Chem.SDWriter(str(output_path))
        molecule.SetProp("_Name", canonical)
        molecule.SetProp("conformer_notice", CANDIDATE_CONFORMER_DISCLAIMER)
        writer.write(molecule, confId=selected_id)
        writer.close()
        result.success = True
        result.structure_path = str(output_path)
        result.smiles = resolved_smiles
        result.builder = builder
        result.force_field = force_field
        result.conformers_generated = len(conformer_ids)
        result.selected_conformer_id = selected_id
        result.descriptors = descriptors
    except Exception as exc:
        result.failure_reason = f"{type(exc).__name__}: {exc}"
    return result


def write_failure_log(results: list[StructureResult], path: str | Path) -> Path:
    failures = [result.to_record() for result in results if not result.success]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def assess_m2_eligibility(
    results: list[StructureResult], labels: list[int] | None
) -> dict[str, Any]:
    """Apply formal-vs-exploratory M2 gates to the structure subset."""

    total = len(results)
    successful_indices = [index for index, result in enumerate(results) if result.success]
    success_count = len(successful_indices)
    success_rate = success_count / total if total else 0.0
    minority_count: int | None = None
    reasons: list[str] = []
    if success_rate < 0.70:
        reasons.append("structure success rate is below 70%")
    if success_count < 50:
        reasons.append("successful structure subset has fewer than 50 peptides")
    if labels is None or len(labels) != total:
        reasons.append("aligned binary labels are unavailable")
    else:
        successful_labels = [int(labels[index]) for index in successful_indices]
        counts = [successful_labels.count(0), successful_labels.count(1)]
        minority_count = min(counts) if successful_labels else 0
        if minority_count < 15:
            reasons.append("successful structure subset minority class has fewer than 15 peptides")
    return {
        "total_samples": total,
        "successful_structures": success_count,
        "success_rate": success_rate,
        "minority_class_count": minority_count,
        "formal_m2_allowed": not reasons,
        "analysis_status": "formal" if not reasons else "exploratory",
        "reasons": reasons,
        "join_key_fields": ["peptide_id", "cyclization_type", "topology"],
    }
