"""Adapters for the exact open-source files used by the S1 pipeline."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import math
import re
import time
from typing import Iterable, Iterator
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .constants import STANDARD_AMINO_ACIDS


AMP_BERT_REPOSITORY = "https://github.com/GIST-CSBL/AMP-BERT"
AMPLIFY_REPOSITORY = "https://github.com/BirolLab/AMPlify"
CYCLICPEPEDIA_REPOSITORY = "https://github.com/dfwlab/cyclicpepedia"
DBAASP_API_ROOT = "https://dbaasp.org"

# Deliberately narrow: these names explicitly assert activity against bacteria.
# We do not infer antibacterial activity from an absent annotation, PubChem
# bioassay status, or the broader Anti-Microbial parent category.
EXPLICIT_ANTIBACTERIAL_FUNCTIONS = frozenset(
    {
        "anti-bacterial",
        "anti-gram+",
        "anti-gram-",
        "anti-mycobacterial",
        "anti-tubercular",
        "anti-mrsa",
    }
)


def read_fasta(path: Path) -> Iterator[tuple[str, str]]:
    """Yield FASTA identifier and sequence, supporting wrapped sequences."""

    identifier: str | None = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            if text.startswith(">"):
                if identifier is not None:
                    yield identifier, "".join(chunks)
                identifier = text[1:].split(maxsplit=1)[0]
                chunks = []
            elif identifier is None:
                raise ValueError(f"sequence before FASTA header in {path}")
            else:
                chunks.append(text)
    if identifier is not None:
        yield identifier, "".join(chunks)


def load_amp_bert(directory: Path) -> list[dict[str, object]]:
    """Read the downloaded AMP-BERT CSV files without changing their labels."""

    rows: list[dict[str, object]] = []
    for path in sorted(directory.glob("*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for source_row in csv.DictReader(handle):
                raw_label = str(source_row.get("AMP", "")).strip().lower()
                if raw_label not in {"true", "false"}:
                    raise ValueError(f"unexpected AMP-BERT label {raw_label!r} in {path.name}")
                identifier = str(source_row.get("", "")).strip()
                rows.append(
                    {
                        "sequence": str(source_row.get("aa_seq", "")).strip(),
                        "label": 1 if raw_label == "true" else 0,
                        "source": "AMP-BERT",
                        "source_dataset": path.name,
                        "source_record_id": identifier,
                        "original_split": "unspecified",
                        "reference": AMP_BERT_REPOSITORY,
                    }
                )
    return rows


def load_amplify(directory: Path) -> list[dict[str, object]]:
    """Read AMPlify FASTA files; labels and splits come only from filenames."""

    rows: list[dict[str, object]] = []
    for path in sorted(directory.glob("*.fa")):
        lowered = path.name.lower()
        if "non_amp" in lowered:
            label = 0
        elif "_amp_" in lowered:
            label = 1
        else:
            raise ValueError(f"cannot derive AMPlify label from filename {path.name}")
        split = "test" if "_test_" in lowered else "train" if "_train_" in lowered else "unspecified"
        for identifier, sequence in read_fasta(path):
            rows.append(
                {
                    "sequence": sequence,
                    "label": label,
                    "source": "AMPlify",
                    "source_dataset": path.name,
                    "source_record_id": identifier,
                    "original_split": split,
                    "reference": AMPLIFY_REPOSITORY,
                }
            )
    return rows


def build_unified_pretrain(raw_root: Path) -> list[dict[str, object]]:
    """Combine source-labelled rows for later cleaning by build_s1_dataset.py."""

    return load_amp_bert(raw_root / "amp_bert") + load_amplify(raw_root / "amplify")


def _xlsx_rows(path: Path) -> list[dict[str, object]]:
    try:
        import openpyxl
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("openpyxl is required to adapt CyclicPepedia workbooks") from exc
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = workbook.worksheets[0]
    iterator = sheet.iter_rows(values_only=True)
    raw_headers = next(iterator)
    headers = ["CPID" if index == 0 and value is None else str(value or f"unused_{index}")
               for index, value in enumerate(raw_headers)]
    result = [dict(zip(headers, values)) for values in iterator]
    workbook.close()
    return result


def _valid_standard_sequence(value: object, *, min_length: int, max_length: int) -> str | None:
    if not isinstance(value, str):
        return None
    sequence = "".join(value.split()).upper()
    if not min_length <= len(sequence) <= max_length:
        return None
    if set(sequence) - STANDARD_AMINO_ACIDS:
        return None
    return sequence


def load_cyclicpepedia_antibacterial_candidates(
    directory: Path,
    *,
    min_length: int = 5,
    max_length: int = 50,
) -> list[dict[str, object]]:
    """Extract only explicitly annotated antibacterial cyclic peptide candidates.

    The output is a positive-candidate inventory, not an MIC-labelled core set.
    In particular, missing function mappings are never converted to negatives.
    """

    function_rows = _xlsx_rows(directory / "Function.xlsx")
    accepted_functions = {
        str(row["Function_ID"]): str(row["Name"])
        for row in function_rows
        if str(row.get("Name", "")).strip().lower() in EXPLICIT_ANTIBACTERIAL_FUNCTIONS
    }
    mapping_rows = _xlsx_rows(directory / "Peptide_to_function.xlsx")
    mappings: dict[str, list[dict[str, object]]] = {}
    for row in mapping_rows:
        function_id = str(row.get("FunctionID", ""))
        if function_id in accepted_functions:
            mappings.setdefault(str(row.get("CPID", "")), []).append(row)

    basic = {str(row["CPID"]): row for row in _xlsx_rows(directory / "Peptide_basic_info.xlsx")}
    sequences = {str(row["CPID"]): row for row in _xlsx_rows(directory / "Peptide_sequence_info.xlsx")}
    structures = {str(row["CPID"]): row for row in _xlsx_rows(directory / "Peptide_structrure_info.xlsx")}

    candidates: list[dict[str, object]] = []
    for cpid in sorted(mappings):
        sequence_row = sequences.get(cpid, {})
        sequence = None
        sequence_field = None
        for field in ("Seq_for_PP", "Tran_one_letter", "One_letter"):
            sequence = _valid_standard_sequence(
                sequence_row.get(field), min_length=min_length, max_length=max_length
            )
            if sequence is not None:
                sequence_field = field
                break
        if sequence is None:
            continue
        peptide = basic.get(cpid, {})
        structure = structures.get(cpid, {})
        functions = mappings[cpid]
        function_ids = sorted({str(row["FunctionID"]) for row in functions})
        evidence_pairs: set[str] = set()
        for row in functions:
            for number in (1, 2, 3):
                evidence = row.get(f"Evidence{number}")
                link = row.get(f"Link{number}")
                if evidence or link:
                    evidence_pairs.add(f"{evidence or ''}|{link or ''}")
        candidates.append(
            {
                "peptide_id": cpid,
                "name": peptide.get("Name", ""),
                "sequence": sequence,
                "sequence_source_field": sequence_field,
                "length": len(sequence),
                "antibacterial_annotation": 1,
                "label_scope": "positive_candidate_only_no_mic_threshold_label",
                "cyclization_type": "cyclic_unspecified",
                "function_ids": "##".join(function_ids),
                "function_names": "##".join(accepted_functions[item] for item in function_ids),
                "function_evidence": "##".join(sorted(evidence_pairs)),
                "pubmed": peptide.get("PubMed", ""),
                "smiles": structure.get("SMILES", ""),
                "structure_evidence": structure.get("Structure_evidence", ""),
                "source": "CyclicPepedia",
                "reference": CYCLICPEPEDIA_REPOSITORY,
                "eligibility_note": "Explicit antibacterial function annotation; MIC confirmation required",
            }
        )
    return candidates


def match_dbaasp_index(
    index_rows: Iterable[dict[str, object]], candidates: Iterable[dict[str, object]]
) -> list[dict[str, object]]:
    """Match DBAASP monomer index records to candidate sequences exactly."""

    candidate_ids: dict[str, list[str]] = {}
    for row in candidates:
        sequence = str(row.get("sequence", "")).strip().upper()
        candidate_ids.setdefault(sequence, []).append(str(row.get("peptide_id", "")))
    matches: list[dict[str, object]] = []
    for row in index_rows:
        sequence = str(row.get("sequence", "")).strip().upper()
        if sequence in candidate_ids:
            matched = dict(row)
            matched["cyclicpepedia_ids"] = "##".join(sorted(candidate_ids[sequence]))
            matches.append(matched)
    return matches


def preserve_censor_on_normalized_activity(
    raw_concentration: object, normalized_activity: object
) -> tuple[str | float | None, str, str]:
    """Transfer an MIC censor operator to DBAASP's normalized ug/mL boundary.

    Returns ``(value, status, exclusion_reason)``. DBAASP exposes a scalar
    normalized activity even for some non-scalar reported values. A range
    cannot be reconstructed from one scalar and is therefore excluded instead
    of being collapsed to a fabricated point estimate.
    """

    raw = str(raw_concentration or "").strip().replace("≤", "<=").replace("≥", ">=")
    try:
        normalized = float(normalized_activity)
    except (TypeError, ValueError):
        normalized = math.nan
    if not math.isfinite(normalized) or normalized <= 0:
        return None, "excluded", "missing_or_invalid_dbaasp_normalized_activity"
    match = re.fullmatch(r"(<=|>=|<|>)?\s*(\d+(?:\.\d+)?)", raw)
    if match:
        operator = match.group(1) or ""
        return f"{operator}{normalized:g}" if operator else normalized, "normalized", ""
    if re.fullmatch(r"\d+(?:\.\d+)?\s*[-–—]\s*\d+(?:\.\d+)?", raw):
        return None, "excluded", "range_cannot_be_reconstructed_from_scalar_activity"
    return None, "excluded", "unparseable_raw_concentration_censor"


def extract_dbaasp_mic_measurements(
    detail: dict[str, object], *, cyclicpepedia_ids: str = ""
) -> list[dict[str, object]]:
    """Extract explicit MIC rows from a cyclic DBAASP detail response.

    ``targetActivities.activity`` is DBAASP's normalized activity value. We
    preserve both it and the originally reported concentration/unit. No
    activity row is created when the record lacks an intrachain bond.
    """

    bonds = detail.get("intrachainBonds") or []
    if not bonds:
        return []
    sequence = str(detail.get("sequence", "")).strip().upper()
    if not sequence:
        return []
    bond_text = json.dumps(bonds, ensure_ascii=False, separators=(",", ":"))
    cycle_names = {
        str((bond.get("cycleType") or {}).get("name", "")).strip().upper()
        for bond in bonds
    }
    bond_names = {
        str((bond.get("type") or {}).get("name", "")).strip().upper()
        for bond in bonds
    }
    if "NCB" in cycle_names or any(
        int(bond.get("position1") or -1) == 1
        and int(bond.get("position2") or -1) == len(sequence)
        and str((bond.get("type") or {}).get("name", "")).strip().upper() == "AMD"
        for bond in bonds
    ):
        cyclization_type = "head_to_tail"
    elif "DSB" in bond_names:
        cyclization_type = "disulfide"
    else:
        cyclization_type = "sidechain_or_other"
    pubmed_ids = "##".join(
        sorted(
            {
                str((article.get("pubmed") or {}).get("pubmedId"))
                for article in (detail.get("articles") or [])
                if (article.get("pubmed") or {}).get("pubmedId")
            }
        )
    )
    rows: list[dict[str, object]] = []
    for activity in detail.get("targetActivities") or []:
        measure_group = str((activity.get("activityMeasureGroup") or {}).get("name", ""))
        measure_value = str(activity.get("activityMeasureValue", ""))
        if measure_group.strip().upper() != "MIC" and measure_value.strip().upper() != "MIC":
            continue
        raw_concentration = activity.get("concentration", "")
        raw_unit = str((activity.get("unit") or {}).get("name", ""))
        normalized = activity.get("activity")
        if normalized in (None, ""):
            # The original value retains its operator and the downstream MIC
            # parser performs unit conversion with an explicit molecular mass.
            mic = raw_concentration
            unit = raw_unit
            conversion_status = "raw_value_requires_downstream_conversion"
            conversion_exclusion_reason = ""
            normalization_source = "raw_reported_value"
        else:
            mic, conversion_status, conversion_exclusion_reason = preserve_censor_on_normalized_activity(
                raw_concentration, normalized
            )
            unit = "ug/mL"
            normalization_source = "DBAASP_targetActivities.activity_with_raw_censor_preserved"
        rows.append(
            {
                "peptide_id": detail.get("dbaaspId") or detail.get("id"),
                "dbaasp_numeric_id": detail.get("id"),
                "cyclicpepedia_ids": cyclicpepedia_ids,
                "sequence": sequence,
                "mic": mic,
                "unit": unit,
                "mic_raw": raw_concentration,
                "unit_raw": raw_unit,
                "normalization_source": normalization_source,
                "conversion_status": conversion_status,
                "conversion_exclusion_reason": conversion_exclusion_reason,
                "target_species": str((activity.get("targetSpecies") or {}).get("name", "")),
                "target_taxon_status": "unverified; retain only bacterial targets for antibacterial modelling",
                "cyclization_type": cyclization_type,
                "intrachain_bonds_json": bond_text,
                "activity_id": activity.get("id", ""),
                "medium": str((activity.get("medium") or {}).get("name", "")),
                "ph": activity.get("ph", ""),
                "ionic_strength": activity.get("ionicStrength", ""),
                "source": "DBAASP",
                "reference": detail.get("url") or f"{DBAASP_API_ROOT}/peptide-card?id={detail.get('dbaaspId', '')}",
                "pubmed": pubmed_ids,
            }
        )
    return rows


class DbaaspClient:
    """Small keyless DBAASP client with bounded pagination and rate limiting."""

    def __init__(self, *, sleep_seconds: float = 0.2, timeout: float = 30.0, retries: int = 3):
        self.sleep_seconds = max(0.0, sleep_seconds)
        self.timeout = timeout
        self.retries = max(1, retries)

    def get_json(self, path: str, params: dict[str, object] | None = None) -> object:
        query = f"?{urlencode(params)}" if params else ""
        request = Request(
            f"{DBAASP_API_ROOT}{path}{query}",
            headers={"Accept": "application/json", "User-Agent": "CycAMP-FusionDemo/0.1"},
        )
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    payload = json.load(response)
                if self.sleep_seconds:
                    time.sleep(self.sleep_seconds)
                return payload
            except Exception as exc:  # network errors vary by Windows runtime
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(1.0 * (attempt + 1))
        assert last_error is not None
        raise last_error

    def iter_index(self, *, page_size: int = 1000) -> Iterator[dict[str, object]]:
        if not 1 <= page_size <= 1000:
            raise ValueError("DBAASP page_size must be between 1 and 1000")
        offset = 0
        while True:
            payload = self.get_json("/peptides", {"limit": page_size, "offset": offset})
            if not isinstance(payload, dict) or "data" not in payload:
                raise ValueError("unexpected DBAASP index response")
            data = payload.get("data") or []
            yield from data
            offset += len(data)
            if not data or offset >= int(payload.get("totalCount", offset)):
                break

    def detail(self, peptide_id: int | str) -> dict[str, object]:
        payload = self.get_json(f"/peptides/{peptide_id}")
        if not isinstance(payload, dict):
            raise ValueError(f"unexpected DBAASP detail response for {peptide_id}")
        return payload
