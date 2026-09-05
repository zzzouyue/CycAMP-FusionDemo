"""Conservative NCBI Taxonomy mapping for DBAASP target species."""

from __future__ import annotations

import re
from typing import Iterable, Mapping


def _normalized_name(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def _binomial(value: object) -> tuple[str, str] | None:
    tokens = re.findall(r"[a-z][a-z.-]*", _normalized_name(value))
    if len(tokens) < 2 or tokens[0] == "candidatus":
        return None
    return tokens[0].rstrip("."), tokens[1].rstrip(".")


def choose_taxon_suggestion(
    target_species: str, payload: Mapping[str, object]
) -> dict[str, object] | None:
    """Choose an exact or binomial-consistent NCBI suggestion.

    A binomial match is used only to verify the domain, not to claim an exact
    strain assignment. Ambiguous or unrelated suggestions are rejected.
    """

    suggestions = list(payload.get("sci_name_and_ids") or [])
    target_normalized = _normalized_name(target_species)
    for index, suggestion in enumerate(suggestions):
        scientific = _normalized_name(suggestion.get("sci_name"))
        matched = _normalized_name(suggestion.get("matched_term"))
        if target_normalized in {scientific, matched}:
            return {**suggestion, "suggestion_rank": index + 1, "match_quality": "exact"}
    target_binomial = _binomial(target_species)
    if target_binomial is None:
        return None
    candidates: list[tuple[int, dict[str, object]]] = []
    for index, suggestion in enumerate(suggestions):
        if _binomial(suggestion.get("sci_name")) == target_binomial:
            candidates.append((index, suggestion))
    if not candidates:
        return None
    # Multiple strain suggestions are not a taxonomy guess here: the first
    # NCBI-ranked one is a proxy solely for domain verification, and the mapping
    # explicitly records binomial-level quality.
    index, suggestion = candidates[0]
    return {**suggestion, "suggestion_rank": index + 1, "match_quality": "binomial_domain_only"}


def parse_taxonomy_report(payload: Mapping[str, object]) -> dict[str, object] | None:
    reports = list(payload.get("reports") or [])
    if len(reports) != 1:
        return None
    taxonomy = reports[0].get("taxonomy") or {}
    classification = taxonomy.get("classification") or {}
    domain = classification.get("domain") or {}
    scientific = taxonomy.get("current_scientific_name") or {}
    if not domain.get("name"):
        return None
    return {
        "ncbi_tax_id": taxonomy.get("tax_id"),
        "ncbi_scientific_name": scientific.get("name", ""),
        "ncbi_rank": taxonomy.get("rank", ""),
        "domain_name": domain.get("name", ""),
        "domain_tax_id": domain.get("id", ""),
    }


def build_taxonomy_mapping(
    target_species: str,
    suggest_payload: Mapping[str, object] | None,
    report_payload: Mapping[str, object] | None,
    *,
    error: str | None = None,
) -> dict[str, object]:
    """Build one auditable mapping without inferring missing classifications."""

    base: dict[str, object] = {
        "target_species": target_species,
        "status": "unmatched",
        "suggestion_count": len((suggest_payload or {}).get("sci_name_and_ids") or []),
        "selected_tax_id": "",
        "selected_sci_name": "",
        "matched_term": "",
        "suggestion_rank": "",
        "match_quality": "",
        "ncbi_tax_id": "",
        "ncbi_scientific_name": "",
        "ncbi_rank": "",
        "domain_name": "",
        "domain_tax_id": "",
        "audit_reason": "no_reliable_ncbi_suggestion",
    }
    if error:
        base.update(status="error", audit_reason=error)
        return base
    if suggest_payload is None:
        return base
    suggestion = choose_taxon_suggestion(target_species, suggest_payload)
    if suggestion is None:
        return base
    base.update(
        selected_tax_id=suggestion.get("tax_id", ""),
        selected_sci_name=suggestion.get("sci_name", ""),
        matched_term=suggestion.get("matched_term", ""),
        suggestion_rank=suggestion.get("suggestion_rank", ""),
        match_quality=suggestion.get("match_quality", ""),
    )
    if report_payload is None:
        base.update(status="unmatched", audit_reason="missing_or_invalid_dataset_report")
        return base
    report = parse_taxonomy_report(report_payload)
    if report is None:
        base.update(status="unmatched", audit_reason="missing_or_ambiguous_dataset_report")
        return base
    base.update(report)
    if str(report["domain_name"]).casefold() == "bacteria":
        base.update(status="bacterial", audit_reason="")
    else:
        base.update(status="non_bacteria", audit_reason="classification.domain.name_is_not_Bacteria")
    return base


def partition_mic_rows(
    rows: Iterable[Mapping[str, object]], mappings: Iterable[Mapping[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Keep only rows whose unique target mapping is explicitly bacterial."""

    by_target = {str(row["target_species"]): dict(row) for row in mappings}
    kept: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    fields = (
        "status", "selected_tax_id", "selected_sci_name", "match_quality",
        "ncbi_tax_id", "ncbi_scientific_name", "ncbi_rank", "domain_name",
        "domain_tax_id", "audit_reason",
    )
    for original in rows:
        row = dict(original)
        target = str(row.get("target_species", ""))
        mapping = by_target.get(target)
        if mapping is None:
            mapping = build_taxonomy_mapping(target, None, None)
            mapping["audit_reason"] = "target_missing_from_mapping"
        for field in fields:
            row[f"taxonomy_{field}"] = mapping.get(field, "")
        if mapping.get("status") == "bacterial" and mapping.get("domain_name") == "Bacteria":
            kept.append(row)
        else:
            audit.append(row)
    return kept, audit
