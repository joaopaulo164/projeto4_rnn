"""Exact-duplicate and overlap audit of the ElectricDevices partitions.

Every series is fingerprinted from its raw 96 values (before any normalization fitted by the
project) with :func:`project4_rnn.data.hashing.hash_series`. The fingerprint ignores the label and
the sample index, so two observations share a hash exactly when their raw series are identical.

The audit never alters labels, never removes observations, and never votes on conflicting labels.
It only measures and reports.
"""

import json
import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from project4_rnn.constants import (
    OFFICIAL_CLASSES,
    SERIES_HASH_ALGORITHM,
    SERIES_HASH_DTYPE,
    SERIES_HASH_ENDIANNESS,
)

logger = logging.getLogger("project4_rnn.audit.duplicates")

PARTITION_NAMES: tuple[str, ...] = ("train", "validation", "test")

PAIRWISE_COMPARISONS: tuple[tuple[str, str], ...] = (
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test"),
)

AUDIT_ARTIFACT_NAMES: tuple[str, ...] = (
    "duplicate_overlap_audit.json",
    "duplicate_overlap_audit.md",
    "duplicate_groups.csv",
    "duplicate_overlap_pairs.csv",
    "duplicate_label_conflicts.csv",
    "duplicate_counts_by_class.csv",
    "duplicate_sample_inventory.csv",
    "official_development_test_overlap.json",
)

# Values reported by the independent external audit, used only for transparent comparison.
# Quantities that depend on the (now replaced) train/validation split are intentionally absent.
EXTERNAL_AUDIT_REFERENCE: dict[str, dict[str, int]] = {
    "global": {
        "samples": 16637,
        "unique_series": 16160,
        "excess_duplicates": 477,
        "duplicate_groups": 136,
        "samples_involved": 613,
        "conflicting_groups": 54,
    },
    "test": {
        "samples": 7711,
        "unique_series": 7460,
        "excess_duplicates": 251,
        "duplicate_groups": 89,
        "samples_involved": 340,
    },
    "development_vs_test": {
        "shared_groups": 57,
        "development_samples": 178,
        "test_samples": 208,
        "groups_with_label_conflict": 41,
    },
}

GroupIndex = dict[str, Counter[int]]


def build_group_index(hashes: Sequence[str], labels: Sequence[int]) -> GroupIndex:
    """Map each series hash to the counter of official labels observed for it."""
    if len(hashes) != len(labels):
        raise ValueError(
            f"hashes ({len(hashes)}) and labels ({len(labels)}) must have equal length"
        )
    index: GroupIndex = {}
    for h, label in zip(hashes, labels, strict=True):
        index.setdefault(h, Counter())[int(label)] += 1
    return index


def merge_group_indexes(*indexes: GroupIndex) -> GroupIndex:
    """Union of several group indexes (counts are added)."""
    merged: GroupIndex = {}
    for index in indexes:
        for h, counter in index.items():
            merged.setdefault(h, Counter()).update(counter)
    return merged


def conflicting_hashes(hashes: Sequence[str], labels: Sequence[int]) -> set[str]:
    """Hashes whose identical series carry more than one distinct official label."""
    return {h for h, c in build_group_index(hashes, labels).items() if len(c) > 1}


def _size(counter: Counter[int]) -> int:
    return int(sum(counter.values()))


def partition_stats(index: GroupIndex) -> dict[str, Any]:
    """Internal duplicate statistics of one partition (or of a union of partitions)."""
    samples = sum(_size(c) for c in index.values())
    duplicated = {h: c for h, c in index.items() if _size(c) > 1}
    conflicting = {h: c for h, c in duplicated.items() if len(c) > 1}
    largest = max((_size(c) for c in index.values()), default=0)
    return {
        "samples": int(samples),
        "unique_series": len(index),
        "excess_duplicates": int(samples - len(index)),
        "duplicate_groups": len(duplicated),
        "samples_involved": int(sum(_size(c) for c in duplicated.values())),
        "largest_group": int(largest),
        "conflicting_groups": len(conflicting),
        "conflicting_samples": int(sum(_size(c) for c in conflicting.values())),
    }


def overlap_stats(index_a: GroupIndex, index_b: GroupIndex) -> dict[str, Any]:
    """Cross-partition overlap statistics between two group indexes."""
    shared = sorted(set(index_a) & set(index_b))
    samples_a = sum(_size(index_a[h]) for h in shared)
    samples_b = sum(_size(index_b[h]) for h in shared)
    pairs = 0
    same_label_pairs = 0
    groups_with_conflict = 0
    for h in shared:
        ca, cb = index_a[h], index_b[h]
        pairs += _size(ca) * _size(cb)
        same_label_pairs += sum(n * cb.get(label, 0) for label, n in ca.items())
        if len(set(ca) | set(cb)) > 1:
            groups_with_conflict += 1
    return {
        "shared_hashes": len(shared),
        "samples_a": int(samples_a),
        "samples_b": int(samples_b),
        "pairs": int(pairs),
        "pairs_same_label": int(same_label_pairs),
        "pairs_different_label": int(pairs - same_label_pairs),
        "groups_with_label_conflict": int(groups_with_conflict),
    }


def sensitivity_masks(
    dev_hashes: Sequence[str],
    dev_labels: Sequence[int],
    test_hashes: Sequence[str],
    test_labels: Sequence[int],
) -> dict[str, np.ndarray]:
    """Boolean keep-masks over the official test set for the two sensitivity analyses.

    ``no_exact_overlap`` keeps test rows whose hash does not occur in the official development
    set. ``without_label_conflicts`` keeps test rows whose hash is not a conflicting group, where
    a conflicting group is a hash carrying more than one official label across the pooled
    development and test sets. The official test set itself is never modified.
    """
    dev_hash_set = set(dev_hashes)
    pooled_conflicts = conflicting_hashes(
        list(dev_hashes) + list(test_hashes), list(dev_labels) + list(test_labels)
    )
    no_overlap = np.fromiter((h not in dev_hash_set for h in test_hashes), dtype=bool)
    no_conflict = np.fromiter((h not in pooled_conflicts for h in test_hashes), dtype=bool)
    return {"no_exact_overlap": no_overlap, "without_label_conflicts": no_conflict}


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    counts = Counter(int(v) for v in labels)
    return {str(c): int(counts.get(c, 0)) for c in OFFICIAL_CLASSES}


def build_duplicate_audit(
    dev_hashes: Sequence[str],
    dev_labels: np.ndarray,
    test_hashes: Sequence[str],
    test_labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    dataset_hash: str | None = None,
    split_hash: str | None = None,
    split_method: str | None = None,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Compute the full duplicate/overlap audit.

    Args:
        dev_hashes: Series hash of every official TRAIN sample (8,926).
        dev_labels: Official labels (1..7) of the official TRAIN samples.
        test_hashes: Series hash of every official TEST sample.
        test_labels: Official labels (1..7) of the official TEST samples.
        train_idx: Indices (into the official TRAIN file) of the internal training partition.
        val_idx: Indices (into the official TRAIN file) of the internal validation partition.
        dataset_hash: Dataset identity hash recorded in the report.
        split_hash: Split identity hash recorded in the report.
        split_method: Split method recorded in the report.

    Returns:
        Tuple (audit_dict, tables) where tables holds the CSV frames keyed by artifact stem.
    """
    dev_labels = np.asarray(dev_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)

    part_hashes: dict[str, list[str]] = {
        "train": [dev_hashes[int(i)] for i in train_idx],
        "validation": [dev_hashes[int(i)] for i in val_idx],
        "test": list(test_hashes),
    }
    part_labels: dict[str, np.ndarray] = {
        "train": dev_labels[train_idx],
        "validation": dev_labels[val_idx],
        "test": test_labels,
    }
    part_original_index: dict[str, np.ndarray] = {
        "train": train_idx,
        "validation": val_idx,
        "test": np.arange(len(test_hashes), dtype=np.int64),
    }
    indexes: dict[str, GroupIndex] = {
        name: build_group_index(part_hashes[name], part_labels[name].tolist())
        for name in PARTITION_NAMES
    }
    development_index = merge_group_indexes(indexes["train"], indexes["validation"])
    global_index = merge_group_indexes(development_index, indexes["test"])

    partitions = {name: partition_stats(indexes[name]) for name in PARTITION_NAMES}
    partitions["development"] = partition_stats(development_index)
    global_stats = partition_stats(global_index)

    overlaps: dict[str, dict[str, Any]] = {}
    for a, b in PAIRWISE_COMPARISONS:
        stats = overlap_stats(indexes[a], indexes[b])
        overlaps[f"{a}_vs_{b}"] = {
            "shared_hashes": stats["shared_hashes"],
            f"{a}_samples_involved": stats["samples_a"],
            f"{b}_samples_involved": stats["samples_b"],
            "pairs": stats["pairs"],
            "pairs_same_label": stats["pairs_same_label"],
            "pairs_different_label": stats["pairs_different_label"],
            "groups_with_label_conflict": stats["groups_with_label_conflict"],
        }

    dev_vs_test_raw = overlap_stats(development_index, indexes["test"])
    development_vs_test = {
        "shared_groups": dev_vs_test_raw["shared_hashes"],
        "development_samples": dev_vs_test_raw["samples_a"],
        "test_samples": dev_vs_test_raw["samples_b"],
        "pairs": dev_vs_test_raw["pairs"],
        "pairs_same_label": dev_vs_test_raw["pairs_same_label"],
        "pairs_different_label": dev_vs_test_raw["pairs_different_label"],
        "groups_with_label_conflict": dev_vs_test_raw["groups_with_label_conflict"],
    }

    # Conflicting groups (pooled over development and test)
    conflict_rows: list[dict[str, Any]] = []
    for h, counter in sorted(global_index.items()):
        if len(counter) <= 1:
            continue
        per_partition = {name: _size(indexes[name].get(h, Counter())) for name in PARTITION_NAMES}
        conflict_rows.append(
            {
                "series_hash": h,
                "total_size": _size(counter),
                "labels": json.dumps({str(k): int(v) for k, v in sorted(counter.items())}),
                "n_distinct_labels": len(counter),
                "train_samples": per_partition["train"],
                "validation_samples": per_partition["validation"],
                "test_samples": per_partition["test"],
                "crosses_partitions": sum(1 for n in per_partition.values() if n > 0) > 1,
            }
        )
    conflicts_df = pd.DataFrame(
        conflict_rows,
        columns=[
            "series_hash",
            "total_size",
            "labels",
            "n_distinct_labels",
            "train_samples",
            "validation_samples",
            "test_samples",
            "crosses_partitions",
        ],
    )
    internal_conflicts = [
        r
        for r in conflict_rows
        if r["test_samples"] == 0 and (r["train_samples"] > 0 or r["validation_samples"] > 0)
    ]
    split_conflicts = [
        r for r in conflict_rows if r["train_samples"] > 0 and r["validation_samples"] > 0
    ]

    # Duplicate groups (any hash observed more than once anywhere)
    group_rows: list[dict[str, Any]] = []
    for h, counter in sorted(global_index.items()):
        total = _size(counter)
        if total <= 1:
            continue
        per_partition = {name: _size(indexes[name].get(h, Counter())) for name in PARTITION_NAMES}
        present = [name for name in PARTITION_NAMES if per_partition[name] > 0]
        group_rows.append(
            {
                "series_hash": h,
                "total_size": total,
                "train_samples": per_partition["train"],
                "validation_samples": per_partition["validation"],
                "test_samples": per_partition["test"],
                "partitions": "|".join(present),
                "crosses_partitions": len(present) > 1,
                "labels": "|".join(str(k) for k in sorted(counter)),
                "n_distinct_labels": len(counter),
                "has_label_conflict": len(counter) > 1,
            }
        )
    groups_df = pd.DataFrame(
        group_rows,
        columns=[
            "series_hash",
            "total_size",
            "train_samples",
            "validation_samples",
            "test_samples",
            "partitions",
            "crosses_partitions",
            "labels",
            "n_distinct_labels",
            "has_label_conflict",
        ],
    )

    # Overlap pairs (one row per shared hash and comparison)
    comparisons: list[tuple[str, GroupIndex, GroupIndex]] = [
        (f"{a}_vs_{b}", indexes[a], indexes[b]) for a, b in PAIRWISE_COMPARISONS
    ]
    comparisons.append(("development_vs_test", development_index, indexes["test"]))
    pair_rows: list[dict[str, Any]] = []
    for name, ia, ib in comparisons:
        for h in sorted(set(ia) & set(ib)):
            ca, cb = ia[h], ib[h]
            pairs = _size(ca) * _size(cb)
            same = sum(n * cb.get(label, 0) for label, n in ca.items())
            pair_rows.append(
                {
                    "comparison": name,
                    "series_hash": h,
                    "samples_a": _size(ca),
                    "samples_b": _size(cb),
                    "labels_a": "|".join(str(k) for k in sorted(ca)),
                    "labels_b": "|".join(str(k) for k in sorted(cb)),
                    "pairs": pairs,
                    "pairs_same_label": same,
                    "pairs_different_label": pairs - same,
                }
            )
    pairs_df = pd.DataFrame(
        pair_rows,
        columns=[
            "comparison",
            "series_hash",
            "samples_a",
            "samples_b",
            "labels_a",
            "labels_b",
            "pairs",
            "pairs_same_label",
            "pairs_different_label",
        ],
    )

    # Duplicates by class and partition
    class_rows: list[dict[str, Any]] = []
    for name in (*PARTITION_NAMES, "development"):
        if name == "development":
            hashes_p = part_hashes["train"] + part_hashes["validation"]
            labels_p = np.concatenate([part_labels["train"], part_labels["validation"]])
            index_p = development_index
        else:
            hashes_p = part_hashes[name]
            labels_p = part_labels[name]
            index_p = indexes[name]
        for cls in OFFICIAL_CLASSES:
            mask = labels_p == cls
            class_hashes = [h for h, m in zip(hashes_p, mask, strict=True) if m]
            n_samples = int(mask.sum())
            n_unique = len(set(class_hashes))
            involved = sum(1 for h in class_hashes if _size(index_p[h]) > 1)
            class_rows.append(
                {
                    "partition": name,
                    "official_class": int(cls),
                    "samples": n_samples,
                    "unique_series": n_unique,
                    "excess_duplicates": n_samples - n_unique,
                    "samples_in_duplicate_groups": int(involved),
                }
            )
    class_df = pd.DataFrame(class_rows)

    # Per-sample inventory
    inventory_rows: list[dict[str, Any]] = []
    global_conflicts = {r["series_hash"] for r in conflict_rows}
    for name in PARTITION_NAMES:
        for h, label, orig in zip(
            part_hashes[name], part_labels[name], part_original_index[name], strict=True
        ):
            inventory_rows.append(
                {
                    "partition": name,
                    "official_file": "TEST" if name == "test" else "TRAIN",
                    "original_index": int(orig),
                    "official_label": int(label),
                    "series_hash": h,
                    "group_size_global": _size(global_index[h]),
                    "group_size_in_partition": _size(indexes[name][h]),
                    "in_conflicting_group": h in global_conflicts,
                }
            )
    inventory_df = pd.DataFrame(inventory_rows)

    reference_comparison = _compare_with_reference(
        global_stats, partitions["test"], development_vs_test
    )

    audit: dict[str, Any] = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "hash_specification": {
            "algorithm": SERIES_HASH_ALGORITHM,
            "dtype": SERIES_HASH_DTYPE,
            "endianness": SERIES_HASH_ENDIANNESS,
            "values_per_series": 96,
            "rounding": "none",
            "includes_label": False,
            "includes_index": False,
            "applied_before_normalization": True,
        },
        "dataset_hash": dataset_hash,
        "split_hash": split_hash,
        "split_method": split_method,
        "global": global_stats,
        "partitions": partitions,
        "pairwise_overlaps": overlaps,
        "development_vs_test": development_vs_test,
        "conflicting_groups": {
            "pooled_dev_and_test": len(conflict_rows),
            "samples_involved": int(sum(r["total_size"] for r in conflict_rows)),
            "classes_involved": sorted(
                {int(k) for r in conflict_rows for k in json.loads(r["labels"])}
            ),
            "inside_development_only": len(internal_conflicts),
            "crossing_train_and_validation": len(split_conflicts),
        },
        "class_distribution": {
            "train": _class_counts(part_labels["train"]),
            "validation": _class_counts(part_labels["validation"]),
            "test": _class_counts(part_labels["test"]),
        },
        "external_reference_comparison": reference_comparison,
        "notes": [
            "Identical series with different official labels cannot be separated "
            "deterministically using only the 96 values.",
            "Labels were never altered, voted on, or removed; internal duplicates are preserved.",
            "Quantities that depend on the previous train/validation split are not compared.",
        ],
    }
    tables = {
        "duplicate_groups": groups_df,
        "duplicate_overlap_pairs": pairs_df,
        "duplicate_label_conflicts": conflicts_df,
        "duplicate_counts_by_class": class_df,
        "duplicate_sample_inventory": inventory_df,
    }
    return audit, tables


def _compare_with_reference(
    global_stats: Mapping[str, Any],
    test_stats: Mapping[str, Any],
    dev_vs_test: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare split-independent recomputed values with the external audit reference."""
    observed: dict[str, dict[str, int]] = {
        "global": {
            "samples": int(global_stats["samples"]),
            "unique_series": int(global_stats["unique_series"]),
            "excess_duplicates": int(global_stats["excess_duplicates"]),
            "duplicate_groups": int(global_stats["duplicate_groups"]),
            "samples_involved": int(global_stats["samples_involved"]),
            "conflicting_groups": int(global_stats["conflicting_groups"]),
        },
        "test": {
            "samples": int(test_stats["samples"]),
            "unique_series": int(test_stats["unique_series"]),
            "excess_duplicates": int(test_stats["excess_duplicates"]),
            "duplicate_groups": int(test_stats["duplicate_groups"]),
            "samples_involved": int(test_stats["samples_involved"]),
        },
        "development_vs_test": {
            "shared_groups": int(dev_vs_test["shared_groups"]),
            "development_samples": int(dev_vs_test["development_samples"]),
            "test_samples": int(dev_vs_test["test_samples"]),
            "groups_with_label_conflict": int(dev_vs_test["groups_with_label_conflict"]),
        },
    }
    result: dict[str, Any] = {"all_match": True, "divergences": [], "checked": 0}
    for section, expected_values in EXTERNAL_AUDIT_REFERENCE.items():
        for key, expected in expected_values.items():
            got = observed[section][key]
            result["checked"] += 1
            if got != expected:
                result["all_match"] = False
                result["divergences"].append(
                    {"section": section, "metric": key, "external": expected, "recomputed": got}
                )
    return result


def render_audit_markdown(audit: Mapping[str, Any]) -> str:
    """Render the audit as a Markdown report (Portuguese)."""
    g = audit["global"]
    lines = [
        "# Auditoria de Duplicatas e Sobreposição — ElectricDevices",
        "",
        "> Auditoria técnica de dados. Não envolve treinamento nem avaliação de modelos.",
        "",
        f"- Gerada em: `{audit['generated_at']}`",
        f"- Hash do dataset: `{audit.get('dataset_hash')}`",
        f"- Hash do split: `{audit.get('split_hash')}`",
        f"- Método do split: `{audit.get('split_method')}`",
        "- Hash das séries: SHA-256 sobre 96 valores float64 little-endian contíguos, sem "
        "arredondamento, sem rótulo e sem índice, calculado antes da normalização.",
        "",
        "## 1. Visão global (desenvolvimento oficial + teste oficial)",
        "",
        "| Métrica | Valor |",
        "|---|---|",
        f"| Amostras | {g['samples']} |",
        f"| Séries únicas | {g['unique_series']} |",
        f"| Excesso de duplicatas | {g['excess_duplicates']} |",
        f"| Grupos duplicados | {g['duplicate_groups']} |",
        f"| Amostras em grupos duplicados | {g['samples_involved']} |",
        f"| Maior grupo | {g['largest_group']} |",
        f"| Grupos com rótulos conflitantes | {g['conflicting_groups']} |",
        "",
        "## 2. Duplicatas internas por partição",
        "",
        "| Partição | Amostras | Únicas | Excesso | Grupos duplicados | Amostras envolvidas | "
        "Maior grupo | Grupos conflitantes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in (*PARTITION_NAMES, "development"):
        p = audit["partitions"][name]
        lines.append(
            f"| {name} | {p['samples']} | {p['unique_series']} | {p['excess_duplicates']} | "
            f"{p['duplicate_groups']} | {p['samples_involved']} | {p['largest_group']} | "
            f"{p['conflicting_groups']} |"
        )

    lines.extend(
        [
            "",
            "## 3. Sobreposição entre partições",
            "",
            "| Comparação | Hashes compartilhados | Pares | Pares mesmo rótulo | "
            "Pares rótulo diferente | Grupos com conflito |",
            "|---|---|---|---|---|---|",
        ]
    )
    for name, o in audit["pairwise_overlaps"].items():
        lines.append(
            f"| {name} | {o['shared_hashes']} | {o['pairs']} | {o['pairs_same_label']} | "
            f"{o['pairs_different_label']} | {o['groups_with_label_conflict']} |"
        )
    d = audit["development_vs_test"]
    lines.extend(
        [
            "",
            "### Desenvolvimento oficial completo × teste oficial",
            "",
            f"- Grupos compartilhados: **{d['shared_groups']}**",
            f"- Amostras de desenvolvimento envolvidas: {d['development_samples']}",
            f"- Amostras de teste envolvidas: {d['test_samples']}",
            f"- Pares: {d['pairs']} (mesmo rótulo: {d['pairs_same_label']}; "
            f"rótulo diferente: {d['pairs_different_label']})",
            f"- Grupos com algum conflito de rótulo: {d['groups_with_label_conflict']}",
            "",
            "## 4. Grupos com rótulos conflitantes",
            "",
        ]
    )
    c = audit["conflicting_groups"]
    lines.extend(
        [
            f"- Grupos conflitantes (desenvolvimento + teste): **{c['pooled_dev_and_test']}**",
            f"- Amostras envolvidas: {c['samples_involved']}",
            f"- Classes envolvidas: {c['classes_involved']}",
            f"- Grupos conflitantes que atravessam treino e validação: "
            f"**{c['crossing_train_and_validation']}**",
            "",
            "Séries idênticas com rótulos diferentes não podem ser separadas de forma "
            "determinística usando somente os 96 valores. Os rótulos oficiais são preservados e "
            "o grupo inteiro permanece em uma única partição.",
            "",
            "## 5. Comparação com a auditoria externa",
            "",
        ]
    )
    ref = audit["external_reference_comparison"]
    if ref["all_match"]:
        lines.append(
            f"Todos os {ref['checked']} valores independentes do split coincidem com a auditoria "
            "externa."
        )
    else:
        lines.append("Divergências encontradas:")
        for div in ref["divergences"]:
            lines.append(
                f"- {div['section']}/{div['metric']}: externo={div['external']} "
                f"recalculado={div['recomputed']}"
            )
    lines.extend(
        [
            "",
            "## 6. Consequências das duplicatas internas",
            "",
            "- Duplicatas internas ao treino aumentam o peso efetivo dessas séries na função de "
            "perda.",
            "- Duplicatas internas à validação inflam ou deflam métricas de validação de forma "
            "correlacionada.",
            "- Duplicatas no teste oficial são mantidas; as análises de sensibilidade futuras "
            "reportam o teste sem hashes presentes no desenvolvimento e sem grupos conflitantes.",
            "- Nenhuma deduplicação automática nem ponderação por grupo é aplicada.",
            "",
        ]
    )
    return "\n".join(lines)


def write_audit_artifacts(
    audit: dict[str, Any],
    tables: Mapping[str, pd.DataFrame],
    output_dir: Path,
) -> list[Path]:
    """Persist the audit artifacts under ``output_dir`` and return the written paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    json_path = output_dir / "duplicate_overlap_audit.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    written.append(json_path)

    md_path = output_dir / "duplicate_overlap_audit.md"
    md_path.write_text(render_audit_markdown(audit), encoding="utf-8")
    written.append(md_path)

    for stem, frame in tables.items():
        csv_path = output_dir / f"{stem}.csv"
        frame.to_csv(csv_path, index=False)
        written.append(csv_path)

    dev_test_path = output_dir / "official_development_test_overlap.json"
    with open(dev_test_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": audit["schema_version"],
                "generated_at": audit["generated_at"],
                "dataset_hash": audit.get("dataset_hash"),
                "development_vs_test": audit["development_vs_test"],
                "test_internal": audit["partitions"]["test"],
                "development_internal": audit["partitions"]["development"],
                "note": (
                    "Official development set (8,926) compared with the official test set "
                    "(7,711). The official test set was not modified."
                ),
            },
            f,
            indent=2,
        )
    written.append(dev_test_path)

    logger.info(f"Duplicate audit artifacts written to {output_dir}")
    return written


def run_duplicate_audit(
    base_config_path: Path = Path("configs/base.yaml"),
    output_dir: Path = Path("artifacts/audits"),
) -> dict[str, Any]:
    """Run the duplicate audit on the real dataset and persist its artifacts.

    This is a pure data audit: it trains nothing and evaluates no model.
    """
    from project4_rnn.config import build_experiment_config
    from project4_rnn.data.hashing import hash_series_matrix
    from project4_rnn.data.loader import load_raw_electricdevices
    from project4_rnn.data.splitting import create_or_load_split

    cfg = build_experiment_config(base_config_path)
    processed_dir = Path(cfg.data.processed_dir)
    X_dev, y_dev, X_test, y_test, manifest = load_raw_electricdevices(
        raw_dir=Path(cfg.data.raw_dir), verify_checksums=True, processed_dir=None
    )
    dev_hashes = hash_series_matrix(X_dev)
    test_hashes = hash_series_matrix(X_test)
    split = create_or_load_split(
        y_train_official=y_dev,
        series_hashes=dev_hashes,
        split_indices_path=processed_dir / "split_indices.json",
        dataset_hash=manifest["dataset_hash"],
        seed=cfg.data.split_seed,
        n_splits=cfg.data.split_n_splits,
        val_ratio=cfg.data.val_split_ratio,
    )
    audit, tables = build_duplicate_audit(
        dev_hashes=dev_hashes,
        dev_labels=y_dev,
        test_hashes=test_hashes,
        test_labels=y_test,
        train_idx=split.train_idx,
        val_idx=split.val_idx,
        dataset_hash=manifest["dataset_hash"],
        split_hash=split.split_hash,
        split_method=str(split.payload["method"]),
    )
    write_audit_artifacts(audit, tables, output_dir)
    return audit
