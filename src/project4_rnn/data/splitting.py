"""Hash-grouped, stratified train/validation split of the official ElectricDevices training set.

Series with identical raw values (same SHA-256 fingerprint) always stay together in a single
partition, so no exact duplicate crosses the train/validation boundary. Rows whose identical
series carry different official labels are never relabeled, voted on, or dropped: the whole group
is kept in one partition.
"""

import json
import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from project4_rnn.constants import (
    NUM_CLASSES,
    OFFICIAL_CLASSES,
    SERIES_HASH_ALGORITHM,
    SERIES_HASH_DTYPE,
    SERIES_HASH_ENDIANNESS,
    SPLIT_METHOD,
    SPLIT_METHOD_VERSION,
    SPLIT_N_SPLITS,
    SPLIT_SEED,
    SPLIT_SELECTION_CRITERION,
    VAL_SPLIT_RATIO,
)
from project4_rnn.reproducibility import compute_object_hash

logger = logging.getLogger("project4_rnn.data.splitting")

MAX_REPORTED_ITEMS = 10


class SplitIntegrityError(Exception):
    """Raised when a train/validation split violates a structural or leakage invariant."""

    def __init__(
        self,
        condition: str,
        expected: Any,
        observed: Any,
        offending: Sequence[Any] | None = None,
        suggested_action: str = "",
    ) -> None:
        self.condition = condition
        self.expected = expected
        self.observed = observed
        self.offending = list(offending) if offending is not None else []
        self.suggested_action = suggested_action

        lines = [
            f"SplitIntegrityError: {condition}",
            f"  Expected: {expected}",
            f"  Observed: {observed}",
        ]
        if self.offending:
            shown = ", ".join(str(item) for item in self.offending[:MAX_REPORTED_ITEMS])
            extra = len(self.offending) - MAX_REPORTED_ITEMS
            suffix = f" (+{extra} more)" if extra > 0 else ""
            lines.append(f"  Offending hashes/indices: {shown}{suffix}")
        if suggested_action:
            lines.append(f"  Suggested action: {suggested_action}")
        super().__init__("\n".join(lines))


@dataclass
class SplitResult:
    """Validated train/validation split with its identity."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    split_hash: str
    payload: dict[str, Any]


def group_ids_from_hashes(series_hashes: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    """Map each sample to the integer id of its identical-series group (sorted by hash)."""
    unique_hashes, inverse = np.unique(
        np.asarray(list(series_hashes), dtype=str), return_inverse=True
    )
    return inverse.astype(np.int64), [str(h) for h in unique_hashes]


def series_hash_digest(series_hashes: Sequence[str]) -> str:
    """Fingerprint of the ordered list of per-sample hashes (index/hash consistency check)."""
    return compute_object_hash(list(series_hashes))


def _class_proportions(labels: np.ndarray) -> dict[int, float]:
    total = len(labels)
    counts = Counter(int(v) for v in labels)
    return {c: (counts.get(c, 0) / total if total else 0.0) for c in OFFICIAL_CLASSES}


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    counts = Counter(int(v) for v in labels)
    return {str(c): int(counts.get(c, 0)) for c in OFFICIAL_CLASSES}


def score_fold(
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    target_val_fraction: float,
) -> float:
    """Objective score of a candidate fold (lower is better).

    score = |val_fraction - target| + L1(val class proportions - official)
            + L1(train class proportions - official)

    It depends only on partition sizes and class distributions, never on model performance.
    """
    official = _class_proportions(y)
    val_props = _class_proportions(y[val_idx])
    train_props = _class_proportions(y[train_idx])
    size_term = abs(len(val_idx) / len(y) - target_val_fraction)
    val_term = sum(abs(val_props[c] - official[c]) for c in OFFICIAL_CLASSES)
    train_term = sum(abs(train_props[c] - official[c]) for c in OFFICIAL_CLASSES)
    return float(size_term + val_term + train_term)


def evaluate_candidate_folds(
    y: np.ndarray,
    group_ids: np.ndarray,
    seed: int,
    n_splits: int,
    target_val_fraction: float,
) -> list[dict[str, Any]]:
    """Evaluate every StratifiedGroupKFold fold as a candidate train/validation split."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    candidates: list[dict[str, Any]] = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(np.zeros(len(y)), y, group_ids)):
        train_sorted = np.sort(train_idx).astype(np.int64)
        val_sorted = np.sort(val_idx).astype(np.int64)
        shared = set(group_ids[train_sorted].tolist()) & set(group_ids[val_sorted].tolist())
        classes_train = {int(v) for v in np.unique(y[train_sorted])}
        classes_val = {int(v) for v in np.unique(y[val_sorted])}
        has_all_classes = classes_train >= set(OFFICIAL_CLASSES) and classes_val >= set(
            OFFICIAL_CLASSES
        )
        candidates.append(
            {
                "fold": fold,
                "train_samples": int(len(train_sorted)),
                "val_samples": int(len(val_sorted)),
                "val_fraction": float(len(val_sorted) / len(y)),
                "shared_hashes": len(shared),
                "all_classes_present": bool(has_all_classes),
                "eligible": bool(len(shared) == 0 and has_all_classes),
                "score": score_fold(y, train_sorted, val_sorted, target_val_fraction),
                "_train_idx": train_sorted,
                "_val_idx": val_sorted,
            }
        )
    return candidates


def _group_summaries(
    y: np.ndarray,
    series_hashes: Sequence[str],
    val_set: set[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Duplicate-group and conflicting-group summaries for the persisted payload."""
    members: dict[str, list[int]] = {}
    for idx, h in enumerate(series_hashes):
        members.setdefault(h, []).append(idx)

    duplicate_groups = {h: v for h, v in members.items() if len(v) > 1}
    conflicting: list[dict[str, Any]] = []
    for h, idxs in sorted(duplicate_groups.items()):
        label_counts = Counter(int(y[i]) for i in idxs)
        if len(label_counts) > 1:
            in_val = [i in val_set for i in idxs]
            partition = "validation" if all(in_val) else "train" if not any(in_val) else "SPLIT"
            conflicting.append(
                {
                    "series_hash": h,
                    "size": len(idxs),
                    "labels": {str(k): int(v) for k, v in sorted(label_counts.items())},
                    "partition": partition,
                }
            )

    duplicates_summary = {
        "groups": len(duplicate_groups),
        "samples_involved": int(sum(len(v) for v in duplicate_groups.values())),
        "excess": int(sum(len(v) - 1 for v in duplicate_groups.values())),
    }
    conflicts_summary = {
        "count": len(conflicting),
        "samples_involved": int(sum(g["size"] for g in conflicting)),
        "classes_involved": sorted({int(k) for g in conflicting for k in g["labels"]}),
        "assignment": conflicting,
    }
    return duplicates_summary, conflicts_summary


def build_split_payload(
    y: np.ndarray,
    series_hashes: Sequence[str],
    dataset_hash: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    seed: int,
    n_splits: int,
    selected_fold: int,
    selected_score: float,
    fold_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the persisted split document (without checksum)."""
    val_set = {int(i) for i in val_idx}
    duplicates_summary, conflicts_summary = _group_summaries(y, series_hashes, val_set)
    train_hashes = {series_hashes[int(i)] for i in train_idx}
    val_hashes = {series_hashes[int(i)] for i in val_idx}

    train_list = [int(i) for i in train_idx]
    val_list = [int(i) for i in val_idx]
    split_hash = compute_object_hash(
        {
            "method": SPLIT_METHOD,
            "method_version": SPLIT_METHOD_VERSION,
            "split_seed": seed,
            "n_splits": n_splits,
            "selected_fold": selected_fold,
            "dataset_hash": dataset_hash,
            "train_indices": train_list,
            "val_indices": val_list,
        }
    )

    return {
        "method": SPLIT_METHOD,
        "method_version": SPLIT_METHOD_VERSION,
        "split_seed": seed,
        "n_splits": n_splits,
        "selected_fold": selected_fold,
        "selection_criterion": SPLIT_SELECTION_CRITERION,
        "selected_score": selected_score,
        "fold_candidates": [
            {k: v for k, v in c.items() if not k.startswith("_")} for c in fold_candidates
        ],
        "hash_algorithm": SERIES_HASH_ALGORITHM,
        "hash_dtype": SERIES_HASH_DTYPE,
        "hash_endianness": SERIES_HASH_ENDIANNESS,
        "series_hash_digest": series_hash_digest(series_hashes),
        "dataset_hash": dataset_hash,
        "total_official_train_samples": int(len(y)),
        "num_groups": len(set(series_hashes)),
        "duplicate_groups": duplicates_summary,
        "conflicting_groups": conflicts_summary,
        "train_samples": len(train_list),
        "val_samples": len(val_list),
        "val_fraction": len(val_list) / len(y),
        "official_class_distribution": _class_counts(y),
        "train_class_distribution": _class_counts(y[train_idx]),
        "val_class_distribution": _class_counts(y[val_idx]),
        "shared_hashes_train_val": len(train_hashes & val_hashes),
        "split_hash": split_hash,
        "generated_at": datetime.now(UTC).isoformat(),
        "train_indices": train_list,
        "val_indices": val_list,
    }


def compute_payload_checksum(payload: dict[str, Any]) -> str:
    """Checksum of the whole split document, excluding the checksum field itself."""
    return compute_object_hash({k: v for k, v in payload.items() if k != "checksum_sha256"})


def save_split_payload(payload: dict[str, Any], path: Path) -> None:
    """Persist the split document with its checksum."""
    document = dict(payload)
    document["checksum_sha256"] = compute_payload_checksum(document)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=2)
    payload["checksum_sha256"] = document["checksum_sha256"]


def verify_split_integrity(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    y: np.ndarray,
    series_hashes: Sequence[str],
) -> None:
    """Fail-fast verification of the structural and leakage invariants of a split.

    Raises:
        SplitIntegrityError: On the first violated condition.
    """
    total = len(y)
    if len(series_hashes) != total:
        raise SplitIntegrityError(
            "one series hash per sample",
            total,
            len(series_hashes),
            suggested_action="Recompute hashes from the same dataset used to create the split.",
        )
    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)

    for name, idx in (("train", train_idx), ("validation", val_idx)):
        counts = Counter(int(i) for i in idx)
        repeated = sorted(i for i, n in counts.items() if n > 1)
        if repeated:
            raise SplitIntegrityError(
                f"no repeated indices inside the {name} partition",
                0,
                len(repeated),
                repeated,
                "Regenerate the split with prepare-data --force.",
            )
        if len(idx) and (int(idx.min()) < 0 or int(idx.max()) >= total):
            raise SplitIntegrityError(
                f"{name} indices within [0, {total - 1}]",
                f"[0, {total - 1}]",
                f"[{int(idx.min())}, {int(idx.max())}]",
                suggested_action="Regenerate the split with prepare-data --force.",
            )

    overlap = sorted(set(train_idx.tolist()) & set(val_idx.tolist()))
    if overlap:
        raise SplitIntegrityError(
            "disjoint train and validation indices",
            0,
            len(overlap),
            overlap,
            "Regenerate the split with prepare-data --force.",
        )

    covered = len(train_idx) + len(val_idx)
    if covered != total:
        raise SplitIntegrityError(
            "train + validation cover every official training sample",
            total,
            covered,
            suggested_action="Regenerate the split with prepare-data --force.",
        )

    for name, idx in (("train", train_idx), ("validation", val_idx)):
        present = sorted({int(v) for v in np.unique(y[idx])})
        if present != list(OFFICIAL_CLASSES):
            raise SplitIntegrityError(
                f"all {NUM_CLASSES} classes present in the {name} partition",
                list(OFFICIAL_CLASSES),
                present,
                suggested_action="Use a different split seed or inspect the label distribution.",
            )

    train_hashes = {series_hashes[int(i)] for i in train_idx}
    val_hashes = {series_hashes[int(i)] for i in val_idx}
    shared = sorted(train_hashes & val_hashes)
    if shared:
        raise SplitIntegrityError(
            "indivisible identical-series groups (zero shared hashes between train and validation)",
            0,
            len(shared),
            shared,
            "Regenerate the split with prepare-data --force (grouped by series hash).",
        )


def verify_split_payload(
    payload: dict[str, Any],
    y: np.ndarray,
    series_hashes: Sequence[str],
    dataset_hash: str,
    seed: int,
    n_splits: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Verify a persisted split document against the current dataset and parameters."""
    if payload.get("method") != SPLIT_METHOD or "method_version" not in payload:
        raise SplitIntegrityError(
            "split document produced by the hash-grouped method",
            SPLIT_METHOD,
            payload.get("method", "legacy (no method field)"),
            suggested_action=(
                "The existing split is obsolete (index-based stratified split). Run "
                "'clean-artifacts --generated-only' or 'prepare-data --force' to regenerate it."
            ),
        )

    stored_checksum = payload.get("checksum_sha256")
    expected_checksum = compute_payload_checksum(payload)
    if stored_checksum != expected_checksum:
        raise SplitIntegrityError(
            "valid split file checksum",
            expected_checksum,
            stored_checksum,
            suggested_action="The split file was modified; regenerate it with prepare-data --force.",
        )

    if payload.get("dataset_hash") != dataset_hash:
        raise SplitIntegrityError(
            "split generated for the current dataset",
            dataset_hash,
            payload.get("dataset_hash"),
            suggested_action="Regenerate the split with prepare-data --force.",
        )
    if payload.get("split_seed") != seed or payload.get("n_splits") != n_splits:
        raise SplitIntegrityError(
            "split parameters equal to the configuration (seed, n_splits)",
            {"split_seed": seed, "n_splits": n_splits},
            {"split_seed": payload.get("split_seed"), "n_splits": payload.get("n_splits")},
            suggested_action="Regenerate the split with prepare-data --force.",
        )
    if payload.get("total_official_train_samples") != len(y):
        raise SplitIntegrityError(
            "split covers the current official training set",
            len(y),
            payload.get("total_official_train_samples"),
            suggested_action="Regenerate the split with prepare-data --force.",
        )
    if payload.get("series_hash_digest") != series_hash_digest(series_hashes):
        raise SplitIntegrityError(
            "indices consistent with the series hashes of the current dataset",
            payload.get("series_hash_digest"),
            series_hash_digest(series_hashes),
            suggested_action="Regenerate the split with prepare-data --force.",
        )

    train_idx = np.asarray(payload["train_indices"], dtype=np.int64)
    val_idx = np.asarray(payload["val_indices"], dtype=np.int64)
    verify_split_integrity(train_idx, val_idx, y, series_hashes)

    recorded = {
        "train_class_distribution": _class_counts(y[train_idx]),
        "val_class_distribution": _class_counts(y[val_idx]),
    }
    for key, observed in recorded.items():
        if payload.get(key) != observed:
            raise SplitIntegrityError(
                f"recorded {key} equal to the recomputed distribution",
                payload.get(key),
                observed,
                suggested_action="Regenerate the split with prepare-data --force.",
            )
    if payload.get("shared_hashes_train_val") != 0:
        raise SplitIntegrityError(
            "recorded shared_hashes_train_val equal to 0",
            0,
            payload.get("shared_hashes_train_val"),
            suggested_action="Regenerate the split with prepare-data --force.",
        )
    return train_idx, val_idx


def read_split_identity(split_indices_path: Path) -> str:
    """Return the split hash of a persisted split after verifying its checksum.

    Raises:
        FileNotFoundError: If the split file does not exist.
        SplitIntegrityError: If the file is legacy or its checksum is invalid.
    """
    split_indices_path = Path(split_indices_path)
    if not split_indices_path.is_file():
        raise FileNotFoundError(f"Split file not found: {split_indices_path}")
    with open(split_indices_path, encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("method") != SPLIT_METHOD or "split_hash" not in payload:
        raise SplitIntegrityError(
            "split document produced by the hash-grouped method",
            SPLIT_METHOD,
            payload.get("method", "legacy (no method field)"),
            suggested_action="Run 'prepare-data --force' to regenerate the split.",
        )
    if payload.get("checksum_sha256") != compute_payload_checksum(payload):
        raise SplitIntegrityError(
            "valid split file checksum",
            compute_payload_checksum(payload),
            payload.get("checksum_sha256"),
            suggested_action="Run 'prepare-data --force' to regenerate the split.",
        )
    return str(payload["split_hash"])


def create_grouped_split(
    y_train_official: np.ndarray,
    series_hashes: Sequence[str],
    dataset_hash: str,
    seed: int = SPLIT_SEED,
    n_splits: int = SPLIT_N_SPLITS,
    val_ratio: float = VAL_SPLIT_RATIO,
) -> SplitResult:
    """Create the hash-grouped stratified split, choosing the best fold deterministically."""
    y = np.asarray(y_train_official, dtype=np.int64)
    if n_splits < 2 or abs(1.0 / n_splits - val_ratio) > 1e-9:
        raise ValueError(
            f"val_split_ratio ({val_ratio}) must equal 1/n_splits ({n_splits}) for "
            "StratifiedGroupKFold"
        )

    group_ids, _ = group_ids_from_hashes(series_hashes)
    candidates = evaluate_candidate_folds(y, group_ids, seed, n_splits, val_ratio)
    eligible = [c for c in candidates if c["eligible"]]
    if not eligible:
        raise SplitIntegrityError(
            "at least one candidate fold with zero shared hashes and all classes in both partitions",
            ">= 1",
            0,
            suggested_action="Inspect the label distribution or change split_seed/n_splits.",
        )

    best = min(eligible, key=lambda c: (c["score"], c["fold"]))
    train_idx: np.ndarray = best["_train_idx"]
    val_idx: np.ndarray = best["_val_idx"]
    verify_split_integrity(train_idx, val_idx, y, series_hashes)

    payload = build_split_payload(
        y=y,
        series_hashes=series_hashes,
        dataset_hash=dataset_hash,
        train_idx=train_idx,
        val_idx=val_idx,
        seed=seed,
        n_splits=n_splits,
        selected_fold=int(best["fold"]),
        selected_score=float(best["score"]),
        fold_candidates=candidates,
    )
    logger.info(
        f"Selected fold {best['fold']} (score={best['score']:.6f}): "
        f"train={len(train_idx)}, val={len(val_idx)}, shared hashes=0"
    )
    return SplitResult(train_idx, val_idx, str(payload["split_hash"]), payload)


def create_or_load_split(
    y_train_official: np.ndarray,
    series_hashes: Sequence[str],
    split_indices_path: Path,
    dataset_hash: str,
    seed: int = SPLIT_SEED,
    n_splits: int = SPLIT_N_SPLITS,
    val_ratio: float = VAL_SPLIT_RATIO,
    force_recreate: bool = False,
) -> SplitResult:
    """Load the persisted split (verifying it) or create and persist a new grouped split.

    Args:
        y_train_official: Official labels (1..7) of the 8,926 development samples.
        series_hashes: Per-sample SHA-256 fingerprint of the raw series.
        split_indices_path: Path to split_indices.json.
        dataset_hash: Identity hash of the dataset the split belongs to.
        seed: Split seed.
        n_splits: Number of StratifiedGroupKFold folds evaluated.
        val_ratio: Target validation fraction (must equal 1/n_splits).
        force_recreate: Regenerate even when a valid split file exists.

    Raises:
        SplitIntegrityError: If an existing split is obsolete, tampered with, or leaks.
    """
    split_indices_path = Path(split_indices_path)
    y = np.asarray(y_train_official, dtype=np.int64)

    if split_indices_path.is_file() and not force_recreate:
        logger.info(f"Loading existing split from {split_indices_path}")
        with open(split_indices_path, encoding="utf-8") as f:
            payload = json.load(f)
        train_idx, val_idx = verify_split_payload(
            payload, y, series_hashes, dataset_hash, seed, n_splits
        )
        return SplitResult(train_idx, val_idx, str(payload["split_hash"]), payload)

    result = create_grouped_split(y, series_hashes, dataset_hash, seed, n_splits, val_ratio)
    save_split_payload(result.payload, split_indices_path)
    logger.info(f"Saved hash-grouped split to {split_indices_path}")
    return result
