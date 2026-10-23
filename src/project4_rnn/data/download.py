"""Dataset acquisition, local verification, and provenance tracking for ECG5000."""

import json
import logging
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from project4_rnn.reproducibility import compute_file_hash

logger = logging.getLogger("project4_rnn.data.download")

# Standard UCR / TimeSeriesClassification dataset URLs
PRIMARY_URL = "https://www.timeseriesclassification.com/aeon-toolkit/ECG5000.zip"
MIRROR_URL = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/UnivData/ECG5000"


def download_file(url: str, dest_path: Path, timeout: int = 30) -> None:
    """Download a remote file using urllib.

    Args:
        url: Direct HTTP/HTTPS URL.
        dest_path: Target local file path.
        timeout: Network timeout in seconds.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    with (
        urllib.request.urlopen(req, timeout=timeout) as response,
        open(dest_path, "wb") as out_file,
    ):
        out_file.write(response.read())


def download_ecg5000(
    raw_dir: Path,
    force_download: bool = False,
    timeout: int = 30,
) -> dict[str, Any]:
    """Download and extract the ECG5000 dataset into raw_dir if not already present.

    Args:
        raw_dir: Directory where raw dataset files are stored.
        force_download: If True, forces redownload even if files exist.
        timeout: Request timeout in seconds.

    Returns:
        Dictionary containing manifest metadata (files, hashes, source, timestamp).
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    raw_dir / "raw_manifest.json"

    # Check for expected txt or arff files
    train_txt = raw_dir / "ECG5000_TRAIN.txt"
    test_txt = raw_dir / "ECG5000_TEST.txt"
    train_arff = raw_dir / "ECG5000_TRAIN.arff"
    test_arff = raw_dir / "ECG5000_TEST.arff"
    train_ts = raw_dir / "ECG5000_TRAIN.ts"
    test_ts = raw_dir / "ECG5000_TEST.ts"

    files_exist = (
        (train_txt.is_file() and test_txt.is_file())
        or (train_arff.is_file() and test_arff.is_file())
        or (train_ts.is_file() and test_ts.is_file())
    )

    if files_exist and not force_download:
        logger.info(f"Raw ECG5000 files found locally in {raw_dir}.")
        return update_raw_manifest(raw_dir, source="local_disk")

    zip_path = raw_dir / "ECG5000.zip"
    source_used: str | None = None

    # Try downloading zip from primary URL
    try:
        logger.info(f"Downloading ECG5000 from primary source: {PRIMARY_URL}")
        download_file(PRIMARY_URL, zip_path, timeout=timeout)
        source_used = PRIMARY_URL
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(raw_dir)
        logger.info(f"Extracted ECG5000 archive to {raw_dir}")
    except Exception as primary_err:
        logger.warning(f"Primary download failed ({primary_err}). Trying mirror sources...")
        try:
            # Try fetching individual text files from mirror
            train_url = f"{MIRROR_URL}/ECG5000_TRAIN.txt"
            test_url = f"{MIRROR_URL}/ECG5000_TEST.txt"
            download_file(train_url, train_txt, timeout=timeout)
            download_file(test_url, test_txt, timeout=timeout)
            source_used = MIRROR_URL
            logger.info(f"Downloaded raw TXT files from mirror: {MIRROR_URL}")
        except Exception as mirror_err:
            if not files_exist:
                raise RuntimeError(
                    f"Failed to acquire ECG5000 dataset automatically. "
                    f"Primary error: {primary_err}. Mirror error: {mirror_err}. "
                    f"Please place ECG5000_TRAIN.txt and ECG5000_TEST.txt (or .arff) in {raw_dir}."
                ) from mirror_err

    return update_raw_manifest(raw_dir, source=source_used or "local_disk")


def update_raw_manifest(raw_dir: Path, source: str) -> dict[str, Any]:
    """Inspect raw files in raw_dir and write raw_manifest.json with hashes.

    Args:
        raw_dir: Path to directory containing raw dataset files.
        source: Provenance string (URL or 'local_disk').

    Returns:
        Manifest dictionary.
    """
    manifest_path = raw_dir / "raw_manifest.json"
    files_info: dict[str, Any] = {}

    for fpath in sorted(raw_dir.glob("*")):
        if (
            fpath.is_file()
            and fpath.name != "raw_manifest.json"
            and not fpath.name.endswith(".zip")
        ):
            files_info[fpath.name] = {
                "size_bytes": fpath.stat().st_size,
                "sha256": compute_file_hash(fpath, "sha256"),
            }

    manifest = {
        "dataset_name": "ECG5000",
        "source": source,
        "acquisition_timestamp": datetime.now(UTC).isoformat(),
        "files": files_info,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Raw data manifest updated at {manifest_path}")
    return manifest
