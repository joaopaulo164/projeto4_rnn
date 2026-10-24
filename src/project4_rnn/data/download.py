"""Dataset acquisition, local verification, and provenance tracking for ElectricDevices."""

import json
import logging
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from project4_rnn.constants import (
    DATASET_NAME,
    DATASET_SOURCE,
    TEST_FILE_NAME,
    TRAIN_FILE_NAME,
)
from project4_rnn.reproducibility import compute_file_hash

logger = logging.getLogger("project4_rnn.data.download")

# Standard UCR / TimeSeriesClassification / sktime dataset URLs
PRIMARY_URL = "https://www.timeseriesclassification.com/aeon-toolkit/ElectricDevices.zip"
FALLBACK_URL = "https://www.timeseriesclassification.com/Downloads/ElectricDevices.zip"
MIRROR_SKTIME_TRAIN = "https://raw.githubusercontent.com/sktime/sktime/main/sktime/datasets/data/ElectricDevices/ElectricDevices_TRAIN.ts"
MIRROR_SKTIME_TEST = "https://raw.githubusercontent.com/sktime/sktime/main/sktime/datasets/data/ElectricDevices/ElectricDevices_TEST.ts"
MIRROR_HFAWAZ_TRAIN = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/UnivData/ElectricDevices/ElectricDevices_TRAIN.txt"
MIRROR_HFAWAZ_TEST = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/UnivData/ElectricDevices/ElectricDevices_TEST.txt"


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


def download_electricdevices(
    raw_dir: Path,
    force_download: bool = False,
    timeout: int = 30,
) -> dict[str, Any]:
    """Download and extract the ElectricDevices dataset into raw_dir if not already present.

    Args:
        raw_dir: Directory where raw dataset files are stored.
        force_download: If True, forces redownload even if files exist.
        timeout: Request timeout in seconds.

    Returns:
        Dictionary containing manifest metadata (files, hashes, source, timestamp).
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    train_ts = raw_dir / TRAIN_FILE_NAME
    test_ts = raw_dir / TEST_FILE_NAME
    train_txt = raw_dir / "ElectricDevices_TRAIN.txt"
    test_txt = raw_dir / "ElectricDevices_TEST.txt"
    train_arff = raw_dir / "ElectricDevices_TRAIN.arff"
    test_arff = raw_dir / "ElectricDevices_TEST.arff"

    files_exist = (
        (train_ts.is_file() and test_ts.is_file())
        or (train_txt.is_file() and test_txt.is_file())
        or (train_arff.is_file() and test_arff.is_file())
    )

    if files_exist and not force_download:
        logger.info(f"Raw ElectricDevices files found locally in {raw_dir}.")
        return update_raw_manifest(raw_dir, source="local_disk")

    zip_path = raw_dir / "ElectricDevices.zip"
    source_used: str | None = None

    # Strategy 1: Try downloading zip from primary URL
    for candidate_zip_url in (PRIMARY_URL, FALLBACK_URL):
        try:
            logger.info(f"Attempting download of ElectricDevices from: {candidate_zip_url}")
            download_file(candidate_zip_url, zip_path, timeout=timeout)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(raw_dir)
            source_used = candidate_zip_url
            logger.info(f"Extracted ElectricDevices archive to {raw_dir}")
            break
        except Exception as zip_err:
            logger.warning(f"Download from {candidate_zip_url} failed: {zip_err}")

    # Strategy 2: Download .ts files from sktime mirror
    if source_used is None and not files_exist:
        try:
            logger.info("Attempting download from sktime mirror...")
            download_file(MIRROR_SKTIME_TRAIN, train_ts, timeout=timeout)
            download_file(MIRROR_SKTIME_TEST, test_ts, timeout=timeout)
            source_used = "sktime_github_mirror"
            logger.info(f"Downloaded .ts files from sktime mirror into {raw_dir}")
        except Exception as sktime_err:
            logger.warning(f"Sktime mirror download failed: {sktime_err}")

    # Strategy 3: Download .txt files from UnivData mirror
    if source_used is None and not files_exist:
        try:
            logger.info("Attempting download from UnivData mirror...")
            download_file(MIRROR_HFAWAZ_TRAIN, train_txt, timeout=timeout)
            download_file(MIRROR_HFAWAZ_TEST, test_txt, timeout=timeout)
            source_used = "univdata_mirror"
            logger.info(f"Downloaded raw TXT files from UnivData mirror into {raw_dir}")
        except Exception as mirror_err:
            if not files_exist:
                raise RuntimeError(
                    f"Failed to acquire ElectricDevices dataset automatically. "
                    f"Please place {TRAIN_FILE_NAME} and {TEST_FILE_NAME} (or .txt/.arff) in {raw_dir}."
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
                "md5": compute_file_hash(fpath, "md5"),
                "sha256": compute_file_hash(fpath, "sha256"),
            }

    manifest = {
        "dataset_name": DATASET_NAME,
        "source": source,
        "archive_provenance": DATASET_SOURCE,
        "acquisition_timestamp": datetime.now(UTC).isoformat(),
        "files": files_info,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Raw data manifest updated at {manifest_path}")
    return manifest
