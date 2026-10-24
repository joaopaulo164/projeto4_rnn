"""Data module for ElectricDevices acquisition, loading, contract validation, and splitting."""

from project4_rnn.data.dataset import ElectricDevicesDataset, create_dataloaders
from project4_rnn.data.download import download_electricdevices, update_raw_manifest
from project4_rnn.data.loader import load_raw_electricdevices
from project4_rnn.data.preprocessing import (
    fit_and_apply_scaler,
    map_internal_to_official_labels,
    map_official_to_internal_labels,
)
from project4_rnn.data.splitting import create_or_load_split
from project4_rnn.data.validation import (
    DatasetContractError,
    save_dataset_manifest,
    validate_dataset_arrays,
)

__all__ = [
    "download_electricdevices",
    "update_raw_manifest",
    "load_raw_electricdevices",
    "DatasetContractError",
    "validate_dataset_arrays",
    "save_dataset_manifest",
    "map_official_to_internal_labels",
    "map_internal_to_official_labels",
    "fit_and_apply_scaler",
    "create_or_load_split",
    "ElectricDevicesDataset",
    "create_dataloaders",
]
