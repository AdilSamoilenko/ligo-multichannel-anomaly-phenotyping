from pathlib import Path

import h5py
import numpy as np


SAMPLE_RATE = 4096


def extract_strain(
    hdf5_path: str | Path,
    start_gps: int,
    end_gps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract a GPS-defined strain interval from a GWOSC O4a HDF5 file.

    Parameters
    ----------
    hdf5_path:
        Path to the GWOSC HDF5 strain file.
    start_gps:
        GPS start time of requested interval.
    end_gps:
        GPS end time of requested interval.

    Returns
    -------
    time:
        Time relative to start_gps, in seconds.
    strain:
        Strain samples.
    """

    if end_gps <= start_gps:
        raise ValueError("end_gps must be greater than start_gps")

    with h5py.File(hdf5_path, "r") as file:
        file_start = int(file["meta/GPSstart"][()])
        duration = int(file["meta/Duration"][()])
        file_end = file_start + duration

        if start_gps < file_start or end_gps > file_end:
            raise ValueError(
                f"Requested interval [{start_gps}, {end_gps}) "
                f"is outside file interval [{file_start}, {file_end})"
            )

        start_sample = int(
            (start_gps - file_start) * SAMPLE_RATE
        )
        end_sample = int(
            (end_gps - file_start) * SAMPLE_RATE
        )

        strain = np.asarray(
            file["strain/Strain"][start_sample:end_sample],
            dtype=np.float64,
        )

    time = np.arange(len(strain), dtype=np.float64) / SAMPLE_RATE

    return time, strain


def validate_strain(
    time: np.ndarray,
    strain: np.ndarray,
) -> dict:
    """
    Run basic integrity checks on an extracted strain segment.
    """

    if len(time) != len(strain):
        raise ValueError("Time and strain arrays have different lengths")

    if len(strain) == 0:
        raise ValueError("Strain array is empty")

    if not np.all(np.isfinite(strain)):
        raise ValueError("Strain contains non-finite values")

    duration = len(strain) / SAMPLE_RATE

    return {
        "samples": int(len(strain)),
        "duration_seconds": float(duration),
        "sample_rate_hz": SAMPLE_RATE,
        "finite": True,
        "mean": float(np.mean(strain)),
        "std": float(np.std(strain)),
        "minimum": float(np.min(strain)),
        "maximum": float(np.max(strain)),
    }