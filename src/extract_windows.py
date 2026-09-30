from pathlib import Path
import csv
import h5py
import numpy as np

MANIFEST = Path("data/H1_O4a_window_manifest.csv")
STRAIN_DIR = Path("data/strain_files")
OUTPUT_DIR = Path("data/windows")
METADATA_FILE = Path("data/H1_O4a_extracted_metadata.csv")

SAMPLE_RATE = 4096


def find_source_file(start_gps):
    candidates = list(STRAIN_DIR.glob(f"*{start_gps}*.hdf5"))

    if candidates:
        return candidates[0]

    for path in STRAIN_DIR.glob("*.hdf5"):
        with h5py.File(path, "r") as file:
            gps_start = int(file["meta/GPSstart"][()])
            duration = int(file["meta/Duration"][()])

        if gps_start <= start_gps < gps_start + duration:
            return path

    raise FileNotFoundError(
        f"No HDF5 file contains GPS time {start_gps}"
    )


def extract_window(source_file, start_gps, end_gps):
    with h5py.File(source_file, "r") as file:
        gps_start = int(file["meta/GPSstart"][()])
        strain = file["strain/Strain"]

        start_index = int(round((start_gps - gps_start) * SAMPLE_RATE))
        end_index = int(round((end_gps - gps_start) * SAMPLE_RATE))

        data = np.asarray(strain[start_index:end_index], dtype=np.float64)

    expected_samples = int((end_gps - start_gps) * SAMPLE_RATE)

    if len(data) != expected_samples:
        raise RuntimeError(
            f"Expected {expected_samples} samples, got {len(data)} "
            f"for {start_gps}-{end_gps}"
        )

    return data


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with MANIFEST.open("r", newline="") as file:
        windows = list(csv.DictReader(file))

    metadata = []

    for i, window in enumerate(windows):
        start_gps = int(window["start_gps"])
        end_gps = int(window["end_gps"])

        output_file = OUTPUT_DIR / f"H1_O4a_window_{i:03d}.npy"

        source_file = find_source_file(start_gps)

        print(
            f"[{i + 1}/{len(windows)}] "
            f"{start_gps}-{end_gps} "
            f"-> {output_file.name}"
        )

        data = extract_window(
            source_file,
            start_gps,
            end_gps,
        )

        np.save(output_file, data)

        metadata.append(
            {
                "window_id": i,
                "detector": "H1",
                "start_gps": start_gps,
                "end_gps": end_gps,
                "duration_seconds": end_gps - start_gps,
                "sample_rate_hz": SAMPLE_RATE,
                "samples": len(data),
                "finite_samples": int(np.isfinite(data).sum()),
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
                "minimum": float(np.min(data)),
                "maximum": float(np.max(data)),
                "source_file": source_file.name,
            }
        )

    with METADATA_FILE.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=metadata[0].keys(),
        )
        writer.writeheader()
        writer.writerows(metadata)

    print()
    print(f"Windows extracted: {len(metadata)}")
    print(f"Saved to: {OUTPUT_DIR}")
    print(f"Metadata: {METADATA_FILE}")


if __name__ == "__main__":
    main()