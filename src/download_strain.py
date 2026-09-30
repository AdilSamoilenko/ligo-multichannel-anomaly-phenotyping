from pathlib import Path
import csv
import requests


MANIFEST = Path("data/H1_O4a_window_manifest.csv")
DATA_DIR = Path("data/strain_files")

API = "https://gwosc.org/api/v2/datasets/O4a_4KHZ_R1/strain-files"


def get_file_info(start_gps: int, end_gps: int):
    params = {
        "start": start_gps,
        "stop": end_gps,
        "detector": "H1",
        "sample-rate": 4,
        "pagesize": 20,
    }

    response = requests.get(API, params=params, timeout=30)
    response.raise_for_status()

    results = response.json()["results"]

    if not results:
        raise RuntimeError(
            f"No GWOSC strain file found for "
            f"{start_gps} to {end_gps}"
        )

    return results[0]


def download_file(url: str, destination: Path):
    if destination.exists():
        print(f"Already exists: {destination.name}")
        return

    print(f"Downloading: {destination.name}")

    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()

        with destination.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)

    print(f"Saved: {destination}")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with MANIFEST.open("r", newline="") as file:
        windows = list(csv.DictReader(file))

    downloaded = {}

    for i, window in enumerate(windows, start=1):
        start = int(window["start_gps"])
        end = int(window["end_gps"])

        info = get_file_info(start, end)

        url = info["hdf5_url"]
        filename = url.split("/")[-1].split("?")[0]
        destination = DATA_DIR / filename

        print(
            f"[{i}/{len(windows)}] "
            f"{start}-{end} -> {filename}"
        )

        download_file(url, destination)

        downloaded[filename] = {
            "gps_start": info["gps_start"],
            "url": url,
            "local_path": str(destination),
        }

    print()
    print(f"Unique HDF5 files required: {len(downloaded)}")


if __name__ == "__main__":
    main()