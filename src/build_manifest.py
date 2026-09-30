from pathlib import Path
import csv


SEGMENTS_FILE = Path("data/H1-O4a-segments.csv")
OUTPUT_FILE = Path("data/H1_O4a_window_manifest.csv")

WINDOW_SECONDS = 10
N_WINDOWS = 30


def load_segments(path: Path):
    segments = []

    with path.open("r", newline="") as file:
        reader = csv.reader(file)

        for row in reader:
            if len(row) < 2:
                continue

            start = int(row[0])
            end = int(row[1])

            if end - start >= WINDOW_SECONDS:
                segments.append((start, end))

    return segments


def build_manifest(segments):
    if not segments:
        raise RuntimeError("No usable observing segments found.")

    total_duration = sum(
        end - start
        for start, end in segments
    )

    spacing = total_duration / N_WINDOWS

    windows = []

    for i in range(N_WINDOWS):
        target = spacing * (i + 0.5)

        cumulative = 0

        for start, end in segments:
            duration = end - start

            if cumulative + duration > target:
                offset = int(target - cumulative)

                window_start = start + offset

                if window_start + WINDOW_SECONDS > end:
                    window_start = end - WINDOW_SECONDS

                windows.append(
                    {
                        "detector": "H1",
                        "start_gps": window_start,
                        "end_gps": window_start + WINDOW_SECONDS,
                        "duration_seconds": WINDOW_SECONDS,
                    }
                )

                break

            cumulative += duration

    return windows


def save_manifest(windows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "detector",
                "start_gps",
                "end_gps",
                "duration_seconds",
            ],
        )

        writer.writeheader()
        writer.writerows(windows)


def main():
    segments = load_segments(SEGMENTS_FILE)

    print(f"Usable segments: {len(segments)}")

    windows = build_manifest(segments)

    save_manifest(windows, OUTPUT_FILE)

    print(f"Windows created: {len(windows)}")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()