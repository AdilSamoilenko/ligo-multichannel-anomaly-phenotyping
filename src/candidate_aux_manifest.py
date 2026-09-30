from pathlib import Path
import csv

OUTPUT = Path("data/H1_O4a_candidate_aux_manifest.csv")

CANDIDATES = [
    {
        "window": 0,
        "start_gps": 1369599133,
        "end_gps": 1369599143,
    },
    {
        "window": 7,
        "start_gps": 1374550465,
        "end_gps": 1374550475,
    },
    {
        "window": 10,
        "start_gps": 1376516149,
        "end_gps": 1376516159,
    },
]

CHANNELS = [
    "H1:IMC-WFS_A_DC_PIT_OUT_DQ",
    "H1:IMC-WFS_A_DC_YAW_OUT_DQ",
    "H1:LSC-REFL_A_RIN_OUT_DQ",
    "H1:PEM-EY_MAINSMON_EBAY_1_DQ",
]

def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for candidate in CANDIDATES:
        for channel in CHANNELS:
            rows.append({
                "window": candidate["window"],
                "start_gps": candidate["start_gps"],
                "end_gps": candidate["end_gps"],
                "channel": channel,
                "status": "available_frame_required",
            })

    with OUTPUT.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "window",
                "start_gps",
                "end_gps",
                "channel",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Candidates: {len(CANDIDATES)}")
    print(f"Channels: {len(CHANNELS)}")
    print(f"Analysis combinations: {len(rows)}")
    print(f"Saved: {OUTPUT}")

if __name__ == "__main__":
    main()