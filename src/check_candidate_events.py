from pathlib import Path
import requests


CANDIDATES = {
    "W0": (1369599133, 1369599143),
    "W7": (1374550465, 1374550475),
    "W10": (1376516149, 1376516159),
}

CATALOGUES = [
    "GWTC-4.0",
    "GWTC-4.1",
    "GWTC-5.0",
    "O4_Discovery_Papers",
]

MARGIN = 30


def get_catalogue(name):
    url = f"https://gwosc.org/eventapi/json/{name}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def extract_gps(event):
    for key in ["GPS", "gps"]:
        value = event.get(key)

        if value is None:
            continue

        try:
            return float(value)
        except (TypeError, ValueError):
            pass

    return None


def main():
    print("=== CATALOGUE EVENT MATCHING ===")
    print(f"Checking ±{MARGIN} seconds around each candidate.")
    print()

    matches = []

    for catalogue in CATALOGUES:
        print(f"Loading {catalogue}...")

        try:
            data = get_catalogue(catalogue)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue

        for event_name, event in data.items():
            if not isinstance(event, dict):
                continue

            gps = extract_gps(event)

            if gps is None:
                continue

            for window, (start, end) in CANDIDATES.items():
                if start - MARGIN <= gps <= end + MARGIN:
                    matches.append(
                        (
                            window,
                            catalogue,
                            event_name,
                            gps,
                        )
                    )

    print()
    print("=== MATCHES ===")

    if not matches:
        print("No catalogue events found within ±30 seconds of any candidate.")
    else:
        for window, catalogue, name, gps in matches:
            print(
                f"{window}: {catalogue} -> "
                f"{name} at GPS {gps}"
            )


if __name__ == "__main__":
    main()