#!/usr/bin/env python3

import json
import math
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np

ROOT = Path(".")
NETWORK = ROOT / "results_overnight/checkpoints/episode_network.json"
OUT = ROOT / "results_overnight/deep_dive"
OUT.mkdir(parents=True, exist_ok=True)

REPORT = OUT / "DEEP_DIVE_REPORT.txt"
JSON_OUT = OUT / "deep_dive_results.json"

GROUPS = {
    "WFS": {"WFS_PIT", "WFS_YAW"},
    "LSC": {"LSC_POP", "LSC_REFL"},
    "OAF": {"OAF_PIT", "OAF_YAW", "OAF_REFL"},
    "PEM": {"PEM_MAINS"},
    "ETMX": {"ETMX_L1", "ETMX_L2", "ETMX_L3"},
    "PI": {"PI_MON"},
    "PCAL": {"PCALX", "PCALY"},
}

def group(ch):
    for g, members in GROUPS.items():
        if ch in members:
            return g
    return "OTHER"

def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)

def write_report(lines):
    REPORT.write_text("\n".join(lines), encoding="utf-8")

results = {
    "metadata": {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "network_file": str(NETWORK),
    },
    "network": {},
    "cross_subsystem": [],
    "within_subsystem": [],
    "strongest": [],
    "sanity": {},
}

report = []

report.append("=" * 80)
report.append("LIGO AUXILIARY CHANNEL OVERNIGHT DEEP DIVE")
report.append("=" * 80)
report.append("")
report.append(f"Created: {results['metadata']['created_utc']}")
report.append(f"Input:   {NETWORK}")
report.append("")

# ----------------------------------------------------------------------
# 1. Load network
# ----------------------------------------------------------------------

report.append("=" * 80)
report.append("1. NETWORK VALIDATION")
report.append("=" * 80)

if not NETWORK.exists():
    report.append("ERROR: episode_network.json does not exist.")
    write_report(report)
    print("\n".join(report))
    raise SystemExit(1)

try:
    x = json.loads(NETWORK.read_text())
except Exception as e:
    report.append(f"ERROR loading JSON: {e}")
    write_report(report)
    raise

channels = list(x.keys())

report.append(f"Channels: {len(channels)}")
report.append("")

for ch in channels:
    report.append(f"  {ch:12s} [{group(ch)}]")

# ----------------------------------------------------------------------
# 2. Extract every valid relationship
# ----------------------------------------------------------------------

rows = []

for a, targets in x.items():
    if not isinstance(targets, dict):
        continue

    for b, d in targets.items():
        if a == b or not isinstance(d, dict):
            continue

        r = d.get("r")
        lag = d.get("lag")

        if not finite(r) or not finite(lag):
            continue

        row = {
            "a": a,
            "b": b,
            "group_a": group(a),
            "group_b": group(b),
            "r": float(r),
            "abs_r": abs(float(r)),
            "lag_seconds": float(lag),
        }

        rows.append(row)

        if group(a) == group(b):
            results["within_subsystem"].append(row)
        else:
            results["cross_subsystem"].append(row)

results["network"] = {
    "channels": channels,
    "valid_relationships": len(rows),
    "cross_subsystem_relationships": len(results["cross_subsystem"]),
    "within_subsystem_relationships": len(results["within_subsystem"]),
}

report.append("")
report.append(f"Valid relationships: {len(rows)}")
report.append(
    f"Cross-subsystem relationships: "
    f"{len(results['cross_subsystem'])}"
)
report.append(
    f"Within-subsystem relationships: "
    f"{len(results['within_subsystem'])}"
)

# ----------------------------------------------------------------------
# 3. Strongest overall relationships
# ----------------------------------------------------------------------

report.append("")
report.append("=" * 80)
report.append("2. STRONGEST OVERALL RELATIONSHIPS")
report.append("=" * 80)

strongest = sorted(rows, key=lambda z: z["abs_r"], reverse=True)

results["strongest"] = strongest[:50]

for i, z in enumerate(strongest[:30], 1):
    report.append(
        f"{i:02d}. "
        f"{z['a']:12s} -> {z['b']:12s} "
        f"r={z['r']:+.6f} "
        f"|r|={z['abs_r']:.6f} "
        f"lag={z['lag_seconds']:+.6f}s "
        f"[{z['group_a']} -> {z['group_b']}]"
    )

# ----------------------------------------------------------------------
# 4. Cross-subsystem relationships
# ----------------------------------------------------------------------

report.append("")
report.append("=" * 80)
report.append("3. STRONGEST CROSS-SUBSYSTEM RELATIONSHIPS")
report.append("=" * 80)

cross = sorted(
    results["cross_subsystem"],
    key=lambda z: z["abs_r"],
    reverse=True
)

for i, z in enumerate(cross, 1):
    report.append(
        f"{i:02d}. "
        f"{z['a']:12s} -> {z['b']:12s} "
        f"r={z['r']:+.6f} "
        f"lag={z['lag_seconds']:+.6f}s "
        f"[{z['group_a']} -> {z['group_b']}]"
    )

# ----------------------------------------------------------------------
# 5. Strong relationships that involve independent-looking systems
# ----------------------------------------------------------------------

interesting_groups = {"PEM", "ETMX", "PCAL", "PI"}

report.append("")
report.append("=" * 80)
report.append("4. CROSS-SUBSYSTEM RELATIONSHIPS INVOLVING PEM / ETMX / PCAL / PI")
report.append("=" * 80)

independent_candidates = []

for z in cross:
    if z["group_a"] in interesting_groups or z["group_b"] in interesting_groups:
        independent_candidates.append(z)

for i, z in enumerate(independent_candidates[:50], 1):
    report.append(
        f"{i:02d}. "
        f"{z['a']:12s} -> {z['b']:12s} "
        f"r={z['r']:+.6f} "
        f"lag={z['lag_seconds']:+.6f}s "
        f"[{z['group_a']} -> {z['group_b']}]"
    )

if not independent_candidates:
    report.append("No valid relationships involving these groups.")

# ----------------------------------------------------------------------
# 6. Sanity checks
# ----------------------------------------------------------------------

report.append("")
report.append("=" * 80)
report.append("5. SANITY CHECKS")
report.append("=" * 80)

same_signal_like = [
    z for z in rows
    if z["abs_r"] >= 0.95
]

near_zero_lag = [
    z for z in rows
    if abs(z["lag_seconds"]) <= 0.02
]

results["sanity"] = {
    "relationships_abs_r_ge_0_95": same_signal_like,
    "relationships_abs_lag_le_0_02": near_zero_lag,
}

report.append(
    f"|r| >= 0.95 relationships: {len(same_signal_like)}"
)
report.append(
    f"|lag| <= 0.02 s relationships: {len(near_zero_lag)}"
)

report.append("")
report.append("Very high correlations should be treated as possible")
report.append("shared/derived/control-system signals until channel")
report.append("provenance is established.")

for z in same_signal_like[:20]:
    report.append(
        f"  {z['a']:12s} -> {z['b']:12s} "
        f"r={z['r']:+.6f}, lag={z['lag_seconds']:+.6f}s"
    )

# ----------------------------------------------------------------------
# 7. Search repository for candidate episode GPS
# ----------------------------------------------------------------------

report.append("")
report.append("=" * 80)
report.append("6. CANDIDATE EPISODE FILE SEARCH")
report.append("=" * 80)

gps_terms = [
    "1376516121",
    "1376516128",
    "1376516096",
]

candidate_files = []

for p in ROOT.rglob("*"):
    if not p.is_file():
        continue

    if ".git" in p.parts:
        continue

    try:
        size = p.stat().st_size
    except Exception:
        continue

    # Avoid reading huge binary files.
    if size > 50_000_000:
        continue

    try:
        data = p.read_bytes()
    except Exception:
        continue

    for term in gps_terms:
        if term.encode() in data:
            candidate_files.append({
                "file": str(p),
                "term": term,
            })

            report.append(
                f"Found {term} in {p}"
            )

            break

if not candidate_files:
    report.append(
        "No small repository file directly contains the candidate GPS strings."
    )

# ----------------------------------------------------------------------
# 8. Inspect known candidate result files
# ----------------------------------------------------------------------

report.append("")
report.append("=" * 80)
report.append("7. EXISTING RESULT FILES")
report.append("=" * 80)

result_candidates = [
    ROOT / "results/H1_O4a_candidate_summary.csv",
    ROOT / "results/anomaly_windows.csv",
    ROOT / "results/tables/candidate_windows.json",
    ROOT / "results/candidate_windows.json",
]

for p in result_candidates:
    if p.exists():
        report.append(f"FOUND: {p}")

        try:
            text = p.read_text(errors="replace")
            matches = []

            for line_no, line in enumerate(text.splitlines(), 1):
                if any(term in line for term in gps_terms):
                    matches.append((line_no, line[:500]))

            if matches:
                report.append(f"  Matching lines: {len(matches)}")
                for line_no, line in matches[:20]:
                    report.append(
                        f"    {line_no}: {line}"
                    )
            else:
                report.append("  No direct GPS match.")

        except Exception as e:
            report.append(f"  Read error: {e}")

# ----------------------------------------------------------------------
# 9. Inspect source code parameters
# ----------------------------------------------------------------------

report.append("")
report.append("=" * 80)
report.append("8. CORRELATION PARAMETER AUDIT")
report.append("=" * 80)

scripts = [
    "episode_correlations.py",
    "lag_correlation_scan.py",
    "lag_significance.py",
    "cross_channel_episode.py",
    "cross_channel_scan.py",
    "episode_multichannel.py",
    "episode_network.py",
]

for name in scripts:
    p = ROOT / name

    if not p.exists():
        continue

    report.append("")
    report.append(f"[{name}]")

    try:
        lines = p.read_text(errors="replace").splitlines()

        keywords = [
            "MAX_LAG",
            "max_lag",
            "window",
            "fs",
            "surrogate",
            "shuffle",
            "lag",
            "correlation",
            "GPS",
            "gps",
        ]

        seen = set()

        for i, line in enumerate(lines, 1):
            lower = line.lower()

            if any(k.lower() in lower for k in keywords):
                if line.strip() and i not in seen:
                    report.append(f"  {i}: {line[:300]}")
                    seen.add(i)

    except Exception as e:
        report.append(f"  Read error: {e}")

# ----------------------------------------------------------------------
# 10. Inspect statistical files
# ----------------------------------------------------------------------

report.append("")
report.append("=" * 80)
report.append("9. STATISTICAL RESULT INVENTORY")
report.append("=" * 80)

stats_dir = ROOT / "results_overnight/statistics"

if stats_dir.exists():
    for p in sorted(stats_dir.glob("*")):
        if not p.is_file():
            continue

        try:
            size = p.stat().st_size
            report.append(f"{p.name:45s} {size:>12,} bytes")
        except Exception:
            pass

# ----------------------------------------------------------------------
# 11. Candidate interpretation flags
# ----------------------------------------------------------------------

report.append("")
report.append("=" * 80)
report.append("10. AUTOMATED CANDIDATE FLAGS")
report.append("=" * 80)

flags = []

for z in cross:
    reasons = []

    if z["abs_r"] >= 0.95:
        reasons.append("near-identical waveform correlation")

    if z["abs_r"] >= 0.75:
        reasons.append("strong cross-subsystem correlation")

    if abs(z["lag_seconds"]) <= 0.02:
        reasons.append("near-zero lag")

    if abs(z["lag_seconds"]) >= 0.25:
        reasons.append("non-trivial temporal offset")

    if reasons:
        flags.append({
            **z,
            "flags": reasons,
        })

for z in flags[:50]:
    report.append(
        f"{z['a']} -> {z['b']}: "
        + "; ".join(z["flags"])
    )

# ----------------------------------------------------------------------
# 12. Scientific interpretation guardrails
# ----------------------------------------------------------------------

report.append("")
report.append("=" * 80)
report.append("11. SCIENTIFIC STATUS")
report.append("=" * 80)

report.append("")
report.append(
    "CURRENT STATUS:"
)
report.append(
    "The analysis identifies strong auxiliary-channel relationships "
    "within the selected episode."
)
report.append("")
report.append(
    "It does NOT establish:"
)
report.append(
    "  - a gravitational-wave detection"
)
report.append(
    "  - a new instrumental coupling"
)
report.append(
    "  - a new transient class"
)
report.append(
    "  - causal direction"
)
report.append(
    "  - statistical significance corrected for the full search"
)
report.append(
    "  - independent replication"
)
report.append(
    "  - novelty relative to the LIGO literature"
)

report.append("")
report.append(
    "The next scientifically meaningful test is independent replication "
    "of a frozen phenotype on data not used for discovery."
)

# ----------------------------------------------------------------------
# 13. Save
# ----------------------------------------------------------------------

JSON_OUT.write_text(
    json.dumps(results, indent=2, allow_nan=False),
    encoding="utf-8"
)

write_report(report)

print("")
print("=" * 80)
print("OVERNIGHT DEEP DIVE COMPLETE")
print("=" * 80)
print(f"Report: {REPORT}")
print(f"JSON:   {JSON_OUT}")
print("")
print("Strongest cross-subsystem relationships:")
for z in cross[:10]:
    print(
        f"  {z['a']:12s} -> {z['b']:12s} "
        f"r={z['r']:+.6f} "
        f"lag={z['lag_seconds']:+.6f}s"
    )

