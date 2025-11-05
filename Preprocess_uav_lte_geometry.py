# preprocess_uav_lte_geometry.py
# Purpose: Preprocess UAV LTE measurement CSVs across multiple altitudes:
# - compute 2D & 3d distanceS and azimuth/elevation from a base station
# - normalize pcell and detected carrier fields to a tidy format
# - optionally filter specific PCIs, combine, and deduplicate
# Output: cleaned per-step CSVs (Filter_1, Filter_2, Filter_3, Filter_4)

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from AltAzRange import AltAzimuthRange  


# ----------------------------- Config dataclasses -----------------------------


@dataclass(frozen=True)
class BSConfig:
    """Base station (observer) geodetic configuration."""
    lat: float
    lon: float
    height_m: float = 30.0  


@dataclass(frozen=True)
class IOConfig:
    """I/O paths and folder structure."""
    input_dir: Path
    out_filter1: Path
    out_filter2_pcell: Path
    out_filter2_detected: Path
    out_filter3: Path
    out_filter4: Path


# ----------------------------- Utility functions -----------------------------


def haversine_2d_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle (2D) distance in meters between two lat/lon points."""
    r = 6371_000.0  # Earth radius in meters
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def find_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """Return first column name that exists in df (case-sensitive)."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ----------------------------- Geometry features -----------------------------


def add_geometry_features(
    df: pd.DataFrame,
    bs: BSConfig,
    lat_col_candidates: Sequence[str] = ("Latitude", "lat", "Lat"),
    lon_col_candidates: Sequence[str] = ("Longitude", "lon", "Lon"),
    alt_col_candidates: Sequence[str] = ("Altitude", "alt", "Alt", "Height"),
    altitude_from_filename: Optional[int] = None,
) -> pd.DataFrame:
    """
    Add 2D distance (Haversine) and, if available, azimuth/elevation relative to BS.
    Falls back gracefully if columns or AltAzRange are not available.

    Parameters
    ----------
    df : pd.DataFrame
        Input raw measurement dataframe.
    bs : BSConfig
        Base station geodetic configuration.
    altitude_from_filename : int | None
        If provided, adds 'Altitude_m' to df using this value (e.g., 20, 25, ...).

    Returns
    -------
    pd.DataFrame
    """
    lat_c = find_col(df, lat_col_candidates)
    lon_c = find_col(df, lon_col_candidates)
    alt_c = find_col(df, alt_col_candidates)

    if altitude_from_filename is not None and "Altitude_m" not in df.columns:
        df["Altitude_m"] = altitude_from_filename

    if lat_c and lon_c:
        df["Distance_2D_m"] = [
            haversine_2d_m(bs.lat, bs.lon, float(la), float(lo)) for la, lo in zip(df[lat_c], df[lon_c])
        ]

        # Optional: azimuth/elevation if AltAzRange is present and altitude exists
        if AltAzimuthRange is not None and (alt_c or "Altitude_m" in df.columns):
            try:
                uav = AltAzimuthRange()
                uav.observer(bs.lat, bs.lon, bs.height_m)
                az_list: List[float] = []
                el_list: List[float] = []
                for _, row in df.iterrows():
                    la = float(row[lat_c])
                    lo = float(row[lon_c])
                    al = float(row[alt_c]) if alt_c else float(row["Altitude_m"])
                    az, el, _rng = uav.target(la, lo, al)  # assumes your helper returns (az, el, range)
                    az_list.append(float(az))
                    el_list.append(float(el))
                df["Azimuth_Angle"] = az_list
                df["Elevation_Angle"] = el_list
            except Exception:
                # If the helper is unavailable or fails, skip angular features
                pass

    return df


# ----------------------------- Normalization (pcell) -----------------------------


def normalize_pcell(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize pcell columns into a tidy set:
    RSRP/RSRQ/PCI/EARFCN/RSSI/Pathloss.

    Expected input column suffix: '(LTE pcell)'
    """
    mapping = {
        "RSRP (LTE pcell)": "RSRP",
        "RSRQ (LTE pcell)": "RSRQ",
        "Physical cell identity (LTE pcell)": "Cell_id",
        "EARFCN (LTE pcell)": "EARFCN",
        "E-UTRAN carrier RSSI (LTE pcell)": "RSSI",
        "Pathloss (LTE pcell)": "Pathloss",
    }
    keep = {k: v for k, v in mapping.items() if k in df.columns}
    out = df.loc[:, list(keep.keys())].rename(columns=keep).copy()
    return out


# ----------------------------- Normalization (detected, wide→long) -----------------------------


def normalize_detected(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 'LTE detected' wide columns with index ' - 1', ' - 2', ... into long/tidy format.

    Example input columns:
      'RSRP (LTE detected) - 1', 'RSRQ (LTE detected) - 1', 'Physical cell identity (LTE detected) - 1', ...
    """
    # Collect indexed field sets
    prefixes = [
        ("RSRP (LTE detected) - {}", "RSRP"),
        ("RSRQ (LTE detected) - {}", "RSRQ"),
        ("Physical cell identity (LTE detected) - {}", "Cell_id"),
        ("EARFCN (LTE detected) - {}", "EARFCN"),
        ("E-UTRAN carrier RSSI (LTE detected) - {}", "RSSI"),
        ("Pathloss (LTE detected) - {}", "Pathloss"),
    ]

    # Find available detect indexes
    indexes: List[int] = []
    for idx in range(1, 16):  # up to 15 carriers; adjust if needed
        any_found = False
        for pat, _alias in prefixes:
            col = pat.format(idx)
            if col in df.columns:
                any_found = True
                break
        if any_found:
            indexes.append(idx)

    # Build long rows
    rows: List[dict] = []
    for _, row in df.iterrows():
        for idx in indexes:
            rec = {}
            for pat, alias in prefixes:
                col = pat.format(idx)
                if col in df.columns:
                    rec[alias] = row[col]
            if rec:  # only keep if something present
                rows.append(rec)

    return pd.DataFrame.from_records(rows) if rows else pd.DataFrame(
        columns=["Cell_id", "EARFCN", "RSRP", "RSRQ", "RSSI", "Pathloss"]
    )


# ----------------------------- Pipeline steps -----------------------------


def step_filter1_process_altitude(
    input_csv: Path, out_csv: Path, bs: BSConfig, altitude_m: Optional[int]
) -> None:
    """Filter_1: per-altitude geometric features + carry raw columns forward."""
    df = pd.read_csv(input_csv)
    df = add_geometry_features(df, bs=bs, altitude_from_filename=altitude_m)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


def step_filter2_split_normalize(
    filter1_csv: Path, out_pcell_csv: Path, out_detected_csv: Path
) -> None:
    """Filter_2: split one file into normalized pcell + detected tables."""
    df = pd.read_csv(filter1_csv)
    pcell = normalize_pcell(df)
    detected = normalize_detected(df)
    out_pcell_csv.parent.mkdir(parents=True, exist_ok=True)
    out_detected_csv.parent.mkdir(parents=True, exist_ok=True)
    pcell.to_csv(out_pcell_csv, index=False)
    detected.to_csv(out_detected_csv, index=False)


def step_filter3_by_pci(
    inputs: Iterable[Path], out_dir: Path, pci_list: Sequence[int]
) -> List[Path]:
    """
    Filter_3: from many normalized files, extract only rows with Cell_id ∈ pci_list.
    Returns list of written file paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for p in inputs:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "Cell_id" not in df.columns:
            continue
        for pci in pci_list:
            sub = df[df["Cell_id"] == int(pci)]
            if sub.empty:
                continue
            out_path = out_dir / f"{p.stem}_pci{pci}.csv"
            sub.to_csv(out_path, index=False)
            written.append(out_path)
    return written


def step_filter4_combine_and_dedup(
    inputs: Iterable[Path],
    out_combined_csv: Path,
    out_dedup_csv: Path,
    dedup_keys: Sequence[str] = ("Distance_2D_m", "Azimuth_Angle", "Elevation_Angle"),
) -> None:
    """Filter_4: combine many CSVs into one and drop duplicates by geometric keys."""
    frames: List[pd.DataFrame] = []
    for p in inputs:
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        # create empty outputs
        out_combined_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(out_combined_csv, index=False)
        pd.DataFrame().to_csv(out_dedup_csv, index=False)
        return

    all_df = pd.concat(frames, ignore_index=True)
    out_combined_csv.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out_combined_csv, index=False)

    keys_present = [k for k in dedup_keys if k in all_df.columns]
    if keys_present:
        dedup_df = all_df.drop_duplicates(subset=keys_present)
    else:
        dedup_df = all_df  # nothing to deduplicate on
    dedup_df.to_csv(out_dedup_csv, index=False)


# ----------------------------- Orchestration -----------------------------


def parse_altitudes(altitudes: Optional[Sequence[int]], remove_80: bool) -> List[int]:
    
    if altitudes:
        lst = sorted(set(int(a) for a in altitudes))
    else:
        lst = list(range(20, 160, 5))
    if remove_80 and 80 in lst:
        lst.remove(80)
    return lst


def discover_input_file(input_dir: Path, altitude_m: int) -> Optional[Path]:
    """Find '<altitude>m.csv' in input_dir (e.g., '20m.csv')."""
    candidate = input_dir / f"{altitude_m}m.csv"
    return candidate if candidate.exists() else None


def build_io(
    input_dir: Path, output_dir: Path
) -> IOConfig:
    return IOConfig(
        input_dir=input_dir,
        out_filter1=output_dir / "Filter_1",
        out_filter2_pcell=output_dir / "Filter_2" / "pcell",
        out_filter2_detected=output_dir / "Filter_2" / "detected",
        out_filter3=output_dir / "Filter_3",
        out_filter4=output_dir / "Filter_4",
    )


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    bs: BSConfig,
    altitudes: Optional[Sequence[int]],
    remove_80: bool,
    pci_filter: Optional[Sequence[int]],
) -> None:
    io = build_io(input_dir, output_dir)
    io.out_filter1.mkdir(parents=True, exist_ok=True)
    io.out_filter2_pcell.mkdir(parents=True, exist_ok=True)
    io.out_filter2_detected.mkdir(parents=True, exist_ok=True)
    io.out_filter3.mkdir(parents=True, exist_ok=True)
    io.out_filter4.mkdir(parents=True, exist_ok=True)

    alt_list = parse_altitudes(altitudes, remove_80)

    # Filter_1: geometry features per altitude file
    filter1_outputs: List[Path] = []
    for alt in alt_list:
        in_file = discover_input_file(io.input_dir, alt)
        if not in_file:
            continue
        out_file = io.out_filter1 / f"{alt}m.csv"
        step_filter1_process_altitude(in_file, out_file, bs, altitude_m=alt)
        filter1_outputs.append(out_file)

    # Filter_2: split pcell and detected (normalized)
    pcell_files: List[Path] = []
    detected_files: List[Path] = []
    for f in filter1_outputs:
        p_out = io.out_filter2_pcell / f"pcell_{f.stem}.csv"
        d_out = io.out_filter2_detected / f"detcell_{f.stem}.csv"
        step_filter2_split_normalize(f, p_out, d_out)
        pcell_files.append(p_out)
        detected_files.append(d_out)

    # Filter_3: optional PCI filtering from normalized files
    filter3_files: List[Path] = []
    if pci_filter:
        filter3_files += step_filter3_by_pci(pcell_files, io.out_filter3, pci_filter)
        filter3_files += step_filter3_by_pci(detected_files, io.out_filter3, pci_filter)

    # Filter_4: combine and deduplicate
    sources_for_filter4 = filter3_files if pci_filter else (pcell_files + detected_files)
    out_all = io.out_filter4 / "all.csv"
    out_all_v2 = io.out_filter4 / "all_v2.csv"
    step_filter4_combine_and_dedup(sources_for_filter4, out_all, out_all_v2)


# ----------------------------- CLI -----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Preprocess UAV LTE measurement CSVs across multiple altitudes:\n"
            "  1) compute geometry (2D distance, az/el if available)\n"
            "  2) normalize pcell + detected (wide→long)\n"
            "  3) optional PCI filtering\n"
            "  4) combine & deduplicate"
        )
    )
    p.add_argument("--input-dir", type=Path, required=True, help="Directory with raw '<alt>m.csv' files (e.g., ../Data)")
    p.add_argument("--output-dir", type=Path, required=True, help="Directory to write Filter_1..Filter_4 outputs")
    p.add_argument("--bs-lat", type=float, required=True, help="Base station latitude (observer)")
    p.add_argument("--bs-lon", type=float, required=True, help="Base station longitude (observer)")
    p.add_argument("--bs-height", type=float, default=30.0, help="Base station height in meters (default: 30)")
    p.
