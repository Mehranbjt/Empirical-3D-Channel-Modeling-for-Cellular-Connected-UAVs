# preprocess_lte_measurements.py
# Purpose: Load raw LTE measurement CSV data, clean & normalize columns, optionally filter by PCI,
#          and (optionally) add a binned signal-strength label for ML training. Saves a clean CSV.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd


COL_TIME = "Time"
COL_RSRP = "RSRP (LTE pcell)"
COL_RSRQ = "RSRQ (LTE pcell)"
COL_PCI = "Physical cell identity (LTE pcell)"

# Normalized column names for ML pipelines
RENAME_MAP = {
    COL_TIME: "time",
    COL_RSRP: "rsrp",
    COL_RSRQ: "rsrq",
    COL_PCI: "pci",
}

# Default bins for RSRP (dBm). 
# Example interpretation:
#   rsrp <= -100: "very_poor"
#   -100 < rsrp <= -90: "poor"
#   -90  < rsrp <= -80: "fair"
#   -80  < rsrp <= -70: "good"
#   rsrp > -70: "excellent"
DEFAULT_RSRP_BINS = [-1_000, -100, -90, -80, -70, 1_000]
DEFAULT_RSRP_LABELS = ["very_poor", "poor", "fair", "good", "excellent"]


def load_data(
    csv_path: Path,
    usecols: Optional[Sequence[str]] = None,
    parse_dates: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Load the raw CSV into a DataFrame.

    Parameters
    ----------
    csv_path : Path
        Path to the input CSV file.
    usecols : Sequence[str] | None
        Subset of columns to load. If None, loads all columns.
    parse_dates : Iterable[str] | None
        Columns to parse as datetime.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    """
    df = pd.read_csv(
        csv_path,
        usecols=usecols,
        parse_dates=list(parse_dates) if parse_dates else None,
        low_memory=False,
    )
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize/rename column names to snake_case for downstream ML pipelines.

    Returns
    -------
    pd.DataFrame
    """
    cols_present = {c: c for c in df.columns}
    renamer = {k: v for k, v in RENAME_MAP.items() if k in cols_present}
    df = df.rename(columns=renamer)
    return df


def basic_clean(
    df: pd.DataFrame,
    sort_by_time: bool = True,
    dropna_subset: Optional[Sequence[str]] = ("rsrp", "rsrq"),
) -> pd.DataFrame:
    """
    Basic cleaning: drop NA in essential columns and sort by time if present.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with normalized column names.
    sort_by_time : bool
        If True and 'time' column exists, sort ascending by time.
    dropna_subset : Sequence[str] | None
        Columns that must be non-null.

    Returns
    -------
    pd.DataFrame
    """
    if dropna_subset:
        cols = [c for c in dropna_subset if c in df.columns]
        if cols:
            df = df.dropna(subset=cols)

    if sort_by_time and "time" in df.columns:
        df = df.sort_values(by="time", ascending=True)

    df = df.reset_index(drop=True)
    return df


def filter_by_pci(df: pd.DataFrame, pci_allowlist: Optional[Iterable[int]] = None) -> pd.DataFrame:
    """
    Optionally filter rows by allowed PCI values.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame.
    pci_allowlist : Iterable[int] | None
        If provided, keep only rows where 'pci' is in this set.

    Returns
    -------
    pd.DataFrame
    """
    if pci_allowlist is None:
        return df
    if "pci" not in df.columns:
        return df

    allow = set(int(p) for p in pci_allowlist)
    return df[df["pci"].isin(allow)].reset_index(drop=True)


def add_rsrp_strength_label(
    df: pd.DataFrame,
    rsrp_col: str = "rsrp",
    bins: Sequence[float] = DEFAULT_RSRP_BINS,
    labels: Sequence[str] = DEFAULT_RSRP_LABELS,
    out_col: str = "signal_strength",
) -> pd.DataFrame:
    """
    Add a categorical signal strength label based on binned RSRP values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame expected to contain the RSRP column.
    rsrp_col : str
        Column name for RSRP in dBm.
    bins : Sequence[float]
        Bin edges for pd.cut (must be monotonically increasing).
    labels : Sequence[str]
        Labels corresponding to the bins (len(labels) == len(bins) - 1).
    out_col : str
        Name of the output categorical label column.

    Returns
    -------
    pd.DataFrame
    """
    if rsrp_col not in df.columns:
        return df

    df[out_col] = pd.cut(df[rsrp_col], bins=bins, labels=labels, include_lowest=True)
    return df


def save_data(df: pd.DataFrame, out_path: Path) -> None:
    """
    Save the cleaned/preprocessed DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Data to save.
    out_path : Path
        Output CSV path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def preprocess(
    input_csv: Path,
    output_csv: Path,
    pci_allowlist: Optional[Iterable[int]] = None,
    label_rsrp: bool = True,
    rsrp_bins: Sequence[float] = DEFAULT_RSRP_BINS,
    rsrp_labels: Sequence[str] = DEFAULT_RSRP_LABELS,
    required_columns: Sequence[str] = (COL_TIME, COL_RSRP, COL_RSRQ, COL_PCI),
) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
      1) Load selected columns and parse time.
      2) Normalize column names.
      3) Drop NA in essential columns, sort by time.
      4) Optional filter by PCI.
      5) Optional add binned signal-strength label.
      6) Save to CSV.

    Returns
    -------
    pd.DataFrame
        The final cleaned DataFrame (also written to disk).
    """
    df = load_data(
        input_csv,
        usecols=required_columns,
        parse_dates=[COL_TIME],
    )
    df = normalize_columns(df)
    df = basic_clean(df, sort_by_time=True, dropna_subset=("rsrp", "rsrq"))
    df = filter_by_pci(df, pci_allowlist=pci_allowlist)

    if label_rsrp:
        df = add_rsrp_strength_label(
            df,
            rsrp_col="rsrp",
            bins=rsrp_bins,
            labels=rsrp_labels,
            out_col="signal_strength",
        )

    save_data(df, output_csv)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess LTE measurement CSV for ML training:\n"
            "- cleans & normalizes columns\n"
            "- optional PCI filtering\n"
            "- optional binned signal strength label\n"
            "- saves clean CSV"
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to raw input CSV (e.g., ../Data/20m.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save cleaned CSV (e.g., ./data/clean/20m_clean.csv)",
    )
    parser.add_argument(
        "--pci",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of allowed PCI values to keep (e.g., --pci 173 110).",
    )
    parser.add_argument(
        "--no-label",
        action="store_true",
        help="Disable adding the binned signal strength label.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess(
        input_csv=args.input,
        output_csv=args.output,
        pci_allowlist=args.pci,
        label_rsrp=not args.no_label,
    )


if __name__ == "__main__":
    main()
