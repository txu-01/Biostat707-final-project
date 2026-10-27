"""
Download and check data for BIOSTAT 707 Final Project.
Cross-platform Python implementation of UCI WDBC dataset download and validation.
"""

import hashlib
import json
from datetime import datetime, timezone
import os
import urllib.request
import zipfile

UCI_URL = "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip"

def download_and_check(dest_dir="data/raw", use_existing=False):
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "wdbc.zip")
    data_path = os.path.join(dest_dir, "wdbc.data")
    names_path = os.path.join(dest_dir, "wdbc.names")
    audit_path = os.path.join(dest_dir, "data-audit.json")

    # Download archive if needed
    if not use_existing or not os.path.exists(data_path):
        if not os.path.exists(zip_path):
            print(f"Downloading dataset from {UCI_URL}...")
            urllib.request.urlretrieve(UCI_URL, zip_path)
            print("Download complete.")
        
        # Extract
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dest_dir)
            print("Extracted archive.")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing data file: {data_path}")

    # Compute SHA256
    sha256 = hashlib.sha256()
    with open(data_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    file_hash = sha256.hexdigest().upper()

    # Read and audit records
    with open(data_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    records = [line.split(",") for line in lines]
    if len(records) == 0:
        raise ValueError("Data file is empty.")

    labels = {}
    unique_ids = set()
    missing_cells = 0
    invalid_numeric_cells = 0
    col_counts = set()

    for rec in records:
        col_counts.add(len(rec))
        if len(rec) != 32:
            raise ValueError(f"Unexpected column count: {len(rec)} (expected 32)")
        
        rec_id = rec[0]
        unique_ids.add(rec_id)

        label = rec[1]
        if label not in ("M", "B"):
            raise ValueError(f"Unexpected label: {label}")
        labels[label] = labels.get(label, 0) + 1

        for val in rec[2:]:
            val_str = val.strip()
            if not val_str or val_str == "?":
                missing_cells += 1
            try:
                num = float(val_str)
                if num != num or abs(num) == float("inf"):
                    invalid_numeric_cells += 1
            except ValueError:
                invalid_numeric_cells += 1

    duplicate_rows = len(lines) - len(set(lines))

    audit = {
        "source": UCI_URL,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": file_hash,
        "rows": len(records),
        "column_counts": sorted(list(col_counts)),
        "labels": [{"label": k, "count": v} for k, v in sorted(labels.items())],
        "unique_ids": len(unique_ids),
        "duplicate_rows": duplicate_rows,
        "missing_cells": missing_cells,
        "invalid_numeric_cells": invalid_numeric_cells
    }

    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    print("Data audit summary:")
    print(json.dumps(audit, indent=2))

    if missing_cells > 0 or invalid_numeric_cells > 0 or duplicate_rows > 0:
        raise ValueError("Data audit failed checks!")

    print("\n[SUCCESS] Raw data verified successfully.")
    return audit

if __name__ == "__main__":
    download_and_check(use_existing=True)
