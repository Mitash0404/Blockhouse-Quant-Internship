# Blockhouse-Quant-Internship

**Purpose:** Build order-flow-imbalance (OFI) features from raw level-III order-book data.

**What it does:**

- Reads a time-sorted CSV of book events (one row per event, multi-level quotes).

- Calculates per-bucket OFI signals: best-level, depth 0-9, integrated (PCA) and cross-asset.

- Writes a tidy feature table (Parquet or CSV) ready for modelling or back-tests.

**Run from terminal:**

`python ofi_features.py first_25000_rows.csv --out ofi_features.parquet`

**Output columns** (In the ofi_features_csv.csv file):

| Name                   | Description                          |
|------------------------|--------------------------------------|
| timestamp, symbol      | bucket midpoint + ticker             |
| ofi_lvl_0 … ofi_lvl_9  | raw depth-wise OFI                   |
| best_ofi               | alias of ofi_lvl_0                   |
| integrated_ofi         | PCA-weighted aggregate               |
| cross_asset_ofi        | integrated OFI from all other symbols|







