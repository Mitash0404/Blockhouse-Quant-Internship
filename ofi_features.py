from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# 1. Helper: one event’s OFI contribution at a given level
def _ofi_event(delta_px, prev_sz, new_sz, side):
    """
    Parameters
    ----------
    delta_px : bid(ask) price_t − price_{t-1}
    prev_sz  : size at t-1
    new_sz   : size at t
    side     : "B" for bid, "A" for ask
    """
    if side == "B":  # bid side
        if delta_px > 0:
            return new_sz
        if delta_px < 0:
            return -prev_sz
        return np.sign(new_sz - prev_sz) * abs(new_sz - prev_sz)
    else:  # ask side (mirror logic)
        if delta_px < 0:
            return new_sz
        if delta_px > 0:
            return -prev_sz
        return -np.sign(new_sz - prev_sz) * abs(new_sz - prev_sz)

# 2. Main builder
def build_ofi_features(
    csv_path: str | Path,
    out_path: str | Path = "ofi_features.parquet",
    bucket_ms: int = 1000,
    max_depth: int = 10,
) -> None:
    # 2-a. Load & pre-format
    df = pd.read_csv(csv_path)
    if "ts_event" not in df.columns:
        raise ValueError("Expected a 'ts_event' column with nanosecond timestamps.")
    df["ts_event"] = pd.to_datetime(df["ts_event"])
    sym_col = "symbol" if "symbol" in df.columns else "instrument_id"
    df.sort_values([sym_col, "ts_event"], inplace=True)

    # 2-b. Compute OFI for each symbol
    all_feat = []
    for sym, g in df.groupby(sym_col, sort=False):
        g = g.set_index("ts_event")

        level_cols = {
            m: {
                "bid_px": g[f"bid_px_{m:02d}"],
                "ask_px": g[f"ask_px_{m:02d}"],
                "bid_sz": g[f"bid_sz_{m:02d}"],
                "ask_sz": g[f"ask_sz_{m:02d}"],
            }
            for m in range(max_depth)
        }

        # build event-level OFI per depth
        feat_evt = pd.DataFrame(index=g.index)
        for m in range(max_depth):
            bp, ap = level_cols[m]["bid_px"], level_cols[m]["ask_px"]
            bs, ass = level_cols[m]["bid_sz"], level_cols[m]["ask_sz"]

            # price / size deltas (prev -> curr)
            dbp, dap = bp.diff(), ap.diff()
            bs_prev, as_prev = bs.shift(), ass.shift()

            # first row: no previous quote
            dbp.iloc[0] = dap.iloc[0] = 0.0
            bs_prev.iloc[0] = bs.iloc[0]
            as_prev.iloc[0] = ass.iloc[0]

            # vectorised OFI calculation
            ofi_bid = _vector_ofi(dbp.values, bs_prev.values, bs.values, side="B")
            ofi_ask = _vector_ofi(dap.values, as_prev.values, ass.values, side="A")
            feat_evt[f"ofi_lvl_{m}"] = ofi_bid + ofi_ask

        # bucket into fixed-width windows (e.g. 1 s) and sum OFI inside each bucket
        feat_evt["bucket"] = feat_evt.index.floor(f"{bucket_ms}ms")
        feat = (
            feat_evt.groupby("bucket").sum(numeric_only=True).reset_index().rename(columns={"bucket": "timestamp"})
        )
        feat[sym_col] = sym
        all_feat.append(feat)

    X = pd.concat(all_feat, ignore_index=True)

    # 2-c. Integrated OFI (PCA across levels 0-9)
    pca = PCA(n_components=1, random_state=0)
    level_cols = [f"ofi_lvl_{m}" for m in range(max_depth)]
    X["integrated_ofi"] = pca.fit_transform(X[level_cols])[:, 0]

    # 2-d. Best-level alias and cross-asset aggregation
    X["best_ofi"] = X["ofi_lvl_0"]
    X["cross_asset_ofi"] = (
        X.groupby("timestamp")["integrated_ofi"].transform("sum") - X["integrated_ofi"]
    )

    # 2-e. Save
    X.to_parquet(out_path, index=False)
    print(f"✓ Features written → {out_path}")

# Vectorised OFI helper for speed
def _vector_ofi(delta_px, prev_sz, new_sz, side):
    out = np.zeros_like(delta_px, dtype=float)

    if side == "B":
        # price↑ → +new_sz
        mask = delta_px > 0
        out[mask] = new_sz[mask]
        # price↓ → –prev_sz
        mask = delta_px < 0
        out[mask] = -prev_sz[mask]
        # same price → ±Δsize
        mask = delta_px == 0
        out[mask] = np.sign(new_sz[mask] - prev_sz[mask]) * np.abs(new_sz[mask] - prev_sz[mask])
    else:  # ask side (mirror)
        mask = delta_px < 0
        out[mask] = new_sz[mask]
        mask = delta_px > 0
        out[mask] = -prev_sz[mask]
        mask = delta_px == 0
        out[mask] = -np.sign(new_sz[mask] - prev_sz[mask]) * np.abs(new_sz[mask] - prev_sz[mask])

    return out

# CLI convenience
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct OFI-based features from a raw LOB CSV.")
    parser.add_argument("csv_path", nargs="?", default="first_25000_rows.csv", help="Input CSV (default = first_25000_rows.csv in CWD)")
    parser.add_argument("--out", dest="out_path", default="ofi_features.parquet", help="Output Parquet file")
    parser.add_argument("--bucket_ms", type=int, default=1000, help="Window size in milliseconds (default 1000)")
    parser.add_argument("--max_depth", type=int, default=10, help="Depth levels to include (default 10 = 0-9)")

    args, _ = parser.parse_known_args()

    build_ofi_features(args.csv_path, args.out_path,
                       args.bucket_ms, args.max_depth)


pd.read_parquet("ofi_features.parquet").to_csv("ofi_features_csv.csv", index=False)
