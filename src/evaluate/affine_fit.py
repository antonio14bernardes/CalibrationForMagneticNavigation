import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from tqdm import tqdm

def get_affine_fits(df: pd.DataFrame, resolution: float = 0.0016666) -> Tuple[np.ndarray, Dict]:
    df = df.copy()

    # --- pos_key ---
    pos_bins = (df[["x", "y", "z"]] / resolution).round().astype(int)
    df["pos_key"] = pos_bins.astype(str).agg(",".join, axis=1)

    # --- active coil (at most one nonzero em_ expected) ---
    em_cols = [c for c in df.columns if c.startswith("em_")]
    candidate = df[em_cols].abs().idxmax(axis=1)
    df["active_coil"] = candidate.where(df[em_cols].ne(0).any(axis=1))

    # Expand rows with no active coil (all em_ == 0) for fitting only
    has_coil = df["active_coil"].notna()
    data_with = df[has_coil].copy()
    data_without = df[~has_coil].copy()

    repeated = pd.concat(
        [data_without.assign(active_coil=coil) for coil in em_cols],
        ignore_index=True,
    )
    data_expanded = pd.concat([data_with, repeated], ignore_index=True)

    data_expanded["pos_coil_key"] = data_expanded["pos_key"] + "_" + data_expanded["active_coil"]

    # --- fit per (pos, coil) ---
    fits: Dict[str, Dict[str, Any]] = {}

    grouped = data_expanded.groupby("pos_coil_key") 
    for pos_coil_id, g in tqdm(grouped, total=len(grouped), desc="Computing affine fits"):
        coil = g["active_coil"].iloc[0]          # e.g. "em_3"
        x = g[coil].to_numpy(float)              # (N,)
        Y = g[["Bx", "By", "Bz"]].to_numpy(float)  # (N,3)

        A = np.c_[x, np.ones_like(x)]            # (N,2)
        beta, *_ = np.linalg.lstsq(A, Y, rcond=None)  # (2,3)

        slope = beta[0]                          # (3,)
        intercept = beta[1]                      # (3,)
        Yhat = A @ beta                          # (N,3)

        err = Y - Yhat
        rse = np.linalg.norm(err, axis=1)        # (N,)
        rmse = np.sqrt(np.mean(rse**2))          # scalar

        sse = np.sum(err**2)
        sst = np.sum((Y - Y.mean(axis=0))**2)
        r2 = 1 - sse / sst if sst > 0 else np.nan

        fits[pos_coil_id] = {
            "fit": {"slope": slope, "intercept": intercept, "rse": rse, "rmse": rmse, "r2": r2},
        }

   # --- assemble predictions for original df rows ---
    preds_array = np.full((len(df), 3), np.nan, dtype=float)

    idx_to_pos = {idx: i for i, idx in enumerate(df.index)}

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Assembling predictions"):
        i = idx_to_pos[idx]
        coil = row["active_coil"]

        # Case 1: active coil exists -> normal prediction
        if pd.notna(coil):
            pos_coil_id = f"{row['pos_key']}_{coil}"
            fit_info = fits.get(pos_coil_id, None)
            if fit_info is None:
                continue

            x = float(row[coil])
            slope = fit_info["fit"]["slope"]
            intercept = fit_info["fit"]["intercept"]
            preds_array[i] = slope * x + intercept
            continue

        # Case 2: no active coil (all em_* == 0) -> use intercept baseline at this position
        intercepts = []
        for c in em_cols:
            pos_coil_id = f"{row['pos_key']}_{c}"
            fit_info = fits.get(pos_coil_id, None)
            if fit_info is not None:
                intercepts.append(fit_info["fit"]["intercept"])

        if intercepts:
            preds_array[i] = np.mean(intercepts, axis=0)
        # else: remain NaN if no fits exist for that position at all

    return preds_array, fits