import sys, os
import pandas as pd
import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

def convert_frames(data_raw: pd.DataFrame) -> pd.DataFrame:
    df = data_raw.copy()

    required = [
        "sensor_id","x","y","z","Bx","By","Bz",
        "cube_x","cube_y","cube_z",
        "qw","qx","qy","qz"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"convert_frames: missing required columns: {missing}")

    # Columns that we consume to build WORLD-frame quantities
    consumed = set(["sensor_id", "x","y","z","Bx","By","Bz",
                    "cube_x","cube_y","cube_z",
                    "qw","qx","qy","qz"])
    passthrough_cols = [c for c in df.columns if c not in consumed]

    # --- Pull out arrays ---
    p_cube = df[["x","y","z"]].to_numpy(float)       # (N, 3)
    B_cube = df[["Bx","By","Bz"]].to_numpy(float)    # (N, 3)
    C_world = df[["cube_x","cube_y","cube_z"]].to_numpy(float)  # (N, 3)

    q = df[["qw","qx","qy","qz"]].to_numpy(float)    # (N, 4)
    w, x, y, z = q.T                                 # each (N,)

    # --- Build rotation matrices R(q) for each row, shape (N, 3, 3) ---
    R = np.empty((len(df), 3, 3), dtype=float)

    # Standard quaternion -> rotation matrix (w, x, y, z)
    R[:, 0, 0] = 1 - 2*(y*y + z*z)
    R[:, 0, 1] = 2*(x*y - z*w)
    R[:, 0, 2] = 2*(x*z + y*w)

    R[:, 1, 0] = 2*(x*y + z*w)
    R[:, 1, 1] = 1 - 2*(x*x + z*z)
    R[:, 1, 2] = 2*(y*z - x*w)

    R[:, 2, 0] = 2*(x*z - y*w)
    R[:, 2, 1] = 2*(y*z + x*w)
    R[:, 2, 2] = 1 - 2*(x*x + y*y)

    # --- Apply rotation & translation ---
    # Positions: p_world = C_world + R @ p_cube
    p_world = np.einsum("nij,nj->ni", R, p_cube) + C_world      # (N, 3)

    # Fields: B_world = R @ B_cube (no translation)
    B_world = np.einsum("nij,nj->ni", R, B_cube)                # (N, 3)

    # --- Build output DataFrame ---
    base = pd.DataFrame({
        "sensor_id": df["sensor_id"].values,
        "x":  p_world[:, 0],
        "y":  p_world[:, 1],
        "z":  p_world[:, 2],
        "Bx": B_world[:, 0],
        "By": B_world[:, 1],
        "Bz": B_world[:, 2],
    })

    out = pd.concat(
        [base.reset_index(drop=True),
         df.loc[:, passthrough_cols].reset_index(drop=True)],
        axis=1
    )

    return out