import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Optional, Tuple
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.tri as mtri
import warnings

def _find_col(df: pd.DataFrame, cands):
    lower_map = {c.lower(): c for c in df.columns}
    for c in cands:
        if c in lower_map:
            return lower_map[c]
    for col in df.columns:
        if re.fullmatch(cands[0], col, flags=re.IGNORECASE):
            return col
    return None

def _voxel_downsample(X: np.ndarray, max_points: int = 100_000) -> np.ndarray:
    """
    Grid/voxel downsample: pick one point per occupied voxel.
    Grid size is chosen so we end up with <= max_points (approx).
    Returns indices into X of selected points.
    """
    n = len(X)
    if n <= max_points:
        return np.arange(n)

    # Compute grid size from volume^(1/3) scaled to target occupancy
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    span = np.maximum(maxs - mins, 1e-12)
    # choose bins per axis so total bins ~ max_points
    bins_total = max_points
    # proportional bins by span so voxels roughly cubic
    ratios = span / (span.prod() ** (1/3))
    bins = np.maximum((ratios * (bins_total ** (1/3))).astype(int), 1)
    # ensure at least 1 bin per axis
    bx, by, bz = bins

    # Compute voxel indices
    # Normalize to [0,1), multiply by bins and floor
    norm = (X - mins) / span
    ijk = np.column_stack([
        np.minimum((norm[:, 0] * bx).astype(int), bx - 1),
        np.minimum((norm[:, 1] * by).astype(int), by - 1),
        np.minimum((norm[:, 2] * bz).astype(int), bz - 1),
    ])

    # Hash voxel to a single int and keep the first point per voxel
    # This strongly reduces points; if still too many, random sample cap
    keys = (ijk[:, 0] * (by * bz) + ijk[:, 1] * bz + ijk[:, 2]).astype(np.int64)
    # first occurrence index for each key
    order = np.argsort(keys, kind="mergesort")  # stable
    sorted_keys = keys[order]
    mask = np.empty_like(sorted_keys, dtype=bool)
    mask[0] = True
    mask[1:] = sorted_keys[1:] != sorted_keys[:-1]
    selected = order[mask]

    if selected.size > max_points:
        # uniformly sample if still above target
        sel = np.random.default_rng(0).choice(selected, size=max_points, replace=False)
        return np.sort(sel)
    return np.sort(selected)



def plot_positions(df: pd.DataFrame,
                          point_size: Optional[float] = None,
                          max_points: int = 200_000,
                          downsample: bool = True,
                          title: Optional[str] = None,
                          column_colored: Optional[str] = None,
                          max_unique_colored: int = 5,
                          x: Optional[str] = "x",
                          y: Optional[str] = "y",
                          z: Optional[str] = "z") -> go.Figure:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if max_unique_colored < 1:
        raise ValueError("max_unique_colored must be >= 1")

    # If user supplies x/y/z (not None), use them directly (no _find_col).
    # Only auto-detect if they explicitly pass None.
    if x is None:
        x_col = _find_col(df, ["x", "pos_x", "px"])
    else:
        x_col = x

    if y is None:
        y_col = _find_col(df, ["y", "pos_y", "py"])
    else:
        y_col = y

    if z is None:
        z_col = _find_col(df, ["z", "pos_z", "pz"])
    else:
        z_col = z

    # Validate resolved columns exist
    missing = [c for c in [x_col, y_col, z_col] if c is None or c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing/invalid position column(s): {missing}. "
            f"Provide valid x/y/z names, or pass None to auto-detect."
        )

    # Build base data, optionally include colored column
    cols = [x_col, y_col, z_col]
    if column_colored is not None:
        if column_colored not in df.columns:
            raise ValueError(f"column_colored='{column_colored}' not found in DataFrame columns.")
        cols.append(column_colored)

    data = df[cols].dropna()
    X = data[[x_col, y_col, z_col]].to_numpy(dtype=float, copy=False)

    color_vals = None
    if column_colored is not None:
        color_vals = data[column_colored].to_numpy(copy=False)

    if downsample and len(X) > max_points:
        idx = _voxel_downsample(X, max_points=max_points)
        X = X[idx, :]
        if color_vals is not None:
            color_vals = color_vals[idx]

    n = len(X)
    if point_size is None:
        point_size = 4.0 if n <= 3_000 else (2.5 if n <= 30_000 else 1.5)

    hoverinfo = "skip" if n > 150_000 else "x+y+z"

    if column_colored is None:
        fig = go.Figure(
            data=go.Scatter3d(
                x=X[:, 0], y=X[:, 1], z=X[:, 2],
                mode="markers",
                marker=dict(size=float(point_size), opacity=0.85),
                hoverinfo=hoverinfo,
            )
        )
        fig.update_layout(
            scene=dict(xaxis_title=x_col, yaxis_title=y_col, zaxis_title=z_col, aspectmode="data"),
            margin=dict(l=0, r=0, t=30, b=0),
            title=title if title else "",
            showlegend=False,
        )
        return fig

    uniques = pd.unique(pd.Series(color_vals))
    n_unique = len(uniques)
    if n_unique > max_unique_colored:
        raise ValueError(
            f"column_colored='{column_colored}' has {n_unique} unique values, "
            f"which exceeds max_unique_colored={max_unique_colored}."
        )

    fig = go.Figure()
    for u in uniques:
        mask = (color_vals == u)
        Xu = X[mask]
        fig.add_trace(
            go.Scatter3d(
                x=Xu[:, 0], y=Xu[:, 1], z=Xu[:, 2],
                mode="markers",
                name=str(u),
                marker=dict(size=float(point_size), opacity=0.85),
                hoverinfo=hoverinfo,
            )
        )

    fig.update_layout(
        scene=dict(xaxis_title=x_col, yaxis_title=y_col, zaxis_title=z_col, aspectmode="data"),
        margin=dict(l=0, r=0, t=30, b=0),
        title=title if title else "",
        showlegend=True,
    )
    return fig


def plot_quiver_slice(
    df,
    currents_vec,
    normal_to="z",          # "x", "y", or "z"
    normal_coordinate=0.0,  # e.g. z = -0.02; if None -> project all points
    normal_margin=0.002,    # ± window around that coord (ignored if normal_coordinate is None)
    current_tol=1e-3,       # tolerance on currents
    scale=1,                # scale B -> arrow length (mm/mT)
    Nmax=50000,             # downsample for plotting
    flag_column=None,       # column name with boolean flags
    flag_color="red",       # color for flagged vectors
):
    """
    Plot a magnetic field quiver on a plane normal to a chosen axis.
    Optionally color flagged vectors differently.

    If `normal_coordinate` is None, *all* points are used and projected
    onto that plane (no slab selection along the normal axis).

    Args
    ----
    df : DataFrame
        Must contain columns: 'x', 'y', 'z', 'Bx', 'By', 'Bz', and em_*.
    currents_vec : dict
        Mapping {em_col_name: target_current}.
    normal_to : {"x", "y", "z"}, default "z"
        Axis normal to the plane you want (e.g. "z" → x-y plane).
    normal_coordinate : float or None
        Coordinate along the normal axis (e.g. z0). If None, no slicing is
        done along that axis and all rows are projected.
    normal_margin : float
        Half-width of the slab: |coord - normal_coordinate| ≤ normal_margin.
        Ignored if normal_coordinate is None.
    current_tol : float, optional
        Allowed deviation in currents.
    scale : float
        Factor to convert B to arrow length. Unit: mm/mT
    Nmax : int
        Max number of points to plot (downsampling).
    flag_column : str, optional
        Column name in df with boolean or {0,1} values.
        If provided, flagged vectors are drawn in flag_color.
    flag_color : color spec, default "red"
        Matplotlib color for flagged vectors.
    """

    # --- Decide which spatial axes and field components to plot ---
    axis_map = {
        "z": {"norm": "z", "coords": ("x", "y"), "fields": ("Bx", "By")},
        "x": {"norm": "x", "coords": ("y", "z"), "fields": ("By", "Bz")},
        "y": {"norm": "y", "coords": ("x", "z"), "fields": ("Bx", "Bz")},
    }

    if normal_to not in axis_map:
        raise ValueError("normal_to must be one of 'x', 'y', or 'z'.")

    cfg = axis_map[normal_to]
    norm_col = cfg["norm"]
    coord1, coord2 = cfg["coords"]
    B1_col, B2_col = cfg["fields"]

    # --- Normal-axis mask ---
    coord_vals = df[norm_col].to_numpy()
    if normal_coordinate is None:
        # Use all finite points along the normal axis (projection)
        mask_norm = np.isfinite(coord_vals)
        plane_desc = f"projection of all points onto {coord1}-{coord2} plane"
    else:
        # Standard slab selection: |coord - normal_coordinate| <= normal_margin
        mask_norm = np.isfinite(coord_vals) & (
            np.abs(coord_vals - normal_coordinate) <= normal_margin
        )
        plane_desc = (
            f"{normal_to} = {normal_coordinate:+.3f} ± {normal_margin}"
        )

    # --- Current mask ---
    em_cols = list(currents_vec.keys())
    if any(c not in df.columns for c in em_cols):
        missing = [c for c in em_cols if c not in df.columns]
        raise KeyError(f"Missing current columns: {missing}")

    diffs = df[em_cols].sub(pd.Series(currents_vec))
    if current_tol is None:
        mask_curr = (diffs == 0).all(axis=1)
    else:
        mask_curr = diffs.abs().le(current_tol).all(axis=1)

    # --- Combined mask ---
    mask = mask_norm & mask_curr

    # --- Extract plane slice / projection ---
    cols_needed = [coord1, coord2, B1_col, B2_col]
    if flag_column is not None and flag_column in df.columns:
        cols_needed.append(flag_column)
    R = df.loc[mask, cols_needed].dropna(
        subset=[coord1, coord2, B1_col, B2_col]
    ).copy()

    # Optional downsampling
    if len(R) > Nmax:
        step = max(1, len(R) // Nmax)
        R = R.iloc[::step, :]

    # --- Prepare arrays for plotting ---
    x = R[coord1].to_numpy()
    y = R[coord2].to_numpy()
    B1 = R[B1_col].to_numpy()
    B2 = R[B2_col].to_numpy()
    Ux = B1 * scale / 1000
    Uy = B2 * scale / 1000

    # --- Flag handling ---
    if flag_column is not None and flag_column in R.columns:
        flagged = R[flag_column].astype(bool).to_numpy()
    else:
        flagged = np.zeros(len(R), dtype=bool)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7, 7))

    # Common quiver style for both flagged and non-flagged
    quiv_kwargs = dict(
        angles="xy",
        scale_units="xy",
        scale=1,
        pivot="mid",
        alpha=0.8,
    )

    nonflag_mask = ~flagged

    # Non-flagged vectors (default color)
    ax.quiver(
        x[nonflag_mask],
        y[nonflag_mask],
        Ux[nonflag_mask],
        Uy[nonflag_mask],
        color="black",
        **quiv_kwargs,
    )

    # Flagged vectors (same style, different color)
    if flagged.any():
        ax.quiver(
            x[flagged],
            y[flagged],
            Ux[flagged],
            Uy[flagged],
            color=flag_color,
            **quiv_kwargs,
        )

    # Scatter background
    ax.scatter(x, y, s=1, alpha=0.2, c="gray")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{coord1} [mm] (→ right)")
    ax.set_ylabel(f"{coord2} [mm] (→ up)")
    ax.grid(True, alpha=0.2)

    # Title
    downsampled_str = f", downsampled to {len(R):,} pts" if len(R) < mask.sum() else ""
    ax.set_title(
        f"Magnetic field in plane normal to {normal_to.upper()} "
        f"({plane_desc})\n"
        f"Field vector scale: {scale} mm/mT{downsampled_str}"
    )

    plt.tight_layout()
    plt.show()

    return R, mask_norm, mask_curr

def plot_quiver_3d(
    df: pd.DataFrame,
    currents_vec,
    current_tol: float = 1e-3,   # tolerance on currents
    scale: float = 1.0,          # scale B -> arrow length (mm/mT)
    Nmax: int = 50_000,          # max number of vectors after downsampling
    arrow_linewidth: float = 2.0, # thickness of the arrows (line width)
    color: str = "#1f77b4",      # line color for arrows
) -> go.Figure:
    """
    Interactive 3D magnetic field 'quiver' using Plotly line segments
    (no cones).

    Args
    ----
    df : DataFrame
        Must contain columns: 'x', 'y', 'z', 'Bx', 'By', 'Bz', and the em_* columns.
    currents_vec : dict
        Mapping {em_col_name: target_current}.
    current_tol : float, optional
        Allowed deviation in currents.
    scale : float
        Factor to convert B to arrow length. Unit: mm/mT
        (we internally do U = B * scale / 1000).
    Nmax : int
        Target max number of vectors after downsampling (via voxel grid).
    arrow_linewidth : float
        Line width for the quiver arrows.
    color : str
        Color of the arrow lines.

    Returns
    -------
    plotly.graph_objects.Figure
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    # --- Check required columns ---
    cols_needed = ["x", "y", "z", "Bx", "By", "Bz"]
    if any(c not in df.columns for c in cols_needed):
        raise KeyError("Expected columns 'x','y','z','Bx','By','Bz' in df.")

    # --- Build current mask (vectorized, same idea as in 2D) ---
    em_cols = list(currents_vec.keys())
    if any(c not in df.columns for c in em_cols):
        missing = [c for c in em_cols if c not in df.columns]
        raise KeyError(f"Missing current columns: {missing}")

    diffs = df[em_cols].sub(pd.Series(currents_vec))
    if current_tol is None:
        mask_curr = (diffs == 0).all(axis=1)
    else:
        mask_curr = diffs.abs().le(current_tol).all(axis=1)

    # --- Extract only what we need and ensure numeric ---
    work = (
        df.loc[mask_curr, cols_needed]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
        .copy()
    )

    if work.empty:
        raise ValueError("No valid rows to plot after current filtering and NaN removal.")

    # --- Downsample via voxel selection if too many vectors ---
    coords = work[["x", "y", "z"]].to_numpy()
    if len(work) > Nmax:
        idx = _voxel_downsample(coords, max_points=Nmax)
        work = work.iloc[idx].reset_index(drop=True)
        coords = coords[idx]

    x_tail = coords[:, 0]
    y_tail = coords[:, 1]
    z_tail = coords[:, 2]

    Bx = work["Bx"].to_numpy()
    By = work["By"].to_numpy()
    Bz = work["Bz"].to_numpy()

    # Physical scaling: B [mT] -> arrow components [mm]
    Ux = Bx * scale / 1000.0
    Uy = By * scale / 1000.0
    Uz = Bz * scale / 1000.0

    x_head = x_tail + Ux
    y_head = y_tail + Uy
    z_head = z_tail + Uz

    n = len(work)

    # Build line segments with NaN separators: (tail -> head -> NaN) for each arrow
    xs = np.empty(3 * n)
    ys = np.empty(3 * n)
    zs = np.empty(3 * n)

    xs[0::3] = x_tail
    xs[1::3] = x_head
    xs[2::3] = np.nan

    ys[0::3] = y_tail
    ys[1::3] = y_head
    ys[2::3] = np.nan

    zs[0::3] = z_tail
    zs[1::3] = z_head
    zs[2::3] = np.nan

    # --- Build Plotly figure with line segments ---
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(width=float(arrow_linewidth), color=color),
                hoverinfo="skip",  # line hover can be noisy for many segments
            )
        ]
    )

    # Optionally add faint points at the tails for context:
    fig.add_trace(
        go.Scatter3d(
            x=x_tail,
            y=y_tail,
            z=z_tail,
            mode="markers",
            marker=dict(size=2.0, opacity=0.4, color=color),
            hoverinfo="x+y+z",
            showlegend=False,
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="x [mm]",
            yaxis_title="y [mm]",
            zaxis_title="z [mm]",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        title=(
            f"3D magnetic field quiver (lines). "
            f"Scale={scale} mm/mT, vectors={n:,}"
        ),
        showlegend=False,
    )

    return fig

