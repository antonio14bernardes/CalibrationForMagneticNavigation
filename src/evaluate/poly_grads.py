import numpy as np
import pandas as pd
import os
from sklearn.neighbors import KDTree
import pickle
from tqdm import tqdm

def get_poly_gradients_dict(df, orders, nn_counts):
    if isinstance(nn_counts, int):
        nn_counts = [nn_counts]
    if isinstance(orders, int):
        orders = [orders]

    gradient_reference_dict = {}

    grads_dict = {}

    em_dict = None  # to cache current-config grouping
    for order in orders:
        orders_dict = {}
        for nn_count in nn_counts:
            print(f"Computing gradients for nn_count={nn_count}, order={order}...")
            grads_array, em_dict = get_poly_gradients(df, order, nn_count, _em_dict_precomputed=em_dict)
            orders_dict[nn_count] = grads_array
        grads_dict[order] = orders_dict

    gradient_reference_dict['gradients'] = grads_dict

    # Include positions, currents, and fields for reference
    gradient_reference_dict['positions'] = df[['x', 'y', 'z']].to_numpy(dtype=float)
    em_cols = [col for col in df.columns if col.startswith("em_")]
    gradient_reference_dict['currents'] = df[em_cols].to_numpy(dtype=float)
    gradient_reference_dict['fields'] = df[['Bx', 'By', 'Bz']].to_numpy(dtype=float)

    return gradient_reference_dict

def get_poly_gradients(df, order, nn_count, _em_dict_precomputed=None):
    em_cols = [c for c in df.columns if c.startswith("em_")]

    # Group indices by current config
    if _em_dict_precomputed is not None:
        em_dict = _em_dict_precomputed
    else:
        em_dict = {}
        for idx, row in tqdm(df[em_cols].iterrows(), desc="Grouping by current configurations"):
            key = tuple(row.to_numpy())
            em_dict.setdefault(key, []).append(idx)

    # Preallocate output in df order
    grads_array = np.empty((len(df), 3, 3), dtype=float)

    for key, idxs in tqdm(em_dict.items(), desc="Computing polynomial gradients"):
        sub_df = df.loc[idxs]

        X_all = sub_df[["x", "y", "z"]].to_numpy(float)
        B_all = sub_df[["Bx", "By", "Bz"]].to_numpy(float)

        tree = KDTree(X_all)

        k = min(nn_count, len(idxs))
        n_features = 1 + 3 * order
        if k < n_features:
            raise ValueError(
                f"Need at least {n_features} neighbors for order={order} "
                f"(got k={k} for current-config {key})."
            )

        for i, idx in enumerate(idxs):
            pos = X_all[i, :].reshape(1, -1)
            _, nn_idxs = tree.query(pos, k=k)
            nn_idxs = nn_idxs.ravel()

            X_nn = X_all[nn_idxs, :]
            B_nn = B_all[nn_idxs, :]

            Xhat_x, Xhat_y, Xhat_z, _ = polynomial_fit(X_nn, B_nn, order)
            J = jacobian_from_poly_fit(pos.ravel(), Xhat_x, Xhat_y, Xhat_z, order)

            grads_array[idx] = J  # <- write into the correct df row

    return grads_array, em_dict

def jacobian_from_poly_fit(pos, Xhat_x, Xhat_y, Xhat_z, order):
    
    Xhat_x = np.asarray(Xhat_x)
    Xhat_y = np.asarray(Xhat_y)
    Xhat_z = np.asarray(Xhat_z)
    pos    = np.asarray(pos)

    if pos.shape != (3,):
        raise ValueError(f"pos must have shape (3,), got {pos.shape}")

    expected_len = 1 + 3 * order
    for name, w in zip(["Xhat_x", "Xhat_y", "Xhat_z"], [Xhat_x, Xhat_y, Xhat_z]):
        if w.shape != (expected_len,):
            raise ValueError(
                f"{name} must have shape ({expected_len},), got {w.shape}"
            )

    x, y, z = pos
    coords = [x, y, z]

    # Helper to compute gradient of one component (Bx or By or Bz)
    def grad_of_component(w):
        """
        Given coeff vector w for one field component, return
        [dB/dx, dB/dy, dB/dz] at (x, y, z).
        """
        grad = np.zeros(3, dtype=float)

        # loop over variables: 0->x, 1->y, 2->z
        for var_idx, coord in enumerate(coords):
            val = 0.0
            # degrees 1..order
            for k in range(1, order + 1):
                coef_idx = 1 + 3*(k-1) + var_idx   # select coeff of coord^k
                coef = w[coef_idx]
                val += k * coef * (coord ** (k-1))
            grad[var_idx] = val

        return grad

    gBx = grad_of_component(Xhat_x)  # [dBx/dx, dBx/dy, dBx/dz]
    gBy = grad_of_component(Xhat_y)  # [dBy/dx, dBy/dy, dBy/dz]
    gBz = grad_of_component(Xhat_z)  # [dBz/dx, dBz/dy, dBz/dz]

    J = np.vstack([gBx, gBy, gBz])   # shape (3, 3)
    return J

def polynomial_fit(X, y, order):
    X = np.asarray(X)
    y = np.asarray(y)

    # Check shapes
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("X array must have shape (N, 3) -> x, y, z")
    if y.ndim != 2 or y.shape[1] != 3:
        raise ValueError("y array must have shape (N, 3) -> Bx, By, Bz")

    N = y.shape[0]

    # Expand for polynomial order: [1, x, y, z, x^2, y^2, z^2, ...]
    exps = [i + 1 for i in range(order)]
    blocks = [np.ones((N, 1))]            # offset term

    for exp in exps:
        blocks.append(X ** exp)

    Phi = np.hstack(blocks)

    # Split outputs
    y_x = y[:, 0]
    y_y = y[:, 1]
    y_z = y[:, 2]

    # Common helper matrix (Phi^T Phi)
    A = Phi.T @ Phi

    # Use solve instead of explicit inverse
    Xhat_x = np.linalg.solve(A, Phi.T @ y_x)
    Xhat_y = np.linalg.solve(A, Phi.T @ y_y)
    Xhat_z = np.linalg.solve(A, Phi.T @ y_z)

    return Xhat_x, Xhat_y, Xhat_z, order

def build_reduced_gradient_reference_dict(
    gradient_reference_dict,
    reduced_positions,
    reduced_currents,
    reduced_fields,
    *,
    decimals=None,
    strict_unique=True,
):
    for k in ("positions", "currents", "fields", "gradients"):
        if k not in gradient_reference_dict:
            raise KeyError(f"gradient_reference_dict missing key '{k}'")

    full_positions = np.asarray(gradient_reference_dict["positions"])
    full_currents  = np.asarray(gradient_reference_dict["currents"])
    full_fields    = np.asarray(gradient_reference_dict["fields"])

    rpos = np.asarray(reduced_positions)
    rcur = np.asarray(reduced_currents)
    rfld = np.asarray(reduced_fields)

    full_combo = np.hstack([full_positions, full_currents, full_fields])
    red_combo  = np.hstack([rpos, rcur, rfld])

    if full_combo.shape[1] != red_combo.shape[1]:
        raise ValueError(
            f"Full combo has {full_combo.shape[1]} cols but reduced combo has {red_combo.shape[1]} cols."
        )

    if decimals is not None:
        full_combo = np.round(full_combo, decimals=decimals)
        red_combo  = np.round(red_combo,  decimals=decimals)

    # contiguous
    full_combo_c = np.ascontiguousarray(full_combo)
    red_combo_c  = np.ascontiguousarray(red_combo)

    # Represent each row as raw bytes (hashable)
    row_nbytes = full_combo_c.dtype.itemsize * full_combo_c.shape[1]
    full_void = full_combo_c.view(np.dtype((np.void, row_nbytes))).ravel()
    red_void  = red_combo_c.view(np.dtype((np.void, row_nbytes))).ravel()

    # Convert to bytes keys so dict hashing is reliable across numpy versions
    full_keys = [v.tobytes() for v in full_void]
    red_keys  = [v.tobytes() for v in red_void]

    if strict_unique:
        # uniqueness check using bytes keys
        uniq, counts = np.unique(np.array(full_keys, dtype=object), return_counts=True)
        if np.any(counts > 1):
            dup_count = int(np.sum(counts > 1))
            raise ValueError(
                f"Full dataset has {dup_count} duplicated rows in (pos+curr+field); mapping ambiguous. "
                "Set strict_unique=False to allow (keeps first occurrence)."
            )

    # Build lookup (bytes -> first index)
    lookup = {}
    for i, k in enumerate(full_keys):
        if (not strict_unique) and (k in lookup):
            continue
        lookup[k] = i

    # Map reduced rows -> full indices
    idx_map = np.empty(len(red_keys), dtype=int)
    missing = []
    for i, k in enumerate(red_keys):
        j = lookup.get(k, None)
        if j is None:
            missing.append(i)
        else:
            idx_map[i] = j

    if missing:
        preview = missing[:10]
        raise KeyError(
            f"{len(missing)} reduced rows not found in full gradient_reference_dict. "
            f"First missing reduced indices: {preview}. "
            f"If float precision is the issue, try decimals=6 or 7."
        )

    # Build reduced dict by slicing
    reduced_dict = {
        "positions": full_positions[idx_map],
        "currents":  full_currents[idx_map],
        "fields":    full_fields[idx_map],
    }

    full_grads = gradient_reference_dict["gradients"]
    grads_out = {}
    for order, nn_dict in full_grads.items():
        nn_out = {}
        for nn_count, arr in nn_dict.items():
            arr = np.asarray(arr)
            if arr.shape[0] != full_positions.shape[0]:
                raise ValueError(
                    f"Gradient array for order={order}, nn_count={nn_count} has N={arr.shape[0]}, "
                    f"but full positions has N={full_positions.shape[0]}."
                )
            nn_out[nn_count] = arr[idx_map]
        grads_out[order] = nn_out

    reduced_dict["gradients"] = grads_out

    # copy other metadata keys if you want (optional)
    for k, v in gradient_reference_dict.items():
        if k in ("positions", "currents", "fields", "gradients"):
            continue
        reduced_dict[k] = v

    return reduced_dict, idx_map