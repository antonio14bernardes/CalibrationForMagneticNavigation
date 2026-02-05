import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.colors import Normalize, LogNorm, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import matplotlib.tri as mtri
import warnings


_PLANE_MAP = {
    "x": {"coords": (1, 2), "labels": ("y", "z")},  # normal to x => plot y-z
    "y": {"coords": (0, 2), "labels": ("x", "z")},  # normal to y => plot x-z
    "z": {"coords": (0, 1), "labels": ("x", "y")},  # normal to z => plot x-y
}
_PROJECTIONS = ["x", "y", "z"]

def plot_rse_heatmap_projection(
    pkg,
    models,                  # list[ModelPhantom]
    *,
    title=None,
    figsize=None,
    continuous=False,         # points (False) or tricontourf (True)
    levels=200,
    cmap="viridis",
    point_size=6,
    point_alpha=0.9,
    shared_colorbar=True,     # shared across the 3 projections within each model figure
    log_color=False,          # if True: LogNorm on color
    show=True,
    verbose_names=True,
):
    """
    Per-model figure: 1x3 projections, colored by RSE = ||pred - true||.

    Dataset selection:
      - If phantom.reduced and reduced exists -> use reduced positions/fields
      - else -> use full

    Predictions:
      - pulled from pkg.get_field_predictions(phantom)
    """
    figs = []
    axes_list = []

    for ph in models:
        pos, _cur, true_fields, used_red = pkg._select_dataset_for_phantom(ph, purpose="plot_rse_heatmap_projection_pkg")
        pos = np.asarray(pos, float)
        tru = np.asarray(true_fields, float)
        pred = np.asarray(pkg.get_field_predictions(ph), float)

        if pred.shape != tru.shape:
            raise ValueError(
                f"Prediction/true shape mismatch for {ph.string(verbose=True)}: pred {pred.shape}, true {tru.shape}"
            )

        rse = np.linalg.norm(pred - tru, axis=1)

        # shared scaling inside each model figure
        if shared_colorbar:
            vmin = float(np.nanmin(rse))
            vmax = float(np.nanmax(rse))
            if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmin == vmax):
                vmin, vmax = 0.0, (vmin if np.isfinite(vmin) else 1.0) + 1e-12
        else:
            vmin = vmax = None

        norm = LogNorm(vmin=max(vmin, 1e-12), vmax=max(vmax, 1e-12)) if (log_color and shared_colorbar) else None

        if figsize is None:
            fig_size = (18, 6)
        else:
            fig_size = figsize

        fig, axes = plt.subplots(1, 3, figsize=fig_size, squeeze=False)
        axes = axes.ravel()

        mappable = None

        for j, proj in enumerate(_PROJECTIONS):
            ax = axes[j]
            c1, c2 = _PLANE_MAP[proj]["coords"]
            lab1, lab2 = _PLANE_MAP[proj]["labels"]

            X = pos[:, c1]
            Y = pos[:, c2]
            m = np.isfinite(X) & np.isfinite(Y) & np.isfinite(rse)

            Xp, Yp, ep = X[m], Y[m], rse[m]

            do_cont = continuous and (Xp.size >= 3)

            if do_cont:
                tri = mtri.Triangulation(Xp, Yp)
                cf = ax.tricontourf(
                    tri, ep, levels=levels, cmap=cmap,
                    vmin=vmin, vmax=vmax, norm=norm
                )
                if mappable is None:
                    mappable = cf
            else:
                sc = ax.scatter(
                    Xp, Yp, c=ep, s=point_size, alpha=point_alpha, cmap=cmap,
                    vmin=vmin, vmax=vmax, norm=norm
                )
                if mappable is None:
                    mappable = sc

            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(lab1)
            ax.set_ylabel(lab2)
            ax.set_title(f"Projection onto {lab1}-{lab2} (normal to {proj.upper()})")
            ax.grid(True, alpha=0.2)

        model_name = ph.string(verbose=verbose_names)
        mode = "continuous" if continuous else "points"
        auto_hdr = f"{'REDUCED' if used_red else 'FULL'} | RSE = ||pred - true|| (mT) | mode={mode}"

        if title is not None:
            fig.suptitle(f"{title} — {model_name}", y=0.995)
            fig.text(0.5, 0.92, auto_hdr, ha="center", va="top")
            top_rect = 0.86
        else:
            fig.suptitle(f"{model_name}\n{auto_hdr}", y=0.98)
            top_rect = 0.93

        if mappable is not None:
            fig.tight_layout(rect=[0, 0, 0.88, top_rect])
            cax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
            cbar = fig.colorbar(mappable, cax=cax)
            cbar.set_label("RSE = ||pred - true|| (mT)")
        else:
            fig.tight_layout(rect=[0, 0, 1, top_rect])

        if show:
            plt.show()
        else:
            figs.append(fig)
            axes_list.append(axes)

    if show:
        return
    return figs, axes_list

def plot_component_error_heatmap_projection(
    pkg,
    models,                  # list[ModelPhantom]
    *,
    title=None,
    figsize=None,
    continuous=False,         # points (False) or tricontourf (True)
    levels=200,
    cmap=None,                # if None: viridis for abs, RdBu_r for signed
    point_size=6,
    point_alpha=0.9,
    shared_colorbar=True,     # shared across all 9 panels within each model figure
    log_color=False,          # only meaningful for abs errors
    use_abs=True,             # True: |pred-true|, False: (pred-true) signed
    show=True,
    verbose_names=True,
):
    """
    Per-model figure: 3x3 grid (rows=components Bx/By/Bz, cols=projections),
    colored by component-wise error.

    Dataset selection:
      - If phantom.reduced and reduced exists -> use reduced positions/fields
      - else -> use full

    Predictions:
      - pulled from pkg.get_field_predictions(phantom)
    """
    figs = []
    axes_list = []

    if cmap is None:
        cmap = "viridis" if use_abs else "RdBu_r"

    comp_names = ["Bx", "By", "Bz"]

    for ph in models:
        pos, _cur, true_fields, used_red = pkg._select_dataset_for_phantom(
            ph, purpose="plot_component_error_heatmap_projection_pkg"
        )
        pos = np.asarray(pos, float)
        tru = np.asarray(true_fields, float)
        pred = np.asarray(pkg.get_field_predictions(ph), float)

        if pred.shape != tru.shape:
            raise ValueError(
                f"Prediction/true shape mismatch for {ph.string(verbose=True)}: pred {pred.shape}, true {tru.shape}"
            )

        diff = pred - tru
        if use_abs:
            diff_plot = np.abs(diff)
        else:
            diff_plot = diff

        # --- build shared norm across all 9 panels (optional) ---
        norm = None
        if shared_colorbar:
            # collect all values that will actually be plotted (respect finite projection masks)
            vals = []
            for proj in _PROJECTIONS:
                c1, c2 = _PLANE_MAP[proj]["coords"]
                X = pos[:, c1]
                Y = pos[:, c2]
                m = np.isfinite(X) & np.isfinite(Y) & np.all(np.isfinite(diff_plot), axis=1)
                if np.any(m):
                    vals.append(diff_plot[m, :].ravel())

            allv = np.concatenate(vals) if vals else np.array([0.0])

            if use_abs:
                vmin = float(np.nanmin(allv))
                vmax = float(np.nanmax(allv))
                if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmin == vmax):
                    vmin, vmax = 0.0, (vmin if np.isfinite(vmin) else 1.0) + 1e-12

                if log_color:
                    # LogNorm requires vmin > 0
                    vmin = max(vmin, 1e-12)
                    vmax = max(vmax, vmin * (1.0 + 1e-12))
                    norm = LogNorm(vmin=vmin, vmax=vmax)
                else:
                    norm = Normalize(vmin=vmin, vmax=vmax)
            else:
                # signed: symmetric about 0
                maxabs = float(np.nanmax(np.abs(allv)))
                if (not np.isfinite(maxabs)) or (maxabs == 0.0):
                    maxabs = 1e-12
                norm = TwoSlopeNorm(vmin=-maxabs, vcenter=0.0, vmax=+maxabs)

        # --- figure ---
        fig_size = (18, 16) if figsize is None else figsize
        fig, axes = plt.subplots(3, 3, figsize=fig_size, squeeze=False)

        mappable = None

        for r in range(3):  # component row
            for c, proj in enumerate(_PROJECTIONS):  # projection col
                ax = axes[r, c]
                c1, c2 = _PLANE_MAP[proj]["coords"]
                lab1, lab2 = _PLANE_MAP[proj]["labels"]

                X = pos[:, c1]
                Y = pos[:, c2]
                m = np.isfinite(X) & np.isfinite(Y) & np.isfinite(diff_plot[:, r])

                Xp, Yp, ep = X[m], Y[m], diff_plot[m, r]
                do_cont = continuous and (Xp.size >= 3)

                if do_cont:
                    tri = mtri.Triangulation(Xp, Yp)
                    cf = ax.tricontourf(tri, ep, levels=levels, cmap=cmap, norm=norm)
                    if mappable is None:
                        mappable = cf
                else:
                    sc = ax.scatter(
                        Xp, Yp, c=ep, s=point_size, alpha=point_alpha, cmap=cmap, norm=norm
                    )
                    if mappable is None:
                        mappable = sc

                ax.set_aspect("equal", adjustable="box")
                ax.set_xlabel(lab1)
                ax.set_ylabel(lab2)
                ax.grid(True, alpha=0.2)

                if r == 0:
                    ax.set_title(f"Projection onto {lab1}-{lab2} (normal to {proj.upper()})")
                if c == 0:
                    ax.set_ylabel(f"{comp_names[r]}  vs  {lab2}")

        model_name = ph.string(verbose=verbose_names)
        mode = "continuous" if continuous else "points"
        err_kind = "|pred - true|" if use_abs else "pred - true"
        scale_kind = "log" if (log_color and use_abs) else "linear"
        auto_hdr = (
            f"{'REDUCED' if used_red else 'FULL'} | component error: {err_kind} (mT) | "
            f"mode={mode} | color={scale_kind}"
        )

        if title is not None:
            fig.suptitle(f"{title} — {model_name}", y=0.995)
            fig.text(0.5, 0.92, auto_hdr, ha="center", va="top")
            top_rect = 0.86
        else:
            fig.suptitle(f"{model_name}\n{auto_hdr}", y=0.98)
            top_rect = 0.93

        # --- shared colorbar per figure ---
        if mappable is not None:
            fig.tight_layout(rect=[0, 0, 0.88, top_rect])
            cax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
            cbar = fig.colorbar(mappable, cax=cax)
            if use_abs:
                cbar.set_label("|pred - true| (mT)")
            else:
                cbar.set_label("pred - true (mT)")
        else:
            fig.tight_layout(rect=[0, 0, 1, top_rect])

        if show:
            plt.show()
        else:
            figs.append(fig)
            axes_list.append(axes)

    if show:
        return
    return figs, axes_list