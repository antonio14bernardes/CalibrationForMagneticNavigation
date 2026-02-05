import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
import pandas as pd

import os, sys

# go to project_dir
project_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "..", ".."))
if project_dir not in sys.path:
    sys.path.insert(0, os.path.join(project_dir, "src"))


def _label_for_phantom(p, verbose=True):
    return p.string(verbose=verbose)

def _true_fields_for_phantom(pkg, phantom):
    """
    Get the true fields array that corresponds to this phantom.
    This relies on your package method that selects the appropriate dataset for the phantom
    (now keyed by dataset_percentage).
    """
    _pos, _cur, true_fields, _used_reduced_like = pkg._select_dataset_for_phantom(
        phantom, purpose="plot_true_fields"
    )
    if true_fields is None:
        raise ValueError("No true fields found in package for this phantom.")
    return np.asarray(true_fields)

def _get_metric_leaf(pkg, phantom):
    """
    Returns the metrics dict for a single phantom:
        pkg.field_metrics[name][dataset_percentage][structure] -> dict of metric arrays/scalars
    where:
        phantom.keys() -> (name, dataset_percentage, structure)
    """
    name, dataset_percentage, structure = phantom.keys()

    if pkg.field_metrics is None:
        raise ValueError("pkg.field_metrics is None. Run pkg.apply_field_metric(...) first.")

    # Be robust to whether the middle key is stored as int or str
    candidates = [
        (name, dataset_percentage, structure),
        (name, int(dataset_percentage), structure),
        (name, str(dataset_percentage), structure),
    ]

    last_err = None
    for n, pct, st in candidates:
        try:
            return pkg.field_metrics[n][pct][st]
        except Exception as e:
            last_err = e

    raise KeyError(
        f"Missing field metrics for model '{phantom.string(verbose=True)}'. "
        f"Tried dataset_percentage keys: {dataset_percentage!r}, {int(dataset_percentage)!r}, {str(dataset_percentage)!r}"
    ) from last_err

def prepare_for_plot_pkg(pkg, models, metric_names, verbose_names=True):
    """
    Build:
      plot_dict[metric_name][label] = metric_value
    where metric_value may be an array (per-sample) or scalar.
    """
    plot_dict = {m: {} for m in metric_names}

    for phantom in models:
        leaf = _get_metric_leaf(pkg, phantom)
        label = _label_for_phantom(phantom, verbose=verbose_names)

        for metric_name in metric_names:
            if metric_name not in leaf:
                raise KeyError(
                    f"Metric '{metric_name}' not found for model '{label}'. "
                    f"Available metrics: {list(leaf.keys())}"
                )
            plot_dict[metric_name][label] = leaf[metric_name]

    return plot_dict


# ----------------------------
# Basic plot suite (2x2)
# ----------------------------

def basic_plot_suite(
    pkg,
    models,
    *,
    verbose=True,
    show=True,
    title=None,
    figsize=(12, 8),
    showfliers=False,
):
    """
    Assumes you've already computed metrics into pkg via:
        pkg.apply_field_metric(metrics.rse)
        pkg.apply_field_metric(metrics.mag_and_angle)

    Produces a 2x2 plot:
      [0,0] boxplot of mag_rel
      [0,1] boxplot of angle
      [1,0] boxplot of rse
      [1,1] bar chart of rmse
    """
    metric_names = ["mag_rel", "angle", "rse", "rmse"]
    plot_dict = prepare_for_plot_pkg(pkg, models, metric_names, verbose_names=verbose)

    labels = [_label_for_phantom(m, verbose=verbose) for m in models]

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    if title is not None:
        fig.suptitle(title, fontsize=14)

    # mag_rel
    mag_data = [np.asarray(plot_dict["mag_rel"][lbl]) for lbl in labels]
    axes[0, 0].boxplot(mag_data, labels=labels, showfliers=showfliers)
    axes[0, 0].set_title("Relative magnitude error")
    axes[0, 0].set_ylabel("Relative error")
    axes[0, 0].tick_params(axis="x", rotation=20)

    # angle
    angle_data = [np.asarray(plot_dict["angle"][lbl]) for lbl in labels]
    axes[0, 1].boxplot(angle_data, labels=labels, showfliers=showfliers)
    axes[0, 1].set_title("Angle error")
    axes[0, 1].set_ylabel("Angle [deg]")
    axes[0, 1].tick_params(axis="x", rotation=20)

    # rse
    rse_data = [np.asarray(plot_dict["rse"][lbl]) for lbl in labels]
    axes[1, 0].boxplot(rse_data, labels=labels, showfliers=showfliers)
    axes[1, 0].set_title("RSE")
    axes[1, 0].set_ylabel("‖pred - true‖ (mT)")
    axes[1, 0].tick_params(axis="x", rotation=20)

    # rmse
    rmse_vals = [float(plot_dict["rmse"][lbl]) for lbl in labels]
    axes[1, 1].bar(labels, rmse_vals)
    axes[1, 1].set_title("RMSE")
    axes[1, 1].set_ylabel("RMSE (mT)")
    axes[1, 1].tick_params(axis="x", rotation=20)

    if show:
        plt.show()
        return

    return fig, axes


# ----------------------------
# Histogram diagnostics suite (per model, 1x2)
# ----------------------------

def hist_plot_suite(
    pkg,
    models,
    *,
    verbose=True,
    show=True,
    title_prefix=None,
    figsize=(10, 4),
    mag_percentiles=(1, 99),
    err_percentiles=(1, 99),
    angle_percentiles=(1, 99),
):
    """
    For each model in `models`, plots:
      [0] |true field| vs mag_rel (log-log 2D hist)
      [1] |true field| vs angle (log-x 2D hist)

    Uses per-phantom true fields selection consistent with your package rules,
    now keyed by dataset_percentage.
    """
    metric_names = ["mag_rel", "angle"]
    plot_dict = prepare_for_plot_pkg(pkg, models, metric_names, verbose_names=verbose)

    figs = []
    axes_list = []

    for phantom in models:
        label = _label_for_phantom(phantom, verbose=verbose)

        true_fields = _true_fields_for_phantom(pkg, phantom)
        true_mag = np.linalg.norm(true_fields, axis=1)

        mag_rel = np.asarray(plot_dict["mag_rel"][label])
        angle   = np.asarray(plot_dict["angle"][label])

        if len(true_mag) != len(mag_rel) or len(true_mag) != len(angle):
            raise ValueError(
                f"Length mismatch for '{label}': true={len(true_mag)}, mag_rel={len(mag_rel)}, angle={len(angle)}"
            )

        eps = 1e-12

        fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

        if title_prefix is not None:
            fig.suptitle(f"{title_prefix}: {label}", fontsize=12)
        else:
            fig.suptitle(label, fontsize=12)

        # ---- Left: |true| vs mag_rel ----
        ax = axes[0]
        mag_min, mag_max = np.percentile(true_mag, mag_percentiles)
        err_min, err_max = np.percentile(mag_rel, err_percentiles)

        mag_min = max(float(mag_min), 1e-9)
        mag_max = max(float(mag_max), mag_min * 1.0001)
        err_min = max(float(err_min), eps)
        err_max = max(float(err_max), err_min * 1.0001)

        mag_bins = np.logspace(np.log10(mag_min), np.log10(mag_max), 50)
        err_bins = np.logspace(np.log10(err_min), np.log10(err_max), 50)

        h = ax.hist2d(true_mag, mag_rel, bins=[mag_bins, err_bins], norm=LogNorm())
        fig.colorbar(h[3], ax=ax, label="Count")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("|true field| (mT)")
        ax.set_ylabel("relative mag error")
        ax.set_title("rel mag vs |true|")

        # ---- Right: |true| vs angle ----
        ax = axes[1]
        ang_min, ang_max = np.percentile(angle, angle_percentiles)
        ang_min = max(float(ang_min), eps)
        ang_max = max(float(ang_max), ang_min * 1.0001)
        ang_bins = np.linspace(ang_min, ang_max, 50)

        h2 = ax.hist2d(true_mag, angle, bins=[mag_bins, ang_bins], norm=LogNorm())
        fig.colorbar(h2[3], ax=ax, label="Count")
        ax.set_xscale("log")
        ax.set_xlabel("|true field| (mT)")
        ax.set_ylabel("angle [deg]")
        ax.set_title("angle vs |true|")

        if show:
            plt.show()

        figs.append(fig)
        axes_list.append(axes)

    if show:
        return

    return figs, axes_list


# ----------------------------
# Train vs test compare suite
# ----------------------------

def train_test_compare_suite(
    train_pkg,
    test_pkg,
    models,
    *,
    verbose=True,
    title=None,
    figsize=(16, 6),
    showfliers=False,
    train_label="Train",
    test_label="Test",
):
    """
    For each phantom in `models`, plot:
      - LEFT: RSE boxplots, ordered as (train, test) for each model
      - RIGHT: RMSE bar chart, paired (train, test) for each model

    Uses dataset_percentage keying implicitly through _get_metric_leaf().
    """

    labels_box = []
    rse_data = []

    model_base_labels = []
    rmse_train = []
    rmse_test = []

    missing = []

    for phantom in models:
        model_label = _label_for_phantom(phantom, verbose=verbose)
        model_base_labels.append(model_label)

        # ---- TRAIN ----
        try:
            leaf_tr = _get_metric_leaf(train_pkg, phantom)
            rse_tr = np.asarray(leaf_tr["rse"])
            rmse_tr = float(leaf_tr["rmse"])
        except Exception as e:
            missing.append((model_label, "train", str(e)))
            rse_tr = None
            rmse_tr = np.nan

        # ---- TEST ----
        try:
            leaf_te = _get_metric_leaf(test_pkg, phantom)
            rse_te = np.asarray(leaf_te["rse"])
            rmse_te = float(leaf_te["rmse"])
        except Exception as e:
            missing.append((model_label, "test", str(e)))
            rse_te = None
            rmse_te = np.nan

        # Interleaved order: train then test
        if rse_tr is not None:
            rse_data.append(rse_tr)
            labels_box.append(f"{model_label} ({train_label})")
        if rse_te is not None:
            rse_data.append(rse_te)
            labels_box.append(f"{model_label} ({test_label})")

        rmse_train.append(rmse_tr)
        rmse_test.append(rmse_te)

    if len(rse_data) == 0:
        raise ValueError("No RSE data found to plot. Did you run apply_field_metric(metrics.rse) on both packages?")

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    if title is not None:
        fig.suptitle(title, fontsize=14)

    # LEFT: RSE boxplots
    ax = axes[0]
    bp = ax.boxplot(rse_data, labels=labels_box, showfliers=showfliers)
    ax.set_title("RSE distributions (train vs test, paired per model)")
    ax.set_ylabel("‖pred - true‖ (mT)")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, alpha=0.2)

    # style train vs test by linestyle
    for i, box in enumerate(bp.get("boxes", [])):
        lbl = labels_box[i].lower()
        if f"({train_label.lower()})" in lbl:
            box.set_linewidth(2.0)
            box.set_linestyle("-")
        elif f"({test_label.lower()})" in lbl:
            box.set_linewidth(2.0)
            box.set_linestyle("--")

    ax.legend(
        handles=[
            Patch(facecolor="none", edgecolor="black", linewidth=2.0, linestyle="-", label=train_label),
            Patch(facecolor="none", edgecolor="black", linewidth=2.0, linestyle="--", label=test_label),
        ],
        frameon=False,
        loc="best",
    )

    # RIGHT: RMSE paired bars
    ax = axes[1]
    x = np.arange(len(models))
    width = 0.38
    ax.bar(x - width/2, rmse_train, width, label=train_label)
    ax.bar(x + width/2, rmse_test,  width, label=test_label)

    ax.set_title("RMSE (train vs test, paired per model)")
    ax.set_ylabel("RMSE (mT)")
    ax.set_xticks(x)
    ax.set_xticklabels(model_base_labels, rotation=30, ha="right")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(frameon=False)

    if missing:
        msg = "\n".join([f"  - {m} ({split}): {err}" for (m, split, err) in missing[:8]])
        if len(missing) > 8:
            msg += f"\n  ... and {len(missing) - 8} more"
        print("[train_test_compare_suite] Some items were missing / failed to load:")
        print(msg)

    plt.show()
    return fig, axes



def plot_train_test_rmse_rank_and_pct_suite(
    train_pkg,
    test_pkg,
    models_rank,
    models_pct,
    *,
    verbose=True,
    title=None,
    figsize=(10, 4.8),
    ylabel_test="Test RMSE (mT)",
    ylabel_gap="Test - Train RMSE (mT)",
    xlabel_rank="Model size rank",
    xlabel_pct="Dataset percentage (%)",
    marker="o",
    ymin_test=None,
    ymax_test=None,
    ymin_gap=None,
    ymax_gap=None,
    legend_out=True,
    legend_ncol=1,
    use_paper_style=False,
    paper_kwargs=None,
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib as mpl

    # NOTE: no sns.set_theme() here — keep globals unchanged

    # ----------------------------
    # Build DF for rank panel
    # ----------------------------
    rows_rank = []
    for ph in models_rank:
        name, dataset_percentage, structure = ph.keys()
        leaf_tr = _get_metric_leaf(train_pkg, ph)
        leaf_te = _get_metric_leaf(test_pkg, ph)

        rmse_train = float(leaf_tr["rmse"])
        rmse_test  = float(leaf_te["rmse"])

        if structure is None:
            structure_tuple = None
            size_score = 1.0
        else:
            structure_tuple = tuple(structure)
            size_score = float(np.prod(np.asarray(structure_tuple, dtype=np.float64)))

        rows_rank.append({
            "name": name,
            "dataset_percentage": int(dataset_percentage),
            "structure": structure_tuple,
            "size_score": size_score,
            "rmse_train": rmse_train,
            "rmse_test": rmse_test,
            "rmse_gap": rmse_test - rmse_train,  # TEST - TRAIN
        })

    df_rank = pd.DataFrame(rows_rank)
    if df_rank.empty:
        raise ValueError("models_rank produced no rows (empty list or missing rmse).")

    df_rank = df_rank.sort_values(["name", "size_score", "structure"], na_position="last").reset_index(drop=True)
    df_rank["size_rank"] = df_rank.groupby("name").cumcount() + 1

    # ----------------------------
    # Build DF for pct panel
    # ----------------------------
    rows_pct = []
    for ph in models_pct:
        name, dataset_percentage, structure = ph.keys()
        leaf_tr = _get_metric_leaf(train_pkg, ph)
        leaf_te = _get_metric_leaf(test_pkg, ph)

        rmse_train = float(leaf_tr["rmse"])
        rmse_test  = float(leaf_te["rmse"])

        structure_tuple = None if structure is None else tuple(structure)

        if structure_tuple is None:
            line_id = name
        else:
            line_id = f"{name}_{'x'.join(map(str, structure_tuple))}"

        rows_pct.append({
            "name": name,
            "line_id": line_id,                     # used only for grouping/connecting
            "dataset_percentage": int(dataset_percentage),
            "structure": structure_tuple,
            "rmse_train": rmse_train,
            "rmse_test": rmse_test,
            "rmse_gap": rmse_test - rmse_train,    # TEST - TRAIN
        })

    df_pct = pd.DataFrame(rows_pct)
    if df_pct.empty:
        raise ValueError("models_pct produced no rows (empty list or missing rmse).")

    df_pct = df_pct.sort_values(["line_id", "dataset_percentage"]).reset_index(drop=True)

    # ----------------------------
    # Palette keyed ONLY by model name (alphabetical assignment)
    # ----------------------------
    model_names = sorted(set(df_rank["name"].unique()).union(set(df_pct["name"].unique())))
    pal = sns.color_palette("deep", n_colors=len(model_names))
    palette = {nm: pal[i] for i, nm in enumerate(model_names)}

    # ----------------------------
    # Legend ORDER = order models were passed in
    # (unique, preserve first appearance)
    # ----------------------------
    legend_order = []
    for ph in list(models_rank) + list(models_pct):
        nm = ph.keys()[0]
        if nm not in legend_order:
            legend_order.append(nm)

    # ----------------------------
    # Optional: local paper rcParams (NO globals)
    # ----------------------------
    rc_ctx = mpl.rc_context()  # default empty context
    fig_w_h_override = None

    if use_paper_style:
        if paper_kwargs is None:
            paper_kwargs = {}
        # import locally so eval_field_plots doesn't hard-depend at import time
        from paper.mpl_settings_old import mpl_paper_format

        fig_w, fig_h, rc = mpl_paper_format(apply=False, **paper_kwargs)
        rc_ctx = mpl.rc_context(rc)

        # If user did NOT explicitly pass figsize, use paper-derived size.
        # (If they did, keep figsize as-is.)
        if figsize is None:
            fig_w_h_override = (fig_w, fig_h)

    # ----------------------------
    # Plotting (inside rc_context if enabled)
    # ----------------------------
    with rc_ctx:
        # choose final figsize
        final_figsize = figsize
        if final_figsize is None and fig_w_h_override is not None:
            final_figsize = fig_w_h_override

        fig = plt.figure(figsize=final_figsize, constrained_layout=True)
        gs = fig.add_gridspec(
            2, 2,
            height_ratios=[1, 1],
            width_ratios=[1, 1],
            wspace=0.15,
            hspace=0.05
        )

        ax_rank_top = fig.add_subplot(gs[0, 0])
        ax_rank_bot = fig.add_subplot(gs[1, 0], sharex=ax_rank_top)

        ax_pct_top  = fig.add_subplot(gs[0, 1], sharey=ax_rank_top)
        ax_pct_bot  = fig.add_subplot(gs[1, 1], sharey=ax_rank_bot, sharex=ax_pct_top)

        def _prettify_ax(ax, show_xticklabels=True):
            ax.grid(True, alpha=0.25, linewidth=0.6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if not show_xticklabels:
                ax.tick_params(axis="x", labelbottom=False)

        legend_handles = {}  # name -> line handle (from rank-top plot)

        # LEFT: rank (one line per name)
        for name, g in df_rank.groupby("name", sort=False):
            g = g.sort_values("size_rank")
            color = palette[name]

            ln, = ax_rank_top.plot(
                g["size_rank"], g["rmse_test"],
                marker=marker, linewidth=plt.rcParams.get("lines.linewidth", 1.8),
                markersize=plt.rcParams.get("lines.markersize", 4.0),
                color=color
            )
            ax_rank_bot.plot(
                g["size_rank"], g["rmse_gap"],
                marker=marker, linewidth=plt.rcParams.get("lines.linewidth", 1.8),
                markersize=plt.rcParams.get("lines.markersize", 4.0),
                color=color
            )

            # keep first handle per name (for legend)
            if name not in legend_handles:
                legend_handles[name] = ln

        # RIGHT: pct (connect per line_id; same color per name)
        for line_id, g in df_pct.groupby("line_id", sort=False):
            g = g.sort_values("dataset_percentage")
            name = g["name"].iloc[0]
            color = palette[name]

            ax_pct_top.plot(
                g["dataset_percentage"], g["rmse_test"],
                marker=marker, linewidth=plt.rcParams.get("lines.linewidth", 1.8),
                markersize=plt.rcParams.get("lines.markersize", 4.0),
                color=color
            )
            ax_pct_bot.plot(
                g["dataset_percentage"], g["rmse_gap"],
                marker=marker, linewidth=plt.rcParams.get("lines.linewidth", 1.8),
                markersize=plt.rcParams.get("lines.markersize", 4.0),
                color=color
            )

        ax_rank_top.set_title("vs model-size rank")
        ax_pct_top.set_title("vs dataset percentage")

        ax_rank_top.set_ylabel(ylabel_test)
        ax_rank_bot.set_ylabel(ylabel_gap)
        ax_rank_bot.set_xlabel(xlabel_rank)
        ax_pct_bot.set_xlabel(xlabel_pct)

        ax_pct_top.tick_params(axis="y", labelleft=False)
        ax_pct_bot.tick_params(axis="y", labelleft=False)

        ax_rank_bot.axhline(0.0, linewidth=1.0, alpha=0.6)
        ax_pct_bot.axhline(0.0, linewidth=1.0, alpha=0.6)

        max_rank = int(df_rank["size_rank"].max())
        ax_rank_bot.set_xticks(np.arange(1, max_rank + 1))

        xticks_pct = np.sort(df_pct["dataset_percentage"].unique())
        ax_pct_bot.set_xticks(xticks_pct)

        if (ymin_test is not None) or (ymax_test is not None):
            ax_rank_top.set_ylim(bottom=ymin_test, top=ymax_test)
        if (ymin_gap is not None) or (ymax_gap is not None):
            ax_rank_bot.set_ylim(bottom=ymin_gap, top=ymax_gap)

        _prettify_ax(ax_rank_top, show_xticklabels=False)
        _prettify_ax(ax_rank_bot, show_xticklabels=True)
        _prettify_ax(ax_pct_top,  show_xticklabels=False)
        _prettify_ax(ax_pct_bot,  show_xticklabels=True)

        if title is None:
            title = "Test RMSE and (Test − Train) gap"
        fig.suptitle(title)

        # ----------------------------
        # Legend labels in PASSED-IN order (but colors still alphabetical)
        # ----------------------------
        leg_labels = [nm for nm in legend_order if nm in legend_handles]
        leg_handles = [legend_handles[nm] for nm in leg_labels]

        if legend_out:
            fig.legend(
                leg_handles, leg_labels,
                frameon=False, loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                ncol=legend_ncol
            )
        else:
            ax_pct_top.legend(leg_handles, leg_labels, frameon=False, loc="best")

        plt.show()

    return fig, (ax_rank_top, ax_rank_bot, ax_pct_top, ax_pct_bot), {"rank": df_rank, "pct": df_pct}