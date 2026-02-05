import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Helpers (gradient metrics)
# ----------------------------

def _label_for_phantom(p, verbose=True):
    return p.string(verbose=verbose)

def _get_grad_metric_leaf(pkg, phantom):
    """
    Returns the gradient-metrics dict for a single phantom:
      pkg._gradient_metrics_dict[base][Reduced/Full][structure] -> dict of metric arrays/scalars
    """
    base, reduced_key, structure = phantom.keys()
    if pkg.gradient_metrics is None:
        raise ValueError("pkg.gradient_metrics is None. Run pkg.apply_gradient_metric(...) first.")
    try:
        return pkg.gradient_metrics[base][reduced_key][structure]
    except Exception as e:
        raise KeyError(f"Missing gradient metrics for model '{phantom.string(verbose=True)}'") from e

def prepare_for_plot_pkg_grad(pkg, models, metric_names, verbose_names=True):
    """
    Build:
      plot_dict[metric_name][label] = metric_value
    """
    plot_dict = {m: {} for m in metric_names}

    for phantom in models:
        leaf = _get_grad_metric_leaf(pkg, phantom)
        label = _label_for_phantom(phantom, verbose=verbose_names)

        for metric_name in metric_names:
            if metric_name not in leaf:
                raise KeyError(
                    f"Gradient metric '{metric_name}' not found for model '{label}'. "
                    f"Available metrics: {list(leaf.keys())}"
                )
            plot_dict[metric_name][label] = leaf[metric_name]

    return plot_dict


# ----------------------------
# Gradient plot suite (1x2)
# ----------------------------

def div_curl_plot(
    pkg,
    models,
    *,
    div_key="div",
    curl_key="curl",
    verbose=True,
    show=True,
    title=None,
    figsize=(12, 4),
    showfliers=False,
    ylabel_div="divergence",
    ylabel_curl="|curl|",
):
    """
    Produces a 1x2 plot:
      LEFT  : boxplot of divergence metric (div_key)
      RIGHT : boxplot of curl magnitude metric (curl_key)

    Assumes you've already computed gradient metrics into pkg via:
        pkg.apply_gradient_metric(metrics.grad_curl_div)   # or equivalent

    Metrics stored under pkg.gradient_metrics[base][Reduced/Full][structure][div_key/curl_key]
    """
    metric_names = [div_key, curl_key]
    plot_dict = prepare_for_plot_pkg_grad(pkg, models, metric_names, verbose_names=verbose)

    labels = [_label_for_phantom(m, verbose=verbose) for m in models]

    div_data  = [np.asarray(plot_dict[div_key][lbl])  for lbl in labels]
    curl_data = [np.asarray(plot_dict[curl_key][lbl]) for lbl in labels]

    # sanity: ensure these are 1D arrays for boxplot
    def _as_1d(a, name, lbl):
        a = np.asarray(a)
        if a.ndim == 0:
            # scalar -> treat as length-1 array
            return a.reshape(1)
        if a.ndim != 1:
            raise ValueError(f"{name} for '{lbl}' must be 1D (per-sample), got shape {a.shape}")
        return a

    div_data  = [_as_1d(a, div_key,  lbl) for a, lbl in zip(div_data, labels)]
    curl_data = [_as_1d(a, curl_key, lbl) for a, lbl in zip(curl_data, labels)]

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    if title is not None:
        fig.suptitle(title, fontsize=14)

    # LEFT: divergence
    axes[0].boxplot(div_data, labels=labels, showfliers=showfliers)
    axes[0].set_title("Divergence error" if div_key == "div" else f"{div_key}")
    axes[0].set_ylabel(ylabel_div)
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(True, alpha=0.2)

    # RIGHT: curl magnitude
    axes[1].boxplot(curl_data, labels=labels, showfliers=showfliers)
    axes[1].set_title("Curl magnitude error" if curl_key in ("curl_mag", "curl") else f"{curl_key}")
    axes[1].set_ylabel(ylabel_curl)
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(True, alpha=0.2)

    if show:
        plt.show()
        return

    return fig, axes