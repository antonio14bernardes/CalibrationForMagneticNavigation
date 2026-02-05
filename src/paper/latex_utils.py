from matplotlib import rc_context

latex_prms_singlecol = {
    "label_size": 9,        # overall font.size
    "tick_size": 8,
    "axes_labelsize": 9,
    "legend_size": 8,
    "title_size": 9,        # axes titles
    "box_font": 8,
    # keep figsize in the plotting function (or set here if you prefer)
}

latex_prms_2img = {
    "label_size": 15,
    "tick_size": 13,
    "axes_labelsize": 15,
    "legend_size": 13,
    "title_size": 17,
    "box_font": 13,
}

latex_prms_3img = {
    "label_size": 22,
    "tick_size": 20,
    "axes_labelsize": 22,
    "legend_size": 20,
    "title_size": 16,
    "box_font": 20,
    "figsize": (24, 7),
}

def rc_context_latex(
    label_size=14,
    tick_size=12,
    axes_labelsize=14,
    legend_size=12,
    title_size=16,
    box_font=12,
    figsize=None,
    usetex=True,
):

    rc = {
        "text.usetex": True,

        # Embed fonts properly in vector outputs
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        "font.family": "serif",

        "text.latex.preamble": r"""
            \usepackage{amsmath}
            \usepackage{bm}

            % Times/Helvetica/Courier (modern, consistent)
            
            \PassOptionsToPackage{scaled=0.92}{helvet}
            \usepackage{helvet}
            \usepackage{courier}

            \providecommand{\field}{\bm{b}}
            \providecommand{\actuation}{\mathcal{A}}
            \renewcommand{\phi}{\varphi}
            \providecommand{\current}{\bm{i}}
            \providecommand{\deg}{\text{\textdegree}}
        """,

        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "axes.labelsize": axes_labelsize,
        "legend.fontsize": legend_size,
        "axes.titlesize": title_size,
        "figure.titlesize": title_size,
        "font.size": label_size,
    }

    if figsize is not None:
        rc["figure.figsize"] = figsize

    return rc_context(rc=rc)