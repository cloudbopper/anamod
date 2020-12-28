"""Functions to visualize data"""

from collections import OrderedDict
from functools import reduce
import json
from types import SimpleNamespace

import cloudpickle
from IPython.display import display
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

from anamod.core import constants
from anamod.core.constants import FDR, POWER, AVERAGE_WINDOW_FDR, AVERAGE_WINDOW_POWER
from anamod.core.constants import ORDERING_ALL_IMPORTANT_FDR, ORDERING_ALL_IMPORTANT_POWER
from anamod.core.constants import ORDERING_IDENTIFIED_IMPORTANT_FDR, ORDERING_IDENTIFIED_IMPORTANT_POWER
from anamod.core.constants import TEMPORAL, WINDOW_OVERLAP, RESULTS, CONFIG
from anamod.core.constants import SYNTHESIZED_FEATURES_FILENAME, ANALYZED_FEATURES_FILENAME, MODEL_WRAPPER_FILENAME
from anamod.core.constants import WINDOW_IMPORTANT_FDR, WINDOW_IMPORTANT_POWER, WINDOW_ORDERING_IMPORTANT_FDR, WINDOW_ORDERING_IMPORTANT_POWER
from anamod.core.constants import OVERALL_SCORES_CORR, WINDOW_SCORES_CORR, OVERALL_RELEVANT_SCORES_CORR, WINDOW_RELEVANT_SCORES_CORR


GROUPS = {"Overall Feature Importance Detection": (FDR, POWER),
          "Ordering Detection (w.r.t. All Important Features)": (ORDERING_ALL_IMPORTANT_FDR, ORDERING_ALL_IMPORTANT_POWER),
          "Ordering Detection (w.r.t. Identified Important Features)": (ORDERING_IDENTIFIED_IMPORTANT_FDR, ORDERING_IDENTIFIED_IMPORTANT_POWER),
          "Average Window Detection": (AVERAGE_WINDOW_FDR, AVERAGE_WINDOW_POWER),
          "Window Importance Detection (w.r.t Identified Important Features)": (WINDOW_IMPORTANT_FDR, WINDOW_IMPORTANT_POWER),
          "Window Ordering Detection (w.r.t Identified Important Features)": (WINDOW_ORDERING_IMPORTANT_FDR, WINDOW_ORDERING_IMPORTANT_POWER),
          "Importance Scores R2 (all features)": (OVERALL_SCORES_CORR, WINDOW_SCORES_CORR),
          "Importance Scores R2 (relevant features)": (OVERALL_RELEVANT_SCORES_CORR, WINDOW_RELEVANT_SCORES_CORR)}

LEGACY_GROUPS = {"Overall Feature Importance Detection": (FDR, POWER),
                 "Temporal Feature Importance Detection": ("Temporal_FDR", "Temporal_Power"),
                 "Average Window Detection": (AVERAGE_WINDOW_FDR, AVERAGE_WINDOW_POWER)}


# Plot types
BOX = "box"
VIOLIN = "violin"
# Importance scores
OVERALL = "Overall"
WINDOW = "Window"


def flatten(nested_list):
    """Flatten nested list"""
    return [item for sub_list in nested_list for item in sub_list]


def get_param_name(param):
    """Convert parameter to appropriate label (to ensure it's treated as a string by the plotting library"""
    try:
        float(param)
        return f"n = {param}"
    except ValueError:
        return param


def visualize_analysis(data, trial_type, analysis_type=TEMPORAL, plot_type=BOX):
    """Visualize outputs"""
    # pylint: disable = invalid-name, too-many-locals
    # TODO: Document a bit better
    # TODO: First, write out a summary of the setup
    results = data[RESULTS]
    output_dirs = flatten(data[CONFIG]["output_dir"].values())
    hovertext = output_dirs  # visible when hovering over any given point (i.e. simulation)
    if analysis_type == TEMPORAL:
        fig = go.Figure()
        x, y = ([], [])
        for param, values in results[WINDOW_OVERLAP].items():  # len(values) = num_trials
            y.extend(values)
            param_name = get_param_name(param)
            x.extend([param_name] * len(values))
        if plot_type == BOX:
            fig.add_trace(go.Box(x=x, y=y, hovertext=hovertext))
        else:
            fig.add_trace(go.Violin(x=x, y=y, hovertext=hovertext))
        layout(fig, title="Average Window Overlap", xaxis_title=trial_type, yaxis_title="Average Overlap", plot_type=plot_type)
        fig.show()
    groups = GROUPS if ORDERING_ALL_IMPORTANT_FDR in results else LEGACY_GROUPS  # Backward-compatibility for old-style ordering results
    means = {}
    for name, group in groups.items():  # Overall, Ordering, Window
        fig = go.Figure()
        for cat in group:  # FDR, Power
            x, y = ([], [])
            key = f"{name}->{cat}"
            means[key] = OrderedDict()
            # Add all values to the same list y, and corresponding param names in x (used to split by param)
            for param, values in results[cat].items():
                y.extend(values)
                param_name = get_param_name(param)
                x.extend([param_name] * len(values))
                means[key][param] = np.mean(values)
            if plot_type == BOX:
                fig.add_trace(go.Box(x=x, y=y, hovertext=hovertext, legendgroup=cat, name=cat))
            else:
                fig.add_trace(go.Violin(x=x, y=y, hovertext=hovertext,
                                        legendgroup=cat, scalegroup=cat, name=cat))
        layout(fig, title=name, xaxis_title=trial_type, yaxis_title="Value", plot_type=plot_type)
        fig.show()
    return means


def layout(fig, title="", xaxis_title="", yaxis_title="", plot_type=BOX):
    """Perform common changes to plot layout"""
    if plot_type == BOX:
        fig.update_traces(opacity=0.6, boxmean=True)
        fig.update_yaxes(range=[-0.1, 1.1])
    else:
        fig.update_traces(box_visible=True, meanline_visible=True, opacity=0.6, points="all")
    fig.update_layout(title={"text": title, "xanchor": "center", "x": 0.5},
                      xaxis_title=xaxis_title, yaxis_title=yaxis_title,
                      font=dict(family="Serif", size=20, color="black"),
                      violinmode="group", template="none")


def visualize_simulation(sim_dir=".", generators=False, scores=False, overlap=False):
    """Detailed analysis of given simulation"""
    # pylint: disable = unused-variable, invalid-name, too-many-locals, line-too-long
    # TODO: Make compatible with hierarchical analysis
    # TODO: Clean up after finalizing what to use
    with open(f"{sim_dir}/{MODEL_WRAPPER_FILENAME}", "rb") as model_file:
        model_wrapper = cloudpickle.load(model_file)
    with open(f"{sim_dir}/{SYNTHESIZED_FEATURES_FILENAME}", "rb") as features_file:
        synthesized_features = cloudpickle.load(features_file)
    with open(f"{sim_dir}/{ANALYZED_FEATURES_FILENAME}", "rb") as features_file:
        analyzed_features = cloudpickle.load(features_file)
    with open(f"{sim_dir}/{constants.SIMULATION_SUMMARY_FILENAME}", "rb") as summary_file:
        summary = json.load(summary_file)

    model = model_wrapper.ground_truth_model
    relevant_features = reduce(set.union, model.relevant_feature_map, set())
    args = SimpleNamespace()
    args.sequence_length = int(summary[constants.CONFIG]["sequence_length"])
    # eval_results = evaluate_temporal(args, model, analyzed_features)

    if generators:
        visualize_generators(synthesized_features, analyzed_features, relevant_features)

    if scores:
        visualize_scores(synthesized_features, analyzed_features)

    if overlap:
        visualize_overlap(synthesized_features, analyzed_features, args.sequence_length)

    return model, synthesized_features, analyzed_features


def visualize_generators(synthesized_features, analyzed_features, relevant_features):
    """Visualize feature generators"""
    num_features = len(synthesized_features)
    assert num_features == len(analyzed_features)
    assert all([synthesized_features[fid].name == analyzed_features[fid].name for fid in range(num_features)])
    for fid in range(num_features):
        sfeature = synthesized_features[fid]
        afeature = analyzed_features[fid]
        print(f"Feature {fid}:\nGenerator:")
        display(sfeature.generator.graph())
        relevant = (fid in relevant_features)
        print(f"Feature relevant: {relevant}")
        print(f"Feature function: {sfeature.aggregation_fn}")
        print("Analysis:")
        print(f"Feature overall important: {afeature.important}")
        print(f"Feature overall importance identified correctly: {afeature.important == relevant}")
        print(f"Feature temporally important: {afeature.ordering_important}")
        print(f"Feature window: {afeature.temporal_window}")
        print(f"Complete analysis: {afeature}\n**************************\n")


def visualize_scores(sfeatures, afeatures):
    """Visualize feature importance scores"""
    visualize_scores_aux(sfeatures, afeatures, OVERALL)
    visualize_scores_aux(sfeatures, afeatures, WINDOW)


def visualize_scores_aux(sfeatures, afeatures, seq_type):
    """Visualize feature importance scores"""
    # pylint: disable = invalid-name
    sns.set(rc={'figure.figsize': (24, 14), 'figure.dpi': 300,
                'font.family': 'Serif', 'font.serif': 'Palatino',
                'legend.fontsize': 20, 'legend.title_fontsize': 24,
                'axes.titlesize': 30, 'axes.labelsize': 24, 'xtick.labelsize': 16, 'ytick.labelsize': 24})

    # First sort features by decreasing order of ground truth effect
    sfeatures, afeatures = zip(*sorted(zip(sfeatures, afeatures),
                                       key=lambda pair: (pair[0].effect_size,
                                                         pair[1].overall_effect_size if seq_type == OVERALL else pair[1].window_effect_size),
                                       reverse=True))

    num_features = len(afeatures)
    seffects = [sfeature.effect_size for sfeature in sfeatures]
    aeffects = [afeature.overall_effect_size if seq_type == OVERALL else afeature.window_effect_size for afeature in afeatures]
    # ids = np.arange(num_features)
    # ax = sns.barplot(x=ids, y={"Ground truth": seffects, "Inferred": aeffects})
    # ax = sns.barplot(x="Feature", y="Effect size", ["Ground truth", "Inferred"],
    #                  data={"Feature": ids, "Effect size": {}"Ground truth": seffects, "Inferred": aeffects})
    # ax = sns.barplot(x="feature", y="effectsize", data=dict(feature=ids, effectsize=aeffects))
    df = pd.DataFrame(columns=["Feature", "Effect size", "Source"])
    for idx in range(num_features):
        df = df.append({"Feature": f"x_{sfeatures[idx].name}", "Effect size": seffects[idx], "Source": "Ground truth"}, ignore_index=True)
        df = df.append({"Feature": f"x_{sfeatures[idx].name}", "Effect size": aeffects[idx], "Source": "Inferred"}, ignore_index=True)
    plt.figure()
    ax = sns.barplot(x="Feature", y="Effect size", hue="Source", data=df)
    ax.set_yscale("log")
    ax.set_ylabel("Scores")
    ax.set_xlabel("Features")
    ax.set_title(f"{seq_type} feature importance scores")


def visualize_overlap(sfeatures, afeatures, seq_length):
    """Visualize window overlap with ground truth model"""
    # pylint: disable = invalid-name, too-many-locals
    # First sort features by decreasing order of ground truth effect
    sfeatures, afeatures = zip(*sorted(zip(sfeatures, afeatures),
                                       key=lambda pair: (pair[0].effect_size, pair[1].window_effect_size), reverse=True))
    max_idx = max(max(np.argwhere([sfeature.effect_size > 0 for sfeature in sfeatures])),
                  max(np.argwhere([afeature.window_effect_size > 0 for afeature in afeatures])))[0]

    sns.set(rc={'figure.figsize': (20, max_idx), 'figure.dpi': 300,
                'font.family': 'Serif', 'font.serif': 'Palatino',
                'axes.labelsize': 24, 'xtick.labelsize': 20, 'ytick.labelsize': 20})

    sfeatures = sfeatures[:max_idx]
    afeatures = afeatures[:max_idx]
    num_features = len(afeatures)
    data = np.zeros((3 * num_features - 1, seq_length))
    hatchdata = np.zeros((3 * num_features - 1, seq_length))
    spacing = np.zeros((3 * num_features - 1, seq_length))
    labels = [""] * (3 * num_features - 1)
    for idx, feature in enumerate(afeatures):
        # Ground truth values
        labels[3 * idx] = f"$\\mathbf{{x_{{{feature.name}}}}}$"
        sfeature = sfeatures[idx]
        if sfeature.effect_size != 0:
            # Relevant feature
            left, right = sfeature.window
            data[3 * idx, left: right + 1] = sfeature.effect_size
            hatchdata[3 * idx, left: right + 1] = sfeature.window_ordering_important
        # Inferred values
        labels[3 * idx + 1] = ""
        if feature.temporal_window:
            left, right = feature.temporal_window
            data[3 * idx + 1, left: right + 1] = max(0, feature.window_effect_size)
        if feature.window_ordering_important:
            hatchdata[3 * idx + 1, left: right + 1] = 1
        # Spacing
        if idx < num_features - 1:
            spacing[3 * idx + 2, :] = 1
    plt.figure()
    ax = sns.heatmap(data,
                     mask=(data == 0), linewidth=2, linecolor="black", cmap="YlOrRd",
                     norm=LogNorm(vmin=data.min(), vmax=data.max()),
                     cbar_kws=dict(label="Importance Score"))
    # Border lines
    ax.axhline(y=0, color='black', linewidth=4)
    ax.axhline(y=3 * num_features - 1, color='black', linewidth=4)
    ax.axvline(x=0, color='black', linewidth=4)
    ax.axvline(x=seq_length, color='black', linewidth=4)
    # Labels
    ax.set_xlabel("Timesteps")
    xticks = np.arange(1, seq_length + 1, 2)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontdict=dict(horizontalalignment="right"), rotation="horizontal")
    ax.set_ylabel("Features")
    yticks = np.arange(1, 3 * max_idx, 1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(labels, rotation="horizontal")
    # Hatch texture for ordering relevance
    x = np.arange(seq_length + 1)
    y = np.arange(3 * num_features)
    z = np.ma.masked_equal(hatchdata, 0)
    ax.pcolor(x, y, z, hatch='//', alpha=0.)
    # Grey foreground for non-relevant timesteps
    z = np.ma.masked_not_equal(data, 0)
    ax.pcolor(x, y, z, cmap="Greys", linewidth=2, edgecolors="Grey")
    # White foreground for spacing between features
    z = np.ma.masked_not_equal(spacing, 1)
    ax.pcolor(x, y, z, cmap="binary", linewidth=2)
    # Fix edges of relevant timesteps
    z = np.ma.masked_equal((data != 0), 0)
    ax.pcolor(x, y, z, linewidth=2, edgecolor="k", facecolor="none", alpha=1.0)


def redundant_code(analyzed_features):
    """Redundant"""
    # pylint: disable = unused-variable, line-too-long, invalid-name
    important_features = [feature for feature in analyzed_features if feature.important]
    temporal_features = [feature for feature in analyzed_features if feature.ordering_important]
    windowed_features = [feature for feature in analyzed_features if feature.temporal_window is not None]

    temporally_important_features = [feature for feature in important_features if feature.ordering_important]

    windowed_features = [feature for feature in analyzed_features if feature.temporal_window is not None]  # noqa: F841
    windowed_important_features = [feature for feature in important_features if feature.temporal_window is not None]  # noqa: F841
    windowed_temporal_features = [feature for feature in temporal_features if feature.temporal_window is not None]  # noqa: F841
    windowed_temporally_important_features = [feature for feature in temporally_important_features if feature.temporal_window is not None]  # noqa: F841
