"""Functions to visualize data"""

from functools import reduce

import cloudpickle
from IPython.display import display
import plotly.graph_objects as go

from anamod.constants import FDR, POWER, AVERAGE_WINDOW_FDR, AVERAGE_WINDOW_POWER
from anamod.constants import ORDERING_ALL_IMPORTANT_FDR, ORDERING_ALL_IMPORTANT_POWER
from anamod.constants import ORDERING_IDENTIFIED_IMPORTANT_FDR, ORDERING_IDENTIFIED_IMPORTANT_POWER
from anamod.constants import TEMPORAL, WINDOW_OVERLAP, RESULTS, CONFIG
from anamod.constants import MODEL_FILENAME, SYNTHESIZED_FEATURES_FILENAME, ANALYZED_FEATURES_FILENAME
from anamod.constants import WINDOW_IMPORTANT_FDR, WINDOW_IMPORTANT_POWER, WINDOW_ORDERING_IMPORTANT_FDR, WINDOW_ORDERING_IMPORTANT_POWER

GROUPS = {"Overall Feature Importance Detection": (FDR, POWER),
          "Ordering Detection (w.r.t. All Important Features)": (ORDERING_ALL_IMPORTANT_FDR, ORDERING_ALL_IMPORTANT_POWER),
          "Ordering Detection (w.r.t. Identified Important Features)": (ORDERING_IDENTIFIED_IMPORTANT_FDR, ORDERING_IDENTIFIED_IMPORTANT_POWER),
          "Average Window Detection": (AVERAGE_WINDOW_FDR, AVERAGE_WINDOW_POWER),
          "Window Importance Detection (w.r.t Identified Important Features)": (WINDOW_IMPORTANT_FDR, WINDOW_IMPORTANT_POWER),
          "Window Ordering Detection (w.r.t Identified Important Features)": (WINDOW_ORDERING_IMPORTANT_FDR, WINDOW_ORDERING_IMPORTANT_POWER)}

LEGACY_GROUPS = {"Overall Feature Importance Detection": (FDR, POWER),
                 "Temporal Feature Importance Detection": ("Temporal_FDR", "Temporal_Power"),
                 "Average Window Detection": (AVERAGE_WINDOW_FDR, AVERAGE_WINDOW_POWER)}


BOX = "box"
VIOLIN = "violin"

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
    for name, group in groups.items():  # Overall, Ordering, Window
        fig = go.Figure()
        for cat in group:  # FDR, Power
            x, y = ([], [])
            # Add all values to the same list y, and corresponding param names in x (used to split by param)
            for param, values in results[cat].items():
                y.extend(values)
                param_name = get_param_name(param)
                x.extend([param_name] * len(values))
            if plot_type == BOX:
                fig.add_trace(go.Box(x=x, y=y, hovertext=hovertext, legendgroup=cat, name=cat))
            else:
                fig.add_trace(go.Violin(x=x, y=y, hovertext=hovertext,
                                        legendgroup=cat, scalegroup=cat, name=cat))
        layout(fig, title=name, xaxis_title=trial_type, yaxis_title="Value", plot_type=plot_type)
        fig.show()


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


def visualize_simulation(sim_dir=".", plot=True):
    """Detailed analysis of given simulation"""
    # pylint: disable = unused-variable, invalid-name, too-many-locals, line-too-long
    # TODO: Make compatible with hierarchical analysis
    # TODO: Clean up after finalizing what to use
    with open(f"{sim_dir}/{MODEL_FILENAME}", "rb") as model_file:
        model = cloudpickle.load(model_file)
    with open(f"{sim_dir}/{SYNTHESIZED_FEATURES_FILENAME}", "rb") as features_file:
        synthesized_features = cloudpickle.load(features_file)
    with open(f"{sim_dir}/{ANALYZED_FEATURES_FILENAME}", "rb") as features_file:
        analyzed_features = cloudpickle.load(features_file)

    relevant_features = reduce(set.union, model.relevant_feature_map, set())
    important_features = [feature for feature in analyzed_features if feature.important]
    temporal_features = [feature for feature in analyzed_features if feature.ordering_important]
    windowed_features = [feature for feature in analyzed_features if feature.temporal_window is not None]

    temporally_important_features = [feature for feature in important_features if feature.ordering_important]

    windowed_features = [feature for feature in analyzed_features if feature.temporal_window is not None]  # noqa: F841
    windowed_important_features = [feature for feature in important_features if feature.temporal_window is not None]  # noqa: F841
    windowed_temporal_features = [feature for feature in temporal_features if feature.temporal_window is not None]  # noqa: F841
    windowed_temporally_important_features = [feature for feature in temporally_important_features if feature.temporal_window is not None]  # noqa: F841

    if plot:
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
            print("Analysis:")
            print(f"Feature overall important: {afeature.important}")
            print(f"Feature overall importance identified correctly: {afeature.important == relevant}")
            print(f"Feature temporally important: {afeature.ordering_important}")
            print(f"Feature window: {afeature.temporal_window}")
            print(f"Complete analysis: {afeature}\n**************************\n")
    return model, synthesized_features, analyzed_features
