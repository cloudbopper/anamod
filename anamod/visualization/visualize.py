"""Functions to visualize data"""

from functools import reduce

import cloudpickle
from IPython.display import display
import plotly.graph_objects as go

from anamod.constants import FDR, POWER, TEMPORAL_FDR, TEMPORAL_POWER, AVERAGE_WINDOW_FDR, AVERAGE_WINDOW_POWER
from anamod.constants import TEMPORAL, WINDOW_OVERLAP, RESULTS, CONFIG, MODEL
from anamod.constants import MODEL_FILENAME, SYNTHESIZED_FEATURES_FILENAME, ANALYZED_FEATURES_FILENAME


GROUPS = {"Overall Feature Importance Detection": (FDR, POWER),
          "Temporal Feature Importance Detection": (TEMPORAL_FDR, TEMPORAL_POWER),
          "Average Window Detection": (AVERAGE_WINDOW_FDR, AVERAGE_WINDOW_POWER)}


def flatten(nested_list):
    """Flatten nested list"""
    return [item for sub_list in nested_list for item in sub_list]


def visualize_analysis(data, trial_type, analysis_type=TEMPORAL):
    """Visualize outputs"""
    # pylint: disable = invalid-name
    # TODO: Document a bit better
    # TODO: First, write out a summary of the setup
    results = data[RESULTS]
    output_dirs = flatten(data[CONFIG]["output_dir"].values())
    models = flatten(data[MODEL]["operation"].values())
    hovertext = list(zip(output_dirs, models))  # visible when hovering over any given point (i.e. simulation)
    if analysis_type == TEMPORAL:
        fig = go.Figure()
        x, y = ([], [])
        for param, values in results[WINDOW_OVERLAP].items():  # len(values) = num_trials
            y.extend(values)
            param_name = f"n = {param}" if param.isdigit() else param
            x.extend([param_name] * len(values))
        fig.add_trace(go.Violin(x=x, y=y, hovertext=hovertext))
        layout(fig, title="Average Window Overlap", xaxis_title=trial_type, yaxis_title="Average Overlap")
        fig.show()
    for name, group in GROUPS.items():  # Overall, Temporal, Window
        fig = go.Figure()
        for cat in group:  # FDR, Power
            x, y = ([], [])
            # Add all values to the same list y, and corresponding param names in x (used to split by param)
            for param, values in results[cat].items():
                y.extend(values)
                param_name = f"n = {param}" if param.isdigit() else param
                x.extend([param_name] * len(values))
            fig.add_trace(go.Violin(x=x, y=y, hovertext=hovertext,
                                    legendgroup=cat, scalegroup=cat, name=cat))
        layout(fig, title=name, xaxis_title=trial_type, yaxis_title="Value")
        fig.show()


def layout(fig, title="", xaxis_title="", yaxis_title=""):
    """Perform common changes to violin plot layout"""
    fig.update_traces(box_visible=True, meanline_visible=True, opacity=0.6, points="all")
    fig.update_layout(title={"text": title, "xanchor": "center", "x": 0.5},
                        xaxis_title=xaxis_title, yaxis_title=yaxis_title,
                        violinmode="group", template="none")


def visualize_simulation(sim_dir="."):
    """Detailed analysis of given simulation"""
    # pylint: disable = unused-variable, invalid-name, too-many-locals
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
    temporal_features = [feature for feature in analyzed_features if feature.temporally_important]
    windowed_features = [feature for feature in analyzed_features if feature.temporal_window is not None]

    temporally_important_features = [feature for feature in important_features if feature.temporally_important]

    windowed_features = [feature for feature in analyzed_features if feature.temporal_window is not None]
    windowed_important_features = [feature for feature in important_features if feature.temporal_window is not None]
    windowed_temporal_features = [feature for feature in temporal_features if feature.temporal_window is not None]
    windowed_temporally_important_features = [feature for feature in temporally_important_features if feature.temporal_window is not None]

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
        print(f"Feature temporally important: {afeature.temporally_important}")
        print(f"Feature window: {afeature.temporal_window}")
        print(f"Complete analysis: {afeature}\n**************************\n")
    return model, synthesized_features, analyzed_features
