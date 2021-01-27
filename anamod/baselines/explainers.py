"""Wrappers around different explainers"""
from abc import ABC

import numpy as np
import lime
import lime.lime_tabular
import sage
from anamod import TemporalModelAnalyzer

# pylint 2.6.0 errors out while running
# pylint: disable = all


class TemporalExplainer(ABC):
    """Global temporal model explainer base class"""
    def __init__(self, predict, data):
        self.predict = predict
        self.data = data
        self.num_instances, self.num_features, self.num_timesteps = data.shape

    def explain(self):
        """Return global explanation of data as array of (num_features * num_timesteps)"""


class AnamodExplainer(TemporalExplainer):
    """Anamod explainer"""
    def __init__(self, predict, data, **kwargs):
        super().__init__(predict, data)
        targets = kwargs["targets"]
        output_dir = kwargs["output_dir"]
        self.analyzer = TemporalModelAnalyzer(predict, data, targets, output_dir=output_dir)

    def explain(self):
        features = self.analyzer.analyze()
        scores = np.zeros((self.num_features, self.num_timesteps))
        for fidx, feature in enumerate(features):
            if not feature.important:
                continue
            left, right = feature.window
            effect_size = feature.window_effect_size
            if not feature.window_important:
                left, right = (0, self.num_timesteps)
                effect_size = feature.overall_effect_size
            scores[fidx][left: right + 1] = effect_size
        return scores


class TabularExplainer(TemporalExplainer):
    """Tabular model explainer"""
    def __init__(self, predict, data):
        super().__init__(predict, data)
        self.num_features_tabular = self.num_features * self.num_timesteps
        self.data = self.data.reshape((self.num_instances, self.num_features_tabular), order="C")  # tabular representation of temporal data
        self.predict = self.predict_tabular(predict)

    def predict_tabular(self, predict):
        """Return function that performs prediction on tabular data"""
        return lambda data_tabular: predict(data_tabular.reshape((len(data_tabular), self.num_features, self.num_timesteps), order="C"))


class LimeExplainer(TabularExplainer):
    """LIME explainer"""
    def __init__(self, predict, data, **kwargs):
        super().__init__(predict, data)
        feature_names = kwargs["feature_names"]
        mode = kwargs.get("mode")
        if mode == "classification":
            self.predict = self.predict_tabular_classification(predict)
        self.explainer = lime.lime_tabular.LimeTabularExplainer(self.data, feature_names=feature_names, mode=mode,
                                                                class_names=["0", "1"], verbose=False, discretize_continuous=False)

    def predict_tabular_classification(self, predict):
        """Special handling for LIME for classification model"""
        def predict_aux(data_tabular):
            pred1 = predict(data_tabular.reshape((len(data_tabular), self.num_features, self.num_timesteps), order="C"))
            pred0 = 1 - pred1
            return np.column_stack((pred0, pred1))
        return predict_aux

    def explain(self):
        scores = np.zeros((self.num_features, self.num_timesteps))
        for instance in self.data:
            lime_exp = self.explainer.explain_instance(instance, self.predict, num_features=self.num_features_tabular)
            for feature_name, score in lime_exp.as_list():
                fidx, tidx = feature_name.split("_")
                scores[int(fidx)][int(tidx)] += np.abs(score)
        scores /= self.num_instances
        return scores


class SageExplainer(TabularExplainer):
    """SAGE explainer"""
    def __init__(self, predict, data, **kwargs):
        super().__init__(predict, data)
        loss_fn = kwargs["loss_fn"]
        self.targets = kwargs["targets"]
        imputer = sage.MarginalImputer(self.predict, self.data[:128])
        self.estimator = sage.PermutationEstimator(imputer, loss_fn)

    def explain(self):
        sage_values = self.estimator(self.data, self.targets, batch_size=512, thresh=0.05)
        return sage_values.values.reshape((self.num_features, self.num_timesteps), order="C")
