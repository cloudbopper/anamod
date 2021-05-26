"""Wrappers around different explainers"""
from abc import ABC

import cloudpickle
import numpy as np
import lime
import lime.lime_tabular
import sage

from anamod.core.constants import FEATURE_IMPORTANCE, BINARY_CROSS_ENTROPY
from anamod import ModelAnalyzer, TemporalModelAnalyzer


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
        loss_function = kwargs["loss_function"]
        self.analyzer = TemporalModelAnalyzer(predict, data, targets, output_dir=output_dir, loss_function=loss_function)

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


class OcclusionZeroExplainer(TemporalExplainer):
    """Feature occlusion over time using zeros"""
    def __init__(self, predict, data, **_kwargs):
        super().__init__(predict, data)

    def explain(self):
        scores = np.zeros((self.num_features, self.num_timesteps))
        pred1 = self.predict(self.data)
        for fidx in range(self.num_features):
            for tidx in range(self.num_timesteps):
                back = self.data[:, fidx, tidx]
                self.data[:, fidx, tidx] = 0
                pred2 = self.predict(self.data)
                scores[fidx][tidx] = np.mean(np.abs(pred1 - pred2))
                self.data[:, fidx, tidx] = back
        return scores


class OcclusionUniformExplainer(TemporalExplainer):
    """Feature occlusion over time using uniform samples, based on FIT, Tonekaboni et al. (2020)"""
    def __init__(self, predict, data, **kwargs):
        super().__init__(predict, data)
        self.num_samples = kwargs.get("num_samples")
        self.rng = np.random.default_rng(kwargs.get("rng_seed"))

    def explain(self):
        scores = np.zeros((self.num_features, self.num_timesteps))
        pred1 = self.predict(self.data)
        for fidx in range(self.num_features):
            for tidx in range(self.num_timesteps):
                back = self.data[:, fidx, tidx]
                for _ in range(self.num_samples):
                    self.data[:, fidx, tidx] = self.rng.uniform(-3, 3, size=self.num_instances)
                    pred2 = self.predict(self.data)
                    scores[fidx][tidx] += np.mean(np.abs(pred1 - pred2))
                self.data[:, fidx, tidx] = back
                scores[fidx][tidx] /= self.num_samples
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


class PermutationTestExplainer(TabularExplainer):
    """Permutation test explainer"""
    def __init__(self, predict, data, **kwargs):
        super().__init__(predict, data)
        targets = kwargs["targets"]
        output_dir = kwargs["output_dir"]
        loss_function = kwargs["loss_function"]
        self.analyzer = ModelAnalyzer(self.predict, self.data, targets, output_dir=output_dir, loss_function=loss_function)

    def explain(self):
        features = self.analyzer.analyze()
        scores = np.zeros(self.num_features_tabular)
        for idx, feature in enumerate(features):
            scores[idx] = feature.effect_size
        return scores.reshape((self.num_features, self.num_timesteps), order="C")


class PermutationTestExplainerFDRControl(TabularExplainer):
    """Permutation test explainer using FDR control - requires previously generated results from PermutationTestExplainer"""
    def __init__(self, predict, data, **kwargs):
        super().__init__(predict, data)
        self.base_explainer_dir = kwargs["base_explainer_dir"]

    def explain(self):
        features_filename = f"{self.base_explainer_dir}/{FEATURE_IMPORTANCE}.cpkl"
        with open(features_filename, "rb") as features_file:
            features = cloudpickle.load(features_file)
            scores = np.zeros(self.num_features_tabular)
            for idx, feature in enumerate(features):
                scores[idx] = feature.effect_size if feature.important else 0
            return scores.reshape((self.num_features, self.num_timesteps), order="C")


class LimeExplainer(TabularExplainer):
    """LIME explainer"""
    def __init__(self, predict, data, **kwargs):
        super().__init__(predict, data)
        self.feature_names = kwargs["feature_names"]
        mode = kwargs.get("mode")
        if mode == "classification":
            self.predict = self.predict_tabular_classification(predict)
        self.explainer = lime.lime_tabular.LimeTabularExplainer(self.data, feature_names=self.feature_names, mode=mode,
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
            fidx = -1
            for feature_name, score in lime_exp.as_list():
                _, tidx = feature_name.split("_")
                tidx = int(tidx)
                fidx += 1 if tidx == 0 else 0
                scores[fidx][tidx] += np.abs(score)
        scores /= self.num_instances
        return scores


# SAGE explainer crashes pylint 2.8.2 with InconsistentMroError: https://github.com/PyCQA/pylint/issues/2188
# Remove SAGE block to enable linter for other code
# pylint: disable = all
class SageExplainer(TabularExplainer):
    """SAGE explainer"""
    def __init__(self, predict, data, **kwargs):
        super().__init__(predict, data)
        loss_fn = kwargs["loss_fn"]
        self.targets = kwargs["targets"]
        imputer = self.get_imputer()
        self.estimator = sage.PermutationEstimator(imputer, loss_fn)

    def get_imputer(self):
        """Get SAGE data imputer"""
        return sage.MarginalImputer(self.predict, self.data[:128])

    def explain(self):
        sage_values = self.estimator(self.data, self.targets, batch_size=512, thresh=0.05)
        return sage_values.values.reshape((self.num_features, self.num_timesteps), order="C")


class SageExplainerMeanImputer(SageExplainer):
    """SAGE explainer using default imputation using data mean (fast but lower accuracy)"""
    def get_imputer(self):
        return sage.DefaultImputer(self.predict, np.mean(self.data, axis=0))


class SageExplainerZeroImputer(SageExplainer):
    """SAGE explainer using default imputation using zeros (fast but lower accuracy)"""
    def get_imputer(self):
        return sage.DefaultImputer(self.predict, np.zeros(self.num_features_tabular))


class CXPlainExplainer(TabularExplainer):
    """CXPlain explainer"""
    class Model:
        """Model wrapper - CXPlain requires model object with predict function"""
        def __init__(self, pred_fn):
            self.pred_fn = pred_fn

        def predict(self, X):  # pylint: disable = invalid-name
            """Predict output on X"""
            return self.pred_fn(X)

    def __init__(self, predict, data, **kwargs):
        super().__init__(predict, data)
        # Only import tensorflow if needed
        # pylint: disable = import-outside-toplevel, no-name-in-module
        import tensorflow as tf
        from tensorflow.python.keras.losses import binary_crossentropy, mean_squared_error
        from cxplain import MLPModelBuilder, ZeroMasking, CXPlain
        tf.compat.v1.disable_v2_behavior()
        model_builder = MLPModelBuilder(num_layers=2, num_units=64, batch_size=256, learning_rate=0.001)
        masking_operation = ZeroMasking()
        loss_fn = binary_crossentropy if kwargs["loss_fn"] == BINARY_CROSS_ENTROPY else mean_squared_error
        self.targets = kwargs["targets"]
        self.explainer = CXPlain(self.Model(self.predict), model_builder, masking_operation, loss_fn)

    def explain(self):
        self.explainer.fit(self.data, self.targets)
        attributions = self.explainer.explain(self.data)
        return np.sum(attributions, axis=0).reshape((self.num_features, self.num_timesteps), order="C")
