"""Loss functions"""

from abc import ABC

import numpy as np

from anamod.core import constants

TARGET_VALUES = {constants.LABELS, constants.BASELINE_PREDICTIONS}


class LossFunction(ABC):
    """Loss function base class"""
    @staticmethod
    def loss(y_true, y_pred):
        """Return vector of losses given true and predicted model values over a list of instances"""


class QuadraticLoss(LossFunction):
    """Quadratic loss function"""
    @staticmethod
    def loss(y_true, y_pred):
        return (y_true - y_pred)**2


class AbsoluteDifferenceLoss(LossFunction):
    """Absolute difference loss function (like quadratic loss, but scales differently)"""
    @staticmethod
    def loss(y_true, y_pred):
        return np.abs(y_true - y_pred)


class ZeroOneLoss(LossFunction):
    """0-1 loss"""
    @staticmethod
    def loss(y_true, y_pred):
        y_true = (y_true > 0.5)
        y_pred = (y_pred > 0.5)
        return (y_true != y_pred).astype(np.int32)


class BinaryCrossEntropy(LossFunction):
    """Binary cross-entropy"""
    @staticmethod
    def loss(y_true, y_pred):
        assert all(y_pred >= 0) and all(y_pred <= 1)
        losses = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
        losses[np.isnan(losses)] = 0  # to handle indeterminate case where y_pred components are zero
        return losses


LOSS_FUNCTIONS = {constants.QUADRATIC_LOSS: QuadraticLoss,
                  constants.ABSOLUTE_DIFFERENCE_LOSS: AbsoluteDifferenceLoss,
                  constants.BINARY_CROSS_ENTROPY: BinaryCrossEntropy,
                  constants.ZERO_ONE_LOSS: ZeroOneLoss}


class Loss():
    """Compute losses given true and predicted model values over a list of instances"""
    def __init__(self, loss_function, targets):
        self._loss_fn = LOSS_FUNCTIONS[loss_function].loss
        self._targets = targets

    def loss_fn(self, predictions):
        """Return loss vector"""
        return self._loss_fn(self._targets, predictions)
