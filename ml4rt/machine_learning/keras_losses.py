"""Custom loss functions for Keras models."""

import numpy
import keras.backend as K
from gewittergefahr.gg_utils import error_checking
from ml4rt.utils import example_utils


def scaled_mse(scaling_factor):
    """Scaled MSE (mean squared error).

    :param scaling_factor: Scaling factor.
    :return: loss: Loss function (defined below).
    """

    def loss(target_tensor, prediction_tensor):
        """Computes loss (scaled MSE).

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: loss: Scaled MSE.
        """

        return scaling_factor * K.mean((prediction_tensor - target_tensor) ** 2)

    return loss


def scaled_mse_for_net_flux(scaling_factor):
    """Scaled MSE (mean squared error) for net flux.

    This method expects two channels: surface downwelling flux and
    top-of-atmosphere (TOA) upwelling flux.  This method penalizes only errors
    in the net flux, which is surface downwelling minus TOA upwelling.

    :param scaling_factor: Scaling factor.
    :return: loss: Loss function (defined below).
    """

    def loss(target_tensor, prediction_tensor):
        """Computes loss (scaled MSE for net flux).

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: loss: Scaled MSE for net flux.
        """

        predicted_net_flux_tensor = (
            prediction_tensor[..., 0] - prediction_tensor[..., 1]
        )
        target_net_flux_tensor = target_tensor[..., 0] - target_tensor[..., 1]

        net_flux_term = K.mean(
            (predicted_net_flux_tensor - target_net_flux_tensor) ** 2
        )
        individual_flux_term = K.mean((prediction_tensor - target_tensor) ** 2)

        return scaling_factor * (net_flux_term + 2 * individual_flux_term)

    return loss


def weighted_mse():
    """Weighted MSE (mean squared error).

    Weight = magnitude of target value.

    :return: loss: Loss function (defined below).
    """

    # TODO(thunderhoser): Maybe apply weights only to heating rate?

    def loss(target_tensor, prediction_tensor):
        """Computes loss (weighted MSE).

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: loss: Weighted MSE.
        """

        return K.mean(target_tensor * (prediction_tensor - target_tensor) ** 2)

    return loss


def dual_weighted_mse(use_lowest_n_heights=None):
    """Dual-weighted MSE (mean squared error).

    Weight = max(magnitude of target value, magnitude of predicted value).

    :param use_lowest_n_heights: Will use this number of heights in the loss
        function, starting at the bottom.  If you want to penalize predictions
        at all heights, make this None.
    :return: loss: Loss function (defined below).
    """

    def loss(target_tensor, prediction_tensor):
        """Computes loss (dual-weighted MSE).

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: loss: Dual-weighted MSE.
        """

        if use_lowest_n_heights is None:
            return K.mean(
                K.maximum(K.abs(target_tensor), K.abs(prediction_tensor)) *
                (prediction_tensor - target_tensor) ** 2
            )

        this_target_tensor = target_tensor[..., :use_lowest_n_heights, :]
        this_prediction_tensor = (
            prediction_tensor[..., :use_lowest_n_heights, :]
        )

        return K.mean(
            K.maximum(K.abs(this_target_tensor), K.abs(this_prediction_tensor))
            * (this_prediction_tensor - this_target_tensor) ** 2
        )

    return loss


def dual_sqrt_weighted_mse():
    """Same as dual-weighted MSE, except that the weight is a square root.

    Weight = sqrt(max(magnitude of target value, magnitude of predicted value))

    :return: loss: Loss function (defined below).
    """

    def loss(target_tensor, prediction_tensor):
        """Computes loss (dual-sqrt-weighted MSE).

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: loss: Dual-sqrt-weighted MSE.
        """

        return K.mean(
            K.sqrt(K.maximum(K.abs(target_tensor), K.abs(prediction_tensor))) *
            (prediction_tensor - target_tensor) ** 2
        )

    return loss


def dual_weighted_mse_equalize_heights(num_examples_per_batch, num_channels):
    """Dual-weighted MSE with equalized heights.

    Each height should have an equal contribution to this loss function.

    :param num_examples_per_batch: Number of examples per batch.
    :param num_channels: Number of channels (target variables).
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_integer(num_examples_per_batch)
    error_checking.assert_is_greater(num_examples_per_batch, 0)
    error_checking.assert_is_integer(num_channels)
    error_checking.assert_is_greater(num_channels, 0)

    def loss(target_tensor, prediction_tensor):
        """Computes loss (dual-weighted MSE with equalized heights).

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: loss: Dual-weighted MSE with equalized heights.
        """

        max_target_tensor = K.max(target_tensor, axis=(0, -1), keepdims=True)
        max_target_tensor = K.clip(
            max_target_tensor, min_value=1e-6, max_value=1e6
        )
        # max_target_tensor = K.repeat_elements(
        #     max_target_tensor, rep=num_examples_per_batch, axis=0
        # )
        # max_target_tensor = K.repeat_elements(
        #     max_target_tensor, rep=num_channels, axis=-1
        # )

        norm_target_tensor = target_tensor / max_target_tensor
        norm_prediction_tensor = prediction_tensor / max_target_tensor

        return K.mean(
            K.maximum(norm_target_tensor, norm_prediction_tensor) *
            (norm_prediction_tensor - norm_target_tensor) ** 2
        )

    return loss


def negative_mse_skill_score():
    """Negative MSE (mean squared error) skill score.

    :return: loss: Loss function (defined below).
    """

    def loss(target_tensor, prediction_tensor):
        """Computes loss (negative MSE skill score).

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: loss: Negative MSE skill score.
        """

        mean_target_tensor = K.mean(target_tensor, axis=0, keepdims=True)

        mse_actual = K.mean((prediction_tensor - target_tensor) ** 2)
        mse_climo = K.mean((prediction_tensor - mean_target_tensor) ** 2)
        return (mse_actual - mse_climo) / mse_climo

    return loss
