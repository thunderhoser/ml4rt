"""Custom loss functions for Keras models."""

import os
import sys
import keras.backend as K

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking


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


def scaled_mse_for_net_fluxes(scaling_factor, down_flux_indices,
                              up_flux_indices):
    """Scaled MSE for net fluxes, both shortwave and longwave.

    B = number of wavelength bands simulated.  The shortwave is one band, and
        the longwave is another.

    :param scaling_factor: Scaling factor.
    :param down_flux_indices: length-B numpy array with channel indices for
        surface downwelling flux.
    :param up_flux_indices: length-B numpy array with channel indices for
        top-of-atmosphere upwelling flux.
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_greater(scaling_factor, 0.)
    error_checking.assert_is_numpy_array(down_flux_indices, num_dimensions=1)
    error_checking.assert_is_integer_numpy_array(down_flux_indices)
    error_checking.assert_is_geq_numpy_array(down_flux_indices, 0)
    error_checking.assert_is_numpy_array(up_flux_indices, num_dimensions=1)
    error_checking.assert_is_integer_numpy_array(up_flux_indices)
    error_checking.assert_is_geq_numpy_array(up_flux_indices, 0)

    def loss(target_tensor, prediction_tensor):
        """Computes loss (scaled MSE for net fluxex).

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: loss: Scaled MSE for net fluxes.
        """

        predicted_net_flux_tensor = (
            prediction_tensor[..., down_flux_indices] -
            prediction_tensor[..., up_flux_indices]
        )
        target_net_flux_tensor = (
            target_tensor[..., down_flux_indices] -
            target_tensor[..., up_flux_indices]
        )

        net_flux_term = K.mean(
            (predicted_net_flux_tensor - target_net_flux_tensor) ** 2
        )
        individual_flux_term = K.mean((prediction_tensor - target_tensor) ** 2)

        return scaling_factor * (net_flux_term + 2 * individual_flux_term)

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


def joined_output_loss(num_heights, flux_scaling_factor):
    """Loss function for joined output.

    This loss function is dual-weighted MSE for heating rates plus scaled MSE
    for fluxes.

    :param num_heights: Number of heights in grid.
    :param flux_scaling_factor: Scaling factor for flux errors.
    :return: loss: Loss function (defined below).
    """

    def loss(target_tensor, prediction_tensor):
        """Computes loss for joined output.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: loss: Scalar.
        """

        target_hr_tensor = target_tensor[..., :num_heights]
        predicted_hr_tensor = prediction_tensor[..., :num_heights]

        first_term = K.mean(
            K.maximum(K.abs(target_hr_tensor), K.abs(predicted_hr_tensor))
            * (predicted_hr_tensor - target_hr_tensor) ** 2
        )

        predicted_net_flux_tensor = (
            prediction_tensor[..., -2] - prediction_tensor[..., -1]
        )
        target_net_flux_tensor = target_tensor[..., -2] - target_tensor[..., -1]

        net_flux_term = K.mean(
            (predicted_net_flux_tensor - target_net_flux_tensor) ** 2
        )
        individual_flux_term = K.mean(
            (prediction_tensor[..., -2:] - target_tensor[..., -2:]) ** 2
        )
        second_term = (
            flux_scaling_factor * (net_flux_term + 2 * individual_flux_term)
        )

        return first_term + second_term

    return loss
