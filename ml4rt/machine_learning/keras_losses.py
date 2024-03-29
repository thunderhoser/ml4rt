"""Custom loss functions for Keras models."""

import numpy
import keras.backend as K
from gewittergefahr.gg_utils import error_checking


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


def dual_weighted_mse(
        use_lowest_n_heights=None, heating_rate_weight_exponent=1.,
        height_weighting_type_string=None):
    """Dual-weighted MSE (mean squared error).

    Weight = max(magnitude of target value, magnitude of predicted value).

    :param use_lowest_n_heights: Will use this number of heights in the loss
        function, starting at the bottom.  If you want to penalize predictions
        at all heights, make this None.
    :param heating_rate_weight_exponent: Exponent on heating-rate weight.
    :param height_weighting_type_string: Type of weighting for height level
        (integer from 1...H, where H = number of heights).  Options are
        "linear", where the weight at the bottom [top] grid level is H [1];
        "log2", where the weight at the bottom [top] grid level is log2(H + 1)
        [log2(2)]; and "log10", where the weight at the bottom [top] grid level
        is log10(H + 9) [log10(10)].
    :return: loss: Loss function (defined below).
    """

    if use_lowest_n_heights is not None:
        error_checking.assert_is_integer(use_lowest_n_heights)
        error_checking.assert_is_greater(use_lowest_n_heights, 0)

    heating_rate_weight_exponent = float(heating_rate_weight_exponent)
    error_checking.assert_is_geq(heating_rate_weight_exponent, 1.)

    if height_weighting_type_string == 'None':
        height_weighting_type_string = None

    if height_weighting_type_string is not None:
        assert height_weighting_type_string in ['linear', 'log2', 'log10']

        num_heights = use_lowest_n_heights
        height_weights = numpy.linspace(
            1, num_heights + 1, num=num_heights, dtype=float
        )[::-1]
        height_weight_matrix = numpy.expand_dims(height_weights, axis=0)
        height_weight_matrix = numpy.expand_dims(height_weights, axis=-1)

        if height_weighting_type_string == 'log2':
            height_weight_matrix = numpy.log2(height_weight_matrix + 1.)
        if height_weighting_type_string == 'log10':
            height_weight_matrix = numpy.log10(height_weight_matrix + 9.)

    def loss(target_tensor, prediction_tensor):
        """Computes loss (dual-weighted MSE).

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: loss: Dual-weighted MSE.
        """

        if use_lowest_n_heights is not None:
            target_tensor = target_tensor[..., :use_lowest_n_heights, :]
            prediction_tensor = prediction_tensor[..., :use_lowest_n_heights, :]

        heating_rate_weight_tensor = K.pow(
            K.maximum(K.abs(target_tensor), K.abs(prediction_tensor)),
            heating_rate_weight_exponent
        )

        error_tensor = (
            heating_rate_weight_tensor *
            (prediction_tensor - target_tensor) ** 2
        )

        if height_weighting_type_string is not None:
            error_tensor = error_tensor * height_weight_matrix

        return K.mean(error_tensor)

    return loss


def dual_weighted_crps():
    """Dual-weighted CRPS (continuous ranked probability score) for htng rates.

    Weight = max(magnitude of target value, magnitude of predicted value).

    :return: loss: Loss function (defined below).
    """

    def loss(target_tensor, prediction_tensor):
        """Computes loss (dual-weighted CRPS).

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: loss: Dual-weighted CRPS.
        """

        weight_tensor = K.maximum(
            K.abs(K.expand_dims(target_tensor, axis=-1)),
            K.abs(prediction_tensor)
        )
        absolute_error_tensor = K.abs(
            prediction_tensor - K.expand_dims(target_tensor, axis=-1)
        )
        mean_prediction_error_tensor = K.mean(
            weight_tensor * absolute_error_tensor, axis=-1
        )

        mean_prediction_diff_tensor = K.map_fn(
            fn=lambda p: K.mean(
                K.maximum(
                    K.abs(K.expand_dims(p, axis=-1)),
                    K.abs(K.expand_dims(p, axis=-2))
                ) *
                K.abs(
                    K.expand_dims(p, axis=-1) -
                    K.expand_dims(p, axis=-2)
                ),
                axis=(-2, -1)
            ),
            elems=prediction_tensor
        )

        # weight_tensor = K.maximum(
        #     K.abs(K.expand_dims(prediction_tensor, axis=-1)),
        #     K.abs(K.expand_dims(prediction_tensor, axis=-2))
        # )
        # prediction_diff_tensor = K.abs(
        #     K.expand_dims(prediction_tensor, axis=-1) -
        #     K.expand_dims(prediction_tensor, axis=-2)
        # )
        # mean_prediction_diff_tensor = K.mean(
        #     weight_tensor * prediction_diff_tensor, axis=(-2, -1)
        # )

        return K.mean(
            mean_prediction_error_tensor - 0.5 * mean_prediction_diff_tensor
        )

    return loss


def unscaled_crps_for_net_flux():
    """Unscaled CRPS (continuous ranked probability score) for net flux.

    This method expects two channels: surface downwelling flux and
    top-of-atmosphere (TOA) upwelling flux.  This method penalizes only errors
    in the net flux, which is surface downwelling minus TOA upwelling.

    :return: loss: Loss function (defined below).
    """

    def loss(target_tensor, prediction_tensor):
        """Computes loss (unscaled CRPS for net flux).

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: loss: Unscaled CRPS for net flux.
        """

        target_net_flux_tensor = target_tensor[..., 0] - target_tensor[..., 1]
        full_target_tensor = K.concatenate((
            target_tensor, K.expand_dims(target_net_flux_tensor, axis=-1)
        ), axis=-1)

        predicted_net_flux_tensor = (
            prediction_tensor[..., 0, :] - prediction_tensor[..., 1, :]
        )
        full_prediction_tensor = K.concatenate((
            prediction_tensor, K.expand_dims(predicted_net_flux_tensor, axis=-2)
        ), axis=-2)

        mean_prediction_error_tensor = K.mean(
            K.abs(
                full_prediction_tensor -
                K.expand_dims(full_target_tensor, axis=-1)
            ),
            axis=-1
        )

        mean_prediction_diff_tensor = K.map_fn(
            fn=lambda p: K.mean(
                K.abs(K.expand_dims(p, axis=-1) - K.expand_dims(p, axis=-2)),
                axis=(-2, -1)
            ),
            elems=full_prediction_tensor
        )

        # prediction_diff_tensor = K.abs(
        #     K.expand_dims(full_prediction_tensor, axis=-1) -
        #     K.expand_dims(full_prediction_tensor, axis=-2)
        # )
        # mean_prediction_diff_tensor = K.mean(
        #     prediction_diff_tensor, axis=(-2, -1)
        # )

        return K.mean(
            mean_prediction_error_tensor - 0.5 * mean_prediction_diff_tensor
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


def dual_weighted_mse_simple():
    """Dual-weighted MSE (mean squared error).

    Weight = max(magnitude of target value, magnitude of predicted value).

    :return: loss: Loss function (defined below).
    """

    def loss(target_tensor, prediction_tensor):
        """Computes loss (dual-weighted MSE).

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: loss: Dual-weighted MSE.
        """

        weight_tensor = K.maximum(K.abs(target_tensor), K.abs(prediction_tensor))
        error_tensor = weight_tensor * (prediction_tensor - target_tensor) ** 2
        return K.mean(error_tensor)

    return loss
