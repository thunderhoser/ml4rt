"""Custom loss functions for Keras models.

Definitions:

E = number of examples
H = number of heights
W = number of wavelengths
T = number of target variables
S = ensemble size
"""

import numpy
from tensorflow.keras import backend as K
from gewittergefahr.gg_utils import error_checking


def scaled_mse_for_net_flux(scaling_factor, band_weights=None):
    """Scaled MSE (mean squared error) for flux variables, including net flux.

    This method assumes a deterministic model and expects two channels: surface
    downwelling flux (SDF) and top-of-atmosphere upwelling flux (TUF).  This
    method penalizes errors in three quantities: SDF, TUF, and net flux (SDF
    minus TUF).

    :param scaling_factor: Scaling factor.
    :param band_weights: length-W numpy array of weights, one for each
        wavelength band.  For the default behaviour (every band weighted the
        same), leave this argument alone.
    :return: loss: Loss function (defined below).
    """

    if band_weights is not None:
        error_checking.assert_is_numpy_array(band_weights, num_dimensions=1)
        error_checking.assert_is_greater_numpy_array(band_weights, 0.)

    def loss(target_tensor, prediction_tensor):
        """Computes loss.

        :param target_tensor: E-by-W-by-T tensor of actual values.
        :param prediction_tensor: E-by-W-by-T tensor of predicted values.
        :return: loss: Scaled MSE for flux variables.
        """

        d = 0
        u = 1

        if band_weights is None:
            net_flux_term = K.mean(K.pow(
                x=(prediction_tensor[..., d] - prediction_tensor[..., u]) -
                (target_tensor[..., d] - target_tensor[..., u]),
                a=2
            ))

            individual_flux_term = K.mean(
                (prediction_tensor - target_tensor) ** 2
            )
        else:
            band_weight_tensor = K.cast(
                K.constant(band_weights), prediction_tensor.dtype
            )
            band_weight_tensor = K.expand_dims(band_weight_tensor, axis=0)
            band_weight_tensor = K.expand_dims(band_weight_tensor, axis=-1)

            net_flux_term = K.mean(
                band_weight_tensor[..., 0] *
                K.pow(
                    x=(prediction_tensor[..., d] - prediction_tensor[..., u]) -
                    (target_tensor[..., d] - target_tensor[..., u]),
                    a=2
                )
            )

            individual_flux_term = K.mean(
                band_weight_tensor * (prediction_tensor - target_tensor) ** 2
            )

        return scaling_factor * (net_flux_term + 2 * individual_flux_term)

    return loss


def scaled_mse_for_net_flux_constrained_bb(scaling_factor, band_weights=None):
    """Scaled MSE (mean squared error) for flux variables, including net flux.

    This method is the same as `scaled_mse_for_net_flux`, except that it
    constrains and penalizes broadband heating rates (broadband = sum over all
    wavelength bands) as well.  Assumes a deterministic model.

    :param scaling_factor: Scaling factor.
    :param band_weights: length-W numpy array of weights, one for each
        wavelength band.  For the default behaviour (every band weighted the
        same), leave this argument alone.
    :return: loss: Loss function (defined below).
    """

    if band_weights is not None:
        error_checking.assert_is_numpy_array(band_weights, num_dimensions=1)
        error_checking.assert_is_greater_numpy_array(band_weights, 0.)

        # Add broadband.
        band_weights = numpy.concatenate([
            band_weights,
            numpy.array([numpy.sum(band_weights)])
        ])

    def loss(target_tensor, prediction_tensor):
        """Computes loss.

        :param target_tensor: E-by-W-by-T tensor of actual values.
        :param prediction_tensor: E-by-W-by-T tensor of predicted values.
        :return: loss: Scaled MSE for flux variables, including broadband.
        """

        # Add broadband.
        target_tensor = K.concatenate([
            target_tensor,
            K.sum(target_tensor, axis=-2, keepdims=True)
        ], axis=-2)

        prediction_tensor = K.concatenate([
            prediction_tensor,
            K.sum(prediction_tensor, axis=-2, keepdims=True)
        ], axis=-2)

        d = 0
        u = 1

        if band_weights is None:
            net_flux_term = K.mean(K.pow(
                x=(prediction_tensor[..., d] - prediction_tensor[..., u]) -
                (target_tensor[..., d] - target_tensor[..., u]),
                a=2
            ))

            individual_flux_term = K.mean(
                (prediction_tensor - target_tensor) ** 2
            )
        else:
            band_weight_tensor = K.cast(
                K.constant(band_weights), prediction_tensor.dtype
            )
            band_weight_tensor = K.expand_dims(band_weight_tensor, axis=0)
            band_weight_tensor = K.expand_dims(band_weight_tensor, axis=-1)

            net_flux_term = K.mean(
                band_weight_tensor[..., 0] *
                K.pow(
                    x=(prediction_tensor[..., d] - prediction_tensor[..., u]) -
                    (target_tensor[..., d] - target_tensor[..., u]),
                    a=2
                )
            )

            individual_flux_term = K.mean(
                band_weight_tensor * (prediction_tensor - target_tensor) ** 2
            )

        return scaling_factor * (net_flux_term + 2 * individual_flux_term)

    return loss


def dual_weighted_mse_fancy(
        use_lowest_n_heights=None, heating_rate_weight_exponent=1.,
        height_weighting_type_string=None):
    """Dual-weighted MSE (mean squared error) for heating rates.

    This method assumes a deterministic model.  The "dual weight" for each data
    point is max(abs(target_heating_rate), abs(predicted_heating_rate)) ** E,
    where E is the input arg `heating_rate_weight_exponent`.

    :param use_lowest_n_heights: Will use this number of heights in the loss
        function, starting at the bottom.  If you want to penalize predictions
        at all heights, make this None.
    :param heating_rate_weight_exponent: See above discussion about the dual
        weight.
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
        height_weight_matrix = numpy.expand_dims(height_weights, axis=-1)

        if height_weighting_type_string == 'log2':
            height_weight_matrix = numpy.log2(height_weight_matrix + 1.)
        if height_weighting_type_string == 'log10':
            height_weight_matrix = numpy.log10(height_weight_matrix + 9.)

    def loss(target_tensor, prediction_tensor):
        """Computes loss.

        :param target_tensor: E-by-H-by-W-by-T tensor of actual values.
        :param prediction_tensor: E-by-H-by-W-by-T tensor of predicted values.
        :return: loss: Dual-weighted MSE for heating rate.
        """

        if use_lowest_n_heights is not None:
            target_tensor = target_tensor[:, :use_lowest_n_heights, ...]
            prediction_tensor = prediction_tensor[:, :use_lowest_n_heights, ...]

        # heating_rate_weight_tensor = K.pow(
        #     x=K.maximum(K.abs(target_tensor), K.abs(prediction_tensor)),
        #     a=heating_rate_weight_exponent
        # )
        #
        # error_tensor = (
        #     heating_rate_weight_tensor *
        #     (prediction_tensor - target_tensor) ** 2
        # )
        #
        # if height_weighting_type_string is not None:
        #     error_tensor = error_tensor * height_weight_matrix

        if height_weighting_type_string is None:
            return K.mean(
                K.pow(
                    x=K.maximum(K.abs(target_tensor), K.abs(prediction_tensor)),
                    a=heating_rate_weight_exponent
                )
                * (prediction_tensor - target_tensor) ** 2
            )

        return K.mean(
            K.pow(
                x=K.maximum(K.abs(target_tensor), K.abs(prediction_tensor)),
                a=heating_rate_weight_exponent
            )
            * height_weight_matrix
            * (prediction_tensor - target_tensor) ** 2
        )

    return loss


def dual_weighted_crps():
    """Dual-weighted CRPS (continuous ranked probability score) for htng rates.

    This method assumes an ensemble model.  The "dual weight" for each data
    point is max(abs(target_heating_rate), abs(predicted_heating_rate)).

    :return: loss: Loss function (defined below).
    """

    def loss(target_tensor, prediction_tensor):
        """Computes loss.

        :param target_tensor: E-by-H-by-W-by-T tensor of actual values.
        :param prediction_tensor: E-by-H-by-W-by-T-by-S tensor of predicted
            values.
        :return: loss: Dual-weighted CRPS for heating rate.
        """

        # E x H x W x T x S
        weight_tensor = K.maximum(
            K.abs(K.expand_dims(target_tensor, axis=-1)),
            K.abs(prediction_tensor)
        )

        # E x H x W x T x S
        absolute_error_tensor = K.abs(
            prediction_tensor - K.expand_dims(target_tensor, axis=-1)
        )

        # E x H x W x T
        mean_prediction_error_tensor = K.mean(
            weight_tensor * absolute_error_tensor, axis=-1
        )

        # E x H x W x T
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
    """Unscaled CRPS for flux variables, including net flux.

    CRPS = continuous ranked probability score

    This method assumes an ensemble model and expects two channels: surface
    downwelling flux (SDF) and top-of-atmosphere upwelling flux (TUF).  This
    method penalizes errors in three quantities: SDF, TUF, and net flux (SDF
    minus TUF).

    :return: loss: Loss function (defined below).
    """

    def loss(target_tensor, prediction_tensor):
        """Computes loss.

        :param target_tensor: E-by-W-by-T tensor of actual values.
        :param prediction_tensor: E-by-W-by-T-by-S tensor of predicted values.
        :return: loss: Unscaled CRPS for flux variables.
        """

        # E x W
        target_net_flux_tensor = target_tensor[..., 0] - target_tensor[..., 1]

        # E x W x T
        full_target_tensor = K.concatenate(
            [target_tensor, K.expand_dims(target_net_flux_tensor, axis=-1)],
            axis=-1
        )

        # E x W x S
        predicted_net_flux_tensor = (
            prediction_tensor[..., 0, :] - prediction_tensor[..., 1, :]
        )

        # E x W x T x S
        full_prediction_tensor = K.concatenate([
            prediction_tensor,
            K.expand_dims(predicted_net_flux_tensor, axis=-2)
        ], axis=-2)

        # E x W x T
        mean_prediction_error_tensor = K.mean(
            K.abs(
                full_prediction_tensor -
                K.expand_dims(full_target_tensor, axis=-1)
            ),
            axis=-1
        )

        # E x W x T
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
    """Loss function for joined output layer.

    A 'joined output layer' contains both heating rates and fluxes.  This method
    assumes a deterministic model.  The loss function computed by this method
    is (dual-weighted MSE for heating rates) +
    flux_scaling_factor * (MSE for flux variables).

    :param num_heights: Number of heights in grid.
    :param flux_scaling_factor: Scaling factor for flux errors.
    :return: loss: Loss function (defined below).
    """

    def loss(target_tensor, prediction_tensor):
        """Computes loss.

        :param target_tensor: E-by-(H + 2)-by-W tensor of actual values.
        :param prediction_tensor: E-by-(H + 2)-by-W tensor of predicted values.
        :return: loss: See above.
        """

        # TODO(thunderhoser): Currently assuming only 1 wavelength.
        # This is a HACK.

        # E x (H + 2)
        target_tensor = target_tensor[..., 0]
        prediction_tensor = prediction_tensor[..., 0]

        # E x H
        target_hr_tensor = target_tensor[..., :num_heights]
        predicted_hr_tensor = prediction_tensor[..., :num_heights]

        # Dual-weighted MSE (scalar)
        first_term = K.mean(
            K.maximum(K.abs(target_hr_tensor), K.abs(predicted_hr_tensor))
            * (predicted_hr_tensor - target_hr_tensor) ** 2
        )

        # length-E
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

        # MSE for flux variables (scalar)
        second_term = (
            flux_scaling_factor * (net_flux_term + 2 * individual_flux_term)
        )

        return first_term + second_term

    return loss


def dual_weighted_mse_simple():
    """Dual-weighted MSE (mean squared error) for heating rates.

    This method assumes a deterministic model.  The "dual weight" for each data
    point is max(abs(target_heating_rate), abs(predicted_heating_rate)).

    :return: loss: Loss function (defined below).
    """

    def loss(target_tensor, prediction_tensor):
        """Computes loss.

        :param target_tensor: E-by-H-by-W-by-T tensor of actual values.
        :param prediction_tensor: E-by-H-by-W-by-T tensor of predicted values.
        :return: loss: Dual-weighted MSE for heating rate.
        """

        return K.mean(
            K.maximum(K.abs(target_tensor), K.abs(prediction_tensor))
            * (prediction_tensor - target_tensor) ** 2
        )

    return loss


def dual_weighted_mse(band_weights=None):
    """Dual-weighted MSE (mean squared error) for heating rates.

    This method assumes a deterministic model.  The "dual weight" for each data
    point is max(abs(target_heating_rate), abs(predicted_heating_rate)).

    :param band_weights: length-W numpy array of weights, one for each
        wavelength band.  For the default behaviour (every band weighted the
        same), leave this argument alone.
    :return: loss: Loss function (defined below).
    """

    if band_weights is not None:
        error_checking.assert_is_numpy_array(band_weights, num_dimensions=1)
        error_checking.assert_is_greater_numpy_array(band_weights, 0.)

    def loss(target_tensor, prediction_tensor):
        """Computes loss.

        :param target_tensor: E-by-H-by-W-by-T tensor of actual values.
        :param prediction_tensor: E-by-H-by-W-by-T tensor of predicted values.
        :return: loss: Dual-weighted MSE for heating rate.
        """

        if band_weights is None:
            return K.mean(
                K.maximum(K.abs(target_tensor), K.abs(prediction_tensor))
                * (prediction_tensor - target_tensor) ** 2
            )

        # E x H x W x T
        dual_weight_tensor = K.maximum(
            K.abs(target_tensor),
            K.abs(prediction_tensor)
        )

        # E x H x W x T
        band_weight_tensor = K.cast(
            K.constant(band_weights), prediction_tensor.dtype
        )
        band_weight_tensor = K.expand_dims(band_weight_tensor, axis=0)
        band_weight_tensor = K.expand_dims(band_weight_tensor, axis=0)
        band_weight_tensor = K.expand_dims(band_weight_tensor, axis=-1)

        return K.mean(
            dual_weight_tensor * band_weight_tensor *
            (prediction_tensor - target_tensor) ** 2
        )

    return loss


def dual_weighted_mse_constrained_bb(band_weights=None):
    """Dual-weighted MSE (mean squared error) for heating rates.

    This method is the same as `dual_weighted_mse`, except that it constrains
    and penalizes broadband heating rates (broadband = sum over all wavelength
    bands) as well.  Assumes a deterministic model.

    :param band_weights: length-W numpy array of weights, one for each
        wavelength band.  For the default behaviour (every band weighted the
        same), leave this argument alone.
    :return: loss: Loss function (defined below).
    """

    if band_weights is not None:
        error_checking.assert_is_numpy_array(band_weights, num_dimensions=1)
        error_checking.assert_is_greater_numpy_array(band_weights, 0.)

        # Add broadband.
        band_weights = numpy.concatenate([
            band_weights,
            numpy.array([numpy.sum(band_weights)])
        ])

    def loss(target_tensor, prediction_tensor):
        """Computes loss.

        :param target_tensor: E-by-H-by-W-by-T tensor of actual values.
        :param prediction_tensor: E-by-H-by-W-by-T tensor of predicted values.
        :return: loss: Dual-weighted MSE for heating rates, including broadband.
        """

        # Add broadband.
        target_tensor = K.concatenate([
            target_tensor,
            K.sum(target_tensor, axis=-2, keepdims=True)
        ], axis=-2)

        prediction_tensor = K.concatenate([
            prediction_tensor,
            K.sum(prediction_tensor, axis=-2, keepdims=True)
        ], axis=-2)

        if band_weights is None:
            return K.mean(
                K.maximum(K.abs(target_tensor), K.abs(prediction_tensor))
                * (prediction_tensor - target_tensor) ** 2
            )

        # E x H x W x T
        dual_weight_tensor = K.maximum(
            K.abs(target_tensor),
            K.abs(prediction_tensor)
        )

        # E x H x W x T
        band_weight_tensor = K.cast(
            K.constant(band_weights), prediction_tensor.dtype
        )
        band_weight_tensor = K.expand_dims(band_weight_tensor, axis=0)
        band_weight_tensor = K.expand_dims(band_weight_tensor, axis=0)
        band_weight_tensor = K.expand_dims(band_weight_tensor, axis=-1)

        return K.mean(
            dual_weight_tensor * band_weight_tensor *
            (prediction_tensor - target_tensor) ** 2
        )

    return loss
