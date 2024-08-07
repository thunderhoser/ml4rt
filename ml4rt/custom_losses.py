"""Custom loss functions for Keras models.

Definitions:

E = number of examples
H = number of heights
W = number of wavelengths
T = number of target variables
S = ensemble size
"""

import os
import sys
from tensorflow.keras import backend as K

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking


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
    :param band_weights: length-(W + 1) numpy array of weights.  The first W
        entries are the weights for individual wavelength bands, and the last
        entry is the weight for the broadband quantity (sum over individual
        wavelength bands).  For the default behaviour (every band weighted the
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


def dual_weighted_mse_simple(min_dual_weight=0.):
    """Dual-weighted MSE (mean squared error) for heating rates.

    This method assumes a deterministic model.  The "dual weight" for each data
    point is max(abs(target_heating_rate), abs(predicted_heating_rate)).

    :param min_dual_weight: Minimum dual weight.
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_geq(min_dual_weight, 0.)

    def loss(target_tensor, prediction_tensor):
        """Computes loss.

        :param target_tensor: E-by-H-by-W-by-T tensor of actual values.
        :param prediction_tensor: E-by-H-by-W-by-T tensor of predicted values.
        :return: loss: Dual-weighted MSE for heating rate.
        """

        return K.mean(
            K.maximum(
                K.maximum(K.abs(target_tensor), K.abs(prediction_tensor)),
                min_dual_weight
            )
            * (prediction_tensor - target_tensor) ** 2
        )

    return loss


def dual_weighted_mse(min_dual_weight=0., band_weights=None):
    """Dual-weighted MSE (mean squared error) for heating rates.

    This method assumes a deterministic model.  The "dual weight" for each data
    point is max(abs(target_heating_rate), abs(predicted_heating_rate)).

    :param min_dual_weight: Minimum dual weight.
    :param band_weights: length-W numpy array of weights, one for each
        wavelength band.  For the default behaviour (every band weighted the
        same), leave this argument alone.
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_geq(min_dual_weight, 0.)

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
                K.maximum(
                    K.maximum(K.abs(target_tensor), K.abs(prediction_tensor)),
                    min_dual_weight
                )
                * (prediction_tensor - target_tensor) ** 2
            )

        # E x H x W x T
        dual_weight_tensor = K.maximum(
            K.maximum(K.abs(target_tensor), K.abs(prediction_tensor)),
            min_dual_weight
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


def dual_weighted_mse_constrained_bb(min_dual_weight=0., band_weights=None):
    """Dual-weighted MSE (mean squared error) for heating rates.

    This method is the same as `dual_weighted_mse`, except that it constrains
    and penalizes broadband heating rates (broadband = sum over all wavelength
    bands) as well.  Assumes a deterministic model.

    :param min_dual_weight: Minimum dual weight.
    :param band_weights: See doc for `scaled_mse_for_net_flux_constrained_bb`.
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_geq(min_dual_weight, 0.)

    if band_weights is not None:
        error_checking.assert_is_numpy_array(band_weights, num_dimensions=1)
        error_checking.assert_is_greater_numpy_array(band_weights, 0.)

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
                K.maximum(
                    K.maximum(K.abs(target_tensor), K.abs(prediction_tensor)),
                    min_dual_weight
                )
                * (prediction_tensor - target_tensor) ** 2
            )

        # E x H x W x T
        dual_weight_tensor = K.maximum(
            K.maximum(K.abs(target_tensor), K.abs(prediction_tensor)),
            min_dual_weight
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
