"""Custom loss functions for Keras models."""

import numpy
import keras.backend as K
from gewittergefahr.gg_utils import error_checking


def constrained_mse(
        toa_up_flux_index, toa_up_flux_weight, surface_down_flux_index,
        surface_down_flux_weight, highest_up_flux_index,
        lowest_down_flux_index, net_flux_weight, for_cnn):
    """Physically constrained MSE (mean squared error).

    This function can be applied only to a dense output layer.

    :param toa_up_flux_index: Array index for top-of-atmosphere upwelling flux.
    :param toa_up_flux_weight: Weight for corresponding part of loss function.
    :param surface_down_flux_index: Array index for surface downwelling flux.
    :param surface_down_flux_weight: Weight for corresponding part of loss
        function.
    :param highest_up_flux_index: Array index for upwelling flux at top of
        vertical profile.
    :param lowest_down_flux_index: Array index for downwelling flux at bottom of
        vertical profile.
    :param net_flux_weight: Weight for part of loss function that deals with net
        flux.
    :param for_cnn: Boolean flag.  If True, function will be applied to dense
        output layer for CNN.  If False, will be applied to dense output layer
        for dense net.
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_integer(toa_up_flux_index)
    error_checking.assert_is_geq(toa_up_flux_index, 0)
    error_checking.assert_is_integer(surface_down_flux_index)
    error_checking.assert_is_geq(surface_down_flux_index, 0)
    error_checking.assert_is_integer(highest_up_flux_index)
    error_checking.assert_is_geq(highest_up_flux_index, 0)
    error_checking.assert_is_integer(lowest_down_flux_index)
    error_checking.assert_is_geq(lowest_down_flux_index, 0)
    error_checking.assert_is_greater(toa_up_flux_weight, 0.)
    error_checking.assert_is_greater(surface_down_flux_weight, 0.)
    error_checking.assert_is_greater(net_flux_weight, 0.)
    error_checking.assert_is_boolean(for_cnn)

    def loss(target_tensor, prediction_tensor):
        """Computes loss (constrained MSE).

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: loss: Constrained MSE.
        """

        if for_cnn:
            these_indices = numpy.array(
                [toa_up_flux_index, surface_down_flux_index], dtype=int
            )

            this_loss = K.mean(K.square(
                target_tensor[:, these_indices] -
                prediction_tensor[:, these_indices]
            ))
        else:
            this_loss = K.mean(K.square(target_tensor - prediction_tensor))

        # Add term for disagreement between upwelling flux at TOA and top of
        # profile.
        this_loss += toa_up_flux_weight * K.mean(K.square(
            prediction_tensor[:, toa_up_flux_index] -
            prediction_tensor[:, highest_up_flux_index]
        ))

        # Add term for disagreement between downwelling flux at surface and
        # bottom of profile.
        this_loss += surface_down_flux_weight * K.mean(K.square(
            prediction_tensor[:, surface_down_flux_index] -
            prediction_tensor[:, lowest_down_flux_index]
        ))

        # Add term for disagreement between target and predicted net fluxes.
        target_net_flux_tensor = (
            target_tensor[:, toa_up_flux_index] -
            target_tensor[:, surface_down_flux_index]
        )
        predicted_net_flux_tensor = (
            prediction_tensor[:, toa_up_flux_index] -
            prediction_tensor[:, surface_down_flux_index]
        )
        this_loss += net_flux_weight * K.mean(K.square(
            target_net_flux_tensor - predicted_net_flux_tensor
        ))

        return this_loss

    return loss
