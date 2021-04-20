"""Custom loss functions for Keras models."""

import os
import sys
import numpy
import keras.backend as K

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import example_utils


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

        return scaling_factor * (net_flux_term + individual_flux_term)

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


def dual_weighted_mse():
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

        return K.mean(
            K.maximum(K.abs(target_tensor), K.abs(prediction_tensor)) *
            (prediction_tensor - target_tensor) ** 2
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


def flux_increment_loss_dense(
        first_up_flux_inc_index, first_down_flux_inc_index,
        net_flux_increment_weight, total_net_flux_weight, use_magnitude_weight,
        heights_m_agl):
    """Loss function for dense net that predict flux increments.

    :param first_up_flux_inc_index: Array index for upwelling-flux increment at
        lowest height.
    :param first_down_flux_inc_index: Array index for downwelling-flux increment
        at lowest height.
    :param net_flux_increment_weight: See doc for
        `flux_increment_loss_not_dense`.
    :param total_net_flux_weight: Same.
    :param use_magnitude_weight: Same.
    :param heights_m_agl: Same.
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_integer(first_up_flux_inc_index)
    error_checking.assert_is_geq(first_up_flux_inc_index, 0)
    error_checking.assert_is_integer(first_down_flux_inc_index)
    error_checking.assert_is_geq(first_down_flux_inc_index, 0)

    error_checking.assert_is_geq(net_flux_increment_weight, 0.)
    error_checking.assert_is_geq(total_net_flux_weight, 0.)
    error_checking.assert_is_greater(
        net_flux_increment_weight + total_net_flux_weight, 0.
    )

    error_checking.assert_is_boolean(use_magnitude_weight)

    edge_heights_m_agl = example_utils.get_grid_cell_edges(heights_m_agl)
    grid_cell_widths_metres = (
        example_utils.get_grid_cell_widths(edge_heights_m_agl)
    )

    num_heights = len(heights_m_agl)
    grid_cell_width_matrix_metres = numpy.reshape(
        grid_cell_widths_metres, (1, num_heights)
    )

    def loss(target_tensor, prediction_tensor):
        """Computes loss.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: loss: Scalar.
        """

        j = first_down_flux_inc_index
        k = first_up_flux_inc_index

        target_net_flux_inc_tensor_w_m03 = (
            target_tensor[..., j:(j + num_heights)] -
            target_tensor[..., k:(k + num_heights)]
        )
        predicted_net_flux_inc_tensor_w_m03 = (
            prediction_tensor[..., j:(j + num_heights)] -
            prediction_tensor[..., k:(k + num_heights)]
        )
        loss = net_flux_increment_weight * (
            predicted_net_flux_inc_tensor_w_m03 -
            target_net_flux_inc_tensor_w_m03
        ) ** 2

        target_net_flux_tensor_w_m02 = K.cumsum(
            target_net_flux_inc_tensor_w_m03 * grid_cell_width_matrix_metres,
            axis=1
        )
        predicted_net_flux_tensor_w_m02 = K.cumsum(
            predicted_net_flux_inc_tensor_w_m03 * grid_cell_width_matrix_metres,
            axis=1
        )
        loss += total_net_flux_weight * (
            predicted_net_flux_tensor_w_m02 - target_net_flux_tensor_w_m02
        ) ** 2

        if use_magnitude_weight:
            loss = loss * K.maximum(
                target_net_flux_inc_tensor_w_m03,
                predicted_net_flux_inc_tensor_w_m03
            )

        return K.mean(loss)

    return loss


def flux_increment_loss_not_dense(
        up_flux_inc_channel_index, down_flux_inc_channel_index,
        net_flux_increment_weight, total_net_flux_weight, use_magnitude_weight,
        heights_m_agl):
    """Loss function for non-dense net that predict flux increments.

    "Flux increments" are DF_down / Dz and DF_up / Dz.

    :param up_flux_inc_channel_index: Channel index for upwelling-flux increment
        (DF_up).
    :param down_flux_inc_channel_index: Channel index for downwelling-flux
        increment (DF_down).
    :param net_flux_increment_weight: Weight for mean squared error (MSE)
        between predicted and actual net-flux increments (DF_net / Dz).
    :param total_net_flux_weight: Weight for MSE between predicted and actual
        net fluxes (F_net).
    :param use_magnitude_weight: Boolean flag.  If True, the loss for each
        element (each example at each height) will be weighted by the magnitude
        of DF_net / Dz (max between predicted and actual).
    :param heights_m_agl: 1-D numpy array of heights in profile (metres above
        ground level).
    :return: loss: Loss function (defined below).
    """

    # TODO(thunderhoser): This loss function should be used only when there are
    # two target variables, unnormalized downwelling- and upwelling-flux
    # increments.

    # TODO(thunderhoser): In the future, I may want to use fictitious Dp
    # (pressure increment) to convert fluxes to heating rates, then directly
    # penalize heating rates.

    error_checking.assert_is_integer(up_flux_inc_channel_index)
    error_checking.assert_is_geq(up_flux_inc_channel_index, 0)
    error_checking.assert_is_integer(down_flux_inc_channel_index)
    error_checking.assert_is_geq(down_flux_inc_channel_index, 0)

    assert up_flux_inc_channel_index != down_flux_inc_channel_index

    error_checking.assert_is_geq(net_flux_increment_weight, 0.)
    error_checking.assert_is_geq(total_net_flux_weight, 0.)
    error_checking.assert_is_greater(
        net_flux_increment_weight + total_net_flux_weight, 0.
    )

    error_checking.assert_is_boolean(use_magnitude_weight)

    edge_heights_m_agl = example_utils.get_grid_cell_edges(heights_m_agl)
    grid_cell_widths_metres = (
        example_utils.get_grid_cell_widths(edge_heights_m_agl)
    )

    num_heights = len(heights_m_agl)
    grid_cell_width_matrix_metres = numpy.reshape(
        grid_cell_widths_metres, (1, num_heights)
    )

    print(grid_cell_width_matrix_metres)

    def loss(target_tensor, prediction_tensor):
        """Computes loss.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: loss: Scalar.
        """

        target_net_flux_inc_tensor_w_m03 = (
            target_tensor[..., down_flux_inc_channel_index] -
            target_tensor[..., up_flux_inc_channel_index]
        )
        predicted_net_flux_inc_tensor_w_m03 = (
            prediction_tensor[..., down_flux_inc_channel_index] -
            prediction_tensor[..., up_flux_inc_channel_index]
        )
        loss = net_flux_increment_weight * (
            predicted_net_flux_inc_tensor_w_m03 -
            target_net_flux_inc_tensor_w_m03
        ) ** 2

        target_net_flux_tensor_w_m02 = K.cumsum(
            target_net_flux_inc_tensor_w_m03 * grid_cell_width_matrix_metres,
            axis=1
        )
        predicted_net_flux_tensor_w_m02 = K.cumsum(
            predicted_net_flux_inc_tensor_w_m03 * grid_cell_width_matrix_metres,
            axis=1
        )
        loss += total_net_flux_weight * (
            predicted_net_flux_tensor_w_m02 - target_net_flux_tensor_w_m02
        ) ** 2

        if use_magnitude_weight:
            loss = loss * K.maximum(
                target_net_flux_inc_tensor_w_m03,
                predicted_net_flux_inc_tensor_w_m03
            )

        return K.mean(loss)

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
