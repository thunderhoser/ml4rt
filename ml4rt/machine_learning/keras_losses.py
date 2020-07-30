"""Custom loss functions for Keras models."""

import numpy
import keras.backend as K
from gewittergefahr.gg_utils import error_checking
from ml4rt.io import example_io


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
            K.maximum(target_tensor, prediction_tensor) *
            (prediction_tensor - target_tensor) ** 2
        )

    return loss


def dual_weighted_mse_equalize_heights():
    """Dual-weighted MSE with equalized heights.

    Each height should have an equal contribution to this loss function.

    :return: loss: Loss function (defined below).
    """

    def loss(target_tensor, prediction_tensor):
        """Computes loss (dual-weighted MSE with equalized heights).

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: loss: Dual-weighted MSE with equalized heights.
        """

        max_target_tensor = K.max(target_tensor, axis=(0, -1), keepdims=True)
        max_target_tensor = K.repeat_elements(
            max_target_tensor, rep=K.shape(target_tensor)[-1],
            axis=-1
        )

        max_prediction_tensor = K.max(
            prediction_tensor, axis=(0, -1), keepdims=True
        )
        max_prediction_tensor = K.repeat_elements(
            max_prediction_tensor, rep=K.shape(target_tensor)[-1],
            axis=-1
        )

        norm_target_tensor = target_tensor / max_target_tensor
        norm_prediction_tensor = prediction_tensor / max_prediction_tensor

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

    edge_heights_m_agl = example_io.get_grid_cell_edges(heights_m_agl)
    grid_cell_widths_metres = (
        example_io.get_grid_cell_widths(edge_heights_m_agl)
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

    edge_heights_m_agl = example_io.get_grid_cell_edges(heights_m_agl)
    grid_cell_widths_metres = (
        example_io.get_grid_cell_widths(edge_heights_m_agl)
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
            this_loss = 0.5 * K.mean(K.square(
                target_tensor[:, toa_up_flux_index] -
                prediction_tensor[:, toa_up_flux_index]
            ))

            this_loss += 0.5 * K.mean(K.square(
                target_tensor[:, surface_down_flux_index] -
                prediction_tensor[:, surface_down_flux_index]
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
