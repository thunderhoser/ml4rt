"""Methods for building U-nets."""

import sys
import os.path
import numpy
import keras
from keras import backend as K

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import architecture_utils
import neural_net

NUM_CONV_LAYERS_PER_BLOCK = 2
FIRST_NUM_FILTERS = 64

NUM_LEVELS_KEY = 'num_levels'
NUM_HEIGHTS_KEY = 'num_heights'
NUM_HEIGHTS_FOR_LOSS_KEY = 'num_heights_for_loss'
NUM_INPUT_CHANNELS_KEY = 'num_input_channels'
CONV_LAYER_DROPOUT_RATES_KEY = 'conv_layer_dropout_rates'
UPCONV_LAYER_DROPOUT_RATES_KEY = 'upconv_layer_dropout_rates'
SKIP_LAYER_DROPOUT_RATES_KEY = 'skip_layer_dropout_rates'
OUTPUT_LAYER_DROPOUT_RATES_KEY = 'output_layer_dropout_rates'
DENSE_LAYER_NEURON_NUMS_KEY = 'dense_layer_neuron_nums'
DENSE_LAYER_DROPOUT_RATES_KEY = 'dense_layer_dropout_rates'
INNER_ACTIV_FUNCTION_KEY = 'inner_activ_function_name'
INNER_ACTIV_FUNCTION_ALPHA_KEY = 'inner_activ_function_alpha'
OUTPUT_ACTIV_FUNCTION_KEY = 'output_activ_function_name'
OUTPUT_ACTIV_FUNCTION_ALPHA_KEY = 'output_activ_function_alpha'
L1_WEIGHT_KEY = 'l1_weight'
L2_WEIGHT_KEY = 'l2_weight'
USE_BATCH_NORM_KEY = 'use_batch_normalization'
ZERO_OUT_TOP_HR_KEY = 'zero_out_top_heating_rate'
HEATING_RATE_INDEX_KEY = 'heating_rate_channel_index'

DEFAULT_ARCHITECTURE_OPTION_DICT = {
    NUM_LEVELS_KEY: 4,
    NUM_HEIGHTS_FOR_LOSS_KEY: None,
    CONV_LAYER_DROPOUT_RATES_KEY: numpy.array([0.5, 0.5, 0.5, 0.5, 0.5]),
    UPCONV_LAYER_DROPOUT_RATES_KEY: numpy.array([0.5, 0.5, 0.5, 0.5]),
    SKIP_LAYER_DROPOUT_RATES_KEY: numpy.array([0.5, 0.5, 0.5, 0.5]),
    OUTPUT_LAYER_DROPOUT_RATES_KEY: numpy.array([numpy.nan, numpy.nan]),
    DENSE_LAYER_NEURON_NUMS_KEY: None,
    DENSE_LAYER_DROPOUT_RATES_KEY: None,
    # DENSE_LAYER_NEURON_NUMS_KEY: numpy.array([1024, 128, 16, 2], dtype=int),
    # DENSE_LAYER_DROPOUT_RATES_KEY: numpy.array([0.5, 0.5, 0.5, numpy.nan]),
    INNER_ACTIV_FUNCTION_KEY: architecture_utils.RELU_FUNCTION_STRING,
    INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    OUTPUT_ACTIV_FUNCTION_KEY: architecture_utils.RELU_FUNCTION_STRING,
    OUTPUT_ACTIV_FUNCTION_ALPHA_KEY: 0.,
    L1_WEIGHT_KEY: 0.,
    L2_WEIGHT_KEY: 0.001,
    USE_BATCH_NORM_KEY: True,
    ZERO_OUT_TOP_HR_KEY: False,
    HEATING_RATE_INDEX_KEY: None
}


def _check_architecture_args(option_dict):
    """Error-checks input arguments for architecture.

    D = number of dense layers
    L = number of levels = number of pooling operations
                         = number of upsampling operations

    If you do not want dense layers, make `dense_layer_neuron_nums` and
    `dense_layer_dropout_rates` be None.

    :param option_dict: Dictionary with the following keys.
    option_dict['num_levels']: L in the above discussion.
    option_dict['num_heights']: Number of heights in grid.
    option_dict['num_heights_for_loss']: Number of heights to use in loss
        function.  Will use only the bottom N height levels, where N is
        `num_heights_for_loss`.
    option_dict['conv_layer_dropout_rates']: length-(L + 1) numpy array with
        dropout rate for each convolutional layer.  Use NaN if you do not want
        dropout for a particular layer.
    option_dict['upconv_layer_dropout_rates']: length-L numpy array with dropout
        rate for each upconvolutional layer.  Use NaN if you do not want dropout
        for a particular layer.
    option_dict['skip_layer_dropout_rates']: length-L numpy array with dropout
        rate for each skip layer.  Use NaN if you do not want dropout for a
        particular layer.
    option_dict['output_layer_dropout_rates']: length-2 numpy array with dropout
        rates for last two layers.  Use NaN if you do not want dropout for a
        particular layer.
    option_dict['dense_layer_neuron_nums']: length-D numpy array with number of
        neurons (features) produced by each dense layer.  The last value in the
        array, dense_layer_neuron_nums[-1], is the number of output scalars (to
        be predicted).
    option_dict['dense_layer_dropout_rates']: length-D numpy array with dropout
        rate for each dense layer.  Use NaN if you do not want dropout for a
        particular layer.
    option_dict['num_input_channels']: Number of input channels.
    option_dict['inner_activ_function_name']: Name of activation function for
        all inner (non-output) layers.  Must be accepted by
        `architecture_utils.check_activation_function`.
    option_dict['inner_activ_function_alpha']: Alpha (slope parameter) for
        activation function for all inner layers.  Applies only to ReLU and eLU.
    option_dict['output_activ_function_name']: Same as
        `inner_activ_function_name` but for output layers (profiles and
        scalars).
    option_dict['output_activ_function_alpha']: Same as
        `inner_activ_function_alpha` but for output layers (profiles and
        scalars).
    option_dict['l1_weight']: Weight for L_1 regularization.
    option_dict['l2_weight']: Weight for L_2 regularization.
    option_dict['use_batch_normalization']: Boolean flag.  If True, will use
        batch normalization after each inner (non-output) layer.
    option_dict['zero_out_top_heating_rate']: Boolean flag.  If True, will
        always predict 0 K day^-1 for top heating rate.
    option_dict['heating_rate_channel_index']: Channel index for heating rate.
        Used only if `zero_out_top_heating_rate = True`.
    :return: option_dict: Same as input, except defaults may have been added.
    """

    orig_option_dict = option_dict.copy()
    option_dict = DEFAULT_ARCHITECTURE_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    error_checking.assert_is_integer(option_dict[NUM_LEVELS_KEY])
    error_checking.assert_is_geq(option_dict[NUM_LEVELS_KEY], 3)
    error_checking.assert_is_leq(option_dict[NUM_LEVELS_KEY], 4)

    error_checking.assert_is_integer(option_dict[NUM_HEIGHTS_KEY])
    error_checking.assert_is_geq(option_dict[NUM_HEIGHTS_KEY], 10)

    if option_dict[NUM_HEIGHTS_FOR_LOSS_KEY] is None:
        option_dict[NUM_HEIGHTS_FOR_LOSS_KEY] = option_dict[NUM_HEIGHTS_KEY] + 0

    error_checking.assert_is_integer(option_dict[NUM_HEIGHTS_FOR_LOSS_KEY])
    error_checking.assert_is_geq(option_dict[NUM_HEIGHTS_FOR_LOSS_KEY], 10)
    error_checking.assert_is_leq(
        option_dict[NUM_HEIGHTS_FOR_LOSS_KEY], option_dict[NUM_HEIGHTS_KEY]
    )

    conv_layer_dropout_rates = option_dict[CONV_LAYER_DROPOUT_RATES_KEY]
    num_levels = option_dict[NUM_LEVELS_KEY]
    expected_dim = numpy.array([num_levels + 1], dtype=int)

    error_checking.assert_is_numpy_array(
        conv_layer_dropout_rates, exact_dimensions=expected_dim
    )
    error_checking.assert_is_leq_numpy_array(
        conv_layer_dropout_rates, 1., allow_nan=True
    )

    upconv_layer_dropout_rates = option_dict[UPCONV_LAYER_DROPOUT_RATES_KEY]
    expected_dim = numpy.array([num_levels], dtype=int)
    error_checking.assert_is_numpy_array(
        upconv_layer_dropout_rates, exact_dimensions=expected_dim
    )
    error_checking.assert_is_leq_numpy_array(
        upconv_layer_dropout_rates, 1., allow_nan=True
    )

    skip_layer_dropout_rates = option_dict[SKIP_LAYER_DROPOUT_RATES_KEY]
    error_checking.assert_is_numpy_array(
        skip_layer_dropout_rates, exact_dimensions=expected_dim
    )
    error_checking.assert_is_leq_numpy_array(
        skip_layer_dropout_rates, 1., allow_nan=True
    )

    output_layer_dropout_rates = option_dict[OUTPUT_LAYER_DROPOUT_RATES_KEY]
    expected_dim = numpy.array([2], dtype=int)
    error_checking.assert_is_numpy_array(
        output_layer_dropout_rates, exact_dimensions=expected_dim
    )
    error_checking.assert_is_leq_numpy_array(
        output_layer_dropout_rates, 1., allow_nan=True
    )

    dense_layer_neuron_nums = option_dict[DENSE_LAYER_NEURON_NUMS_KEY]
    dense_layer_dropout_rates = option_dict[DENSE_LAYER_DROPOUT_RATES_KEY]
    has_dense_layers = not (
        dense_layer_neuron_nums is None and dense_layer_dropout_rates is None
    )

    if has_dense_layers:
        error_checking.assert_is_integer_numpy_array(dense_layer_neuron_nums)
        error_checking.assert_is_numpy_array(
            dense_layer_neuron_nums, num_dimensions=1
        )
        error_checking.assert_is_geq_numpy_array(dense_layer_neuron_nums, 1)

        num_dense_layers = len(dense_layer_neuron_nums)
        these_dimensions = numpy.array([num_dense_layers], dtype=int)

        error_checking.assert_is_numpy_array(
            dense_layer_dropout_rates, exact_dimensions=these_dimensions
        )
        error_checking.assert_is_leq_numpy_array(
            dense_layer_dropout_rates, 1., allow_nan=True
        )

    error_checking.assert_is_integer(option_dict[NUM_INPUT_CHANNELS_KEY])
    error_checking.assert_is_geq(option_dict[NUM_INPUT_CHANNELS_KEY], 1)
    error_checking.assert_is_geq(option_dict[L1_WEIGHT_KEY], 0.)
    error_checking.assert_is_geq(option_dict[L2_WEIGHT_KEY], 0.)
    error_checking.assert_is_boolean(option_dict[USE_BATCH_NORM_KEY])
    error_checking.assert_is_boolean(option_dict[ZERO_OUT_TOP_HR_KEY])

    if option_dict[ZERO_OUT_TOP_HR_KEY]:
        error_checking.assert_is_integer(option_dict[HEATING_RATE_INDEX_KEY])
        error_checking.assert_is_geq(option_dict[HEATING_RATE_INDEX_KEY], 0)

    return option_dict


def _zero_top_heights_function(num_heights_to_zero):
    """Returns function that zeroes out model predictions at the top heights.

    :param num_heights_to_zero: Will zero out predictions for this many heights
        at top of profile.
    :return: zeroing_function: Function handle (see below).
    """

    def zeroing_function(prediction_tensor):
        """Zeroes out model predictions at the top height levels.

        :param prediction_tensor: Keras tensor with model predictions.
        :return: prediction_tensor: Same as input but with predictions zeroed
            out at top height levels.
        """

        zero_tensor = K.greater_equal(
            prediction_tensor[:, -num_heights_to_zero:, :], 1e12
        )
        zero_tensor = K.cast(zero_tensor, dtype=K.floatx())

        return K.concatenate((
            prediction_tensor[:, :-num_heights_to_zero, :], zero_tensor
        ), axis=1)

    return zeroing_function


def _zero_top_heating_rate_function(heating_rate_channel_index, height_index):
    """Returns function that zeroes predicted heating rate at top of profile.

    :param heating_rate_channel_index: Channel index for heating rate.
    :param height_index: Will zero out heating rate at this height.
    :return: zeroing_function: Function handle (see below).
    """

    def zeroing_function(orig_prediction_tensor):
        """Zeroes out predicted heating rate at top of profile.

        :param orig_prediction_tensor: Keras tensor with model predictions.
        :return: new_prediction_tensor: Same as input but with top heating rate
            zeroed out.
        """

        num_heights = orig_prediction_tensor.get_shape().as_list()[-2]
        num_channels = orig_prediction_tensor.get_shape().as_list()[-1]

        zero_tensor = K.greater_equal(
            orig_prediction_tensor[
                ..., height_index, heating_rate_channel_index
            ],
            1e12
        )
        zero_tensor = K.cast(zero_tensor, dtype=K.floatx())

        heating_rate_tensor = K.concatenate((
            orig_prediction_tensor[..., heating_rate_channel_index][
                ..., :height_index
            ],
            K.expand_dims(zero_tensor, axis=-1)
        ), axis=-1)

        if height_index != num_heights - 1:
            heating_rate_tensor = K.concatenate((
                heating_rate_tensor,
                orig_prediction_tensor[..., heating_rate_channel_index][
                    ..., (height_index + 1):
                ]
            ), axis=-1)

        new_prediction_tensor = K.concatenate((
            orig_prediction_tensor[..., :heating_rate_channel_index],
            K.expand_dims(heating_rate_tensor, axis=-1)
        ), axis=-1)

        if heating_rate_channel_index == num_channels - 1:
            return new_prediction_tensor

        return K.concatenate((
            new_prediction_tensor,
            orig_prediction_tensor[..., (heating_rate_channel_index + 1):]
        ), axis=-1)

    return zeroing_function


def create_model(option_dict, vector_loss_function, num_output_channels=1,
                 scalar_loss_function=None):
    """Creates U-net.

    This method sets up the architecture, loss function, and optimizer -- and
    compiles the model -- but does not train it.

    Architecture taken from:
    https://github.com/zhixuhao/unet/blob/master/model.py

    :param option_dict: See doc for `_check_architecture_args`.
    :param vector_loss_function: Loss function for vector outputs.
    :param num_output_channels: Number of output channels.
    :param scalar_loss_function: Loss function scalar outputs.  If there are no
        dense layers, leave this alone.
    :return: model_object: Instance of `keras.models.Model`, with the
        aforementioned architecture.
    """

    option_dict = _check_architecture_args(option_dict)
    error_checking.assert_is_integer(num_output_channels)
    error_checking.assert_is_greater(num_output_channels, 0)

    num_levels = option_dict[NUM_LEVELS_KEY]
    num_heights = option_dict[NUM_HEIGHTS_KEY]
    num_heights_for_loss = option_dict[NUM_HEIGHTS_FOR_LOSS_KEY]
    conv_layer_dropout_rates = option_dict[CONV_LAYER_DROPOUT_RATES_KEY]
    upconv_layer_dropout_rates = option_dict[UPCONV_LAYER_DROPOUT_RATES_KEY]
    skip_layer_dropout_rates = option_dict[SKIP_LAYER_DROPOUT_RATES_KEY]
    output_layer_dropout_rates = option_dict[OUTPUT_LAYER_DROPOUT_RATES_KEY]
    dense_layer_neuron_nums = option_dict[DENSE_LAYER_NEURON_NUMS_KEY]
    dense_layer_dropout_rates = option_dict[DENSE_LAYER_DROPOUT_RATES_KEY]
    num_input_channels = option_dict[NUM_INPUT_CHANNELS_KEY]
    inner_activ_function_name = option_dict[INNER_ACTIV_FUNCTION_KEY]
    inner_activ_function_alpha = option_dict[INNER_ACTIV_FUNCTION_ALPHA_KEY]
    output_activ_function_name = option_dict[OUTPUT_ACTIV_FUNCTION_KEY]
    output_activ_function_alpha = option_dict[OUTPUT_ACTIV_FUNCTION_ALPHA_KEY]
    l1_weight = option_dict[L1_WEIGHT_KEY]
    l2_weight = option_dict[L2_WEIGHT_KEY]
    use_batch_normalization = option_dict[USE_BATCH_NORM_KEY]
    zero_out_top_heating_rate = option_dict[ZERO_OUT_TOP_HR_KEY]
    heating_rate_channel_index = option_dict[HEATING_RATE_INDEX_KEY]

    any_dense_layers = dense_layer_neuron_nums is not None

    input_layer_object = keras.layers.Input(
        shape=(num_heights, num_input_channels)
    )
    regularizer_object = architecture_utils.get_weight_regularizer(
        l1_weight=l1_weight, l2_weight=l2_weight
    )

    conv_layer_by_level = [None] * (num_levels + 1)
    pooling_layer_by_level = [None] * num_levels
    num_filters = 0

    for i in range(num_levels + 1):
        if i == 0:
            num_filters = FIRST_NUM_FILTERS + 0
        else:
            num_filters *= 2

        for j in range(NUM_CONV_LAYERS_PER_BLOCK):
            if j == 0:
                if i == 0:
                    this_input_layer_object = input_layer_object
                else:
                    this_input_layer_object = pooling_layer_by_level[i - 1]
            else:
                this_input_layer_object = conv_layer_by_level[i]

            conv_layer_by_level[i] = architecture_utils.get_1d_conv_layer(
                num_kernel_rows=3, num_rows_per_stride=1,
                num_filters=num_filters,
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object
            )(this_input_layer_object)

            conv_layer_by_level[i] = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha
            )(conv_layer_by_level[i])

            if conv_layer_dropout_rates[i] > 0:
                conv_layer_by_level[i] = architecture_utils.get_dropout_layer(
                    dropout_fraction=conv_layer_dropout_rates[i]
                )(conv_layer_by_level[i])

            if use_batch_normalization:
                conv_layer_by_level[i] = (
                    architecture_utils.get_batch_norm_layer()(
                        conv_layer_by_level[i]
                    )
                )

        if i == num_levels:
            break

        pooling_layer_by_level[i] = architecture_utils.get_1d_pooling_layer(
            num_rows_in_window=2, num_rows_per_stride=2,
            pooling_type_string=architecture_utils.MAX_POOLING_STRING
        )(conv_layer_by_level[i])

    if any_dense_layers:
        num_dense_layers = len(dense_layer_neuron_nums)
        dense_output_layer_object = architecture_utils.get_flattening_layer()(
            conv_layer_by_level[-1]
        )
    else:
        num_dense_layers = 0
        dense_output_layer_object = None

    for j in range(num_dense_layers):
        dense_output_layer_object = architecture_utils.get_dense_layer(
            num_output_units=dense_layer_neuron_nums[j]
        )(dense_output_layer_object)

        if j == num_dense_layers - 1:
            this_name = (
                None if dense_layer_dropout_rates[j] > 0 else 'dense_output'
            )

            dense_output_layer_object = (
                architecture_utils.get_activation_layer(
                    activation_function_string=output_activ_function_name,
                    alpha_for_relu=output_activ_function_alpha,
                    alpha_for_elu=output_activ_function_alpha,
                    layer_name=this_name
                )(dense_output_layer_object)
            )
        else:
            dense_output_layer_object = (
                architecture_utils.get_activation_layer(
                    activation_function_string=inner_activ_function_name,
                    alpha_for_relu=inner_activ_function_alpha,
                    alpha_for_elu=inner_activ_function_alpha
                )(dense_output_layer_object)
            )

        if dense_layer_dropout_rates[j] > 0:
            this_name = (
                'dense_output' if j == num_dense_layers - 1 else None
            )

            dense_output_layer_object = (
                architecture_utils.get_dropout_layer(
                    dropout_fraction=dense_layer_dropout_rates[j],
                    layer_name=this_name
                )(dense_output_layer_object)
            )

        if use_batch_normalization and j != num_dense_layers - 1:
            dense_output_layer_object = (
                architecture_utils.get_batch_norm_layer()(
                    dense_output_layer_object
                )
            )

    upconv_layer_by_level = [None] * num_levels
    skip_layer_by_level = [None] * num_levels
    merged_layer_by_level = [None] * num_levels
    num_filters = int(numpy.round(float(num_filters) / 2))

    this_layer_object = keras.layers.UpSampling1D(
        size=2
    )(conv_layer_by_level[num_levels])

    i = num_levels - 1
    upconv_layer_by_level[i] = architecture_utils.get_1d_conv_layer(
        num_kernel_rows=2, num_rows_per_stride=1, num_filters=num_filters,
        padding_type_string=architecture_utils.YES_PADDING_STRING,
        weight_regularizer=regularizer_object
    )(this_layer_object)

    if upconv_layer_dropout_rates[i] > 0:
        upconv_layer_by_level[i] = architecture_utils.get_dropout_layer(
            dropout_fraction=upconv_layer_dropout_rates[i]
        )(upconv_layer_by_level[i])

    num_upconv_heights = upconv_layer_by_level[i].get_shape()[1]
    num_desired_heights = conv_layer_by_level[i].get_shape()[1]

    if num_desired_heights == num_upconv_heights + 1:
        upconv_layer_by_level[i] = keras.layers.ZeroPadding1D(
            padding=(0, 1)
        )(upconv_layer_by_level[i])

    merged_layer_by_level[i] = keras.layers.Concatenate(axis=-1)(
        [conv_layer_by_level[i], upconv_layer_by_level[i]]
    )

    level_indices = numpy.linspace(
        0, num_levels - 1, num=num_levels, dtype=int
    )[::-1]

    for i in level_indices:
        for j in range(NUM_CONV_LAYERS_PER_BLOCK):
            if j == 0:
                this_input_layer_object = merged_layer_by_level[i]
            else:
                this_input_layer_object = skip_layer_by_level[i]

            skip_layer_by_level[i] = architecture_utils.get_1d_conv_layer(
                num_kernel_rows=3, num_rows_per_stride=1,
                num_filters=num_filters,
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object
            )(this_input_layer_object)

            skip_layer_by_level[i] = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha
            )(skip_layer_by_level[i])

            if skip_layer_dropout_rates[i] > 0:
                skip_layer_by_level[i] = architecture_utils.get_dropout_layer(
                    dropout_fraction=skip_layer_dropout_rates[i]
                )(skip_layer_by_level[i])

            if use_batch_normalization:
                skip_layer_by_level[i] = (
                    architecture_utils.get_batch_norm_layer()(
                        skip_layer_by_level[i]
                    )
                )

        if i == 0:
            skip_layer_by_level[i] = architecture_utils.get_1d_conv_layer(
                num_kernel_rows=3, num_rows_per_stride=1,
                num_filters=2 * num_output_channels,
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object
            )(skip_layer_by_level[i])

            skip_layer_by_level[i] = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha
            )(skip_layer_by_level[i])

            if output_layer_dropout_rates[0] > 0:
                skip_layer_by_level[i] = architecture_utils.get_dropout_layer(
                    dropout_fraction=output_layer_dropout_rates[0]
                )(skip_layer_by_level[i])

            if use_batch_normalization:
                skip_layer_by_level[i] = (
                    architecture_utils.get_batch_norm_layer()(
                        skip_layer_by_level[i]
                    )
                )

            break

        this_layer_object = keras.layers.UpSampling1D(
            size=2
        )(skip_layer_by_level[i])

        num_filters = int(numpy.round(float(num_filters) / 2))

        upconv_layer_by_level[i - 1] = architecture_utils.get_1d_conv_layer(
            num_kernel_rows=2, num_rows_per_stride=1, num_filters=num_filters,
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object
        )(this_layer_object)

        if upconv_layer_dropout_rates[i - 1] > 0:
            upconv_layer_by_level[i - 1] = architecture_utils.get_dropout_layer(
                dropout_fraction=upconv_layer_dropout_rates[i - 1]
            )(upconv_layer_by_level[i - 1])

        num_upconv_heights = upconv_layer_by_level[i - 1].get_shape()[1]
        num_desired_heights = conv_layer_by_level[i - 1].get_shape()[1]

        if num_desired_heights == num_upconv_heights + 1:
            upconv_layer_by_level[i - 1] = keras.layers.ZeroPadding1D(
                padding=(0, 1)
            )(upconv_layer_by_level[i - 1])

        merged_layer_by_level[i - 1] = keras.layers.Concatenate(axis=-1)(
            [conv_layer_by_level[i - 1], upconv_layer_by_level[i - 1]]
        )

    skip_layer_by_level[0] = architecture_utils.get_1d_conv_layer(
        num_kernel_rows=1, num_rows_per_stride=1,
        num_filters=num_output_channels,
        padding_type_string=architecture_utils.YES_PADDING_STRING,
        weight_regularizer=regularizer_object
    )(skip_layer_by_level[0])

    this_name = (
        None if num_heights > num_heights_for_loss or zero_out_top_heating_rate
        else 'conv_output'
    )

    skip_layer_by_level[0] = architecture_utils.get_activation_layer(
        activation_function_string=output_activ_function_name,
        alpha_for_relu=output_activ_function_alpha,
        alpha_for_elu=output_activ_function_alpha,
        layer_name=None if output_layer_dropout_rates[1] > 0 else this_name
    )(skip_layer_by_level[0])

    if output_layer_dropout_rates[1] > 0:
        skip_layer_by_level[0] = architecture_utils.get_dropout_layer(
            dropout_fraction=output_layer_dropout_rates[1], layer_name=this_name
        )(skip_layer_by_level[0])

    if num_heights > num_heights_for_loss:
        print('Heating rate at top {0:d} heights will always be zero!'.format(
            num_heights - num_heights_for_loss
        ))

        this_function = (
            _zero_top_heights_function(num_heights - num_heights_for_loss)
        )
        this_name = None if zero_out_top_heating_rate else 'conv_output'

        skip_layer_by_level[0] = keras.layers.Lambda(
            this_function, name=this_name
        )(skip_layer_by_level[0])

    if zero_out_top_heating_rate:
        print('Heating rate at height index {0:d} will always be zero!'.format(
            num_heights_for_loss - 1
        ))

        this_function = _zero_top_heating_rate_function(
            heating_rate_channel_index=heating_rate_channel_index,
            height_index=num_heights_for_loss - 1
        )

        skip_layer_by_level[0] = keras.layers.Lambda(
            this_function, name='conv_output'
        )(skip_layer_by_level[0])

    if any_dense_layers:
        model_object = keras.models.Model(
            inputs=input_layer_object,
            outputs=[skip_layer_by_level[0], dense_output_layer_object]
        )
    else:
        model_object = keras.models.Model(
            inputs=input_layer_object, outputs=skip_layer_by_level[0]
        )

    if any_dense_layers:
        loss_dict = {
            'conv_output': vector_loss_function,
            'dense_output': scalar_loss_function
        }

        model_object.compile(
            loss=loss_dict, optimizer=keras.optimizers.Adam(),
            metrics=neural_net.METRIC_FUNCTION_LIST
        )
    else:
        model_object.compile(
            loss=vector_loss_function, optimizer=keras.optimizers.Adam(),
            metrics=neural_net.METRIC_FUNCTION_LIST
        )

    model_object.summary()
    return model_object
