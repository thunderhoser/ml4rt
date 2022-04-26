"""Methods for building U-nets."""

import os
import sys
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

INPUT_DIMENSIONS_KEY = 'input_dimensions'
NUM_LEVELS_KEY = 'num_levels'
CONV_LAYER_COUNTS_KEY = 'num_conv_layers_by_level'
CHANNEL_COUNTS_KEY = 'num_channels_by_level'
ENCODER_DROPOUT_RATES_KEY = 'encoder_dropout_rate_by_level'
UPCONV_DROPOUT_RATES_KEY = 'upconv_dropout_rate_by_level'
SKIP_DROPOUT_RATES_KEY = 'skip_dropout_rate_by_level'
INCLUDE_PENULTIMATE_KEY = 'include_penultimate_conv'
PENULTIMATE_DROPOUT_RATE_KEY = 'penultimate_conv_dropout_rate'
DENSE_LAYER_NEURON_NUMS_KEY = 'dense_layer_neuron_nums'
DENSE_LAYER_DROPOUT_RATES_KEY = 'dense_layer_dropout_rates'
INNER_ACTIV_FUNCTION_KEY = 'inner_activ_function_name'
INNER_ACTIV_FUNCTION_ALPHA_KEY = 'inner_activ_function_alpha'
OUTPUT_ACTIV_FUNCTION_KEY = 'output_activ_function_name'
OUTPUT_ACTIV_FUNCTION_ALPHA_KEY = 'output_activ_function_alpha'
L1_WEIGHT_KEY = 'l1_weight'
L2_WEIGHT_KEY = 'l2_weight'
USE_BATCH_NORM_KEY = 'use_batch_normalization'

DEFAULT_ARCHITECTURE_OPTION_DICT = {
    NUM_LEVELS_KEY: 4,
    CONV_LAYER_COUNTS_KEY: numpy.full(5, 2, dtype=int),
    CHANNEL_COUNTS_KEY: numpy.array([64, 128, 256, 512, 1024], dtype=int),
    ENCODER_DROPOUT_RATES_KEY: numpy.full(5, 0.),
    UPCONV_DROPOUT_RATES_KEY: numpy.full(4, 0.),
    SKIP_DROPOUT_RATES_KEY: numpy.full(4, 0.),
    INCLUDE_PENULTIMATE_KEY: True,
    PENULTIMATE_DROPOUT_RATE_KEY: 0.,
    DENSE_LAYER_NEURON_NUMS_KEY: None,
    DENSE_LAYER_DROPOUT_RATES_KEY: None,
    INNER_ACTIV_FUNCTION_KEY: architecture_utils.RELU_FUNCTION_STRING,
    INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    OUTPUT_ACTIV_FUNCTION_KEY: architecture_utils.SIGMOID_FUNCTION_STRING,
    OUTPUT_ACTIV_FUNCTION_ALPHA_KEY: 0.,
    L1_WEIGHT_KEY: 0.,
    L2_WEIGHT_KEY: 0.001,
    USE_BATCH_NORM_KEY: True
}


def _check_args(option_dict):
    """Error-checks input arguments.

    L = number of levels in encoder = number of levels in decoder
    D = number of dense layers

    :param option_dict: Dictionary with the following keys.
    option_dict['input_dimensions']: numpy array with input dimensions
        (num_heights, num_channels).
    option_dict['num_levels']: L in the above discussion.
    option_dict['num_conv_layers_by_level']: length-(L + 1) numpy array with
        number of conv layers at each level.
    option_dict['num_channels_by_level']: length-(L + 1) numpy array with number
        of channels at each level.
    option_dict['encoder_dropout_rate_by_level']: length-(L + 1) numpy array
        with dropout rate for conv layers in encoder at each level.
    option_dict['upconv_dropout_rate_by_level']: length-L numpy array
        with dropout rate for upconv layers at each level.
    option_dict['skip_dropout_rate_by_level']: length-L numpy array with dropout
        rate for conv layer after skip connection at each level.
    option_dict['include_penultimate_conv']: Boolean flag.  If True, will put in
        extra conv layer (with 3 x 3 filter) before final pixelwise conv.
    option_dict['penultimate_conv_dropout_rate']: Dropout rate for penultimate
        conv layer.
    option_dict['dense_layer_neuron_nums']: length-D numpy array with number of
        neurons in each dense layer.
    option_dict['dense_layer_dropout_rates']: length-D numpy array with dropout
        rate for each dense layer.
    option_dict['inner_activ_function_name']: Name of activation function for
        all inner (non-output) layers.  Must be accepted by
        `architecture_utils.check_activation_function`.
    option_dict['inner_activ_function_alpha']: Alpha (slope parameter) for
        activation function for all inner layers.  Applies only to ReLU and eLU.
    option_dict['output_activ_function_name']: Same as
        `inner_activ_function_name` but for output layer.
    option_dict['output_activ_function_alpha']: Same as
        `inner_activ_function_alpha` but for output layer.
    option_dict['l1_weight']: Weight for L_1 regularization.
    option_dict['l2_weight']: Weight for L_2 regularization.
    option_dict['use_batch_normalization']: Boolean flag.  If True, will use
        batch normalization after each inner (non-output) conv layer.

    :return: option_dict: Same as input, except defaults may have been added.
    """

    orig_option_dict = option_dict.copy()
    option_dict = DEFAULT_ARCHITECTURE_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    input_dimensions = option_dict[INPUT_DIMENSIONS_KEY]
    error_checking.assert_is_numpy_array(
        input_dimensions, exact_dimensions=numpy.array([2], dtype=int)
    )
    error_checking.assert_is_integer_numpy_array(input_dimensions)
    error_checking.assert_is_greater_numpy_array(input_dimensions, 0)

    num_levels = option_dict[NUM_LEVELS_KEY]
    error_checking.assert_is_integer(num_levels)
    error_checking.assert_is_geq(num_levels, 2)

    expected_dim = numpy.array([num_levels + 1], dtype=int)

    num_conv_layers_by_level = option_dict[CONV_LAYER_COUNTS_KEY]
    error_checking.assert_is_numpy_array(
        num_conv_layers_by_level, exact_dimensions=expected_dim
    )
    error_checking.assert_is_integer_numpy_array(num_conv_layers_by_level)
    error_checking.assert_is_greater_numpy_array(num_conv_layers_by_level, 0)

    num_channels_by_level = option_dict[CHANNEL_COUNTS_KEY]
    error_checking.assert_is_numpy_array(
        num_channels_by_level, exact_dimensions=expected_dim
    )
    error_checking.assert_is_integer_numpy_array(num_channels_by_level)
    error_checking.assert_is_greater_numpy_array(num_channels_by_level, 0)

    encoder_dropout_rate_by_level = option_dict[ENCODER_DROPOUT_RATES_KEY]
    error_checking.assert_is_numpy_array(
        encoder_dropout_rate_by_level, exact_dimensions=expected_dim
    )
    error_checking.assert_is_leq_numpy_array(
        encoder_dropout_rate_by_level, 1., allow_nan=True
    )

    expected_dim = numpy.array([num_levels], dtype=int)

    upconv_dropout_rate_by_level = option_dict[UPCONV_DROPOUT_RATES_KEY]
    error_checking.assert_is_numpy_array(
        upconv_dropout_rate_by_level, exact_dimensions=expected_dim
    )
    error_checking.assert_is_leq_numpy_array(
        upconv_dropout_rate_by_level, 1., allow_nan=True
    )

    skip_dropout_rate_by_level = option_dict[SKIP_DROPOUT_RATES_KEY]
    error_checking.assert_is_numpy_array(
        skip_dropout_rate_by_level, exact_dimensions=expected_dim
    )
    error_checking.assert_is_leq_numpy_array(
        skip_dropout_rate_by_level, 1., allow_nan=True
    )

    error_checking.assert_is_boolean(option_dict[INCLUDE_PENULTIMATE_KEY])
    error_checking.assert_is_leq(
        option_dict[PENULTIMATE_DROPOUT_RATE_KEY], 1., allow_nan=True
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
        expected_dim = numpy.array([num_dense_layers], dtype=int)

        error_checking.assert_is_numpy_array(
            dense_layer_dropout_rates, exact_dimensions=expected_dim
        )
        error_checking.assert_is_leq_numpy_array(
            dense_layer_dropout_rates, 1., allow_nan=True
        )

    error_checking.assert_is_geq(option_dict[L1_WEIGHT_KEY], 0.)
    error_checking.assert_is_geq(option_dict[L2_WEIGHT_KEY], 0.)
    error_checking.assert_is_boolean(option_dict[USE_BATCH_NORM_KEY])

    return option_dict


def zero_top_heating_rate_function(heating_rate_channel_index, height_index):
    """Returns function that zeroes predicted heating rate at top of profile.

    :param heating_rate_channel_index: Channel index for heating rate.
    :param height_index: Will zero out heating rate at this height.
    :return: zeroing_function: Function handle (see below).
    """

    error_checking.assert_is_integer(heating_rate_channel_index)
    error_checking.assert_is_geq(heating_rate_channel_index, 0)
    error_checking.assert_is_integer(height_index)
    error_checking.assert_is_geq(height_index, 0)

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

    option_dict = _check_args(option_dict)

    input_dimensions = option_dict[INPUT_DIMENSIONS_KEY]
    num_levels = option_dict[NUM_LEVELS_KEY]
    num_conv_layers_by_level = option_dict[CONV_LAYER_COUNTS_KEY]
    num_channels_by_level = option_dict[CHANNEL_COUNTS_KEY]
    encoder_dropout_rate_by_level = option_dict[ENCODER_DROPOUT_RATES_KEY]
    upconv_dropout_rate_by_level = option_dict[UPCONV_DROPOUT_RATES_KEY]
    skip_dropout_rate_by_level = option_dict[SKIP_DROPOUT_RATES_KEY]
    include_penultimate_conv = option_dict[INCLUDE_PENULTIMATE_KEY]
    penultimate_conv_dropout_rate = option_dict[PENULTIMATE_DROPOUT_RATE_KEY]
    dense_layer_neuron_nums = option_dict[DENSE_LAYER_NEURON_NUMS_KEY]
    dense_layer_dropout_rates = option_dict[DENSE_LAYER_DROPOUT_RATES_KEY]
    inner_activ_function_name = option_dict[INNER_ACTIV_FUNCTION_KEY]
    inner_activ_function_alpha = option_dict[INNER_ACTIV_FUNCTION_ALPHA_KEY]
    output_activ_function_name = option_dict[OUTPUT_ACTIV_FUNCTION_KEY]
    output_activ_function_alpha = option_dict[OUTPUT_ACTIV_FUNCTION_ALPHA_KEY]
    l1_weight = option_dict[L1_WEIGHT_KEY]
    l2_weight = option_dict[L2_WEIGHT_KEY]
    use_batch_normalization = option_dict[USE_BATCH_NORM_KEY]

    has_dense_layers = dense_layer_neuron_nums is not None

    input_layer_object = keras.layers.Input(
        shape=tuple(input_dimensions.tolist())
    )
    regularizer_object = architecture_utils.get_weight_regularizer(
        l1_weight=l1_weight, l2_weight=l2_weight
    )

    conv_layer_by_level = [None] * (num_levels + 1)
    pooling_layer_by_level = [None] * num_levels

    for i in range(num_levels + 1):
        for k in range(num_conv_layers_by_level[i]):
            if k == 0:
                if i == 0:
                    this_input_layer_object = input_layer_object
                else:
                    this_input_layer_object = pooling_layer_by_level[i - 1]
            else:
                this_input_layer_object = conv_layer_by_level[i]

            this_name = 'block{0:d}_conv{1:d}'.format(i, k)
            conv_layer_by_level[i] = architecture_utils.get_1d_conv_layer(
                num_kernel_rows=3, num_rows_per_stride=1,
                num_filters=num_channels_by_level[i],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object, layer_name=this_name
            )(this_input_layer_object)

            this_name = 'block{0:d}_conv{1:d}_activation'.format(i, k)
            conv_layer_by_level[i] = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha, layer_name=this_name
            )(conv_layer_by_level[i])

            if encoder_dropout_rate_by_level[i] > 0:
                this_name = 'block{0:d}_conv{1:d}_dropout'.format(i, k)

                conv_layer_by_level[i] = architecture_utils.get_dropout_layer(
                    dropout_fraction=encoder_dropout_rate_by_level[i],
                    layer_name=this_name
                )(conv_layer_by_level[i])

            if use_batch_normalization:
                this_name = 'block{0:d}_conv{1:d}_bn'.format(i, k)

                conv_layer_by_level[i] = (
                    architecture_utils.get_batch_norm_layer(
                        layer_name=this_name
                    )(conv_layer_by_level[i])
                )

        if i == num_levels:
            break

        this_name = 'block{0:d}_pooling'.format(i)

        pooling_layer_by_level[i] = architecture_utils.get_1d_pooling_layer(
            num_rows_in_window=2, num_rows_per_stride=2,
            pooling_type_string=architecture_utils.MAX_POOLING_STRING,
            layer_name=this_name
        )(conv_layer_by_level[i])

    upconv_layer_by_level = [None] * num_levels
    skip_layer_by_level = [None] * num_levels
    merged_layer_by_level = [None] * num_levels

    level_indices = numpy.linspace(
        0, num_levels - 1, num=num_levels, dtype=int
    )[::-1]

    for i in level_indices:
        this_name = 'block{0:d}_upsampling'.format(i)

        if i == num_levels - 1:
            upconv_layer_by_level[i] = keras.layers.UpSampling1D(
                size=2, name=this_name
            )(conv_layer_by_level[i + 1])
        else:
            upconv_layer_by_level[i] = keras.layers.UpSampling1D(
                size=2, name=this_name
            )(skip_layer_by_level[i + 1])

        this_name = 'block{0:d}_upconv'.format(i)
        upconv_layer_by_level[i] = architecture_utils.get_1d_conv_layer(
            num_kernel_rows=2, num_rows_per_stride=1,
            num_filters=num_channels_by_level[i],
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object, layer_name=this_name
        )(upconv_layer_by_level[i])

        this_name = 'block{0:d}_upconv_activation'.format(i)
        upconv_layer_by_level[i] = architecture_utils.get_activation_layer(
            activation_function_string=inner_activ_function_name,
            alpha_for_relu=inner_activ_function_alpha,
            alpha_for_elu=inner_activ_function_alpha, layer_name=this_name
        )(upconv_layer_by_level[i])

        if upconv_dropout_rate_by_level[i] > 0:
            this_name = 'block{0:d}_upconv_dropout'.format(i)

            upconv_layer_by_level[i] = architecture_utils.get_dropout_layer(
                dropout_fraction=upconv_dropout_rate_by_level[i],
                layer_name=this_name
            )(upconv_layer_by_level[i])

        num_upconv_heights = upconv_layer_by_level[i].get_shape()[1]
        num_desired_heights = conv_layer_by_level[i].get_shape()[1]

        if num_desired_heights == num_upconv_heights + 1:
            this_name = 'block{0:d}_upconv_padding'.format(i)

            upconv_layer_by_level[i] = keras.layers.ZeroPadding1D(
                padding=(0, 1), name=this_name
            )(upconv_layer_by_level[i])

        this_name = 'block{0:d}_concat'.format(i)
        merged_layer_by_level[i] = keras.layers.Concatenate(
            name=this_name, axis=-1
        )(
            [conv_layer_by_level[i], upconv_layer_by_level[i]]
        )

        for k in range(num_conv_layers_by_level[i]):
            if k == 0:
                this_input_layer_object = merged_layer_by_level[i]
            else:
                this_input_layer_object = skip_layer_by_level[i]

            this_name = 'block{0:d}_skipconv{1:d}'.format(i, k)
            skip_layer_by_level[i] = architecture_utils.get_1d_conv_layer(
                num_kernel_rows=3, num_rows_per_stride=1,
                num_filters=num_channels_by_level[i],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object, layer_name=this_name
            )(this_input_layer_object)

            this_name = 'block{0:d}_skipconv{1:d}_activation'.format(i, k)
            skip_layer_by_level[i] = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha, layer_name=this_name
            )(skip_layer_by_level[i])

            if skip_dropout_rate_by_level[i] > 0:
                this_name = 'block{0:d}_skipconv{1:d}_dropout'.format(i, k)

                skip_layer_by_level[i] = architecture_utils.get_dropout_layer(
                    dropout_fraction=skip_dropout_rate_by_level[i],
                    layer_name=this_name
                )(skip_layer_by_level[i])

            if use_batch_normalization:
                this_name = 'block{0:d}_skipconv{1:d}_bn'.format(i, k)

                skip_layer_by_level[i] = (
                    architecture_utils.get_batch_norm_layer(
                        layer_name=this_name
                    )(skip_layer_by_level[i])
                )

    if include_penultimate_conv:
        skip_layer_by_level[0] = architecture_utils.get_1d_conv_layer(
            num_kernel_rows=3, num_rows_per_stride=1,
            num_filters=2 * num_output_channels,
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object, layer_name='penultimate_conv'
        )(skip_layer_by_level[0])

        skip_layer_by_level[0] = architecture_utils.get_activation_layer(
            activation_function_string=inner_activ_function_name,
            alpha_for_relu=inner_activ_function_alpha,
            alpha_for_elu=inner_activ_function_alpha,
            layer_name='penultimate_conv_activation'
        )(skip_layer_by_level[0])

        if penultimate_conv_dropout_rate > 0:
            skip_layer_by_level[0] = architecture_utils.get_dropout_layer(
                dropout_fraction=penultimate_conv_dropout_rate,
                layer_name='penultimate_conv_dropout'
            )(skip_layer_by_level[0])

        if use_batch_normalization:
            skip_layer_by_level[0] = architecture_utils.get_batch_norm_layer(
                layer_name='penultimate_conv_bn'
            )(skip_layer_by_level[0])

    conv_output_layer_object = architecture_utils.get_1d_conv_layer(
        num_kernel_rows=1, num_rows_per_stride=1,
        num_filters=num_output_channels,
        padding_type_string=architecture_utils.YES_PADDING_STRING,
        weight_regularizer=regularizer_object, layer_name='last_conv'
    )(skip_layer_by_level[0])

    conv_output_layer_object = architecture_utils.get_activation_layer(
        activation_function_string=output_activ_function_name,
        alpha_for_relu=output_activ_function_alpha,
        alpha_for_elu=output_activ_function_alpha,
        layer_name='last_conv_activation'
    )(conv_output_layer_object)

    # this_function = zero_top_heating_rate_function(
    #     heating_rate_channel_index=0, height_index=input_dimensions[0] - 1
    # )
    #
    # conv_output_layer_object = keras.layers.Lambda(
    #     this_function, name='conv_output'
    # )(conv_output_layer_object)

    if has_dense_layers:
        num_dense_layers = len(dense_layer_neuron_nums)
        dense_output_layer_object = architecture_utils.get_flattening_layer()(
            conv_layer_by_level[-1]
        )
    else:
        num_dense_layers = 0
        dense_output_layer_object = None

    for j in range(num_dense_layers):
        this_name = 'dense{0:d}'.format(j)
        dense_output_layer_object = architecture_utils.get_dense_layer(
            num_output_units=dense_layer_neuron_nums[j], layer_name=this_name
        )(dense_output_layer_object)

        if j == num_dense_layers - 1:
            if dense_layer_dropout_rates[j] > 0:
                this_name = 'dense{0:d}_activation'.format(j)
            else:
                this_name = 'dense_output'

            dense_output_layer_object = (
                architecture_utils.get_activation_layer(
                    activation_function_string=output_activ_function_name,
                    alpha_for_relu=output_activ_function_alpha,
                    alpha_for_elu=output_activ_function_alpha,
                    layer_name=this_name
                )(dense_output_layer_object)
            )
        else:
            this_name = 'dense{0:d}_activation'.format(j)

            dense_output_layer_object = (
                architecture_utils.get_activation_layer(
                    activation_function_string=inner_activ_function_name,
                    alpha_for_relu=inner_activ_function_alpha,
                    alpha_for_elu=inner_activ_function_alpha,
                    layer_name=this_name
                )(dense_output_layer_object)
            )

        if dense_layer_dropout_rates[j] > 0:
            if j == num_dense_layers - 1:
                this_name = 'dense_output'
            else:
                this_name = 'dense{0:d}_dropout'.format(j)

            dense_output_layer_object = (
                architecture_utils.get_dropout_layer(
                    dropout_fraction=dense_layer_dropout_rates[j],
                    layer_name=this_name
                )(dense_output_layer_object)
            )

        if use_batch_normalization and j != num_dense_layers - 1:
            this_name = 'dense{0:d}_bn'.format(j)

            dense_output_layer_object = architecture_utils.get_batch_norm_layer(
                layer_name=this_name
            )(dense_output_layer_object)

    if has_dense_layers:
        model_object = keras.models.Model(
            inputs=input_layer_object,
            outputs=[conv_output_layer_object, dense_output_layer_object]
        )
    else:
        model_object = keras.models.Model(
            inputs=input_layer_object, outputs=conv_output_layer_object
        )

    if has_dense_layers:
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
