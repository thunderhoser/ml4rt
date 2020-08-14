"""Methods for building U-nets."""

import sys
import os.path
import numpy
import keras
from keras import backend as K

# THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
#     os.path.join(os.getcwd(), os.path.expanduser(__file__))
# ))
# sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

from ml4rt import neural_net
from ml4rt import error_checking

KERNEL_INITIALIZER_NAME = 'glorot_uniform'
BIAS_INITIALIZER_NAME = 'zeros'

MAX_POOLING_STRING = 'max'
MEAN_POOLING_STRING = 'mean'
YES_PADDING_STRING = 'same'
NO_PADDING_STRING = 'valid'

DEFAULT_ALPHA_FOR_ELU = 1.
DEFAULT_ALPHA_FOR_RELU = 0.2
ELU_FUNCTION_STRING = 'elu'
RELU_FUNCTION_STRING = 'relu'

DEFAULT_L1_WEIGHT = 0.
DEFAULT_L2_WEIGHT = 0.001

NUM_HEIGHTS_KEY = 'num_heights'
DENSE_LAYER_NEURON_NUMS_KEY = 'dense_layer_neuron_nums'
DENSE_LAYER_DROPOUT_RATES_KEY = 'dense_layer_dropout_rates'
NUM_HEIGHTS_FOR_LOSS_KEY = 'num_heights_for_loss'
NUM_INPUT_CHANNELS_KEY = 'num_input_channels'
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
    NUM_HEIGHTS_FOR_LOSS_KEY: None,
    DENSE_LAYER_NEURON_NUMS_KEY: None,
    DENSE_LAYER_DROPOUT_RATES_KEY: None,
    # DENSE_LAYER_NEURON_NUMS_KEY: numpy.array([1024, 128, 16, 2], dtype=int),
    # DENSE_LAYER_DROPOUT_RATES_KEY: numpy.array([0.5, 0.5, 0.5, numpy.nan]),
    INNER_ACTIV_FUNCTION_KEY: RELU_FUNCTION_STRING,
    INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    OUTPUT_ACTIV_FUNCTION_KEY: RELU_FUNCTION_STRING,
    OUTPUT_ACTIV_FUNCTION_ALPHA_KEY: 0.,
    L1_WEIGHT_KEY: 0.,
    L2_WEIGHT_KEY: 0.001,
    USE_BATCH_NORM_KEY: True,
    ZERO_OUT_TOP_HR_KEY: False,
    HEATING_RATE_INDEX_KEY: None
}


def _check_architecture_args(option_dict):
    """Error-checks input arguments for architecture.

    :param option_dict: See doc for `create_model`.
    :return: option_dict: Same as input, except defaults may have been added.
    """

    orig_option_dict = option_dict.copy()
    option_dict = DEFAULT_ARCHITECTURE_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    error_checking.assert_is_integer(option_dict[NUM_HEIGHTS_KEY])
    error_checking.assert_is_geq(option_dict[NUM_HEIGHTS_KEY], 10)

    if option_dict[NUM_HEIGHTS_FOR_LOSS_KEY] is None:
        option_dict[NUM_HEIGHTS_FOR_LOSS_KEY] = option_dict[NUM_HEIGHTS_KEY] + 0

    error_checking.assert_is_integer(option_dict[NUM_HEIGHTS_FOR_LOSS_KEY])
    error_checking.assert_is_geq(option_dict[NUM_HEIGHTS_FOR_LOSS_KEY], 10)
    error_checking.assert_is_leq(
        option_dict[NUM_HEIGHTS_FOR_LOSS_KEY], option_dict[NUM_HEIGHTS_KEY]
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


def _get_weight_regularizer(
        l1_weight=DEFAULT_L1_WEIGHT, l2_weight=DEFAULT_L2_WEIGHT):
    """Creates regularizer for network weights.

    :param l1_weight: L1 regularization weight.  This "weight" is not to be
        confused with those being regularized (weights learned by the net).
    :param l2_weight: L2 regularization weight.
    :return: regularizer_object: Instance of `keras.regularizers.l1_l2`.
    """

    return keras.regularizers.l1_l2(l1=l1_weight, l2=l2_weight)


def _get_1d_conv_layer(
        num_kernel_rows, num_rows_per_stride, num_filters,
        padding_type_string=NO_PADDING_STRING, weight_regularizer=None,
        layer_name=None):
    """Creates layer for 1-D convolution.

    :param num_kernel_rows: See doc for `_check_convolution_options`.
    :param num_rows_per_stride: Same.
    :param num_filters: Same.
    :param padding_type_string: Same.
    :param weight_regularizer: Will be used to regularize weights in the new
        layer.  This may be instance of `keras.regularizers` or None (if you
        want no regularization).
    :param layer_name: Layer name (string).  If None, will use default name in
        Keras.
    :return: layer_object: Instance of `keras.layers.Conv1D`.
    """

    return keras.layers.Conv1D(
        filters=num_filters, kernel_size=(num_kernel_rows,),
        strides=(num_rows_per_stride,), padding=padding_type_string,
        dilation_rate=(1,), activation=None, use_bias=True,
        kernel_initializer=KERNEL_INITIALIZER_NAME,
        bias_initializer=BIAS_INITIALIZER_NAME,
        kernel_regularizer=weight_regularizer,
        bias_regularizer=weight_regularizer, name=layer_name
    )


def _get_1d_pooling_layer(num_rows_in_window, num_rows_per_stride,
                          pooling_type_string=MAX_POOLING_STRING):
    """Creates layer for 1-D pooling.

    :param num_rows_in_window: See doc for `_check_pooling_options`.
    :param num_rows_per_stride: Same.
    :param pooling_type_string: Same.
    :return: layer_object: Instance of `keras.layers.MaxPooling1D` or
        `keras.layers.AveragePooling1D`.
    """

    if pooling_type_string == MAX_POOLING_STRING:
        return keras.layers.MaxPooling1D(
            pool_size=num_rows_in_window, strides=num_rows_per_stride,
            padding=NO_PADDING_STRING
        )

    return keras.layers.AveragePooling1D(
        pool_size=num_rows_in_window, strides=num_rows_per_stride,
        padding=NO_PADDING_STRING
    )


def _get_activation_layer(
        activation_function_string, alpha_for_elu=DEFAULT_ALPHA_FOR_ELU,
        alpha_for_relu=DEFAULT_ALPHA_FOR_RELU, layer_name=None):
    """Creates activation layer.

    :param activation_function_string: See doc for `check_activation_function`.
    :param alpha_for_elu: Same.
    :param alpha_for_relu: Same.
    :param layer_name: Layer name (string).  If None, will use default name in
        Keras.
    :return: layer_object: Instance of `keras.layers.Activation`,
        `keras.layers.ELU`, or `keras.layers.LeakyReLU`.
    """

    if activation_function_string == ELU_FUNCTION_STRING:
        return keras.layers.ELU(alpha=alpha_for_elu, name=layer_name)

    if activation_function_string == RELU_FUNCTION_STRING:
        if alpha_for_relu == 0:
            return keras.layers.ReLU(name=layer_name)

        return keras.layers.LeakyReLU(alpha=alpha_for_relu, name=layer_name)

    return keras.layers.Activation(activation_function_string, name=layer_name)


def _get_batch_norm_layer():
    """Creates batch-normalization layer.

    :return: Instance of `keras.layers.BatchNormalization`.
    """

    return keras.layers.BatchNormalization(
        axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True
    )


def _get_dropout_layer(dropout_fraction, layer_name=None):
    """Creates dropout layer.

    :param dropout_fraction: Fraction of weights to drop.
    :param layer_name: Layer name (string).  If None, will use default name in
        Keras.
    :return: layer_object: Instance of `keras.layers.Dropout`.
    """

    return keras.layers.Dropout(rate=dropout_fraction, name=layer_name)


def _get_flattening_layer():
    """Creates flattening layer.

    :return: layer_object: Instance of `keras.layers.Flatten`.
    """

    return keras.layers.Flatten()


def _get_dense_layer(num_output_units, weight_regularizer=None,
                     layer_name=None):
    """Creates dense (fully connected) layer.

    :param num_output_units: Number of output units (or "features" or
        "neurons").
    :param weight_regularizer: See doc for `get_1d_conv_layer`.
    :param layer_name: Layer name (string).  If None, will use default name in
        Keras.
    :return: layer_object: Instance of `keras.layers.Dense`.
    """

    return keras.layers.Dense(
        num_output_units, activation=None, use_bias=True,
        kernel_initializer=KERNEL_INITIALIZER_NAME,
        bias_initializer=BIAS_INITIALIZER_NAME,
        kernel_regularizer=weight_regularizer,
        bias_regularizer=weight_regularizer, name=layer_name
    )


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

    D = number of dense layers

    If you do not want dense layers, make `dense_layer_neuron_nums` and
    `dense_layer_dropout_rates` be None.

    :param option_dict: Dictionary with the following keys.
    option_dict['num_heights']: Number of height levels.
    option_dict['num_heights_for_loss']: Number of height levels to use in loss
        function.  Will use only the bottom N height levels, where N is
        `num_heights_for_loss`.
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

    :param vector_loss_function: Loss function for vector outputs.
    :param num_output_channels: Number of output channels.
    :param scalar_loss_function: Loss function scalar outputs.  If there are no
        dense layers, leave this alone.
    :return: model_object: Instance of `keras.models.Model`, with the
        aforementioned architecture.
    """

    # TODO(thunderhoser): Generalize this method a bit.

    option_dict = _check_architecture_args(option_dict)
    error_checking.assert_is_integer(num_output_channels)
    error_checking.assert_is_greater(num_output_channels, 0)

    num_heights = option_dict[NUM_HEIGHTS_KEY]
    num_heights_for_loss = option_dict[NUM_HEIGHTS_FOR_LOSS_KEY]
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
    regularizer_object = _get_weight_regularizer(
        l1_weight=l1_weight, l2_weight=l2_weight
    )

    conv_layer1_object = None

    for i in range(2):
        if i == 0:
            this_input_layer_object = input_layer_object
        else:
            this_input_layer_object = conv_layer1_object

        conv_layer1_object = _get_1d_conv_layer(
            num_kernel_rows=3, num_rows_per_stride=1, num_filters=64,
            padding_type_string=YES_PADDING_STRING,
            weight_regularizer=regularizer_object
        )(this_input_layer_object)

        conv_layer1_object = _get_activation_layer(
            activation_function_string=inner_activ_function_name,
            alpha_for_relu=inner_activ_function_alpha,
            alpha_for_elu=inner_activ_function_alpha
        )(conv_layer1_object)

        conv_layer1_object = _get_dropout_layer(
            dropout_fraction=0.5
        )(conv_layer1_object)

        if use_batch_normalization:
            conv_layer1_object = _get_batch_norm_layer()(
                conv_layer1_object
            )

    pooling_layer1_object = _get_1d_pooling_layer(
        num_rows_in_window=2, num_rows_per_stride=2,
        pooling_type_string=MAX_POOLING_STRING
    )(conv_layer1_object)

    conv_layer2_object = None

    for i in range(2):
        if i == 0:
            this_input_layer_object = pooling_layer1_object
        else:
            this_input_layer_object = conv_layer2_object

        conv_layer2_object = _get_1d_conv_layer(
            num_kernel_rows=3, num_rows_per_stride=1, num_filters=128,
            padding_type_string=YES_PADDING_STRING,
            weight_regularizer=regularizer_object
        )(this_input_layer_object)

        conv_layer2_object = _get_activation_layer(
            activation_function_string=inner_activ_function_name,
            alpha_for_relu=inner_activ_function_alpha,
            alpha_for_elu=inner_activ_function_alpha
        )(conv_layer2_object)

        conv_layer2_object = _get_dropout_layer(
            dropout_fraction=0.5
        )(conv_layer2_object)

        if use_batch_normalization:
            conv_layer2_object = _get_batch_norm_layer()(
                conv_layer2_object
            )

    pooling_layer2_object = _get_1d_pooling_layer(
        num_rows_in_window=2, num_rows_per_stride=2,
        pooling_type_string=MAX_POOLING_STRING
    )(conv_layer2_object)

    conv_layer3_object = None

    for i in range(2):
        if i == 0:
            this_input_layer_object = pooling_layer2_object
        else:
            this_input_layer_object = conv_layer3_object

        conv_layer3_object = _get_1d_conv_layer(
            num_kernel_rows=3, num_rows_per_stride=1, num_filters=256,
            padding_type_string=YES_PADDING_STRING,
            weight_regularizer=regularizer_object
        )(this_input_layer_object)

        conv_layer3_object = _get_activation_layer(
            activation_function_string=inner_activ_function_name,
            alpha_for_relu=inner_activ_function_alpha,
            alpha_for_elu=inner_activ_function_alpha
        )(conv_layer3_object)

        conv_layer3_object = _get_dropout_layer(
            dropout_fraction=0.5
        )(conv_layer3_object)

        if use_batch_normalization:
            conv_layer3_object = _get_batch_norm_layer()(
                conv_layer3_object
            )

    pooling_layer3_object = _get_1d_pooling_layer(
        num_rows_in_window=2, num_rows_per_stride=2,
        pooling_type_string=MAX_POOLING_STRING
    )(conv_layer3_object)

    conv_layer4_object = None

    for i in range(2):
        if i == 0:
            this_input_layer_object = pooling_layer3_object
        else:
            this_input_layer_object = conv_layer4_object

        conv_layer4_object = _get_1d_conv_layer(
            num_kernel_rows=3, num_rows_per_stride=1, num_filters=512,
            padding_type_string=YES_PADDING_STRING,
            weight_regularizer=regularizer_object
        )(this_input_layer_object)

        conv_layer4_object = _get_activation_layer(
            activation_function_string=inner_activ_function_name,
            alpha_for_relu=inner_activ_function_alpha,
            alpha_for_elu=inner_activ_function_alpha
        )(conv_layer4_object)

        conv_layer4_object = _get_dropout_layer(
            dropout_fraction=0.5
        )(conv_layer4_object)

        if use_batch_normalization:
            conv_layer4_object = _get_batch_norm_layer()(
                conv_layer4_object
            )

    pooling_layer4_object = _get_1d_pooling_layer(
        num_rows_in_window=2, num_rows_per_stride=2,
        pooling_type_string=MAX_POOLING_STRING
    )(conv_layer4_object)

    conv_layer5_object = None

    for i in range(2):
        if i == 0:
            this_input_layer_object = pooling_layer4_object
        else:
            this_input_layer_object = conv_layer5_object

        conv_layer5_object = _get_1d_conv_layer(
            num_kernel_rows=3, num_rows_per_stride=1, num_filters=1024,
            padding_type_string=YES_PADDING_STRING,
            weight_regularizer=regularizer_object
        )(this_input_layer_object)

        conv_layer5_object = _get_activation_layer(
            activation_function_string=inner_activ_function_name,
            alpha_for_relu=inner_activ_function_alpha,
            alpha_for_elu=inner_activ_function_alpha
        )(conv_layer5_object)

        conv_layer5_object = _get_dropout_layer(
            dropout_fraction=0.5
        )(conv_layer5_object)

        if use_batch_normalization:
            conv_layer5_object = _get_batch_norm_layer()(
                conv_layer5_object
            )

    if any_dense_layers:
        num_dense_layers = len(dense_layer_neuron_nums)
        dense_output_layer_object = _get_flattening_layer()(
            conv_layer5_object
        )

        for i in range(num_dense_layers):
            dense_output_layer_object = _get_dense_layer(
                num_output_units=dense_layer_neuron_nums[i]
            )(dense_output_layer_object)

            if i == num_dense_layers - 1:
                this_name = (
                    None if dense_layer_dropout_rates[i] > 0 else 'dense_output'
                )

                dense_output_layer_object = (
                    _get_activation_layer(
                        activation_function_string=output_activ_function_name,
                        alpha_for_relu=output_activ_function_alpha,
                        alpha_for_elu=output_activ_function_alpha,
                        layer_name=this_name
                    )(dense_output_layer_object)
                )
            else:
                dense_output_layer_object = (
                    _get_activation_layer(
                        activation_function_string=inner_activ_function_name,
                        alpha_for_relu=inner_activ_function_alpha,
                        alpha_for_elu=inner_activ_function_alpha
                    )(dense_output_layer_object)
                )

            if dense_layer_dropout_rates[i] > 0:
                this_name = (
                    'dense_output' if i == num_dense_layers - 1 else None
                )

                dense_output_layer_object = (
                    _get_dropout_layer(
                        dropout_fraction=dense_layer_dropout_rates[i],
                        layer_name=this_name
                    )(dense_output_layer_object)
                )

            if use_batch_normalization and i != num_dense_layers - 1:
                dense_output_layer_object = (
                    _get_batch_norm_layer()(
                        dense_output_layer_object
                    )
                )
    else:
        dense_output_layer_object = None

    this_layer_object = keras.layers.UpSampling1D(size=2)

    upconv_layer4_object = _get_1d_conv_layer(
        num_kernel_rows=2, num_rows_per_stride=1, num_filters=512,
        padding_type_string=YES_PADDING_STRING,
        weight_regularizer=regularizer_object
    )(this_layer_object(conv_layer5_object))

    num_upconv_heights = upconv_layer4_object.get_shape()[1]
    num_desired_heights = conv_layer4_object.get_shape()[1]
    if num_desired_heights == num_upconv_heights + 1:
        upconv_layer4_object = keras.layers.ZeroPadding1D(
            padding=(0, 1)
        )(upconv_layer4_object)

    merged_layer4_object = keras.layers.Concatenate(axis=-1)(
        [conv_layer4_object, upconv_layer4_object]
    )

    second_conv_layer4_object = None

    for i in range(2):
        if i == 0:
            this_input_layer_object = merged_layer4_object
        else:
            this_input_layer_object = second_conv_layer4_object

        second_conv_layer4_object = _get_1d_conv_layer(
            num_kernel_rows=3, num_rows_per_stride=1, num_filters=512,
            padding_type_string=YES_PADDING_STRING,
            weight_regularizer=regularizer_object
        )(this_input_layer_object)

        second_conv_layer4_object = _get_activation_layer(
            activation_function_string=inner_activ_function_name,
            alpha_for_relu=inner_activ_function_alpha,
            alpha_for_elu=inner_activ_function_alpha
        )(second_conv_layer4_object)

        if use_batch_normalization:
            second_conv_layer4_object = (
                _get_batch_norm_layer()(
                    second_conv_layer4_object
                )
            )

    this_layer_object = keras.layers.UpSampling1D(size=2)

    upconv_layer3_object = _get_1d_conv_layer(
        num_kernel_rows=2, num_rows_per_stride=1, num_filters=256,
        padding_type_string=YES_PADDING_STRING,
        weight_regularizer=regularizer_object
    )(this_layer_object(second_conv_layer4_object))

    num_upconv_heights = upconv_layer3_object.get_shape()[1]
    num_desired_heights = conv_layer3_object.get_shape()[1]
    if num_desired_heights == num_upconv_heights + 1:
        upconv_layer3_object = keras.layers.ZeroPadding1D(
            padding=(0, 1)
        )(upconv_layer3_object)

    merged_layer3_object = keras.layers.Concatenate(axis=-1)(
        [conv_layer3_object, upconv_layer3_object]
    )

    second_conv_layer3_object = None

    for i in range(2):
        if i == 0:
            this_input_layer_object = merged_layer3_object
        else:
            this_input_layer_object = second_conv_layer3_object

        second_conv_layer3_object = _get_1d_conv_layer(
            num_kernel_rows=3, num_rows_per_stride=1, num_filters=256,
            padding_type_string=YES_PADDING_STRING,
            weight_regularizer=regularizer_object
        )(this_input_layer_object)

        second_conv_layer3_object = _get_activation_layer(
            activation_function_string=inner_activ_function_name,
            alpha_for_relu=inner_activ_function_alpha,
            alpha_for_elu=inner_activ_function_alpha
        )(second_conv_layer3_object)

        if use_batch_normalization:
            second_conv_layer3_object = (
                _get_batch_norm_layer()(
                    second_conv_layer3_object
                )
            )

    this_layer_object = keras.layers.UpSampling1D(size=2)

    upconv_layer2_object = _get_1d_conv_layer(
        num_kernel_rows=2, num_rows_per_stride=1, num_filters=128,
        padding_type_string=YES_PADDING_STRING,
        weight_regularizer=regularizer_object
    )(this_layer_object(second_conv_layer3_object))

    num_upconv_heights = upconv_layer2_object.get_shape()[1]
    num_desired_heights = conv_layer2_object.get_shape()[1]
    if num_desired_heights == num_upconv_heights + 1:
        upconv_layer2_object = keras.layers.ZeroPadding1D(
            padding=(0, 1)
        )(upconv_layer2_object)

    merged_layer2_object = keras.layers.Concatenate(axis=-1)(
        [conv_layer2_object, upconv_layer2_object]
    )

    second_conv_layer2_object = None

    for i in range(2):
        if i == 0:
            this_input_layer_object = merged_layer2_object
        else:
            this_input_layer_object = second_conv_layer2_object

        second_conv_layer2_object = _get_1d_conv_layer(
            num_kernel_rows=3, num_rows_per_stride=1, num_filters=128,
            padding_type_string=YES_PADDING_STRING,
            weight_regularizer=regularizer_object
        )(this_input_layer_object)

        second_conv_layer2_object = _get_activation_layer(
            activation_function_string=inner_activ_function_name,
            alpha_for_relu=inner_activ_function_alpha,
            alpha_for_elu=inner_activ_function_alpha
        )(second_conv_layer2_object)

        if use_batch_normalization:
            second_conv_layer2_object = (
                _get_batch_norm_layer()(
                    second_conv_layer2_object
                )
            )

    this_layer_object = keras.layers.UpSampling1D(size=2)

    upconv_layer1_object = _get_1d_conv_layer(
        num_kernel_rows=2, num_rows_per_stride=1, num_filters=64,
        padding_type_string=YES_PADDING_STRING,
        weight_regularizer=regularizer_object
    )(this_layer_object(second_conv_layer2_object))

    num_upconv_heights = upconv_layer1_object.get_shape()[1]
    num_desired_heights = conv_layer1_object.get_shape()[1]
    if num_desired_heights == num_upconv_heights + 1:
        upconv_layer1_object = keras.layers.ZeroPadding1D(
            padding=(0, 1)
        )(upconv_layer1_object)

    merged_layer1_object = keras.layers.Concatenate(axis=-1)(
        [conv_layer1_object, upconv_layer1_object]
    )

    second_conv_layer1_object = None

    for i in range(3):
        if i == 0:
            this_input_layer_object = merged_layer1_object
        else:
            this_input_layer_object = second_conv_layer1_object

        second_conv_layer1_object = _get_1d_conv_layer(
            num_kernel_rows=3, num_rows_per_stride=1,
            num_filters=2 * num_output_channels if i == 2 else 64,
            padding_type_string=YES_PADDING_STRING,
            weight_regularizer=regularizer_object
        )(this_input_layer_object)

        second_conv_layer1_object = _get_activation_layer(
            activation_function_string=inner_activ_function_name,
            alpha_for_relu=inner_activ_function_alpha,
            alpha_for_elu=inner_activ_function_alpha
        )(second_conv_layer1_object)

        if use_batch_normalization:
            second_conv_layer1_object = (
                _get_batch_norm_layer()(
                    second_conv_layer1_object
                )
            )

    second_conv_layer1_object = _get_1d_conv_layer(
        num_kernel_rows=1, num_rows_per_stride=1,
        num_filters=num_output_channels,
        padding_type_string=YES_PADDING_STRING,
        weight_regularizer=regularizer_object
    )(second_conv_layer1_object)

    this_name = (
        None if num_heights > num_heights_for_loss or zero_out_top_heating_rate
        else 'conv_output'
    )

    second_conv_layer1_object = _get_activation_layer(
        activation_function_string=output_activ_function_name,
        alpha_for_relu=output_activ_function_alpha,
        alpha_for_elu=output_activ_function_alpha,
        layer_name=this_name
    )(second_conv_layer1_object)

    if num_heights > num_heights_for_loss:
        this_name = None if zero_out_top_heating_rate else 'conv_output'

        print('Heating rate at top {0:d} heights will always be zero!'.format(
            num_heights - num_heights_for_loss
        ))

        this_function = (
            _zero_top_heights_function(num_heights - num_heights_for_loss)
        )
        second_conv_layer1_object = keras.layers.Lambda(
            this_function, name=this_name
        )(second_conv_layer1_object)

    if zero_out_top_heating_rate:
        print('Heating rate at height index {0:d} will always be zero!'.format(
            num_heights_for_loss - 1
        ))

        this_function = _zero_top_heating_rate_function(
            heating_rate_channel_index=heating_rate_channel_index,
            height_index=num_heights_for_loss - 1
        )
        second_conv_layer1_object = keras.layers.Lambda(
            this_function, name='conv_output'
        )(second_conv_layer1_object)

    if any_dense_layers:
        model_object = keras.models.Model(
            inputs=input_layer_object,
            outputs=[second_conv_layer1_object, dense_output_layer_object]
        )
    else:
        model_object = keras.models.Model(
            inputs=input_layer_object, outputs=second_conv_layer1_object
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
