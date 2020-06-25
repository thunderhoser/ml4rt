"""Methods for building convolutional neural networks (CNN)."""

import numpy
import keras
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import architecture_utils
from ml4rt.machine_learning import neural_net

NUM_HEIGHTS_KEY = 'num_heights'
NUM_INPUT_CHANNELS_KEY = 'num_input_channels'
CONV_LAYER_CHANNEL_NUMS_KEY = 'conv_layer_channel_nums'
CONV_LAYER_DROPOUT_RATES_KEY = 'conv_layer_dropout_rates'
CONV_LAYER_FILTER_SIZES_KEY = 'conv_layer_filter_sizes'
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
    CONV_LAYER_CHANNEL_NUMS_KEY: numpy.array([80, 80, 80, 3], dtype=int),
    CONV_LAYER_DROPOUT_RATES_KEY: numpy.array([0.5, 0.5, 0.5, numpy.nan]),
    CONV_LAYER_FILTER_SIZES_KEY: numpy.array([5, 5, 5, 5], dtype=int),
    DENSE_LAYER_NEURON_NUMS_KEY: numpy.array([409, 29, 2], dtype=int),
    DENSE_LAYER_DROPOUT_RATES_KEY: numpy.array([0.5, 0.5, numpy.nan]),
    INNER_ACTIV_FUNCTION_KEY: architecture_utils.RELU_FUNCTION_STRING,
    INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    OUTPUT_ACTIV_FUNCTION_KEY: architecture_utils.RELU_FUNCTION_STRING,
    OUTPUT_ACTIV_FUNCTION_ALPHA_KEY: 0.,
    L1_WEIGHT_KEY: 0.,
    L2_WEIGHT_KEY: 0.001,
    USE_BATCH_NORM_KEY: True
}


def _check_architecture_args(option_dict):
    """Error-checks input args for CNN architecture.

    :param option_dict: See doc for `create_model`.
    :return: option_dict: Same as input, except defaults may have been added.
    """

    orig_option_dict = option_dict.copy()
    option_dict = DEFAULT_ARCHITECTURE_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    num_heights = option_dict[NUM_HEIGHTS_KEY]
    error_checking.assert_is_integer(num_heights)
    error_checking.assert_is_geq(num_heights, 10)

    num_input_channels = option_dict[NUM_INPUT_CHANNELS_KEY]
    error_checking.assert_is_integer(num_input_channels)
    error_checking.assert_is_geq(num_input_channels, 1)

    conv_layer_channel_nums = option_dict[CONV_LAYER_CHANNEL_NUMS_KEY]
    error_checking.assert_is_integer_numpy_array(conv_layer_channel_nums)
    error_checking.assert_is_numpy_array(
        conv_layer_channel_nums, num_dimensions=1
    )
    error_checking.assert_is_geq_numpy_array(conv_layer_channel_nums, 1)

    num_conv_layers = len(conv_layer_channel_nums)
    these_dimensions = numpy.array([num_conv_layers], dtype=int)

    conv_layer_dropout_rates = option_dict[CONV_LAYER_DROPOUT_RATES_KEY]
    error_checking.assert_is_numpy_array(
        conv_layer_dropout_rates, exact_dimensions=these_dimensions
    )
    error_checking.assert_is_leq_numpy_array(
        conv_layer_dropout_rates, 1., allow_nan=True
    )

    # TODO(thunderhoser): Also make sure filter sizes are odd?
    conv_layer_filter_sizes = option_dict[CONV_LAYER_FILTER_SIZES_KEY]
    error_checking.assert_is_integer_numpy_array(conv_layer_filter_sizes)
    error_checking.assert_is_numpy_array(
        conv_layer_filter_sizes, exact_dimensions=these_dimensions
    )
    error_checking.assert_is_geq_numpy_array(conv_layer_filter_sizes, 3)

    dense_layer_neuron_nums = option_dict[DENSE_LAYER_NEURON_NUMS_KEY]
    error_checking.assert_is_integer_numpy_array(dense_layer_neuron_nums)
    error_checking.assert_is_numpy_array(
        dense_layer_neuron_nums, num_dimensions=1
    )
    error_checking.assert_is_geq_numpy_array(dense_layer_neuron_nums, 1)

    num_dense_layers = len(dense_layer_neuron_nums)
    these_dimensions = numpy.array([num_dense_layers], dtype=int)

    dense_layer_dropout_rates = option_dict[DENSE_LAYER_DROPOUT_RATES_KEY]
    error_checking.assert_is_numpy_array(
        dense_layer_dropout_rates, exact_dimensions=these_dimensions
    )
    error_checking.assert_is_leq_numpy_array(
        dense_layer_dropout_rates, 1., allow_nan=True
    )

    l1_weight = option_dict[L1_WEIGHT_KEY]
    error_checking.assert_is_geq(l1_weight, 0.)

    l2_weight = option_dict[L2_WEIGHT_KEY]
    error_checking.assert_is_geq(l2_weight, 0.)

    use_batch_normalization = option_dict[USE_BATCH_NORM_KEY]
    error_checking.assert_is_boolean(use_batch_normalization)

    return option_dict


def create_model(option_dict, custom_loss_dict=None):
    """Creates CNN (convolutional neural net).

    This method sets up the architecture, loss function, and optimizer -- and
    compiles the model -- but does not train it.

    C = number of convolutional layers
    D = number of dense layers

    :param option_dict: Dictionary with the following keys.
    option_dict['num_heights']: Number of height levels.
    option_dict['num_input_channels']: Number of input channels.
    option_dict['conv_layer_channel_nums']: length-C numpy array with number of
        channels (filters) produced by each conv layer.  The last value in the
        array, conv_layer_channel_nums[-1], is the number of output channels
        (profiles to be predicted).
    option_dict['conv_layer_dropout_rates']: length-C numpy array with dropout
        rate for each conv layer.  Use NaN if you do not want dropout for a
        particular layer.
    option_dict['conv_layer_filter_sizes']: length-C numpy array with filter
        size (number of heights) for each conv layer.
    option_dict['dense_layer_neuron_nums']: length-D numpy array with number of
        neurons (features) produced by each dense layer.  The last value in the
        array, dense_layer_neuron_nums[-1], is the number of output scalars (to
        be predicted).
    option_dict['dense_layer_dropout_rates']: length-D numpy array with dropout
        rate for each dense layer.  Use NaN if you do not want dropout for a
        particular layer.
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

    :param custom_loss_dict: See doc for `neural_net.get_custom_loss_function`.
        If you do not want a custom loss function, make this None.
    :return: model_object: Untrained instance of `keras.models.Model`.
    """

    # TODO(thunderhoser): Allow for no dense layers.

    option_dict = _check_architecture_args(option_dict)
    use_custom_loss = custom_loss_dict is not None

    if use_custom_loss:
        dense_loss_function = neural_net.get_custom_loss_function(
            custom_loss_dict=custom_loss_dict,
            net_type_string=neural_net.CNN_TYPE_STRING
        )
    else:
        dense_loss_function = None

    num_heights = option_dict[NUM_HEIGHTS_KEY]
    num_input_channels = option_dict[NUM_INPUT_CHANNELS_KEY]
    conv_layer_channel_nums = option_dict[CONV_LAYER_CHANNEL_NUMS_KEY]
    conv_layer_dropout_rates = option_dict[CONV_LAYER_DROPOUT_RATES_KEY]
    conv_layer_filter_sizes = option_dict[CONV_LAYER_FILTER_SIZES_KEY]
    dense_layer_neuron_nums = option_dict[DENSE_LAYER_NEURON_NUMS_KEY]
    dense_layer_dropout_rates = option_dict[DENSE_LAYER_DROPOUT_RATES_KEY]
    inner_activ_function_name = option_dict[INNER_ACTIV_FUNCTION_KEY]
    inner_activ_function_alpha = option_dict[INNER_ACTIV_FUNCTION_ALPHA_KEY]
    output_activ_function_name = option_dict[OUTPUT_ACTIV_FUNCTION_KEY]
    output_activ_function_alpha = option_dict[OUTPUT_ACTIV_FUNCTION_ALPHA_KEY]
    l1_weight = option_dict[L1_WEIGHT_KEY]
    l2_weight = option_dict[L2_WEIGHT_KEY]
    use_batch_normalization = option_dict[USE_BATCH_NORM_KEY]

    input_layer_object = keras.layers.Input(
        shape=(num_heights, num_input_channels)
    )
    regularizer_object = architecture_utils.get_weight_regularizer(
        l1_weight=l1_weight, l2_weight=l2_weight
    )

    num_conv_layers = len(conv_layer_channel_nums)
    conv_output_layer_object = None
    dense_input_layer_object = None

    for i in range(num_conv_layers):
        if conv_output_layer_object is None:
            this_input_layer_object = input_layer_object
        else:
            this_input_layer_object = conv_output_layer_object

        if i == num_conv_layers - 1:
            dense_input_layer_object = conv_output_layer_object

        conv_output_layer_object = architecture_utils.get_1d_conv_layer(
            num_kernel_rows=conv_layer_filter_sizes[i], num_rows_per_stride=1,
            num_filters=conv_layer_channel_nums[i],
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object
        )(this_input_layer_object)

        if i == num_conv_layers - 1:
            conv_output_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=output_activ_function_name,
                alpha_for_relu=output_activ_function_alpha,
                alpha_for_elu=output_activ_function_alpha,
                layer_name='conv_output'
            )(conv_output_layer_object)
        else:
            conv_output_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha
            )(conv_output_layer_object)

        if conv_layer_dropout_rates[i] > 0:
            conv_output_layer_object = architecture_utils.get_dropout_layer(
                dropout_fraction=conv_layer_dropout_rates[i]
            )(conv_output_layer_object)

        if use_batch_normalization and i != num_conv_layers - 1:
            conv_output_layer_object = (
                architecture_utils.get_batch_norm_layer()(
                    conv_output_layer_object
                )
            )

    num_dense_layers = len(dense_layer_neuron_nums)
    dense_output_layer_object = architecture_utils.get_flattening_layer()(
        dense_input_layer_object
    )

    for i in range(num_dense_layers):
        dense_output_layer_object = architecture_utils.get_dense_layer(
            num_output_units=dense_layer_neuron_nums[i]
        )(dense_output_layer_object)

        if i == num_dense_layers - 1:
            dense_output_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=output_activ_function_name,
                alpha_for_relu=output_activ_function_alpha,
                alpha_for_elu=output_activ_function_alpha,
                layer_name=None if use_custom_loss else 'dense_output'
            )(dense_output_layer_object)
        else:
            dense_output_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha
            )(dense_output_layer_object)

        if dense_layer_dropout_rates[i] > 0:
            dense_output_layer_object = architecture_utils.get_dropout_layer(
                dropout_fraction=dense_layer_dropout_rates[i]
            )(dense_output_layer_object)

        if use_batch_normalization and i != num_dense_layers - 1:
            dense_output_layer_object = (
                architecture_utils.get_batch_norm_layer()(
                    dense_output_layer_object
                )
            )

    if use_custom_loss:
        k = custom_loss_dict[neural_net.UP_FLUX_CHANNEL_INDEX_KEY] + 0

        highest_up_flux_layer_object = keras.layers.Lambda(
            lambda x: x[:, -1, k:(k + 1)]
        )(conv_output_layer_object)

        k = custom_loss_dict[neural_net.DOWN_FLUX_CHANNEL_INDEX_KEY] + 0

        lowest_down_flux_layer_object = keras.layers.Lambda(
            lambda x: x[:, -0, k:(k + 1)]
        )(conv_output_layer_object)

        this_list = [
            highest_up_flux_layer_object, lowest_down_flux_layer_object,
            dense_output_layer_object
        ]

        dense_output_layer_object = keras.layers.Concatenate(
            axis=-1, name='dense_output'
        )(this_list)

    model_object = keras.models.Model(
        inputs=input_layer_object,
        outputs=[conv_output_layer_object, dense_output_layer_object]
    )

    if use_custom_loss:
        loss_dict = {
            'conv_output': keras.losses.mse,
            'dense_output': dense_loss_function
        }

        model_object.compile(
            loss=loss_dict, optimizer=keras.optimizers.Adam(),
            metrics=neural_net.METRIC_FUNCTION_LIST
        )
    else:
        model_object.compile(
            loss=keras.losses.mse, optimizer=keras.optimizers.Adam(),
            metrics=neural_net.METRIC_FUNCTION_LIST
        )

    model_object.summary()
    return model_object
