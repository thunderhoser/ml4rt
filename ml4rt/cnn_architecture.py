"""Methods for building convolutional neural networks (CNN)."""

import os
import sys
import numpy
import keras

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import architecture_utils
import neural_net
import u_net_architecture

NUM_HEIGHTS_KEY = 'num_heights'
NUM_INPUT_CHANNELS_KEY = 'num_input_channels'
CONV_LAYER_CHANNEL_NUMS_KEY = 'conv_layer_channel_nums'
CONV_LAYER_DROPOUT_RATES_KEY = 'conv_layer_dropout_rates'
CONV_LAYER_FILTER_SIZES_KEY = 'conv_layer_filter_sizes'
DENSE_LAYER_NEURON_NUMS_KEY = 'dense_layer_neuron_nums'
DENSE_LAYER_DROPOUT_RATES_KEY = 'dense_layer_dropout_rates'
INNER_ACTIV_FUNCTION_KEY = 'inner_activ_function_name'
INNER_ACTIV_FUNCTION_ALPHA_KEY = 'inner_activ_function_alpha'
CONV_OUTPUT_ACTIV_FUNC_KEY = 'conv_output_activ_func_name'
CONV_OUTPUT_ACTIV_FUNC_ALPHA_KEY = 'conv_output_activ_func_alpha'
DENSE_OUTPUT_ACTIV_FUNC_KEY = 'dense_output_activ_func_name'
DENSE_OUTPUT_ACTIV_FUNC_ALPHA_KEY = 'dense_output_activ_func_alpha'
L1_WEIGHT_KEY = 'l1_weight'
L2_WEIGHT_KEY = 'l2_weight'
USE_BATCH_NORM_KEY = 'use_batch_normalization'
USE_RESIDUAL_BLOCKS_KEY = 'use_residual_blocks'

VECTOR_LOSS_FUNCTION_KEY = 'vector_loss_function'
SCALAR_LOSS_FUNCTION_KEY = 'scalar_loss_function'

DEFAULT_ARCHITECTURE_OPTION_DICT = {
    CONV_LAYER_CHANNEL_NUMS_KEY:
        numpy.array([32, 32, 64, 64, 128, 128, 3], dtype=int),
    CONV_LAYER_DROPOUT_RATES_KEY:
        numpy.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, numpy.nan]),
    CONV_LAYER_FILTER_SIZES_KEY: numpy.array([3, 3, 3, 3, 3, 3, 3], dtype=int),
    DENSE_LAYER_NEURON_NUMS_KEY: numpy.array([559, 33, 2], dtype=int),
    DENSE_LAYER_DROPOUT_RATES_KEY: numpy.array([0.5, 0.5, numpy.nan]),
    INNER_ACTIV_FUNCTION_KEY: architecture_utils.RELU_FUNCTION_STRING,
    INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    CONV_OUTPUT_ACTIV_FUNC_KEY: None,
    CONV_OUTPUT_ACTIV_FUNC_ALPHA_KEY: 0.,
    DENSE_OUTPUT_ACTIV_FUNC_KEY: architecture_utils.RELU_FUNCTION_STRING,
    DENSE_OUTPUT_ACTIV_FUNC_ALPHA_KEY: 0.,
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

    error_checking.assert_is_integer(option_dict[NUM_HEIGHTS_KEY])
    error_checking.assert_is_geq(option_dict[NUM_HEIGHTS_KEY], 10)
    error_checking.assert_is_integer(option_dict[NUM_INPUT_CHANNELS_KEY])
    error_checking.assert_is_geq(option_dict[NUM_INPUT_CHANNELS_KEY], 1)

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

    conv_layer_filter_sizes = option_dict[CONV_LAYER_FILTER_SIZES_KEY]
    error_checking.assert_is_integer_numpy_array(conv_layer_filter_sizes)
    error_checking.assert_is_numpy_array(
        conv_layer_filter_sizes, exact_dimensions=these_dimensions
    )
    error_checking.assert_is_geq_numpy_array(conv_layer_filter_sizes, 3)

    # Make sure filter sizes are odd.
    these_filter_sizes = (
        2 * numpy.floor(conv_layer_filter_sizes.astype(float) / 2) + 1
    ).astype(int)
    assert numpy.array_equal(these_filter_sizes, conv_layer_filter_sizes)

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

    error_checking.assert_is_geq(option_dict[L1_WEIGHT_KEY], 0.)
    error_checking.assert_is_geq(option_dict[L2_WEIGHT_KEY], 0.)
    error_checking.assert_is_boolean(option_dict[USE_BATCH_NORM_KEY])
    error_checking.assert_is_boolean(option_dict[USE_RESIDUAL_BLOCKS_KEY])

    return option_dict


def create_model(option_dict):
    """Creates CNN (convolutional neural net).

    This method sets up the architecture, loss function, and optimizer -- and
    compiles the model -- but does not train it.

    C = number of convolutional layers
    D = number of dense layers

    If you do not want dense layers, make `dense_layer_neuron_nums` and
    `dense_layer_dropout_rates` be None.

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
    option_dict['conv_output_activ_func_name']: Same as
        `inner_activ_function_name` but for conv output layer.  Use `None` for
        no activation function.
    option_dict['conv_output_activ_func_alpha']: Same as
        `inner_activ_function_alpha` but for conv output layer.
    option_dict['dense_output_activ_func_name']: Same as
        `inner_activ_function_name` but for dense output layer.  Use `None` for
        no activation function.
    option_dict['dense_output_activ_func_alpha']: Same as
        `inner_activ_function_alpha` but for dense output layer.
    option_dict['l1_weight']: Weight for L_1 regularization.
    option_dict['l2_weight']: Weight for L_2 regularization.
    option_dict['use_batch_normalization']: Boolean flag.  If True, will use
        batch normalization after each inner (non-output) layer.
    option_dict['use_residual_blocks']: Boolean flag.  If True, will use
        residual blocks (basic conv blocks) throughout the architecture.
    option_dict['vector_loss_function']: Loss function for vector targets.
    option_dict['scalar_loss_function']: Loss function for scalar targets.

    :return: model_object: Untrained instance of `keras.models.Model`.
    """

    option_dict = _check_architecture_args(option_dict)

    num_heights = option_dict[NUM_HEIGHTS_KEY]
    num_input_channels = option_dict[NUM_INPUT_CHANNELS_KEY]
    conv_layer_channel_nums = option_dict[CONV_LAYER_CHANNEL_NUMS_KEY]
    conv_layer_dropout_rates = option_dict[CONV_LAYER_DROPOUT_RATES_KEY]
    conv_layer_filter_sizes = option_dict[CONV_LAYER_FILTER_SIZES_KEY]
    dense_layer_neuron_nums = option_dict[DENSE_LAYER_NEURON_NUMS_KEY]
    dense_layer_dropout_rates = option_dict[DENSE_LAYER_DROPOUT_RATES_KEY]
    inner_activ_function_name = option_dict[INNER_ACTIV_FUNCTION_KEY]
    inner_activ_function_alpha = option_dict[INNER_ACTIV_FUNCTION_ALPHA_KEY]
    conv_output_activ_func_name = option_dict[CONV_OUTPUT_ACTIV_FUNC_KEY]
    conv_output_activ_func_alpha = option_dict[CONV_OUTPUT_ACTIV_FUNC_ALPHA_KEY]
    dense_output_activ_func_name = option_dict[DENSE_OUTPUT_ACTIV_FUNC_KEY]
    dense_output_activ_func_alpha = (
        option_dict[DENSE_OUTPUT_ACTIV_FUNC_ALPHA_KEY]
    )
    l1_weight = option_dict[L1_WEIGHT_KEY]
    l2_weight = option_dict[L2_WEIGHT_KEY]
    use_batch_normalization = option_dict[USE_BATCH_NORM_KEY]
    use_residual_blocks = option_dict[USE_RESIDUAL_BLOCKS_KEY]

    vector_loss_function = option_dict[VECTOR_LOSS_FUNCTION_KEY]
    scalar_loss_function = option_dict[SCALAR_LOSS_FUNCTION_KEY]

    has_dense_layers = dense_layer_neuron_nums is not None

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
            if conv_output_layer_object is None:
                dense_input_layer_object = input_layer_object
            else:
                dense_input_layer_object = conv_output_layer_object

        if i == num_conv_layers - 1:
            conv_output_layer_object = u_net_architecture.get_conv_block(
                input_layer_object=this_input_layer_object,
                do_residual=use_residual_blocks,
                num_conv_layers=1,
                filter_size_px=conv_layer_filter_sizes[i],
                num_filters=conv_layer_channel_nums[i],
                regularizer_object=regularizer_object,
                activation_function_name=conv_output_activ_func_name,
                activation_function_alpha=conv_output_activ_func_alpha,
                dropout_rates=conv_layer_dropout_rates[i],
                monte_carlo_dropout_flags=False,
                use_batch_norm=False,
                basic_layer_name='conv{0:d}'.format(i)
            )
        else:
            conv_output_layer_object = u_net_architecture.get_conv_block(
                input_layer_object=this_input_layer_object,
                do_residual=use_residual_blocks,
                num_conv_layers=1,
                filter_size_px=conv_layer_filter_sizes[i],
                num_filters=conv_layer_channel_nums[i],
                regularizer_object=regularizer_object,
                activation_function_name=inner_activ_function_name,
                activation_function_alpha=inner_activ_function_alpha,
                dropout_rates=conv_layer_dropout_rates[i],
                monte_carlo_dropout_flags=False,
                use_batch_norm=use_batch_normalization,
                basic_layer_name='conv{0:d}'.format(i)
            )

    if has_dense_layers:
        num_dense_layers = len(dense_layer_neuron_nums)
        dense_output_layer_object = architecture_utils.get_flattening_layer()(
            dense_input_layer_object
        )

        for i in range(num_dense_layers):
            dense_output_layer_object = architecture_utils.get_dense_layer(
                num_output_units=dense_layer_neuron_nums[i],
                layer_name=None
            )(dense_output_layer_object)

            if i == num_dense_layers - 1:
                if dense_output_activ_func_name is not None:
                    dense_output_layer_object = (
                        architecture_utils.get_activation_layer(
                            activation_function_string=
                            dense_output_activ_func_name,
                            alpha_for_relu=dense_output_activ_func_alpha,
                            alpha_for_elu=dense_output_activ_func_alpha,
                            layer_name=None
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

            if dense_layer_dropout_rates[i] > 0:
                dense_output_layer_object = (
                    architecture_utils.get_dropout_layer(
                        dropout_fraction=dense_layer_dropout_rates[i],
                        layer_name=None
                    )(dense_output_layer_object)
                )

            if use_batch_normalization and i != num_dense_layers - 1:
                dense_output_layer_object = (
                    architecture_utils.get_batch_norm_layer()(
                        dense_output_layer_object
                    )
                )
    else:
        dense_output_layer_object = None

    conv_output_layer_object = keras.layers.Reshape(
        target_shape=(num_heights, 1, 1)
    )(conv_output_layer_object)

    conv_output_layer_object = u_net_architecture.zero_top_heating_rate(
        input_layer_object=conv_output_layer_object,
        ensemble_size=1,
        output_layer_name='conv_output'
    )

    dense_output_layer_object = keras.layers.Reshape(
        target_shape=(1, dense_layer_neuron_nums[i]),
        name='dense_output'
    )(dense_output_layer_object)

    if has_dense_layers:
        loss_dict = {
            'conv_output': vector_loss_function,
            'dense_output': scalar_loss_function
        }

        model_object = keras.models.Model(
            inputs=input_layer_object,
            outputs=[conv_output_layer_object, dense_output_layer_object]
        )

        model_object.compile(
            loss=loss_dict,
            optimizer=keras.optimizers.Nadam(),
            metrics=neural_net.METRIC_FUNCTION_LIST
        )
    else:
        model_object = keras.models.Model(
            inputs=input_layer_object, outputs=conv_output_layer_object
        )

        model_object.compile(
            loss=vector_loss_function,
            optimizer=keras.optimizers.Nadam(),
            metrics=neural_net.METRIC_FUNCTION_LIST
        )

    model_object.summary()
    return model_object
