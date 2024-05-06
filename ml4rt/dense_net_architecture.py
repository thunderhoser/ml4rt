"""Methods for building dense neural networks."""

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
NUM_FLUX_COMPONENTS_KEY = 'num_flux_components'
HIDDEN_LAYER_NEURON_NUMS_KEY = 'hidden_layer_neuron_nums'
HIDDEN_LAYER_DROPOUT_RATES_KEY = 'hidden_layer_dropout_rates'
INNER_ACTIV_FUNCTION_KEY = 'inner_activ_function_name'
INNER_ACTIV_FUNCTION_ALPHA_KEY = 'inner_activ_function_alpha'
HEATING_RATE_ACTIV_FUNC_KEY = 'conv_output_activ_func_name'
HEATING_RATE_ACTIV_FUNC_ALPHA_KEY = 'conv_output_activ_func_alpha'
FLUX_ACTIV_FUNC_KEY = 'dense_output_activ_func_name'
FLUX_ACTIV_FUNC_ALPHA_KEY = 'dense_output_activ_func_alpha'
L1_WEIGHT_KEY = 'l1_weight'
L2_WEIGHT_KEY = 'l2_weight'
USE_BATCH_NORM_KEY = 'use_batch_normalization'

VECTOR_LOSS_FUNCTION_KEY = 'vector_loss_function'
SCALAR_LOSS_FUNCTION_KEY = 'scalar_loss_function'

DEFAULT_ARCHITECTURE_OPTION_DICT = {
    NUM_HEIGHTS_KEY: 73,
    NUM_FLUX_COMPONENTS_KEY: 2,
    HIDDEN_LAYER_NEURON_NUMS_KEY: numpy.array([1000, 605, 366], dtype=int),
    HIDDEN_LAYER_DROPOUT_RATES_KEY: numpy.array([0.5, 0.5, 0.5]),
    INNER_ACTIV_FUNCTION_KEY: architecture_utils.RELU_FUNCTION_STRING,
    INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    HEATING_RATE_ACTIV_FUNC_KEY: None,
    HEATING_RATE_ACTIV_FUNC_ALPHA_KEY: 0.,
    FLUX_ACTIV_FUNC_KEY: architecture_utils.RELU_FUNCTION_STRING,
    FLUX_ACTIV_FUNC_ALPHA_KEY: 0.,
    L1_WEIGHT_KEY: 0.,
    L2_WEIGHT_KEY: 0.001,
    USE_BATCH_NORM_KEY: True
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
    error_checking.assert_is_greater(option_dict[NUM_HEIGHTS_KEY], 0)

    error_checking.assert_is_integer(option_dict[NUM_FLUX_COMPONENTS_KEY])
    error_checking.assert_is_geq(option_dict[NUM_FLUX_COMPONENTS_KEY], 0)
    error_checking.assert_is_leq(option_dict[NUM_FLUX_COMPONENTS_KEY], 2)

    error_checking.assert_is_integer(option_dict[NUM_HEIGHTS_KEY])
    error_checking.assert_is_geq(option_dict[NUM_HEIGHTS_KEY], 10)
    error_checking.assert_is_integer(option_dict[NUM_INPUT_CHANNELS_KEY])
    error_checking.assert_is_geq(option_dict[NUM_INPUT_CHANNELS_KEY], 1)

    hidden_layer_neuron_nums = option_dict[HIDDEN_LAYER_NEURON_NUMS_KEY]
    error_checking.assert_is_integer_numpy_array(hidden_layer_neuron_nums)
    error_checking.assert_is_numpy_array(
        hidden_layer_neuron_nums, num_dimensions=1
    )
    error_checking.assert_is_geq_numpy_array(hidden_layer_neuron_nums, 1)

    num_hidden_layers = len(hidden_layer_neuron_nums)
    these_dimensions = numpy.array([num_hidden_layers], dtype=int)

    hidden_layer_dropout_rates = option_dict[HIDDEN_LAYER_DROPOUT_RATES_KEY]
    error_checking.assert_is_numpy_array(
        hidden_layer_dropout_rates, exact_dimensions=these_dimensions
    )
    error_checking.assert_is_leq_numpy_array(
        hidden_layer_dropout_rates, 1., allow_nan=True
    )

    error_checking.assert_is_geq(option_dict[L1_WEIGHT_KEY], 0.)
    error_checking.assert_is_geq(option_dict[L2_WEIGHT_KEY], 0.)
    error_checking.assert_is_boolean(option_dict[USE_BATCH_NORM_KEY])

    return option_dict


def create_model(option_dict):
    """Creates dense net.

    This method sets up the architecture, loss function, and optimizer -- and
    compiles the model -- but does not train it.

    H = number of hidden layers

    :param option_dict: Dictionary with the following keys.
    option_dict['num_heights']: Number of height levels.
    option_dict['num_input_channels']: Number of input channels.
    option_dict['num_flux_components']: Number of scalar flux components to
        predict.
    option_dict['hidden_layer_neuron_nums']: length-H numpy array with number of
        neurons (features) produced by each hidden layer.  The last value in the
        array, hidden_layer_neuron_nums[-1], is the number of output scalars (to
        be predicted).
    option_dict['hidden_layer_dropout_rates']: length-H numpy array with dropout
        rate for each hidden layer.  Use NaN if you do not want dropout for a
        particular layer.
    option_dict['inner_activ_function_name']: Name of activation function for
        all inner (non-output) layers.  Must be accepted by
        `architecture_utils.check_activation_function`.
    option_dict['inner_activ_function_alpha']: Alpha (slope parameter) for
        activation function for all inner layers.  Applies only to ReLU and eLU.
    option_dict['conv_output_activ_func_name']: Same as
        `inner_activ_function_name` but for heating-rate predictions.  Use
        `None` for no activation function.
    option_dict['conv_output_activ_func_alpha']: Same as
        `inner_activ_function_alpha` but for heating-rate predictions.
    option_dict['dense_output_activ_func_name']: Same as
        `inner_activ_function_name` but for flux predictions.  Use `None` for
        no activation function.
    option_dict['dense_output_activ_func_alpha']: Same as
        `inner_activ_function_alpha` but for flux predictions.
    option_dict['l1_weight']: Weight for L_1 regularization.
    option_dict['l2_weight']: Weight for L_2 regularization.
    option_dict['use_batch_normalization']: Boolean flag.  If True, will use
        batch normalization after each inner (non-output) layer.
    option_dict['vector_loss_function']: Loss function for vector targets.
    option_dict['scalar_loss_function']: Loss function for scalar targets.

    :return: model_object: Untrained instance of `keras.models.Model`.
    """

    option_dict = _check_architecture_args(option_dict)

    num_heights = option_dict[NUM_HEIGHTS_KEY]
    num_input_channels = option_dict[NUM_INPUT_CHANNELS_KEY]
    num_flux_components = option_dict[NUM_FLUX_COMPONENTS_KEY]
    hidden_layer_neuron_nums = option_dict[HIDDEN_LAYER_NEURON_NUMS_KEY]
    hidden_layer_dropout_rates = option_dict[HIDDEN_LAYER_DROPOUT_RATES_KEY]
    inner_activ_function_name = option_dict[INNER_ACTIV_FUNCTION_KEY]
    inner_activ_function_alpha = option_dict[INNER_ACTIV_FUNCTION_ALPHA_KEY]
    conv_output_activ_func_name = option_dict[HEATING_RATE_ACTIV_FUNC_KEY]
    conv_output_activ_func_alpha = (
        option_dict[HEATING_RATE_ACTIV_FUNC_ALPHA_KEY]
    )
    dense_output_activ_func_name = option_dict[FLUX_ACTIV_FUNC_KEY]
    dense_output_activ_func_alpha = option_dict[FLUX_ACTIV_FUNC_ALPHA_KEY]
    l1_weight = option_dict[L1_WEIGHT_KEY]
    l2_weight = option_dict[L2_WEIGHT_KEY]
    use_batch_normalization = option_dict[USE_BATCH_NORM_KEY]

    vector_loss_function = option_dict[VECTOR_LOSS_FUNCTION_KEY]
    scalar_loss_function = option_dict[SCALAR_LOSS_FUNCTION_KEY]

    input_layer_object = keras.layers.Input(
        shape=(num_heights, num_input_channels)
    )
    regularizer_object = architecture_utils.get_weight_regularizer(
        l1_weight=l1_weight, l2_weight=l2_weight
    )

    layer_object = keras.layers.Reshape(
        target_shape=(num_heights * num_input_channels,)
    )(input_layer_object)

    num_hidden_layers = len(hidden_layer_neuron_nums)

    for i in range(num_hidden_layers):
        layer_object = architecture_utils.get_dense_layer(
            num_output_units=hidden_layer_neuron_nums[i],
            weight_regularizer=regularizer_object
        )(layer_object)

        layer_object = architecture_utils.get_activation_layer(
            activation_function_string=inner_activ_function_name,
            alpha_for_relu=inner_activ_function_alpha,
            alpha_for_elu=inner_activ_function_alpha
        )(layer_object)

        if hidden_layer_dropout_rates[i] > 0:
            layer_object = architecture_utils.get_dropout_layer(
                dropout_fraction=hidden_layer_dropout_rates[i]
            )(layer_object)

        if use_batch_normalization:
            layer_object = architecture_utils.get_batch_norm_layer()(
                layer_object
            )

    conv_output_layer_object = architecture_utils.get_dense_layer(
        num_output_units=num_heights,
        weight_regularizer=regularizer_object
    )(layer_object)

    if conv_output_activ_func_name is not None:
        conv_output_layer_object = architecture_utils.get_activation_layer(
            activation_function_string=conv_output_activ_func_name,
            alpha_for_relu=conv_output_activ_func_alpha,
            alpha_for_elu=conv_output_activ_func_alpha
        )(conv_output_layer_object)

    conv_output_layer_object = keras.layers.Reshape(
        target_shape=(num_heights, 1, 1)
    )(conv_output_layer_object)

    conv_output_layer_object = u_net_architecture.zero_top_heating_rate(
        input_layer_object=conv_output_layer_object,
        ensemble_size=1,
        output_layer_name='conv_output'
    )

    if num_flux_components > 0:
        dense_output_layer_object = architecture_utils.get_dense_layer(
            num_output_units=num_flux_components,
            weight_regularizer=regularizer_object,
            layer_name=None
        )(layer_object)

        if dense_output_activ_func_name is not None:
            dense_output_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=dense_output_activ_func_name,
                alpha_for_relu=dense_output_activ_func_alpha,
                alpha_for_elu=dense_output_activ_func_alpha,
                layer_name=None
            )(dense_output_layer_object)

        dense_output_layer_object = keras.layers.Reshape(
            target_shape=(1, num_flux_components),
            name='dense_output'
        )(dense_output_layer_object)

        model_object = keras.models.Model(
            inputs=input_layer_object,
            outputs=[conv_output_layer_object, dense_output_layer_object]
        )

        loss_dict = {
            'conv_output': vector_loss_function,
            'dense_output': scalar_loss_function
        }

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
