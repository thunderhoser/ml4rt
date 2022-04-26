"""Methods for building dense neural networks."""

import numpy
import keras
from keras import backend as K
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import architecture_utils
from ml4rt.machine_learning import neural_net

NUM_INPUTS_KEY = 'num_inputs'
NUM_HEIGHTS_KEY = 'num_heights'
NUM_FLUX_COMPONENTS_KEY = 'num_flux_components'
HIDDEN_LAYER_NEURON_NUMS_KEY = 'hidden_layer_neuron_nums'
HIDDEN_LAYER_DROPOUT_RATES_KEY = 'hidden_layer_dropout_rates'
INNER_ACTIV_FUNCTION_KEY = 'inner_activ_function_name'
INNER_ACTIV_FUNCTION_ALPHA_KEY = 'inner_activ_function_alpha'
OUTPUT_ACTIV_FUNCTION_KEY = 'output_activ_function_name'
OUTPUT_ACTIV_FUNCTION_ALPHA_KEY = 'output_activ_function_alpha'
L1_WEIGHT_KEY = 'l1_weight'
L2_WEIGHT_KEY = 'l2_weight'
USE_BATCH_NORM_KEY = 'use_batch_normalization'

DEFAULT_ARCHITECTURE_OPTION_DICT = {
    NUM_HEIGHTS_KEY: 73,
    NUM_FLUX_COMPONENTS_KEY: 2,
    HIDDEN_LAYER_NEURON_NUMS_KEY: numpy.array([1000, 605, 366], dtype=int),
    HIDDEN_LAYER_DROPOUT_RATES_KEY: numpy.array([0.5, 0.5, 0.5]),
    INNER_ACTIV_FUNCTION_KEY: architecture_utils.RELU_FUNCTION_STRING,
    INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    OUTPUT_ACTIV_FUNCTION_KEY: architecture_utils.RELU_FUNCTION_STRING,
    OUTPUT_ACTIV_FUNCTION_ALPHA_KEY: 0.,
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

    error_checking.assert_is_integer(option_dict[NUM_INPUTS_KEY])
    error_checking.assert_is_geq(option_dict[NUM_INPUTS_KEY], 10)

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


def _zero_top_heating_rate_function():
    """Returns function that zeroes predicted heating rate at top of profile.

    :return: zeroing_function: Function handle (see below).
    """

    def zeroing_function(orig_prediction_tensor):
        """Zeroes out predicted heating rate at top of profile.

        :param orig_prediction_tensor: Keras tensor with model predictions.
        :return: new_prediction_tensor: Same as input but with top heating rate
            zeroed out.
        """

        zero_tensor = K.greater_equal(orig_prediction_tensor[..., -1], 1e12)
        zero_tensor = K.cast(zero_tensor, dtype=K.floatx())

        new_prediction_tensor = K.concatenate((
            orig_prediction_tensor[..., :-1],
            K.expand_dims(zero_tensor, axis=-1)
        ), axis=-1)

        return new_prediction_tensor

    return zeroing_function


def create_model(option_dict, heating_rate_loss_function, flux_loss_function):
    """Creates dense net.

    This method sets up the architecture, loss function, and optimizer -- and
    compiles the model -- but does not train it.

    H = number of hidden layers

    :param option_dict: Dictionary with the following keys.
    option_dict['num_inputs']: Number of input variables (predictors).
    option_dict['num_heights']: Number of heights at which to predict heating
        rate.
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

    :param heating_rate_loss_function: Function handle.
    :param flux_loss_function: Function handle.
    :return: model_object: Untrained instance of `keras.models.Model`.
    """

    option_dict = _check_architecture_args(option_dict)

    num_inputs = option_dict[NUM_INPUTS_KEY]
    num_heights = option_dict[NUM_HEIGHTS_KEY]
    num_flux_components = option_dict[NUM_FLUX_COMPONENTS_KEY]
    hidden_layer_neuron_nums = option_dict[HIDDEN_LAYER_NEURON_NUMS_KEY]
    hidden_layer_dropout_rates = option_dict[HIDDEN_LAYER_DROPOUT_RATES_KEY]
    inner_activ_function_name = option_dict[INNER_ACTIV_FUNCTION_KEY]
    inner_activ_function_alpha = option_dict[INNER_ACTIV_FUNCTION_ALPHA_KEY]
    output_activ_function_name = option_dict[OUTPUT_ACTIV_FUNCTION_KEY]
    output_activ_function_alpha = option_dict[OUTPUT_ACTIV_FUNCTION_ALPHA_KEY]
    l1_weight = option_dict[L1_WEIGHT_KEY]
    l2_weight = option_dict[L2_WEIGHT_KEY]
    use_batch_normalization = option_dict[USE_BATCH_NORM_KEY]

    input_layer_object = keras.layers.Input(shape=(num_inputs,))
    regularizer_object = architecture_utils.get_weight_regularizer(
        l1_weight=l1_weight, l2_weight=l2_weight
    )

    num_hidden_layers = len(hidden_layer_neuron_nums)
    layer_object = None

    for i in range(num_hidden_layers):
        if layer_object is None:
            this_input_layer_object = input_layer_object
        else:
            this_input_layer_object = layer_object

        layer_object = architecture_utils.get_dense_layer(
            num_output_units=hidden_layer_neuron_nums[i],
            weight_regularizer=regularizer_object
        )(this_input_layer_object)

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

    heating_rate_layer_object = architecture_utils.get_dense_layer(
        num_output_units=num_heights,
        weight_regularizer=regularizer_object
    )(layer_object)

    heating_rate_layer_object = architecture_utils.get_activation_layer(
        activation_function_string=output_activ_function_name,
        alpha_for_relu=output_activ_function_alpha,
        alpha_for_elu=output_activ_function_alpha
    )(heating_rate_layer_object)

    # this_function = _zero_top_heating_rate_function()
    # heating_rate_layer_object = keras.layers.Lambda(
    #     this_function, name='conv_output'
    # )(heating_rate_layer_object)

    if num_flux_components > 0:
        flux_layer_object = architecture_utils.get_dense_layer(
            num_output_units=num_flux_components,
            weight_regularizer=regularizer_object
        )(layer_object)

        flux_layer_object = architecture_utils.get_activation_layer(
            activation_function_string=output_activ_function_name,
            alpha_for_relu=output_activ_function_alpha,
            alpha_for_elu=output_activ_function_alpha,
            layer_name='dense_output'
        )(flux_layer_object)

        model_object = keras.models.Model(
            inputs=input_layer_object,
            outputs=[heating_rate_layer_object, flux_layer_object]
        )

        loss_dict = {
            'conv_output': heating_rate_loss_function,
            'dense_output': flux_loss_function
        }

        model_object.compile(
            loss=loss_dict, optimizer=keras.optimizers.Adam(),
            metrics=neural_net.METRIC_FUNCTION_LIST
        )
    else:
        model_object = keras.models.Model(
            inputs=input_layer_object, outputs=heating_rate_layer_object
        )
        model_object.compile(
            loss=heating_rate_loss_function, optimizer=keras.optimizers.Adam(),
            metrics=neural_net.METRIC_FUNCTION_LIST
        )

    model_object.summary()
    return model_object
