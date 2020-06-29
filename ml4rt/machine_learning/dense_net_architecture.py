"""Methods for building dense neural networks."""

import numpy
import keras
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import architecture_utils
from ml4rt.machine_learning import neural_net
from ml4rt.machine_learning import keras_losses as custom_losses

NUM_INPUTS_KEY = 'num_inputs'
DENSE_LAYER_NEURON_NUMS_KEY = 'dense_layer_neuron_nums'
DENSE_LAYER_DROPOUT_RATES_KEY = 'dense_layer_dropout_rates'
INNER_ACTIV_FUNCTION_KEY = 'inner_activ_function_name'
INNER_ACTIV_FUNCTION_ALPHA_KEY = 'inner_activ_function_alpha'
OUTPUT_ACTIV_FUNCTION_KEY = 'output_activ_function_name'
OUTPUT_ACTIV_FUNCTION_ALPHA_KEY = 'output_activ_function_alpha'
L1_WEIGHT_KEY = 'l1_weight'
L2_WEIGHT_KEY = 'l2_weight'
USE_BATCH_NORM_KEY = 'use_batch_normalization'
USE_MSESS_LOSS_KEY = 'use_msess_loss'

DEFAULT_ARCHITECTURE_OPTION_DICT = {
    DENSE_LAYER_NEURON_NUMS_KEY: numpy.array([1000, 605, 366, 221], dtype=int),
    DENSE_LAYER_DROPOUT_RATES_KEY: numpy.array([0.5, 0.5, 0.5, numpy.nan]),
    INNER_ACTIV_FUNCTION_KEY: architecture_utils.RELU_FUNCTION_STRING,
    INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    OUTPUT_ACTIV_FUNCTION_KEY: architecture_utils.RELU_FUNCTION_STRING,
    OUTPUT_ACTIV_FUNCTION_ALPHA_KEY: 0.,
    L1_WEIGHT_KEY: 0.,
    L2_WEIGHT_KEY: 0.001,
    USE_BATCH_NORM_KEY: True,
    USE_MSESS_LOSS_KEY: False
}


def _check_architecture_args(option_dict):
    """Error-checks input arguments for architecture.

    :param option_dict: See doc for `create_model`.
    :return: option_dict: Same as input, except defaults may have been added.
    """

    orig_option_dict = option_dict.copy()
    option_dict = DEFAULT_ARCHITECTURE_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    error_checking.assert_is_integer(option_dict[NUM_INPUTS_KEY])
    error_checking.assert_is_geq(option_dict[NUM_INPUTS_KEY], 10)

    dense_layer_neuron_nums = option_dict[DENSE_LAYER_NEURON_NUMS_KEY]
    error_checking.assert_is_integer_numpy_array(dense_layer_neuron_nums)
    error_checking.assert_is_numpy_array(
        dense_layer_neuron_nums, num_dimensions=1
    )
    error_checking.assert_is_geq_numpy_array(dense_layer_neuron_nums, 1)

    num_layers = len(dense_layer_neuron_nums)
    these_dimensions = numpy.array([num_layers], dtype=int)

    dense_layer_dropout_rates = option_dict[DENSE_LAYER_DROPOUT_RATES_KEY]
    error_checking.assert_is_numpy_array(
        dense_layer_dropout_rates, exact_dimensions=these_dimensions
    )
    error_checking.assert_is_leq_numpy_array(
        dense_layer_dropout_rates, 1., allow_nan=True
    )

    error_checking.assert_is_geq(option_dict[L1_WEIGHT_KEY], 0.)
    error_checking.assert_is_geq(option_dict[L2_WEIGHT_KEY], 0.)
    error_checking.assert_is_boolean(option_dict[USE_BATCH_NORM_KEY])
    error_checking.assert_is_boolean(option_dict[USE_MSESS_LOSS_KEY])

    return option_dict


def create_model(option_dict, custom_loss_dict):
    """Creates dense net.

    This method sets up the architecture, loss function, and optimizer -- and
    compiles the model -- but does not train it.

    D = number of dense layers

    :param option_dict: Dictionary with the following keys.
    option_dict['num_inputs']: Number of input variables (predictors).
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
    option_dict['use_msess_loss']: Boolean flag.  If True, will use MSE (mean
        squared error) skill score as loss function.

    :param custom_loss_dict: See doc for `neural_net.get_custom_loss_function`.
        If you do not want a custom loss function, make this None.

    :return: model_object: Untrained instance of `keras.models.Model`.
    """

    option_dict = _check_architecture_args(option_dict=option_dict)

    use_msess_loss = option_dict[USE_MSESS_LOSS_KEY]
    use_custom_loss = custom_loss_dict is not None

    if use_custom_loss:
        loss_function = neural_net.get_custom_loss_function(
            custom_loss_dict=custom_loss_dict,
            net_type_string=neural_net.DENSE_NET_TYPE_STRING
        )
    elif use_msess_loss:
        loss_function = custom_losses.negative_mse_skill_score()
    else:
        loss_function = keras.losses.mse

    num_inputs = option_dict[NUM_INPUTS_KEY]
    dense_layer_neuron_nums = option_dict[DENSE_LAYER_NEURON_NUMS_KEY]
    dense_layer_dropout_rates = option_dict[DENSE_LAYER_DROPOUT_RATES_KEY]
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

    num_layers = len(dense_layer_neuron_nums)
    layer_object = None

    for i in range(num_layers):
        if layer_object is None:
            this_input_layer_object = input_layer_object
        else:
            this_input_layer_object = layer_object

        layer_object = architecture_utils.get_dense_layer(
            num_output_units=dense_layer_neuron_nums[i],
            weight_regularizer=regularizer_object
        )(this_input_layer_object)

        if i == num_layers - 1:
            layer_object = architecture_utils.get_activation_layer(
                activation_function_string=output_activ_function_name,
                alpha_for_relu=output_activ_function_alpha,
                alpha_for_elu=output_activ_function_alpha
            )(layer_object)
        else:
            layer_object = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha
            )(layer_object)

        if dense_layer_dropout_rates[i] > 0:
            layer_object = architecture_utils.get_dropout_layer(
                dropout_fraction=dense_layer_dropout_rates[i]
            )(layer_object)

        if use_batch_normalization and i != num_layers - 1:
            layer_object = architecture_utils.get_batch_norm_layer()(
                layer_object
            )

    model_object = keras.models.Model(
        inputs=input_layer_object, outputs=layer_object
    )

    model_object.compile(
        loss=loss_function, optimizer=keras.optimizers.Adam(),
        metrics=neural_net.METRIC_FUNCTION_LIST
    )

    model_object.summary()
    return model_object
