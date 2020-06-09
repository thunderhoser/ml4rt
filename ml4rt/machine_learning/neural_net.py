"""Methods for building, training, and applying neural nets."""

import numpy
import keras
from gewittergefahr.deep_learning import architecture_utils

DEFAULT_L1_WEIGHT = 0.
DEFAULT_L2_WEIGHT = 0.001

DEFAULT_CONV_LAYER_CHANNEL_NUMS = numpy.array([80, 80, 80, 3], dtype=int)
DEFAULT_CONV_LAYER_DROPOUT_RATES = numpy.array([0.5, 0.5, 0.5, numpy.nan])
DEFAULT_DENSE_NEURON_NUMS_FOR_CNN = numpy.array([409, 29, 2], dtype=int)
DEFAULT_DENSE_DROPOUT_RATES_FOR_CNN = numpy.array([0.5, 0.5, numpy.nan])

# TODO(thunderhoser): Probably want to change this.
DEFAULT_DENSE_NEURON_NUMS_FOR_DNN = numpy.array([1000, 409, 29, 2], dtype=int)
DEFAULT_DENSE_DROPOUT_RATES_FOR_DNN = numpy.array([0.5, 0.5, 0.5, numpy.nan])

DEFAULT_INNER_ACTIV_FUNCTION_NAME = architecture_utils.RELU_FUNCTION_STRING
DEFAULT_INNER_ACTIV_FUNCTION_ALPHA = 0.2
DEFAULT_OUTPUT_ACTIV_FUNCTION_NAME = architecture_utils.RELU_FUNCTION_STRING
DEFAULT_OUTPUT_ACTIV_FUNCTION_ALPHA = 0.

# TODO(thunderhoser): Filter size needs to be an option.


def make_dense_net(
        num_inputs,
        dense_layer_neuron_nums=DEFAULT_DENSE_NEURON_NUMS_FOR_DNN,
        dense_layer_dropout_rates=DEFAULT_DENSE_DROPOUT_RATES_FOR_DNN,
        inner_activ_function_name=DEFAULT_INNER_ACTIV_FUNCTION_NAME,
        inner_activ_function_alpha=DEFAULT_INNER_ACTIV_FUNCTION_ALPHA,
        output_activ_function_name=DEFAULT_OUTPUT_ACTIV_FUNCTION_NAME,
        output_activ_function_alpha=DEFAULT_OUTPUT_ACTIV_FUNCTION_ALPHA,
        l1_weight=DEFAULT_L1_WEIGHT, l2_weight=DEFAULT_L2_WEIGHT,
        use_batch_normalization=True):
    """Makes dense (fully connected) neural net.

    :param num_inputs: Number of input variables (predictors).
    :param dense_layer_neuron_nums: See doc for `make_cnn`.
    :param dense_layer_dropout_rates: Same.
    :param inner_activ_function_name: Same.
    :param inner_activ_function_alpha: Same.
    :param output_activ_function_name: Same.
    :param output_activ_function_alpha: Same.
    :param l1_weight: Same.
    :param l2_weight: Same.
    :param use_batch_normalization: Same.
    :return: model_object: Same.
    """

    input_layer_object = keras.layers.Input(shape=(num_inputs,))
    regularizer_object = architecture_utils.get_weight_regularizer(
        l1_weight=l1_weight, l2_weight=l2_weight
    )

    num_dense_layers = len(dense_layer_neuron_nums)
    layer_object = None

    for i in range(num_dense_layers):
        if layer_object is None:
            this_input_layer_object = input_layer_object
        else:
            this_input_layer_object = layer_object

        layer_object = architecture_utils.get_dense_layer(
            num_output_units=dense_layer_neuron_nums[i],
            weight_regularizer=regularizer_object
        )(this_input_layer_object)

        if i == num_dense_layers - 1:
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

        if use_batch_normalization and i != num_dense_layers - 1:
            layer_object = architecture_utils.get_batch_norm_layer()(
                layer_object
            )

    model_object = keras.models.Model(
        inputs=input_layer_object, outputs=layer_object
    )

    # TODO(thunderhoser): Add bias to metrics.
    model_object.compile(
        loss=keras.losses.mse, optimizer=keras.optimizers.Adam(),
        metrics=['mae']
    )

    model_object.summary()
    return model_object


def make_cnn(
        num_heights, num_input_channels,
        conv_layer_channel_nums=DEFAULT_CONV_LAYER_CHANNEL_NUMS,
        conv_layer_dropout_rates=DEFAULT_CONV_LAYER_DROPOUT_RATES,
        dense_layer_neuron_nums=DEFAULT_DENSE_NEURON_NUMS_FOR_CNN,
        dense_layer_dropout_rates=DEFAULT_DENSE_DROPOUT_RATES_FOR_CNN,
        inner_activ_function_name=DEFAULT_INNER_ACTIV_FUNCTION_NAME,
        inner_activ_function_alpha=DEFAULT_INNER_ACTIV_FUNCTION_ALPHA,
        output_activ_function_name=DEFAULT_OUTPUT_ACTIV_FUNCTION_NAME,
        output_activ_function_alpha=DEFAULT_OUTPUT_ACTIV_FUNCTION_ALPHA,
        l1_weight=DEFAULT_L1_WEIGHT, l2_weight=DEFAULT_L2_WEIGHT,
        use_batch_normalization=True):
    """Makes CNN (convolutional neural net).

    This method only sets up the architecture, loss function, and optimizer,
    then compiles the model.  This method does *not* train the model.

    C = number of convolutional layers
    D = number of dense layers

    :param num_heights: Number of height levels.
    :param num_input_channels: Number of input channels.
    :param conv_layer_channel_nums: length-C numpy array with number of channels
        (filters) produced by each conv layer.  The last value in the array,
        conv_layer_channel_nums[-1], is the number of output channels (profiles
        to be predicted).
    :param conv_layer_dropout_rates: length-C numpy array with dropout rate for
        each conv layer.  Use NaN if you do not want dropout for a particular
        layer.
    :param dense_layer_neuron_nums: length-D numpy array with number of neurons
        (features) produced by each dense layer.  The last value in the array,
        dense_layer_neuron_nums[-1], is the number of output scalars (to be
        predicted).
    :param dense_layer_dropout_rates: length-D numpy array with dropout rate for
        each dense layer.  Use NaN if you do not want dropout for a particular
        layer.
    :param inner_activ_function_name: Name of activation function for all inner
        (non-output) layers.  Must be accepted by ``.
    :param inner_activ_function_alpha: Alpha (slope parameter) for activation
        function for all inner layers.  Applies only to ReLU and eLU.
    :param output_activ_function_name: Same as `inner_activ_function_name` but
        for output layers (profiles and scalars).
    :param output_activ_function_alpha: Same as `inner_activ_function_alpha` but
        for output layers (profiles and scalars).
    :param l1_weight: Weight for L_1 regularization.
    :param l2_weight: Weight for L_2 regularization.
    :param use_batch_normalization: Boolean flag.  If True, will use batch
        normalization after each inner (non-output) layer.
    :return: model_object: Untrained instance of `keras.models.Model`.
    """

    # TODO(thunderhoser): Check input args.

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
            num_kernel_rows=5, num_rows_per_stride=1,
            num_filters=conv_layer_channel_nums[i],
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object
        )(this_input_layer_object)

        if i == num_conv_layers - 1:
            conv_output_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=output_activ_function_name,
                alpha_for_relu=output_activ_function_alpha,
                alpha_for_elu=output_activ_function_alpha
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
                alpha_for_elu=output_activ_function_alpha
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

    model_object = keras.models.Model(
        inputs=input_layer_object,
        outputs=[conv_output_layer_object, dense_output_layer_object]
    )

    # TODO(thunderhoser): Add bias to metrics.
    model_object.compile(
        loss=keras.losses.mse, optimizer=keras.optimizers.Adam(),
        metrics=['mae']
    )

    model_object.summary()
    return model_object
