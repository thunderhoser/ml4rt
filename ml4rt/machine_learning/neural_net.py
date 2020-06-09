"""Methods for building, training, and applying neural nets."""

import numpy
import keras
from gewittergefahr.deep_learning import architecture_utils

DEFAULT_L1_WEIGHT = 0.
DEFAULT_L2_WEIGHT = 0.001

DEFAULT_CONV_LAYER_CHANNEL_NUMS = numpy.array([80, 80, 80, 3], dtype=int)
DEFAULT_CONV_LAYER_DROPOUT_RATES = numpy.array([0.5, 0.5, 0.5, numpy.nan])
DEFAULT_DENSE_LAYER_NEURON_NUMS = numpy.array([409, 29, 2], dtype=int)
DEFAULT_DENSE_LAYER_DROPOUT_RATES = numpy.array([0.5, 0.5, numpy.nan])
DEFAULT_INNER_ACTIVN_LAYER_OBJECT = architecture_utils.get_activation_layer(
    activation_function_string=architecture_utils.RELU_FUNCTION_STRING,
    alpha_for_relu=0.2
)
DEFAULT_OUTPUT_ACTIVN_LAYER_OBJECT = architecture_utils.get_activation_layer(
    activation_function_string=architecture_utils.RELU_FUNCTION_STRING,
    alpha_for_relu=0.
)

# TODO(thunderhoser): Filter size needs to be an option.


def make_cnn(
        num_heights, num_input_channels,
        conv_layer_channel_nums=DEFAULT_CONV_LAYER_CHANNEL_NUMS,
        conv_layer_dropout_rates=DEFAULT_CONV_LAYER_DROPOUT_RATES,
        dense_layer_neuron_nums=DEFAULT_DENSE_LAYER_NEURON_NUMS,
        dense_layer_dropout_rates=DEFAULT_DENSE_LAYER_DROPOUT_RATES,
        inner_activn_layer_object=DEFAULT_INNER_ACTIVN_LAYER_OBJECT,
        output_activn_layer_object=DEFAULT_OUTPUT_ACTIVN_LAYER_OBJECT,
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
    :param inner_activn_layer_object: Keras layer specifying activation function
        for all inner (non-output) layers.
    :param output_activn_layer_object: Keras layer specifying activation
        function for both output layers (profiles and scalars).
    :param l1_weight: Weight for L_1 regularization.
    :param l2_weight: Weight for L_2 regularization.
    :param use_batch_normalization: Boolean flag.  If True, will use batch
        normalization after each inner (non-output) layer.
    :return: model_object: Untrained instance of `keras.models.Model`.
    """

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
            conv_output_layer_object = output_activn_layer_object(
                conv_output_layer_object
            )
        else:
            conv_output_layer_object = inner_activn_layer_object(
                conv_output_layer_object
            )

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
            dense_output_layer_object = output_activn_layer_object(
                dense_output_layer_object
            )
        else:
            dense_output_layer_object = inner_activn_layer_object(
                dense_output_layer_object
            )

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
