"""Methods for building U-nets."""

import keras
from keras import backend as K
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import architecture_utils
from ml4rt.machine_learning import neural_net

NUM_HEIGHTS_KEY = 'num_heights'
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
            orig_prediction_tensor[:, height_index, heating_rate_channel_index],
            1e12
        )
        zero_tensor = K.cast(zero_tensor, dtype=K.floatx())

        print(zero_tensor.get_shape().as_list())

        heating_rate_tensor = K.concatenate((
            orig_prediction_tensor[:, heating_rate_channel_index][
                :, :height_index
            ],
            K.expand_dims(zero_tensor, axis=-1)
        ), axis=-1)

        print(heating_rate_tensor.get_shape().as_list())

        if height_index != num_heights - 1:
            heating_rate_tensor = K.concatenate((
                heating_rate_tensor,
                orig_prediction_tensor[:, heating_rate_channel_index][
                    :, (height_index + 1):
                ]
            ), axis=-1)

        print(heating_rate_tensor.get_shape().as_list())

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


def create_model(option_dict, loss_function):
    """Creates U-net.

    This method sets up the architecture, loss function, and optimizer -- and
    compiles the model -- but does not train it.

    Architecture taken from:
    https://github.com/zhixuhao/unet/blob/master/model.py

    :param option_dict: Dictionary with the following keys.
    option_dict['num_heights']: Number of height levels.
    option_dict['num_heights_for_loss']: Number of height levels to use in loss
        function.  Will use only the bottom N height levels, where N is
        `num_heights_for_loss`.
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

    :param loss_function: Function handle.
    :return: model_object: Instance of `keras.models.Model`, with the
        aforementioned architecture.
    """

    # TODO(thunderhoser): Generalize this method a bit.

    option_dict = _check_architecture_args(option_dict)

    num_heights = option_dict[NUM_HEIGHTS_KEY]
    num_heights_for_loss = option_dict[NUM_HEIGHTS_FOR_LOSS_KEY]
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

    input_layer_object = keras.layers.Input(
        shape=(num_heights, num_input_channels)
    )
    regularizer_object = architecture_utils.get_weight_regularizer(
        l1_weight=l1_weight, l2_weight=l2_weight
    )

    conv_layer1_object = None

    for i in range(2):
        if i == 0:
            this_input_layer_object = input_layer_object
        else:
            this_input_layer_object = conv_layer1_object

        conv_layer1_object = architecture_utils.get_1d_conv_layer(
            num_kernel_rows=3, num_rows_per_stride=1, num_filters=64,
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object
        )(this_input_layer_object)

        conv_layer1_object = architecture_utils.get_activation_layer(
            activation_function_string=inner_activ_function_name,
            alpha_for_relu=inner_activ_function_alpha,
            alpha_for_elu=inner_activ_function_alpha
        )(conv_layer1_object)

        conv_layer1_object = architecture_utils.get_dropout_layer(
            dropout_fraction=0.5
        )(conv_layer1_object)

        if use_batch_normalization:
            conv_layer1_object = architecture_utils.get_batch_norm_layer()(
                conv_layer1_object
            )

    pooling_layer1_object = architecture_utils.get_1d_pooling_layer(
        num_rows_in_window=2, num_rows_per_stride=2,
        pooling_type_string=architecture_utils.MAX_POOLING_STRING
    )(conv_layer1_object)

    conv_layer2_object = None

    for i in range(2):
        if i == 0:
            this_input_layer_object = pooling_layer1_object
        else:
            this_input_layer_object = conv_layer2_object

        conv_layer2_object = architecture_utils.get_1d_conv_layer(
            num_kernel_rows=3, num_rows_per_stride=1, num_filters=128,
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object
        )(this_input_layer_object)

        conv_layer2_object = architecture_utils.get_activation_layer(
            activation_function_string=inner_activ_function_name,
            alpha_for_relu=inner_activ_function_alpha,
            alpha_for_elu=inner_activ_function_alpha
        )(conv_layer2_object)

        conv_layer2_object = architecture_utils.get_dropout_layer(
            dropout_fraction=0.5
        )(conv_layer2_object)

        if use_batch_normalization:
            conv_layer2_object = architecture_utils.get_batch_norm_layer()(
                conv_layer2_object
            )

    pooling_layer2_object = architecture_utils.get_1d_pooling_layer(
        num_rows_in_window=2, num_rows_per_stride=2,
        pooling_type_string=architecture_utils.MAX_POOLING_STRING
    )(conv_layer2_object)

    conv_layer3_object = None

    for i in range(2):
        if i == 0:
            this_input_layer_object = pooling_layer2_object
        else:
            this_input_layer_object = conv_layer3_object

        conv_layer3_object = architecture_utils.get_1d_conv_layer(
            num_kernel_rows=3, num_rows_per_stride=1, num_filters=256,
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object
        )(this_input_layer_object)

        conv_layer3_object = architecture_utils.get_activation_layer(
            activation_function_string=inner_activ_function_name,
            alpha_for_relu=inner_activ_function_alpha,
            alpha_for_elu=inner_activ_function_alpha
        )(conv_layer3_object)

        conv_layer3_object = architecture_utils.get_dropout_layer(
            dropout_fraction=0.5
        )(conv_layer3_object)

        if use_batch_normalization:
            conv_layer3_object = architecture_utils.get_batch_norm_layer()(
                conv_layer3_object
            )

    pooling_layer3_object = architecture_utils.get_1d_pooling_layer(
        num_rows_in_window=2, num_rows_per_stride=2,
        pooling_type_string=architecture_utils.MAX_POOLING_STRING
    )(conv_layer3_object)

    conv_layer4_object = None

    for i in range(2):
        if i == 0:
            this_input_layer_object = pooling_layer3_object
        else:
            this_input_layer_object = conv_layer4_object

        conv_layer4_object = architecture_utils.get_1d_conv_layer(
            num_kernel_rows=3, num_rows_per_stride=1, num_filters=512,
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object
        )(this_input_layer_object)

        conv_layer4_object = architecture_utils.get_activation_layer(
            activation_function_string=inner_activ_function_name,
            alpha_for_relu=inner_activ_function_alpha,
            alpha_for_elu=inner_activ_function_alpha
        )(conv_layer4_object)

        conv_layer4_object = architecture_utils.get_dropout_layer(
            dropout_fraction=0.5
        )(conv_layer4_object)

        if use_batch_normalization:
            conv_layer4_object = architecture_utils.get_batch_norm_layer()(
                conv_layer4_object
            )

    pooling_layer4_object = architecture_utils.get_1d_pooling_layer(
        num_rows_in_window=2, num_rows_per_stride=2,
        pooling_type_string=architecture_utils.MAX_POOLING_STRING
    )(conv_layer4_object)

    conv_layer5_object = None

    for i in range(2):
        if i == 0:
            this_input_layer_object = pooling_layer4_object
        else:
            this_input_layer_object = conv_layer5_object

        conv_layer5_object = architecture_utils.get_1d_conv_layer(
            num_kernel_rows=3, num_rows_per_stride=1, num_filters=1024,
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object
        )(this_input_layer_object)

        conv_layer5_object = architecture_utils.get_activation_layer(
            activation_function_string=inner_activ_function_name,
            alpha_for_relu=inner_activ_function_alpha,
            alpha_for_elu=inner_activ_function_alpha
        )(conv_layer5_object)

        conv_layer5_object = architecture_utils.get_dropout_layer(
            dropout_fraction=0.5
        )(conv_layer5_object)

        if use_batch_normalization:
            conv_layer5_object = architecture_utils.get_batch_norm_layer()(
                conv_layer5_object
            )

    this_layer_object = keras.layers.UpSampling1D(size=2)

    upconv_layer4_object = architecture_utils.get_1d_conv_layer(
        num_kernel_rows=2, num_rows_per_stride=1, num_filters=512,
        padding_type_string=architecture_utils.YES_PADDING_STRING,
        weight_regularizer=regularizer_object
    )(this_layer_object(conv_layer5_object))

    merged_layer4_object = keras.layers.Concatenate(axis=-1)(
        [conv_layer4_object, upconv_layer4_object]
    )

    second_conv_layer4_object = None

    for i in range(2):
        if i == 0:
            this_input_layer_object = merged_layer4_object
        else:
            this_input_layer_object = second_conv_layer4_object

        second_conv_layer4_object = architecture_utils.get_1d_conv_layer(
            num_kernel_rows=3, num_rows_per_stride=1, num_filters=512,
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object
        )(this_input_layer_object)

        second_conv_layer4_object = architecture_utils.get_activation_layer(
            activation_function_string=inner_activ_function_name,
            alpha_for_relu=inner_activ_function_alpha,
            alpha_for_elu=inner_activ_function_alpha
        )(second_conv_layer4_object)

        if use_batch_normalization:
            second_conv_layer4_object = (
                architecture_utils.get_batch_norm_layer()(
                    second_conv_layer4_object
                )
            )

    this_layer_object = keras.layers.UpSampling1D(size=2)

    upconv_layer3_object = architecture_utils.get_1d_conv_layer(
        num_kernel_rows=2, num_rows_per_stride=1, num_filters=256,
        padding_type_string=architecture_utils.YES_PADDING_STRING,
        weight_regularizer=regularizer_object
    )(this_layer_object(second_conv_layer4_object))

    merged_layer3_object = keras.layers.Concatenate(axis=-1)(
        [conv_layer3_object, upconv_layer3_object]
    )

    second_conv_layer3_object = None

    for i in range(2):
        if i == 0:
            this_input_layer_object = merged_layer3_object
        else:
            this_input_layer_object = second_conv_layer3_object

        second_conv_layer3_object = architecture_utils.get_1d_conv_layer(
            num_kernel_rows=3, num_rows_per_stride=1, num_filters=256,
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object
        )(this_input_layer_object)

        second_conv_layer3_object = architecture_utils.get_activation_layer(
            activation_function_string=inner_activ_function_name,
            alpha_for_relu=inner_activ_function_alpha,
            alpha_for_elu=inner_activ_function_alpha
        )(second_conv_layer3_object)

        if use_batch_normalization:
            second_conv_layer3_object = (
                architecture_utils.get_batch_norm_layer()(
                    second_conv_layer3_object
                )
            )

    this_layer_object = keras.layers.UpSampling1D(size=2)

    upconv_layer2_object = architecture_utils.get_1d_conv_layer(
        num_kernel_rows=2, num_rows_per_stride=1, num_filters=128,
        padding_type_string=architecture_utils.YES_PADDING_STRING,
        weight_regularizer=regularizer_object
    )(this_layer_object(second_conv_layer3_object))

    merged_layer2_object = keras.layers.Concatenate(axis=-1)(
        [conv_layer2_object, upconv_layer2_object]
    )

    second_conv_layer2_object = None

    for i in range(2):
        if i == 0:
            this_input_layer_object = merged_layer2_object
        else:
            this_input_layer_object = second_conv_layer2_object

        second_conv_layer2_object = architecture_utils.get_1d_conv_layer(
            num_kernel_rows=3, num_rows_per_stride=1, num_filters=128,
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object
        )(this_input_layer_object)

        second_conv_layer2_object = architecture_utils.get_activation_layer(
            activation_function_string=inner_activ_function_name,
            alpha_for_relu=inner_activ_function_alpha,
            alpha_for_elu=inner_activ_function_alpha
        )(second_conv_layer2_object)

        if use_batch_normalization:
            second_conv_layer2_object = (
                architecture_utils.get_batch_norm_layer()(
                    second_conv_layer2_object
                )
            )

    this_layer_object = keras.layers.UpSampling1D(size=2)

    upconv_layer1_object = architecture_utils.get_1d_conv_layer(
        num_kernel_rows=2, num_rows_per_stride=1, num_filters=64,
        padding_type_string=architecture_utils.YES_PADDING_STRING,
        weight_regularizer=regularizer_object
    )(this_layer_object(second_conv_layer2_object))

    merged_layer1_object = keras.layers.Concatenate(axis=-1)(
        [conv_layer1_object, upconv_layer1_object]
    )

    second_conv_layer1_object = None

    for i in range(3):
        if i == 0:
            this_input_layer_object = merged_layer1_object
        else:
            this_input_layer_object = second_conv_layer1_object

        second_conv_layer1_object = architecture_utils.get_1d_conv_layer(
            num_kernel_rows=3, num_rows_per_stride=1,
            num_filters=4 if i == 2 else 64,
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object
        )(this_input_layer_object)

        second_conv_layer1_object = architecture_utils.get_activation_layer(
            activation_function_string=inner_activ_function_name,
            alpha_for_relu=inner_activ_function_alpha,
            alpha_for_elu=inner_activ_function_alpha
        )(second_conv_layer1_object)

        if use_batch_normalization:
            second_conv_layer1_object = (
                architecture_utils.get_batch_norm_layer()(
                    second_conv_layer1_object
                )
            )

    second_conv_layer1_object = architecture_utils.get_1d_conv_layer(
        num_kernel_rows=1, num_rows_per_stride=1, num_filters=2,
        padding_type_string=architecture_utils.YES_PADDING_STRING,
        weight_regularizer=regularizer_object
    )(second_conv_layer1_object)

    second_conv_layer1_object = architecture_utils.get_activation_layer(
        activation_function_string=output_activ_function_name,
        alpha_for_relu=output_activ_function_alpha,
        alpha_for_elu=output_activ_function_alpha
    )(second_conv_layer1_object)

    if num_heights > num_heights_for_loss:
        this_function = (
            _zero_top_heights_function(num_heights - num_heights_for_loss)
        )
        second_conv_layer1_object = keras.layers.Lambda(this_function)(
            second_conv_layer1_object
        )

    if zero_out_top_heating_rate:
        this_function = _zero_top_heating_rate_function(
            heating_rate_channel_index=heating_rate_channel_index,
            height_index=num_heights - 1
        )
        second_conv_layer1_object = keras.layers.Lambda(this_function)(
            second_conv_layer1_object
        )

    model_object = keras.models.Model(
        input=input_layer_object, output=second_conv_layer1_object
    )

    model_object.compile(
        loss=loss_function, optimizer=keras.optimizers.Adam(),
        metrics=neural_net.METRIC_FUNCTION_LIST
    )

    model_object.summary()
    return model_object
