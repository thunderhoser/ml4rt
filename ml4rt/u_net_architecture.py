"""Methods for building U-nets."""

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

INPUT_DIMENSIONS_KEY = 'input_dimensions'
NUM_LEVELS_KEY = 'num_levels'
CONV_LAYER_COUNTS_KEY = 'num_conv_layers_by_level'
CHANNEL_COUNTS_KEY = 'num_channels_by_level'
ENCODER_DROPOUT_RATES_KEY = 'encoder_dropout_rate_by_level'
ENCODER_MC_DROPOUT_FLAGS_KEY = 'encoder_mc_dropout_flag_by_level'
UPCONV_DROPOUT_RATES_KEY = 'upconv_dropout_rate_by_level'
UPCONV_MC_DROPOUT_FLAGS_KEY = 'upconv_mc_dropout_flag_by_level'
SKIP_DROPOUT_RATES_KEY = 'skip_dropout_rate_by_level'
SKIP_MC_DROPOUT_FLAGS_KEY = 'skip_mc_dropout_flag_by_level'
INCLUDE_PENULTIMATE_KEY = 'include_penultimate_conv'
PENULTIMATE_DROPOUT_RATE_KEY = 'penultimate_conv_dropout_rate'
PENULTIMATE_MC_DROPOUT_FLAG_KEY = 'penultimate_conv_mc_dropout_flag'
DENSE_LAYER_NEURON_NUMS_KEY = 'dense_layer_neuron_nums'
DENSE_LAYER_DROPOUT_RATES_KEY = 'dense_layer_dropout_rates'
DENSE_LAYER_MC_DROPOUT_FLAGS_KEY = 'dense_layer_mc_dropout_flags'
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

NUM_OUTPUT_WAVELENGTHS_KEY = 'num_output_wavelengths'
VECTOR_LOSS_FUNCTION_KEY = 'vector_loss_function'
SCALAR_LOSS_FUNCTION_KEY = 'scalar_loss_function'
JOINED_LOSS_FUNCTION_KEY = 'joined_loss_function'
OPTIMIZER_FUNCTION_KEY = 'optimizer_function'
USE_DEEP_SUPERVISION_KEY = 'use_deep_supervision'
ENSEMBLE_SIZE_KEY = 'ensemble_size'

DO_INLINE_NORMALIZATION_KEY = 'do_inline_normalization'
PW_LINEAR_UNIF_MODEL_KEY = 'pw_linear_unif_model_file_name'
SCALAR_PREDICTORS_KEY = 'scalar_predictor_names'
VECTOR_PREDICTORS_KEY = 'vector_predictor_names'
HEIGHTS_KEY = 'heights_m_agl'

DEFAULT_ARCHITECTURE_OPTION_DICT = {
    NUM_LEVELS_KEY: 4,
    CONV_LAYER_COUNTS_KEY: numpy.full(5, 2, dtype=int),
    CHANNEL_COUNTS_KEY: numpy.array([64, 128, 256, 512, 1024], dtype=int),
    ENCODER_DROPOUT_RATES_KEY: numpy.full(5, 0.),
    ENCODER_MC_DROPOUT_FLAGS_KEY: numpy.full(5, 0, dtype=bool),
    UPCONV_DROPOUT_RATES_KEY: numpy.full(4, 0.),
    UPCONV_MC_DROPOUT_FLAGS_KEY: numpy.full(4, 0, dtype=bool),
    SKIP_DROPOUT_RATES_KEY: numpy.full(4, 0.),
    SKIP_MC_DROPOUT_FLAGS_KEY: numpy.full(4, 0, dtype=bool),
    INCLUDE_PENULTIMATE_KEY: True,
    PENULTIMATE_DROPOUT_RATE_KEY: 0.,
    PENULTIMATE_MC_DROPOUT_FLAG_KEY: False,
    DENSE_LAYER_NEURON_NUMS_KEY: None,
    DENSE_LAYER_DROPOUT_RATES_KEY: None,
    DENSE_LAYER_MC_DROPOUT_FLAGS_KEY: None,
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


def check_args(option_dict):
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
    option_dict['encoder_mc_dropout_flag_by_level']: length-(L + 1) numpy array
        of Boolean flags, indicating which layers will use Monte Carlo dropout.
    option_dict['upconv_dropout_rate_by_level']: length-L numpy array
        with dropout rate for upconv layers at each level.
    option_dict['upconv_mc_dropout_flag_by_level']: length-L numpy array of
        Boolean flags, indicating which layers will use Monte Carlo dropout.
    option_dict['skip_dropout_rate_by_level']: length-L numpy array with dropout
        rate for conv layer after skip connection at each level.
    option_dict['skip_mc_dropout_flag_by_level']: length-L numpy array of
        Boolean flags, indicating which layers will use Monte Carlo dropout.
    option_dict['include_penultimate_conv']: Boolean flag.  If True, will put in
        extra conv layer (with 3 x 3 filter) before final pixelwise conv.
    option_dict['penultimate_conv_dropout_rate']: Dropout rate for penultimate
        conv layer.
    option_dict['penultimate_conv_mc_dropout_flag']: Boolean flag, indicating
        whether or not penultimate conv layer will use MC dropout.
    option_dict['dense_layer_neuron_nums']: length-D numpy array with number of
        neurons in each dense layer.
    option_dict['dense_layer_dropout_rates']: length-D numpy array with dropout
        rate for each dense layer.
    option_dict['dense_layer_mc_dropout_flags']: length-D numpy array of Boolean
        flags, indicating which layers will use Monte Carlo dropout.
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
        batch normalization after each inner (non-output) conv layer.
    option_dict['use_residual_blocks']: Boolean flag.  If True, will use
        residual blocks (basic conv blocks) throughout the architecture.
    option_dict['num_output_wavelengths']: Number of output wavelengths.
    option_dict['vector_loss_function']: Loss function for vector targets.
    option_dict['scalar_loss_function']: Loss function for scalar targets.
    option_dict['use_deep_supervision']: Boolean flag.
    option_dict['ensemble_size']: Number of ensemble members in output (both
        vector and scalar predictions).
    option_dict['do_inline_normalization']: Boolean flag.  If True, will
        normalize predictors inside the NN architecture.
    option_dict['pw_linear_unif_model_file_name']:
        [used only if do_inline_normalization == True]
        Path to file with piecewise-linear models that approximate the full
        uniformization step.  This will be read by
        `normalization.read_piecewise_linear_models_for_unif`
    option_dict['scalar_predictor_names']:
        [used only if do_inline_normalization == True]
        1-D list with names of scalar predictors, in the order that they appear
        in the input tensor.
    option_dict['vector_predictor_names']:
        [used only if do_inline_normalization == True]
        1-D list with names of vector predictors, in the order that they appear
        in the input tensor.
    option_dict['heights_m_agl']:
        [used only if do_inline_normalization == True]
        1-D numpy array of heights (metres above ground level), in the order
        that they appear in the input tensor.

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

    encoder_mc_dropout_flag_by_level = option_dict[ENCODER_MC_DROPOUT_FLAGS_KEY]
    error_checking.assert_is_numpy_array(
        encoder_mc_dropout_flag_by_level, exact_dimensions=expected_dim
    )
    error_checking.assert_is_boolean_numpy_array(
        encoder_mc_dropout_flag_by_level
    )

    expected_dim = numpy.array([num_levels], dtype=int)

    upconv_dropout_rate_by_level = option_dict[UPCONV_DROPOUT_RATES_KEY]
    error_checking.assert_is_numpy_array(
        upconv_dropout_rate_by_level, exact_dimensions=expected_dim
    )
    error_checking.assert_is_leq_numpy_array(
        upconv_dropout_rate_by_level, 1., allow_nan=True
    )

    upconv_mc_dropout_flag_by_level = option_dict[UPCONV_MC_DROPOUT_FLAGS_KEY]
    error_checking.assert_is_numpy_array(
        upconv_mc_dropout_flag_by_level, exact_dimensions=expected_dim
    )
    error_checking.assert_is_boolean_numpy_array(
        upconv_mc_dropout_flag_by_level
    )

    skip_dropout_rate_by_level = option_dict[SKIP_DROPOUT_RATES_KEY]
    error_checking.assert_is_numpy_array(
        skip_dropout_rate_by_level, exact_dimensions=expected_dim
    )
    error_checking.assert_is_leq_numpy_array(
        skip_dropout_rate_by_level, 1., allow_nan=True
    )

    skip_mc_dropout_flag_by_level = option_dict[SKIP_MC_DROPOUT_FLAGS_KEY]
    error_checking.assert_is_numpy_array(
        skip_mc_dropout_flag_by_level, exact_dimensions=expected_dim
    )
    error_checking.assert_is_boolean_numpy_array(
        skip_mc_dropout_flag_by_level
    )

    error_checking.assert_is_boolean(option_dict[INCLUDE_PENULTIMATE_KEY])
    error_checking.assert_is_leq(
        option_dict[PENULTIMATE_DROPOUT_RATE_KEY], 1., allow_nan=True
    )
    error_checking.assert_is_boolean(
        option_dict[PENULTIMATE_MC_DROPOUT_FLAG_KEY]
    )

    dense_layer_neuron_nums = option_dict[DENSE_LAYER_NEURON_NUMS_KEY]
    dense_layer_dropout_rates = option_dict[DENSE_LAYER_DROPOUT_RATES_KEY]
    dense_layer_mc_dropout_flags = option_dict[DENSE_LAYER_MC_DROPOUT_FLAGS_KEY]
    has_dense_layers = not (
        dense_layer_neuron_nums is None
        and dense_layer_dropout_rates is None
        and dense_layer_mc_dropout_flags is None
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

        error_checking.assert_is_numpy_array(
            dense_layer_mc_dropout_flags, exact_dimensions=expected_dim
        )
        error_checking.assert_is_boolean_numpy_array(
            dense_layer_mc_dropout_flags
        )

    error_checking.assert_is_geq(option_dict[L1_WEIGHT_KEY], 0.)
    error_checking.assert_is_geq(option_dict[L2_WEIGHT_KEY], 0.)
    error_checking.assert_is_boolean(option_dict[USE_BATCH_NORM_KEY])

    if USE_RESIDUAL_BLOCKS_KEY not in option_dict:
        option_dict[USE_RESIDUAL_BLOCKS_KEY] = False

    error_checking.assert_is_boolean(option_dict[USE_RESIDUAL_BLOCKS_KEY])

    error_checking.assert_is_integer(option_dict[NUM_OUTPUT_WAVELENGTHS_KEY])
    error_checking.assert_is_greater(option_dict[NUM_OUTPUT_WAVELENGTHS_KEY], 0)
    error_checking.assert_is_boolean(option_dict[USE_DEEP_SUPERVISION_KEY])
    error_checking.assert_is_integer(option_dict[ENSEMBLE_SIZE_KEY])
    error_checking.assert_is_greater(option_dict[ENSEMBLE_SIZE_KEY], 0)
    error_checking.assert_is_boolean(option_dict[DO_INLINE_NORMALIZATION_KEY])

    if OPTIMIZER_FUNCTION_KEY not in option_dict:
        option_dict[OPTIMIZER_FUNCTION_KEY] = keras.optimizers.Nadam()

    if not option_dict[DO_INLINE_NORMALIZATION_KEY]:
        option_dict[PW_LINEAR_UNIF_MODEL_KEY] = None
        option_dict[SCALAR_PREDICTORS_KEY] = None
        option_dict[VECTOR_PREDICTORS_KEY] = None
        option_dict[HEIGHTS_KEY] = None

        return option_dict

    error_checking.assert_is_string(option_dict[PW_LINEAR_UNIF_MODEL_KEY])
    error_checking.assert_is_string_list(option_dict[SCALAR_PREDICTORS_KEY])
    error_checking.assert_is_string_list(option_dict[VECTOR_PREDICTORS_KEY])
    error_checking.assert_is_numpy_array(
        option_dict[HEIGHTS_KEY], num_dimensions=1
    )
    error_checking.assert_is_greater_numpy_array(option_dict[HEIGHTS_KEY], 0)

    num_predictors = len(
        option_dict[SCALAR_PREDICTORS_KEY] + option_dict[VECTOR_PREDICTORS_KEY]
    )
    error_checking.assert_equals(num_predictors, input_dimensions[1])
    num_heights = len(option_dict[HEIGHTS_KEY])
    error_checking.assert_equals(num_heights, input_dimensions[0])

    return option_dict


def zero_top_heating_rate(input_layer_object, ensemble_size, output_layer_name):
    """Zeroes out heating rate at top of atmosphere.

    :param input_layer_object: Input layer, containing predicted heating rates.
    :param ensemble_size: Number of ensemble members.
    :param output_layer_name: Name of output layer.
    :return: output_layer_object: Same as input but with zeros at TOA.
    """

    if ensemble_size > 1:
        cropping_arg = ((0, 1), (0, 0), (0, 0))
        output_layer_object = keras.layers.Cropping3D(
            cropping=cropping_arg
        )(input_layer_object)

        output_layer_object = keras.layers.ZeroPadding3D(
            padding=cropping_arg, name=output_layer_name
        )(output_layer_object)
    else:
        cropping_arg = ((0, 1), (0, 0))
        output_layer_object = keras.layers.Cropping2D(
            cropping=cropping_arg
        )(input_layer_object)

        output_layer_object = keras.layers.ZeroPadding2D(
            padding=cropping_arg, name=output_layer_name
        )(output_layer_object)

    return output_layer_object


def get_conv_block(
        input_layer_object, do_residual, num_conv_layers, filter_size_px,
        num_filters, regularizer_object,
        activation_function_name, activation_function_alpha,
        dropout_rates, monte_carlo_dropout_flags, use_batch_norm,
        basic_layer_name):
    """Creates convolutional block.

    L = number of conv layers

    :param input_layer_object: Input layer to block.
    :param do_residual: Boolean flag.  If True (False), this will be a residual
        (basic convolutional) block.
    :param num_conv_layers: Number of conv layers in block.
    :param filter_size_px: Filter size (the same for every conv layer).
    :param num_filters: Number of filters -- same for every conv layer.
    :param regularizer_object: Regularizer for conv layers (instance of
        `keras.regularizers.l1_l2` or similar).
    :param activation_function_name: Name of activation function -- same for
        every conv layer.  Must be accepted by
        `architecture_utils.check_activation_function`.
    :param activation_function_alpha: Alpha (slope parameter) for activation
        function -- same for every conv layer.  Applies only to ReLU and eLU.
    :param dropout_rates: Dropout rates for conv layers.  This can be a scalar
        (applied to every conv layer) or length-L numpy array.
    :param monte_carlo_dropout_flags: Boolean flag, indicating whether dropout
        is standard or Monte Carlo.  This can be a single value (applied to
        every conv layer) or length-L numpy array.
    :param use_batch_norm: Boolean flag.  If True, will use batch normalization.
    :param basic_layer_name: Basic layer name.  Each layer name will be made
        unique by adding a suffix.
    :return: output_layer_object: Output layer from block.
    """

    # Process input args.
    if do_residual:
        num_conv_layers = max([num_conv_layers, 2])

    try:
        _ = len(dropout_rates)
    except:
        dropout_rates = numpy.full(num_conv_layers, dropout_rates)
        monte_carlo_dropout_flags = numpy.full(
            num_conv_layers, monte_carlo_dropout_flags, dtype=bool
        )

    if len(dropout_rates) < num_conv_layers:
        dropout_rates = numpy.concatenate([
            dropout_rates, dropout_rates[[-1]]
        ])
        monte_carlo_dropout_flags = numpy.concatenate([
            monte_carlo_dropout_flags, monte_carlo_dropout_flags[[-1]]
        ])

    assert len(dropout_rates) == num_conv_layers
    assert len(monte_carlo_dropout_flags) == num_conv_layers

    # Do actual stuff.
    current_layer_object = None

    for i in range(num_conv_layers):
        if i == 0:
            this_input_layer_object = input_layer_object
        else:
            this_input_layer_object = current_layer_object

        this_name = '{0:s}_conv{1:d}'.format(basic_layer_name, i)
        current_layer_object = architecture_utils.get_1d_conv_layer(
            num_kernel_rows=filter_size_px,
            num_rows_per_stride=1,
            num_filters=num_filters,
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object,
            layer_name=this_name
        )(this_input_layer_object)

        if i == num_conv_layers - 1 and do_residual:
            if input_layer_object.shape[-1] == num_filters:
                new_layer_object = input_layer_object
            else:
                this_name = '{0:s}_preresidual_conv'.format(basic_layer_name)
                new_layer_object = architecture_utils.get_1d_conv_layer(
                    num_kernel_rows=filter_size_px,
                    num_rows_per_stride=1,
                    num_filters=num_filters,
                    padding_type_string=architecture_utils.YES_PADDING_STRING,
                    weight_regularizer=regularizer_object,
                    layer_name=this_name
                )(input_layer_object)

            this_name = '{0:s}_residual'.format(basic_layer_name)
            current_layer_object = keras.layers.Add(name=this_name)([
                current_layer_object, new_layer_object
            ])

        if activation_function_name is not None:
            this_name = '{0:s}_activ{1:d}'.format(basic_layer_name, i)
            current_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=activation_function_name,
                alpha_for_relu=activation_function_alpha,
                alpha_for_elu=activation_function_alpha,
                layer_name=this_name
            )(current_layer_object)

        if dropout_rates[i] > 0:
            this_name = '{0:s}_dropout{1:d}'.format(basic_layer_name, i)

            current_layer_object = architecture_utils.get_dropout_layer(
                dropout_fraction=dropout_rates[i], layer_name=this_name
            )(
                current_layer_object,
                training=bool(monte_carlo_dropout_flags[i])
            )

        if use_batch_norm:
            this_name = '{0:s}_bn{1:d}'.format(basic_layer_name, i)
            current_layer_object = architecture_utils.get_batch_norm_layer(
                layer_name=this_name
            )(current_layer_object)

    return current_layer_object


def create_model(option_dict):
    """Creates U-net.

    This method sets up the architecture, loss function, and optimizer -- and
    compiles the model -- but does not train it.

    Architecture taken from:
    https://github.com/zhixuhao/unet/blob/master/model.py

    :param option_dict: See doc for `check_args`.
    :return: model_object: Instance of `keras.models.Model`, with the
        aforementioned architecture.
    """

    option_dict[USE_DEEP_SUPERVISION_KEY] = False
    option_dict[ENSEMBLE_SIZE_KEY] = 1
    option_dict[DO_INLINE_NORMALIZATION_KEY] = False
    option_dict = check_args(option_dict)

    input_dimensions = option_dict[INPUT_DIMENSIONS_KEY]
    num_levels = option_dict[NUM_LEVELS_KEY]
    num_conv_layers_by_level = option_dict[CONV_LAYER_COUNTS_KEY]
    num_channels_by_level = option_dict[CHANNEL_COUNTS_KEY]
    encoder_dropout_rate_by_level = option_dict[ENCODER_DROPOUT_RATES_KEY]
    encoder_mc_dropout_flag_by_level = option_dict[ENCODER_MC_DROPOUT_FLAGS_KEY]
    upconv_dropout_rate_by_level = option_dict[UPCONV_DROPOUT_RATES_KEY]
    upconv_mc_dropout_flag_by_level = option_dict[UPCONV_MC_DROPOUT_FLAGS_KEY]
    skip_dropout_rate_by_level = option_dict[SKIP_DROPOUT_RATES_KEY]
    skip_mc_dropout_flag_by_level = option_dict[SKIP_MC_DROPOUT_FLAGS_KEY]
    include_penultimate_conv = option_dict[INCLUDE_PENULTIMATE_KEY]
    penultimate_conv_dropout_rate = option_dict[PENULTIMATE_DROPOUT_RATE_KEY]
    penultimate_conv_mc_dropout_flag = (
        option_dict[PENULTIMATE_MC_DROPOUT_FLAG_KEY]
    )
    dense_layer_neuron_nums = option_dict[DENSE_LAYER_NEURON_NUMS_KEY]
    dense_layer_dropout_rates = option_dict[DENSE_LAYER_DROPOUT_RATES_KEY]
    dense_layer_mc_dropout_flags = option_dict[DENSE_LAYER_MC_DROPOUT_FLAGS_KEY]
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

    # TODO(thunderhoser): Need to implement residual blocks for this method.
    assert use_residual_blocks == False

    num_output_wavelengths = option_dict[NUM_OUTPUT_WAVELENGTHS_KEY]
    vector_loss_function = option_dict[VECTOR_LOSS_FUNCTION_KEY]
    scalar_loss_function = option_dict[SCALAR_LOSS_FUNCTION_KEY]
    ensemble_size = option_dict[ENSEMBLE_SIZE_KEY]

    # TODO(thunderhoser): Need to use optimizer_function arg for this method.

    if ensemble_size > 1:
        include_penultimate_conv = False
        metric_function_list = []
    else:
        metric_function_list = neural_net.METRIC_FUNCTION_LIST

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
                this_mc_flag = bool(encoder_mc_dropout_flag_by_level[i])

                conv_layer_by_level[i] = architecture_utils.get_dropout_layer(
                    dropout_fraction=encoder_dropout_rate_by_level[i],
                    layer_name=this_name
                )(conv_layer_by_level[i], training=this_mc_flag)

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
            this_mc_flag = bool(upconv_mc_dropout_flag_by_level[i])

            upconv_layer_by_level[i] = architecture_utils.get_dropout_layer(
                dropout_fraction=upconv_dropout_rate_by_level[i],
                layer_name=this_name
            )(upconv_layer_by_level[i], training=this_mc_flag)

        num_upconv_heights = upconv_layer_by_level[i].shape[1]
        num_desired_heights = conv_layer_by_level[i].shape[1]

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
                this_mc_flag = bool(skip_mc_dropout_flag_by_level[i])

                skip_layer_by_level[i] = architecture_utils.get_dropout_layer(
                    dropout_fraction=skip_dropout_rate_by_level[i],
                    layer_name=this_name
                )(skip_layer_by_level[i], training=this_mc_flag)

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
            num_filters=2 * num_output_wavelengths * ensemble_size,
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
            this_mc_flag = bool(penultimate_conv_mc_dropout_flag)

            skip_layer_by_level[0] = architecture_utils.get_dropout_layer(
                dropout_fraction=penultimate_conv_dropout_rate,
                layer_name='penultimate_conv_dropout'
            )(skip_layer_by_level[0], training=this_mc_flag)

        if use_batch_normalization:
            skip_layer_by_level[0] = architecture_utils.get_batch_norm_layer(
                layer_name='penultimate_conv_bn'
            )(skip_layer_by_level[0])

    conv_output_layer_object = architecture_utils.get_1d_conv_layer(
        num_kernel_rows=1, num_rows_per_stride=1,
        num_filters=num_output_wavelengths * ensemble_size,
        padding_type_string=architecture_utils.YES_PADDING_STRING,
        weight_regularizer=regularizer_object, layer_name='last_conv'
    )(skip_layer_by_level[0])

    if ensemble_size > 1:
        conv_output_layer_object = keras.layers.Reshape(
            target_shape=
            (input_dimensions[0], num_output_wavelengths, 1, ensemble_size)
        )(conv_output_layer_object)
    else:
        conv_output_layer_object = keras.layers.Reshape(
            target_shape=
            (input_dimensions[0], num_output_wavelengths, 1)
        )(conv_output_layer_object)

    if conv_output_activ_func_name is not None:
        conv_output_layer_object = architecture_utils.get_activation_layer(
            activation_function_string=conv_output_activ_func_name,
            alpha_for_relu=conv_output_activ_func_alpha,
            alpha_for_elu=conv_output_activ_func_alpha,
            layer_name='last_conv_activation'
        )(conv_output_layer_object)

    # Zero out heating rate at top of atmosphere.
    conv_output_layer_object = zero_top_heating_rate(
        input_layer_object=conv_output_layer_object,
        ensemble_size=ensemble_size,
        output_layer_name='conv_output'
    )

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
            if (
                    dense_layer_dropout_rates[j] <= 0 and
                    dense_output_activ_func_name is None
            ):
                this_name = 'dense_output'
            else:
                this_name = 'dense{0:d}_reshape'.format(j)

            num_dense_output_vars = (
                float(dense_layer_neuron_nums[j]) /
                (ensemble_size * num_output_wavelengths)
            )
            assert numpy.isclose(
                num_dense_output_vars, numpy.round(num_dense_output_vars),
                atol=1e-6
            )
            num_dense_output_vars = int(numpy.round(num_dense_output_vars))

            if ensemble_size > 1:
                these_dim = (
                    num_output_wavelengths, num_dense_output_vars, ensemble_size
                )
            else:
                these_dim = (num_output_wavelengths, num_dense_output_vars)

            dense_output_layer_object = keras.layers.Reshape(
                target_shape=these_dim, name=this_name
            )(dense_output_layer_object)

            if dense_output_activ_func_name is not None:
                if dense_layer_dropout_rates[j] > 0:
                    this_name = 'dense{0:d}_activation'.format(j)
                else:
                    this_name = 'dense_output'

                dense_output_layer_object = (
                    architecture_utils.get_activation_layer(
                        activation_function_string=dense_output_activ_func_name,
                        alpha_for_relu=dense_output_activ_func_alpha,
                        alpha_for_elu=dense_output_activ_func_alpha,
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

            this_mc_flag = bool(dense_layer_mc_dropout_flags[j])

            dense_output_layer_object = (
                architecture_utils.get_dropout_layer(
                    dropout_fraction=dense_layer_dropout_rates[j],
                    layer_name=this_name
                )(dense_output_layer_object, training=this_mc_flag)
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
        metric_dict = {
            'conv_output': metric_function_list,
            'dense_output': metric_function_list
        }

        model_object.compile(
            loss=loss_dict, optimizer=keras.optimizers.Nadam(),
            metrics=metric_dict
        )
    else:
        model_object.compile(
            loss=vector_loss_function, optimizer=keras.optimizers.Nadam(),
            metrics=metric_function_list
        )

    model_object.summary()
    return model_object
