"""Methods for building U-net++.

Based on: https://github.com/longuyen97/UnetPlusPlus/blob/master/unetpp.py
"""

import os
import sys
import numpy
import keras
import keras.layers

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import architecture_utils
import neural_net
import u_net_architecture
import inline_normalization

POINT_ESTIMATE_TYPE_STRING = 'point_estimate'
FLIPOUT_TYPE_STRING = 'flipout'
REPARAMETERIZATION_TYPE_STRING = 'reparameterization'
VALID_CONV_LAYER_TYPE_STRINGS = [
    POINT_ESTIMATE_TYPE_STRING, FLIPOUT_TYPE_STRING,
    REPARAMETERIZATION_TYPE_STRING
]

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

NUM_OUTPUT_WAVELENGTHS_KEY = 'num_output_wavelengths'
VECTOR_LOSS_FUNCTION_KEY = 'vector_loss_function'
SCALAR_LOSS_FUNCTION_KEY = 'scalar_loss_function'
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
    USE_BATCH_NORM_KEY: True,
    USE_DEEP_SUPERVISION_KEY: False,
    DO_INLINE_NORMALIZATION_KEY: False,
    PW_LINEAR_UNIF_MODEL_KEY: None,
    SCALAR_PREDICTORS_KEY: None,
    VECTOR_PREDICTORS_KEY: None,
    HEIGHTS_KEY: None
}

KL_SCALING_FACTOR_KEY = 'kl_divergence_scaling_factor'
UPCONV_BNN_LAYER_TYPES_KEY = 'upconv_layer_type_string_by_level'
SKIP_BNN_LAYER_TYPES_KEY = 'skip_layer_type_string_by_level'
PENULTIMATE_BNN_LAYER_TYPE_KEY = 'penultimate_conv_layer_type_string'
DENSE_BNN_LAYER_TYPES_KEY = 'dense_layer_type_strings'
CONV_OUTPUT_BNN_LAYER_TYPE_KEY = 'conv_output_layer_type_string'


def _check_layer_type(layer_type_string):
    """Ensures that convolutional-layer type is valid.

    :param layer_type_string: Layer type (must be in list
        VALID_CONV_LAYER_TYPE_STRINGS).
    :raises ValueError: if
        `layer_type_string not in VALID_CONV_LAYER_TYPE_STRINGS`.
    """

    error_checking.assert_is_string(layer_type_string)

    if layer_type_string not in VALID_CONV_LAYER_TYPE_STRINGS:
        error_string = (
            'Valid conv-layer types (listed below) do not include "{0:s}":'
            '\n{1:s}'
        ).format(layer_type_string, str(VALID_CONV_LAYER_TYPE_STRINGS))

        raise ValueError(error_string)


def _check_args(option_dict):
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

    error_checking.assert_is_integer(option_dict[NUM_OUTPUT_WAVELENGTHS_KEY])
    error_checking.assert_is_greater(option_dict[NUM_OUTPUT_WAVELENGTHS_KEY], 0)
    error_checking.assert_is_boolean(option_dict[USE_DEEP_SUPERVISION_KEY])
    error_checking.assert_is_integer(option_dict[ENSEMBLE_SIZE_KEY])
    error_checking.assert_is_greater(option_dict[ENSEMBLE_SIZE_KEY], 0)
    error_checking.assert_is_boolean(option_dict[DO_INLINE_NORMALIZATION_KEY])

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


def create_model(option_dict):
    """Creates U-net++.

    This method sets up the architecture, loss function, and optimizer -- and
    compiles the model -- but does not train it.

    Architecture based on:
    https://github.com/longuyen97/UnetPlusPlus/blob/master/unetpp.py

    M = number of rows in grid
    N = number of columns in grid

    :param option_dict: See doc for `_check_args`.
    :return: model_object: Instance of `keras.models.Model`, with the
        aforementioned architecture.
    """

    option_dict = _check_args(option_dict)

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

    num_output_wavelengths = option_dict[NUM_OUTPUT_WAVELENGTHS_KEY]
    vector_loss_function = option_dict[VECTOR_LOSS_FUNCTION_KEY]
    scalar_loss_function = option_dict[SCALAR_LOSS_FUNCTION_KEY]
    use_deep_supervision = option_dict[USE_DEEP_SUPERVISION_KEY]
    ensemble_size = option_dict[ENSEMBLE_SIZE_KEY]
    do_inline_normalization = option_dict[DO_INLINE_NORMALIZATION_KEY]
    pw_linear_unif_model_file_name = option_dict[PW_LINEAR_UNIF_MODEL_KEY]
    scalar_predictor_names = option_dict[SCALAR_PREDICTORS_KEY]
    vector_predictor_names = option_dict[VECTOR_PREDICTORS_KEY]
    heights_m_agl = option_dict[HEIGHTS_KEY]

    if ensemble_size > 1:
        # include_penultimate_conv = False
        use_deep_supervision = False

    has_dense_layers = dense_layer_neuron_nums is not None
    input_layer_object = keras.layers.Input(
        shape=tuple(input_dimensions.tolist())
    )

    if do_inline_normalization:
        normalized_input_layer_object = (
            inline_normalization.create_normalization_layers(
                input_layer_object=input_layer_object,
                pw_linear_unif_model_file_name=pw_linear_unif_model_file_name,
                vector_predictor_names=vector_predictor_names,
                scalar_predictor_names=scalar_predictor_names,
                heights_m_agl=heights_m_agl
            )
        )
    else:
        normalized_input_layer_object = None

    regularizer_object = architecture_utils.get_weight_regularizer(
        l1_weight=l1_weight, l2_weight=l2_weight
    )

    last_conv_layer_matrix = numpy.full(
        (num_levels + 1, num_levels + 1), '', dtype=object
    )
    pooling_layer_by_level = [None] * num_levels

    for i in range(num_levels + 1):
        for k in range(num_conv_layers_by_level[i]):
            if k == 0:
                if i == 0:
                    if do_inline_normalization:
                        this_input_layer_object = normalized_input_layer_object
                    else:
                        this_input_layer_object = input_layer_object
                else:
                    this_input_layer_object = pooling_layer_by_level[i - 1]
            else:
                this_input_layer_object = last_conv_layer_matrix[i, 0]

            this_name = 'block{0:d}-{1:d}_conv{2:d}'.format(i, 0, k)

            last_conv_layer_matrix[i, 0] = architecture_utils.get_1d_conv_layer(
                num_kernel_rows=3, num_rows_per_stride=1,
                num_filters=num_channels_by_level[i],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object, layer_name=this_name
            )(this_input_layer_object)

            this_name = 'block{0:d}-{1:d}_conv{2:d}_activation'.format(i, 0, k)

            last_conv_layer_matrix[i, 0] = (
                architecture_utils.get_activation_layer(
                    activation_function_string=inner_activ_function_name,
                    alpha_for_relu=inner_activ_function_alpha,
                    alpha_for_elu=inner_activ_function_alpha,
                    layer_name=this_name
                )(last_conv_layer_matrix[i, 0])
            )

            if encoder_dropout_rate_by_level[i] > 0:
                this_name = 'block{0:d}-{1:d}_conv{2:d}_dropout'.format(i, 0, k)
                this_mc_flag = bool(encoder_mc_dropout_flag_by_level[i])

                last_conv_layer_matrix[i, 0] = (
                    architecture_utils.get_dropout_layer(
                        dropout_fraction=encoder_dropout_rate_by_level[i],
                        layer_name=this_name
                    )(last_conv_layer_matrix[i, 0], training=this_mc_flag)
                )

            if use_batch_normalization:
                this_name = 'block{0:d}-{1:d}_conv{2:d}_bn'.format(i, 0, k)

                last_conv_layer_matrix[i, 0] = (
                    architecture_utils.get_batch_norm_layer(
                        layer_name=this_name
                    )(last_conv_layer_matrix[i, 0])
                )

        if i != num_levels:
            this_name = 'block{0:d}-{1:d}_pooling'.format(i, 0)

            pooling_layer_by_level[i] = architecture_utils.get_1d_pooling_layer(
                num_rows_in_window=2, num_rows_per_stride=2,
                pooling_type_string=architecture_utils.MAX_POOLING_STRING,
                layer_name=this_name
            )(last_conv_layer_matrix[i, 0])

        i_new = i + 0
        j = 0

        while i_new > 0:
            i_new -= 1
            j += 1

            this_name = 'block{0:d}-{1:d}_upsampling'.format(i_new, j)

            this_layer_object = keras.layers.UpSampling1D(
                size=2, name=this_name
            )(last_conv_layer_matrix[i_new + 1, j - 1])

            this_name = 'block{0:d}-{1:d}_upconv'.format(i_new, j)

            this_layer_object = architecture_utils.get_1d_conv_layer(
                num_kernel_rows=3, num_rows_per_stride=1,
                num_filters=num_channels_by_level[i_new],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object, layer_name=this_name
            )(this_layer_object)

            this_name = 'block{0:d}-{1:d}_upconv_activation'.format(i_new, j)

            this_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha,
                layer_name=this_name
            )(this_layer_object)

            if upconv_dropout_rate_by_level[i_new] > 0:
                this_name = 'block{0:d}-{1:d}_upconv_dropout'.format(i_new, j)
                this_mc_flag = bool(upconv_mc_dropout_flag_by_level[i_new])

                this_layer_object = architecture_utils.get_dropout_layer(
                    dropout_fraction=upconv_dropout_rate_by_level[i_new],
                    layer_name=this_name
                )(this_layer_object, training=this_mc_flag)

            num_upconv_heights = this_layer_object.get_shape()[1]
            num_desired_heights = (
                last_conv_layer_matrix[i_new, 0].get_shape()[1]
            )

            if num_desired_heights == num_upconv_heights + 1:
                this_name = 'block{0:d}-{1:d}_upconv_padding'.format(i_new, j)

                this_layer_object = keras.layers.ZeroPadding1D(
                    padding=(0, 1), name=this_name
                )(this_layer_object)

            last_conv_layer_matrix[i_new, j] = this_layer_object

            this_name = 'block{0:d}-{1:d}_skip'.format(i_new, j)

            last_conv_layer_matrix[i_new, j] = keras.layers.Concatenate(
                axis=-1, name=this_name
            )(last_conv_layer_matrix[i_new, :(j + 1)].tolist())

            for k in range(num_conv_layers_by_level[i_new]):
                this_name = 'block{0:d}-{1:d}_skipconv{2:d}'.format(i_new, j, k)

                last_conv_layer_matrix[i_new, j] = (
                    architecture_utils.get_1d_conv_layer(
                        num_kernel_rows=3, num_rows_per_stride=1,
                        num_filters=num_channels_by_level[i_new],
                        padding_type_string=
                        architecture_utils.YES_PADDING_STRING,
                        weight_regularizer=regularizer_object,
                        layer_name=this_name
                    )(last_conv_layer_matrix[i_new, j])
                )

                this_name = 'block{0:d}-{1:d}_skipconv{2:d}_activation'.format(
                    i_new, j, k
                )

                last_conv_layer_matrix[i_new, j] = (
                    architecture_utils.get_activation_layer(
                        activation_function_string=inner_activ_function_name,
                        alpha_for_relu=inner_activ_function_alpha,
                        alpha_for_elu=inner_activ_function_alpha,
                        layer_name=this_name
                    )(last_conv_layer_matrix[i_new, j])
                )

                if (
                        skip_dropout_rate_by_level[i_new] > 0
                        and i_new + j == num_levels
                ):
                    this_name = 'block{0:d}-{1:d}_skipconv{2:d}_dropout'.format(
                        i_new, j, k
                    )
                    this_mc_flag = bool(skip_mc_dropout_flag_by_level[i_new])

                    last_conv_layer_matrix[i_new, j] = (
                        architecture_utils.get_dropout_layer(
                            dropout_fraction=skip_dropout_rate_by_level[i_new],
                            layer_name=this_name
                        )(
                            last_conv_layer_matrix[i_new, j],
                            training=this_mc_flag
                        )
                    )

                if use_batch_normalization:
                    this_name = 'block{0:d}-{1:d}_skipconv{2:d}_bn'.format(
                        i_new, j, k
                    )

                    last_conv_layer_matrix[i_new, j] = (
                        architecture_utils.get_batch_norm_layer(
                            layer_name=this_name
                        )(last_conv_layer_matrix[i_new, j])
                    )

    if include_penultimate_conv:
        last_conv_layer_matrix[0, -1] = architecture_utils.get_1d_conv_layer(
            num_kernel_rows=3, num_rows_per_stride=1,
            num_filters=2 * num_output_wavelengths * ensemble_size,
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object, layer_name='penultimate_conv'
        )(last_conv_layer_matrix[0, -1])

        last_conv_layer_matrix[0, -1] = architecture_utils.get_activation_layer(
            activation_function_string=inner_activ_function_name,
            alpha_for_relu=inner_activ_function_alpha,
            alpha_for_elu=inner_activ_function_alpha,
            layer_name='penultimate_conv_activation'
        )(last_conv_layer_matrix[0, -1])

        if penultimate_conv_dropout_rate > 0:
            this_mc_flag = bool(penultimate_conv_mc_dropout_flag)

            last_conv_layer_matrix[0, -1] = (
                architecture_utils.get_dropout_layer(
                    dropout_fraction=penultimate_conv_dropout_rate,
                    layer_name='penultimate_conv_dropout'
                )(last_conv_layer_matrix[0, -1], training=this_mc_flag)
            )

        if use_batch_normalization:
            last_conv_layer_matrix[0, -1] = (
                architecture_utils.get_batch_norm_layer(
                    layer_name='penultimate_conv_bn'
                )(last_conv_layer_matrix[0, -1])
            )

    conv_output_layer_object = architecture_utils.get_1d_conv_layer(
        num_kernel_rows=1, num_rows_per_stride=1,
        num_filters=num_output_wavelengths * ensemble_size,
        padding_type_string=architecture_utils.YES_PADDING_STRING,
        weight_regularizer=regularizer_object, layer_name='last_conv'
    )(last_conv_layer_matrix[0, -1])

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

    this_function = u_net_architecture.zero_top_heating_rate_function(
        height_index=input_dimensions[0] - 1
    )

    conv_output_layer_object = keras.layers.Lambda(
        this_function, name='conv_output'
    )(conv_output_layer_object)

    output_layer_objects = [conv_output_layer_object]
    loss_dict = {'conv_output': vector_loss_function}

    deep_supervision_layer_objects = [None] * num_levels

    if use_deep_supervision:
        for i in range(1, num_levels):
            this_name = 'deepsup{0:d}_conv'.format(i)

            deep_supervision_layer_objects[i] = (
                architecture_utils.get_1d_conv_layer(
                    num_kernel_rows=1, num_rows_per_stride=1,
                    num_filters=num_output_wavelengths,
                    padding_type_string=architecture_utils.YES_PADDING_STRING,
                    weight_regularizer=regularizer_object, layer_name=this_name
                )(last_conv_layer_matrix[0, i])
            )

            if conv_output_activ_func_name is not None:
                this_name = 'deepsup{0:d}_activation'.format(i)

                deep_supervision_layer_objects[i] = (
                    architecture_utils.get_activation_layer(
                        activation_function_string=conv_output_activ_func_name,
                        alpha_for_relu=conv_output_activ_func_alpha,
                        alpha_for_elu=conv_output_activ_func_alpha,
                        layer_name=this_name
                    )(deep_supervision_layer_objects[i])
                )

            deep_supervision_layer_objects[i] = keras.layers.Reshape(
                target_shape=
                (input_dimensions[0], num_output_wavelengths, 1)
            )(deep_supervision_layer_objects[i])

            this_function = u_net_architecture.zero_top_heating_rate_function(
                height_index=input_dimensions[0] - 1
            )
            this_name = 'deepsup{0:d}_output'.format(i)

            deep_supervision_layer_objects[i] = keras.layers.Lambda(
                this_function, name=this_name
            )(deep_supervision_layer_objects[i])

            output_layer_objects.append(deep_supervision_layer_objects[i])
            loss_dict[this_name] = vector_loss_function

    if has_dense_layers:
        num_dense_layers = len(dense_layer_neuron_nums)
        dense_output_layer_object = architecture_utils.get_flattening_layer()(
            last_conv_layer_matrix[-1, 0]
        )
    else:
        num_dense_layers = 0
        dense_output_layer_object = None

    for j in range(num_dense_layers):
        dense_output_layer_object = architecture_utils.get_dense_layer(
            num_output_units=dense_layer_neuron_nums[j]
        )(dense_output_layer_object)

        if j == num_dense_layers - 1:
            if (
                    dense_layer_dropout_rates[j] <= 0 and
                    dense_output_activ_func_name is None
            ):
                this_name = 'dense_output'
            else:
                this_name = None

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
                this_name = (
                    None if dense_layer_dropout_rates[j] > 0 else 'dense_output'
                )

                dense_output_layer_object = (
                    architecture_utils.get_activation_layer(
                        activation_function_string=dense_output_activ_func_name,
                        alpha_for_relu=dense_output_activ_func_alpha,
                        alpha_for_elu=dense_output_activ_func_alpha,
                        layer_name=this_name
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

        if dense_layer_dropout_rates[j] > 0:
            this_name = (
                'dense_output' if j == num_dense_layers - 1 else None
            )
            this_mc_flag = bool(dense_layer_mc_dropout_flags[j])

            dense_output_layer_object = (
                architecture_utils.get_dropout_layer(
                    dropout_fraction=dense_layer_dropout_rates[j],
                    layer_name=this_name
                )(dense_output_layer_object, training=this_mc_flag)
            )

        if use_batch_normalization and j != num_dense_layers - 1:
            dense_output_layer_object = (
                architecture_utils.get_batch_norm_layer()(
                    dense_output_layer_object
                )
            )

    if ensemble_size > 1:
        metric_function_list = []
    else:
        metric_function_list = neural_net.METRIC_FUNCTION_LIST

    if has_dense_layers:
        output_layer_objects.insert(1, dense_output_layer_object)
        loss_dict['dense_output'] = scalar_loss_function

    model_object = keras.models.Model(
        inputs=input_layer_object, outputs=output_layer_objects
    )
    model_object.compile(
        loss=loss_dict, optimizer=keras.optimizers.Adam(),
        metrics=metric_function_list
    )

    model_object.summary()
    return model_object


def create_model_1output_layer(
        option_dict, loss_function, num_output_wavelengths):
    """Same as `create_model`, except that output layers are joined into one.

    :param option_dict: See doc for `_check_args`.
    :param loss_function: Loss function.
    :param num_output_wavelengths: Number of output wavelengths.
    :return: model_object: Instance of `keras.models.Model`.
    """

    option_dict = _check_args(option_dict)

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

    has_dense_layers = dense_layer_neuron_nums is not None

    input_layer_object = keras.layers.Input(
        shape=tuple(input_dimensions.tolist())
    )
    regularizer_object = architecture_utils.get_weight_regularizer(
        l1_weight=l1_weight, l2_weight=l2_weight
    )

    last_conv_layer_matrix = numpy.full(
        (num_levels + 1, num_levels + 1), '', dtype=object
    )
    pooling_layer_by_level = [None] * num_levels

    for i in range(num_levels + 1):
        for k in range(num_conv_layers_by_level[i]):
            if k == 0:
                if i == 0:
                    this_input_layer_object = input_layer_object
                else:
                    this_input_layer_object = pooling_layer_by_level[i - 1]
            else:
                this_input_layer_object = last_conv_layer_matrix[i, 0]

            this_name = 'block{0:d}-{1:d}_conv{2:d}'.format(i, 0, k)

            last_conv_layer_matrix[i, 0] = architecture_utils.get_1d_conv_layer(
                num_kernel_rows=3, num_rows_per_stride=1,
                num_filters=num_channels_by_level[i],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object, layer_name=this_name
            )(this_input_layer_object)

            this_name = 'block{0:d}-{1:d}_conv{2:d}_activation'.format(i, 0, k)

            last_conv_layer_matrix[i, 0] = (
                architecture_utils.get_activation_layer(
                    activation_function_string=inner_activ_function_name,
                    alpha_for_relu=inner_activ_function_alpha,
                    alpha_for_elu=inner_activ_function_alpha,
                    layer_name=this_name
                )(last_conv_layer_matrix[i, 0])
            )

            if encoder_dropout_rate_by_level[i] > 0:
                this_name = 'block{0:d}-{1:d}_conv{2:d}_dropout'.format(i, 0, k)
                this_mc_flag = bool(encoder_mc_dropout_flag_by_level[i])

                last_conv_layer_matrix[i, 0] = (
                    architecture_utils.get_dropout_layer(
                        dropout_fraction=encoder_dropout_rate_by_level[i],
                        layer_name=this_name
                    )(last_conv_layer_matrix[i, 0], training=this_mc_flag)
                )

            if use_batch_normalization:
                this_name = 'block{0:d}-{1:d}_conv{2:d}_bn'.format(i, 0, k)

                last_conv_layer_matrix[i, 0] = (
                    architecture_utils.get_batch_norm_layer(
                        layer_name=this_name
                    )(last_conv_layer_matrix[i, 0])
                )

        if i != num_levels:
            this_name = 'block{0:d}-{1:d}_pooling'.format(i, 0)

            pooling_layer_by_level[i] = architecture_utils.get_1d_pooling_layer(
                num_rows_in_window=2, num_rows_per_stride=2,
                pooling_type_string=architecture_utils.MAX_POOLING_STRING,
                layer_name=this_name
            )(last_conv_layer_matrix[i, 0])

        i_new = i + 0
        j = 0

        while i_new > 0:
            i_new -= 1
            j += 1

            this_name = 'block{0:d}-{1:d}_upsampling'.format(i_new, j)

            this_layer_object = keras.layers.UpSampling1D(
                size=2, name=this_name
            )(last_conv_layer_matrix[i_new + 1, j - 1])

            this_name = 'block{0:d}-{1:d}_upconv'.format(i_new, j)

            this_layer_object = architecture_utils.get_1d_conv_layer(
                num_kernel_rows=3, num_rows_per_stride=1,
                num_filters=num_channels_by_level[i_new],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object, layer_name=this_name
            )(this_layer_object)

            this_name = 'block{0:d}-{1:d}_upconv_activation'.format(i_new, j)

            this_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha,
                layer_name=this_name
            )(this_layer_object)

            if upconv_dropout_rate_by_level[i_new] > 0:
                this_name = 'block{0:d}-{1:d}_upconv_dropout'.format(i_new, j)
                this_mc_flag = bool(upconv_mc_dropout_flag_by_level[i_new])

                this_layer_object = architecture_utils.get_dropout_layer(
                    dropout_fraction=upconv_dropout_rate_by_level[i_new],
                    layer_name=this_name
                )(this_layer_object, training=this_mc_flag)

            num_upconv_heights = this_layer_object.get_shape()[1]
            num_desired_heights = (
                last_conv_layer_matrix[i_new, 0].get_shape()[1]
            )

            if num_desired_heights == num_upconv_heights + 1:
                this_name = 'block{0:d}-{1:d}_upconv_padding'.format(i_new, j)

                this_layer_object = keras.layers.ZeroPadding1D(
                    padding=(0, 1), name=this_name
                )(this_layer_object)

            last_conv_layer_matrix[i_new, j] = this_layer_object

            this_name = 'block{0:d}-{1:d}_skip'.format(i_new, j)

            last_conv_layer_matrix[i_new, j] = keras.layers.Concatenate(
                axis=-1, name=this_name
            )(last_conv_layer_matrix[i_new, :(j + 1)].tolist())

            for k in range(num_conv_layers_by_level[i_new]):
                this_name = 'block{0:d}-{1:d}_skipconv{2:d}'.format(i_new, j, k)

                last_conv_layer_matrix[i_new, j] = (
                    architecture_utils.get_1d_conv_layer(
                        num_kernel_rows=3, num_rows_per_stride=1,
                        num_filters=num_channels_by_level[i_new],
                        padding_type_string=
                        architecture_utils.YES_PADDING_STRING,
                        weight_regularizer=regularizer_object,
                        layer_name=this_name
                    )(last_conv_layer_matrix[i_new, j])
                )

                this_name = 'block{0:d}-{1:d}_skipconv{2:d}_activation'.format(
                    i_new, j, k
                )

                last_conv_layer_matrix[i_new, j] = (
                    architecture_utils.get_activation_layer(
                        activation_function_string=inner_activ_function_name,
                        alpha_for_relu=inner_activ_function_alpha,
                        alpha_for_elu=inner_activ_function_alpha,
                        layer_name=this_name
                    )(last_conv_layer_matrix[i_new, j])
                )

                if (
                        skip_dropout_rate_by_level[i_new] > 0
                        and i_new + j == num_levels
                ):
                    this_name = 'block{0:d}-{1:d}_skipconv{2:d}_dropout'.format(
                        i_new, j, k
                    )
                    this_mc_flag = bool(skip_mc_dropout_flag_by_level[i_new])

                    last_conv_layer_matrix[i_new, j] = (
                        architecture_utils.get_dropout_layer(
                            dropout_fraction=skip_dropout_rate_by_level[i_new],
                            layer_name=this_name
                        )(
                            last_conv_layer_matrix[i_new, j],
                            training=this_mc_flag
                        )
                    )

                if use_batch_normalization:
                    this_name = 'block{0:d}-{1:d}_skipconv{2:d}_bn'.format(
                        i_new, j, k
                    )

                    last_conv_layer_matrix[i_new, j] = (
                        architecture_utils.get_batch_norm_layer(
                            layer_name=this_name
                        )(last_conv_layer_matrix[i_new, j])
                    )

    if include_penultimate_conv:
        last_conv_layer_matrix[0, -1] = architecture_utils.get_1d_conv_layer(
            num_kernel_rows=3, num_rows_per_stride=1,
            num_filters=2 * num_output_wavelengths,
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object, layer_name='penultimate_conv'
        )(last_conv_layer_matrix[0, -1])

        last_conv_layer_matrix[0, -1] = architecture_utils.get_activation_layer(
            activation_function_string=inner_activ_function_name,
            alpha_for_relu=inner_activ_function_alpha,
            alpha_for_elu=inner_activ_function_alpha,
            layer_name='penultimate_conv_activation'
        )(last_conv_layer_matrix[0, -1])

        if penultimate_conv_dropout_rate > 0:
            this_mc_flag = bool(penultimate_conv_mc_dropout_flag)

            last_conv_layer_matrix[0, -1] = (
                architecture_utils.get_dropout_layer(
                    dropout_fraction=penultimate_conv_dropout_rate,
                    layer_name='penultimate_conv_dropout'
                )(last_conv_layer_matrix[0, -1], training=this_mc_flag)
            )

        if use_batch_normalization:
            last_conv_layer_matrix[0, -1] = (
                architecture_utils.get_batch_norm_layer(
                    layer_name='penultimate_conv_bn'
                )(last_conv_layer_matrix[0, -1])
            )

    conv_output_layer_object = architecture_utils.get_1d_conv_layer(
        num_kernel_rows=1, num_rows_per_stride=1,
        num_filters=num_output_wavelengths,
        padding_type_string=architecture_utils.YES_PADDING_STRING,
        weight_regularizer=regularizer_object, layer_name='last_conv'
    )(last_conv_layer_matrix[0, -1])

    if conv_output_activ_func_name is not None:
        conv_output_layer_object = architecture_utils.get_activation_layer(
            activation_function_string=conv_output_activ_func_name,
            alpha_for_relu=conv_output_activ_func_alpha,
            alpha_for_elu=conv_output_activ_func_alpha,
            layer_name='last_conv_activation'
        )(conv_output_layer_object)

    this_function = u_net_architecture.zero_top_heating_rate_function(
        height_index=input_dimensions[0] - 1
    )

    conv_output_layer_object = keras.layers.Lambda(
        this_function, name='conv_output'
    )(conv_output_layer_object)

    if has_dense_layers:
        num_dense_layers = len(dense_layer_neuron_nums)
        dense_output_layer_object = architecture_utils.get_flattening_layer()(
            last_conv_layer_matrix[-1, 0]
        )
    else:
        num_dense_layers = 0
        dense_output_layer_object = None

    for j in range(num_dense_layers):
        dense_output_layer_object = architecture_utils.get_dense_layer(
            num_output_units=dense_layer_neuron_nums[j]
        )(dense_output_layer_object)

        if j == num_dense_layers - 1:
            if (
                    dense_layer_dropout_rates[j] <= 0 and
                    dense_output_activ_func_name is None
            ):
                this_name = 'dense_output'
            else:
                this_name = None

            num_dense_output_vars = (
                float(dense_layer_neuron_nums[j]) / num_output_wavelengths
            )
            assert numpy.isclose(
                num_dense_output_vars, numpy.round(num_dense_output_vars),
                atol=1e-6
            )
            num_dense_output_vars = int(numpy.round(num_dense_output_vars))

            dense_output_layer_object = keras.layers.Reshape(
                target_shape=(num_output_wavelengths, num_dense_output_vars),
                name=this_name
            )(dense_output_layer_object)

            if dense_output_activ_func_name is not None:
                this_name = (
                    None if dense_layer_dropout_rates[j] > 0 else 'dense_output'
                )

                dense_output_layer_object = (
                    architecture_utils.get_activation_layer(
                        activation_function_string=dense_output_activ_func_name,
                        alpha_for_relu=dense_output_activ_func_alpha,
                        alpha_for_elu=dense_output_activ_func_alpha,
                        layer_name=this_name
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

        if dense_layer_dropout_rates[j] > 0:
            this_name = (
                'dense_output' if j == num_dense_layers - 1 else None
            )
            this_mc_flag = bool(dense_layer_mc_dropout_flags[j])

            dense_output_layer_object = (
                architecture_utils.get_dropout_layer(
                    dropout_fraction=dense_layer_dropout_rates[j],
                    layer_name=this_name
                )(dense_output_layer_object, training=this_mc_flag)
            )

        if use_batch_normalization and j != num_dense_layers - 1:
            dense_output_layer_object = (
                architecture_utils.get_batch_norm_layer()(
                    dense_output_layer_object
                )
            )

    if has_dense_layers:
        conv_output_layer_object = keras.layers.Permute(dims=(1, 2))(
            conv_output_layer_object
        )
        output_layer_object = keras.layers.Concatenate(axis=-1)(
            [conv_output_layer_object, dense_output_layer_object]
        )
        model_object = keras.models.Model(
            inputs=input_layer_object, outputs=output_layer_object
        )
    else:
        model_object = keras.models.Model(
            inputs=input_layer_object, outputs=conv_output_layer_object
        )

    model_object.compile(
        loss=loss_function, optimizer=keras.optimizers.Adam(),
        metrics=neural_net.METRIC_FUNCTION_LIST
    )

    model_object.summary()
    return model_object
