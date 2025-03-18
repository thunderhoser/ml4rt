"""Methods for building U-nets."""

import os
import sys
import numpy
import keras
import tensorflow

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import architecture_utils
import example_utils
import normalization
import neural_net
import convnext

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
USE_CONVNEXT_V1_BLOCKS_KEY = 'use_convnext_v1_blocks'
USE_CONVNEXT_V2_BLOCKS_KEY = 'use_convnext_v2_blocks'
SIMPLIFY_CONVNEXT_KEY = 'simplify_convnext_blocks'
SIMPLIFY_OUTPUT_LAYER_KEY = 'simplify_output_layer'
MEAN_VALUE_MATRIX_KEY = 'mean_value_matrix'
STDEV_MATRIX_KEY = 'stdev_matrix'
HEATING_RATE_MASK_KEY = 'heating_rate_mask_matrix'
FLUX_MASK_KEY = 'flux_mask_matrix'

NUM_OUTPUT_WAVELENGTHS_KEY = 'num_output_wavelengths'
VECTOR_LOSS_FUNCTION_KEY = 'vector_loss_function'
SCALAR_LOSS_FUNCTION_KEY = 'scalar_loss_function'
JOINED_LOSS_FUNCTION_KEY = 'joined_loss_function'
OPTIMIZER_FUNCTION_KEY = 'optimizer_function'
USE_DEEP_SUPERVISION_KEY = 'use_deep_supervision'
ENSEMBLE_SIZE_KEY = 'ensemble_size'

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
    MEAN_VALUE_MATRIX_KEY: None,
    STDEV_MATRIX_KEY: None,
    HEATING_RATE_MASK_KEY: None,
    FLUX_MASK_KEY: None
}


class ZScoreNormalization(keras.layers.Layer):
    """This layer performs z-score normalization, using pre-computed params."""

    def __init__(self, mean_value_matrix, stdev_matrix, **kwargs):
        """Constructor.

        H = number of heights
        P = number of predictor variables

        :param mean_value_matrix: H-by-P numpy array of mean values.
        :param stdev_matrix: H-by-P numpy array of standard deviations.
        :param kwargs: Keyword arguments.
        """

        super(ZScoreNormalization, self).__init__(**kwargs)

        self.mean_value_matrix = tensorflow.Variable(
            tensorflow.convert_to_tensor(
                mean_value_matrix, dtype=tensorflow.float32
            ),
            trainable=False
        )

        self.stdev_matrix = tensorflow.Variable(
            tensorflow.convert_to_tensor(
                stdev_matrix, dtype=tensorflow.float32
            ),
            trainable=False
        )

    def call(self, inputs):
        """Main method.  This is where the layer does its thing.

        :param inputs: Input tensor, in physical units.
        :return: output_tensor: Output tensor, in z-score units.
        """

        inputs = (
            (inputs - self.mean_value_matrix[None, :, :]) /
            self.stdev_matrix[None, :, :]
        )

        inputs = tensorflow.where(
            self.stdev_matrix[None, :, :] < 1e-12,
            0.,
            inputs
        )

        return inputs

    def get_config(self):
        """Returns layer configuration.

        :return: config: Dictionary with configuration stuff.
        """

        config = super().get_config()
        config.update({
            'mean_value_matrix': self.mean_value_matrix.numpy().tolist(),
            'stdev_matrix': self.stdev_matrix.numpy().tolist()
        })
        return config


class HeatingRateMask(keras.layers.Layer):
    """This layer zeroes out some heating rates via a Boolean mask."""

    def __init__(self, mask_matrix, **kwargs):
        """Constructor.

        H = number of heights
        W = number of wavelengths

        :param mask_matrix: H-by-W numpy array of integers (0 or 1).
        :param kwargs: Keyword arguments.
        """

        super(HeatingRateMask, self).__init__(**kwargs)

        self.mask_matrix = tensorflow.Variable(
            tensorflow.convert_to_tensor(
                mask_matrix, dtype=tensorflow.float32
            ),
            trainable=False
        )
        self.mask_dimensions = self.mask_matrix.shape

    def call(self, inputs):
        """Main method.  This is where the layer does its thing.

        :param inputs: Input tensor of heating rates.
        :return: output_tensor: Output tensor of heating rates, with some zeroed
            out.
        """

        mask_matrix_expanded = tensorflow.reshape(
            self.mask_matrix,
            [1, self.mask_dimensions[0], self.mask_dimensions[1], 1]
        )
        mask_matrix_expanded = tensorflow.broadcast_to(
            mask_matrix_expanded, tensorflow.shape(inputs)
        )
        return inputs * mask_matrix_expanded

    def get_config(self):
        """Returns layer configuration.

        :return: config: Dictionary with configuration stuff.
        """

        config = super().get_config()
        config.update({
            'mask_matrix': self.mask_matrix.numpy().tolist(),
            'mask_dimensions': self.mask_dimensions.as_list()
        })
        return config


class FluxMask(keras.layers.Layer):
    """This layer zeroes out some fluxes via a Boolean mask."""

    def __init__(self, mask_matrix, **kwargs):
        """Constructor.

        W = number of wavelengths
        F = number of flux variables

        :param mask_matrix: W-by-F numpy array of integers (0 or 1).
        :param kwargs: Keyword arguments.
        """

        super(FluxMask, self).__init__(**kwargs)

        self.mask_matrix = tensorflow.Variable(
            tensorflow.convert_to_tensor(
                mask_matrix, dtype=tensorflow.float32
            ),
            trainable=False
        )
        self.mask_dimensions = self.mask_matrix.shape

    def call(self, inputs):
        """Main method.  This is where the layer does its thing.

        :param inputs: Input tensor of fluxes.
        :return: output_tensor: Output tensor of fluxes, with some zeroed out.
        """

        mask_matrix_expanded = tensorflow.reshape(
            self.mask_matrix,
            [1, self.mask_dimensions[0], self.mask_dimensions[1]]
        )
        mask_matrix_expanded = tensorflow.broadcast_to(
            mask_matrix_expanded, tensorflow.shape(inputs)
        )
        return inputs * mask_matrix_expanded

    def get_config(self):
        """Returns layer configuration.

        :return: config: Dictionary with configuration stuff.
        """

        config = super().get_config()
        config.update({
            'mask_matrix': self.mask_matrix.numpy().tolist(),
            'mask_dimensions': self.mask_dimensions.as_list()
        })
        return config


def check_args(option_dict):
    """Error-checks input arguments.

    L = number of levels in encoder = number of levels in decoder
    D = number of dense layers

    H = number of heights
    W = number of wavelengths
    F = number of flux variables
    V = number of vector predictors
    S = number of scalar predictors
    P = V + S = total number of predictors

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
        residual blocks throughout the architecture.
    option_dict['use_convnext_v1_blocks']: Boolean flag.  If True, will use
        simple ConvNeXt (version 1) blocks throughout the architecture.
    option_dict['use_convnext_v2_blocks']: Boolean flag.  If True, will use
        ConvNeXt-v2 blocks throughout the architecture.
    option_dict["simplify_convnext_blocks"]: Boolean flag.  If True, will use
        simple version of either ConvNeXt-v1 or ConvNeXt-v2 -- without layer
        normalization or global residual normalization, respectively.
    option_dict["simplify_output_layer"]: Boolean flag.  If True, the output
        layer will not come from a ConvNeXt block, even if ConvNeXt is generally
        used throughout the network.
    option_dict["mean_value_matrix"]: H-by-P numpy array of mean values to use
        for z-score normalization.  If you are not doing z-score normalization
        inside the neural net with a custom layer, make this None.
    option_dict["stdev_matrix"]: Same as "mean_value_matrix" but with standard
        deviations.
    option_dict["heating_rate_mask_matrix"]: H-by-W numpy array of Boolean flags
        (0 or 1), to be used for zeroing out heating rates that are always zero.
        If you do not want masking, make this None.
    option_dict["flux_mask_matrix"]: W-by-F numpy array of Boolean flags
        (0 or 1), to be used for zeroing out fluxes that are always zero.
        If you do not want masking, make this None.
    option_dict['num_output_wavelengths']: Number of output wavelengths.
    option_dict['vector_loss_function']: Loss function for vector targets.
    option_dict['scalar_loss_function']: Loss function for scalar targets.
    option_dict['optimizer_function']: Optimizer.
    option_dict['use_deep_supervision']: Boolean flag.
    option_dict['ensemble_size']: Number of ensemble members in output (both
        vector and scalar predictions).

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

        this_denom = (
            option_dict[ENSEMBLE_SIZE_KEY] *
            option_dict[NUM_OUTPUT_WAVELENGTHS_KEY]
        )

        num_flux_vars = float(dense_layer_neuron_nums[-1]) / this_denom
        assert numpy.isclose(
            num_flux_vars, numpy.round(num_flux_vars), atol=1e-6
        )
        num_flux_vars = int(numpy.round(num_flux_vars))
    else:
        num_flux_vars = 0

    error_checking.assert_is_geq(option_dict[L1_WEIGHT_KEY], 0.)
    error_checking.assert_is_geq(option_dict[L2_WEIGHT_KEY], 0.)
    error_checking.assert_is_boolean(option_dict[USE_BATCH_NORM_KEY])

    if USE_RESIDUAL_BLOCKS_KEY not in option_dict:
        option_dict[USE_RESIDUAL_BLOCKS_KEY] = False
    if USE_CONVNEXT_V1_BLOCKS_KEY not in option_dict:
        option_dict[USE_CONVNEXT_V1_BLOCKS_KEY] = False
    if USE_CONVNEXT_V2_BLOCKS_KEY not in option_dict:
        option_dict[USE_CONVNEXT_V2_BLOCKS_KEY] = False
    if SIMPLIFY_CONVNEXT_KEY not in option_dict:
        option_dict[SIMPLIFY_CONVNEXT_KEY] = False
    if SIMPLIFY_OUTPUT_LAYER_KEY not in option_dict:
        option_dict[SIMPLIFY_OUTPUT_LAYER_KEY] = False

    error_checking.assert_is_boolean(option_dict[USE_RESIDUAL_BLOCKS_KEY])
    error_checking.assert_is_boolean(option_dict[USE_CONVNEXT_V1_BLOCKS_KEY])
    error_checking.assert_is_boolean(option_dict[USE_CONVNEXT_V2_BLOCKS_KEY])
    error_checking.assert_is_boolean(option_dict[SIMPLIFY_CONVNEXT_KEY])
    error_checking.assert_is_boolean(option_dict[SIMPLIFY_OUTPUT_LAYER_KEY])

    sum_of_flags = (
        int(option_dict[USE_RESIDUAL_BLOCKS_KEY]) +
        int(option_dict[USE_CONVNEXT_V1_BLOCKS_KEY]) +
        int(option_dict[USE_CONVNEXT_V2_BLOCKS_KEY])
    )
    assert sum_of_flags <= 1

    error_checking.assert_is_integer(option_dict[NUM_OUTPUT_WAVELENGTHS_KEY])
    error_checking.assert_is_greater(option_dict[NUM_OUTPUT_WAVELENGTHS_KEY], 0)
    error_checking.assert_is_boolean(option_dict[USE_DEEP_SUPERVISION_KEY])
    error_checking.assert_is_integer(option_dict[ENSEMBLE_SIZE_KEY])
    error_checking.assert_is_greater(option_dict[ENSEMBLE_SIZE_KEY], 0)

    if (
            option_dict[MEAN_VALUE_MATRIX_KEY] is None
            or option_dict[STDEV_MATRIX_KEY] is None
    ):
        option_dict[MEAN_VALUE_MATRIX_KEY] = None
        option_dict[STDEV_MATRIX_KEY] = None
    else:
        error_checking.assert_is_numpy_array_without_nan(
            option_dict[MEAN_VALUE_MATRIX_KEY]
        )
        error_checking.assert_is_numpy_array(
            option_dict[MEAN_VALUE_MATRIX_KEY],
            exact_dimensions=input_dimensions
        )

        error_checking.assert_is_numpy_array_without_nan(
            option_dict[STDEV_MATRIX_KEY]
        )
        error_checking.assert_is_numpy_array(
            option_dict[STDEV_MATRIX_KEY],
            exact_dimensions=input_dimensions
        )

    if option_dict[HEATING_RATE_MASK_KEY] is not None:
        error_checking.assert_is_integer_numpy_array(
            option_dict[HEATING_RATE_MASK_KEY]
        )
        error_checking.assert_is_geq_numpy_array(
            option_dict[HEATING_RATE_MASK_KEY], 0
        )
        error_checking.assert_is_leq_numpy_array(
            option_dict[HEATING_RATE_MASK_KEY], 1
        )

        expected_dim = numpy.array(
            [input_dimensions[0], option_dict[NUM_OUTPUT_WAVELENGTHS_KEY]],
            dtype=int
        )
        error_checking.assert_is_numpy_array(
            option_dict[HEATING_RATE_MASK_KEY], exact_dimensions=expected_dim
        )

    if not has_dense_layers:
        option_dict[FLUX_MASK_KEY] = None

    if option_dict[FLUX_MASK_KEY] is not None:
        error_checking.assert_is_integer_numpy_array(option_dict[FLUX_MASK_KEY])
        error_checking.assert_is_geq_numpy_array(option_dict[FLUX_MASK_KEY], 0)
        error_checking.assert_is_leq_numpy_array(option_dict[FLUX_MASK_KEY], 1)

        expected_dim = numpy.array(
            [option_dict[NUM_OUTPUT_WAVELENGTHS_KEY], num_flux_vars], dtype=int
        )
        error_checking.assert_is_numpy_array(
            option_dict[FLUX_MASK_KEY], exact_dimensions=expected_dim
        )

    if OPTIMIZER_FUNCTION_KEY not in option_dict:
        option_dict[OPTIMIZER_FUNCTION_KEY] = keras.optimizers.AdamW()

    return option_dict


def get_normalization_params(vector_predictor_names, scalar_predictor_names,
                             heights_m_agl, normalization_file_name):
    """Returns normalization params.

    Normalization params returned by this method will be used as inputs to the
    custom ZScoreNormalization layer.

    V = number of vector predictors
    S = number of scalar predictors
    P = V + S = number of total predictors
    H = number of heights

    :param vector_predictor_names: length-V list of predictor names.
    :param scalar_predictor_names: length-S list of predictor names.
    :param heights_m_agl: length-H numpy array of heights (metres above ground
        level).
    :param normalization_file_name: Path to normalization file (will be read by
        `normalization.read_file`).
    :return: mean_value_matrix: H-by-P numpy array of mean values.
    :return: stdev_matrix: H-by-P numpy array of standard deviations.
    """

    error_checking.assert_is_string_list(vector_predictor_names)
    assert all([
        f in example_utils.ALL_VECTOR_PREDICTOR_NAMES
        for f in vector_predictor_names
    ])

    error_checking.assert_is_string_list(scalar_predictor_names)
    assert all([
        f in example_utils.ALL_SCALAR_PREDICTOR_NAMES
        for f in scalar_predictor_names
    ])

    print('Reading normalization params from: "{0:s}"...'.format(
        normalization_file_name
    ))
    norm_param_table_xarray = normalization.read_params(normalization_file_name)
    npt = norm_param_table_xarray

    predictor_names = vector_predictor_names + scalar_predictor_names
    num_predictors = len(predictor_names)
    num_heights = len(heights_m_agl)
    mean_value_matrix = numpy.full((num_heights, num_predictors), numpy.nan)
    stdev_matrix = numpy.full((num_heights, num_predictors), numpy.nan)

    for h in range(num_heights):
        for p in range(num_predictors):
            if predictor_names[p] in vector_predictor_names:
                h_new = example_utils.match_heights(
                    heights_m_agl=npt.coords[normalization.HEIGHT_DIM].values,
                    desired_height_m_agl=heights_m_agl[h]
                )
                p_new = numpy.where(
                    npt.coords[normalization.VECTOR_PREDICTOR_DIM].values ==
                    predictor_names[p]
                )[0][0]

                mean_value_matrix[h, p] = npt[
                    normalization.VECTOR_PREDICTOR_MEAN_KEY
                ].values[h_new, p_new]

                stdev_matrix[h, p] = npt[
                    normalization.VECTOR_PREDICTOR_STDEV_KEY
                ].values[h_new, p_new]
            else:
                p_new = numpy.where(
                    npt.coords[normalization.SCALAR_PREDICTOR_DIM].values ==
                    predictor_names[p]
                )[0][0]

                mean_value_matrix[h, p] = npt[
                    normalization.SCALAR_PREDICTOR_MEAN_KEY
                ].values[p_new]

                stdev_matrix[h, p] = npt[
                    normalization.SCALAR_PREDICTOR_STDEV_KEY
                ].values[p_new]

    return mean_value_matrix, stdev_matrix


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
        input_layer_object, do_residual, do_convnext_v1, do_convnext_v2,
        simplify_convnext, num_conv_layers,
        filter_size_px, num_filters, regularizer_object,
        activation_function_name, activation_function_alpha,
        dropout_rates, monte_carlo_dropout_flags, use_batch_norm,
        basic_layer_name):
    """Creates convolutional block.

    L = number of conv layers

    If `do_residual == do_convnext_v1 == do_convnext_v2 == False`, this will be
    a basic convolutional block.

    :param input_layer_object: Input layer to block.
    :param do_residual: Boolean flag.  If True, this will be a residual block.
    :param do_convnext_v1: Boolean flag.  If True, this will be a simple
        ConvNeXt (version 1) block.
    :param do_convnext_v2: Boolean flag.  If True, this will be a ConvNeXt-v2
        block.
    :param simplify_convnext: Boolean flag.  If True, will use simple version of
        either ConvNeXt-v1 or ConvNeXt-v2 -- without layer normalization or
        global residual normalization, respectively.
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

    if do_convnext_v2:
        for i in range(num_conv_layers):
            if i == 0:
                this_input_layer_object = input_layer_object
            else:
                this_input_layer_object = current_layer_object

            current_layer_object = convnext.get_convnext_v2_block(
                input_layer_object=this_input_layer_object,
                num_conv_layers=1,
                filter_size_px=filter_size_px,
                num_filters=num_filters,
                regularizer_object=regularizer_object,
                do_activation=True,
                dropout_rate=dropout_rates[i],
                use_simple_version=simplify_convnext,
                basic_layer_name=(
                    basic_layer_name if num_conv_layers == 1
                    else '{0:s}_{1:d}'.format(basic_layer_name, i)
                )
            )

        return current_layer_object

    if do_convnext_v1:
        for i in range(num_conv_layers):
            if i == 0:
                this_input_layer_object = input_layer_object
            else:
                this_input_layer_object = current_layer_object

            current_layer_object = convnext.get_convnext_v1_block(
                input_layer_object=this_input_layer_object,
                num_conv_layers=1,
                filter_size_px=filter_size_px,
                num_filters=num_filters,
                regularizer_object=regularizer_object,
                do_activation=True,
                dropout_rate=dropout_rates[i],
                use_simple_version=simplify_convnext,
                basic_layer_name=(
                    basic_layer_name if num_conv_layers == 1
                    else '{0:s}_{1:d}'.format(basic_layer_name, i)
                )
            )

        return current_layer_object

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
    use_convnext_v1_blocks = option_dict[USE_CONVNEXT_V1_BLOCKS_KEY]
    use_convnext_v2_blocks = option_dict[USE_CONVNEXT_V2_BLOCKS_KEY]
    simplify_convnext_blocks = option_dict[SIMPLIFY_CONVNEXT_KEY]
    simplify_output_layer = option_dict[SIMPLIFY_OUTPUT_LAYER_KEY]

    mean_value_matrix = option_dict[MEAN_VALUE_MATRIX_KEY]
    stdev_matrix = option_dict[STDEV_MATRIX_KEY]
    heating_rate_mask_matrix = option_dict[HEATING_RATE_MASK_KEY]
    flux_mask_matrix = option_dict[FLUX_MASK_KEY]

    num_output_wavelengths = option_dict[NUM_OUTPUT_WAVELENGTHS_KEY]
    vector_loss_function = option_dict[VECTOR_LOSS_FUNCTION_KEY]
    scalar_loss_function = option_dict[SCALAR_LOSS_FUNCTION_KEY]
    optimizer_function = option_dict[OPTIMIZER_FUNCTION_KEY]
    ensemble_size = option_dict[ENSEMBLE_SIZE_KEY]

    if ensemble_size > 1:
        metric_function_list = []
    else:
        metric_function_list = neural_net.METRIC_FUNCTION_LIST

    has_dense_layers = dense_layer_neuron_nums is not None
    if has_dense_layers:
        num_dense_output_vars = int(numpy.round(
            float(dense_layer_neuron_nums[-1]) /
            (ensemble_size * num_output_wavelengths)
        ))
    else:
        num_dense_output_vars = 0

    main_input_layer_object = keras.layers.Input(
        shape=tuple(input_dimensions.tolist()),
        name=neural_net.MAIN_PREDICTORS_KEY
    )

    if mean_value_matrix is None:
        main_layer_object = main_input_layer_object
    else:
        main_layer_object = ZScoreNormalization(
            mean_value_matrix=mean_value_matrix, stdev_matrix=stdev_matrix
        )(main_input_layer_object)

    regularizer_object = architecture_utils.get_weight_regularizer(
        l1_weight=l1_weight, l2_weight=l2_weight
    )
    conv_layer_by_level = [None] * (num_levels + 1)
    pooling_layer_by_level = [None] * num_levels

    for i in range(num_levels + 1):
        if i == 0:
            this_input_layer_object = main_layer_object
        else:
            this_input_layer_object = pooling_layer_by_level[i - 1]

        conv_layer_by_level[i] = get_conv_block(
            input_layer_object=this_input_layer_object,
            do_residual=use_residual_blocks,
            do_convnext_v1=use_convnext_v1_blocks,
            do_convnext_v2=use_convnext_v2_blocks,
            simplify_convnext=simplify_convnext_blocks,
            num_conv_layers=num_conv_layers_by_level[i],
            filter_size_px=3,
            num_filters=num_channels_by_level[i],
            regularizer_object=regularizer_object,
            activation_function_name=inner_activ_function_name,
            activation_function_alpha=inner_activ_function_alpha,
            dropout_rates=encoder_dropout_rate_by_level[i],
            monte_carlo_dropout_flags=encoder_mc_dropout_flag_by_level[i],
            use_batch_norm=use_batch_normalization,
            basic_layer_name='encoder{0:d}'.format(i)
        )

        if i == num_levels:
            break

        this_name = 'encoder{0:d}_pooling'.format(i)
        pooling_layer_by_level[i] = architecture_utils.get_1d_pooling_layer(
            num_rows_in_window=2,
            num_rows_per_stride=2,
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
        this_name = 'decoder{0:d}_upsampling'.format(i)

        if i == num_levels - 1:
            this_layer_object = keras.layers.UpSampling1D(
                size=2, name=this_name
            )(conv_layer_by_level[i + 1])
        else:
            this_layer_object = keras.layers.UpSampling1D(
                size=2, name=this_name
            )(skip_layer_by_level[i + 1])

        num_upconv_heights = this_layer_object.shape[1]
        num_desired_heights = conv_layer_by_level[i].shape[1]

        if num_desired_heights == num_upconv_heights + 1:
            this_name = 'decoder{0:d}_padding'.format(i)
            this_layer_object = keras.layers.ZeroPadding1D(
                padding=(0, 1), name=this_name
            )(this_layer_object)

        upconv_layer_by_level[i] = get_conv_block(
            input_layer_object=this_layer_object,
            do_residual=use_residual_blocks,
            do_convnext_v1=use_convnext_v1_blocks,
            do_convnext_v2=use_convnext_v2_blocks,
            simplify_convnext=simplify_convnext_blocks,
            num_conv_layers=1,
            filter_size_px=3,
            num_filters=num_channels_by_level[i],
            regularizer_object=regularizer_object,
            activation_function_name=inner_activ_function_name,
            activation_function_alpha=inner_activ_function_alpha,
            dropout_rates=upconv_dropout_rate_by_level[i],
            monte_carlo_dropout_flags=upconv_mc_dropout_flag_by_level[i],
            use_batch_norm=use_batch_normalization,
            basic_layer_name='decoder{0:d}'.format(i)
        )

        this_name = 'decoder{0:d}_skip'.format(i)
        merged_layer_by_level[i] = keras.layers.Concatenate(
            axis=-1, name=this_name
        )(
            [conv_layer_by_level[i], upconv_layer_by_level[i]]
        )

        skip_layer_by_level[i] = get_conv_block(
            input_layer_object=merged_layer_by_level[i],
            do_residual=use_residual_blocks,
            do_convnext_v1=use_convnext_v1_blocks,
            do_convnext_v2=use_convnext_v2_blocks,
            simplify_convnext=simplify_convnext_blocks,
            num_conv_layers=num_conv_layers_by_level[i],
            filter_size_px=3,
            num_filters=num_channels_by_level[i],
            regularizer_object=regularizer_object,
            activation_function_name=inner_activ_function_name,
            activation_function_alpha=inner_activ_function_alpha,
            dropout_rates=skip_dropout_rate_by_level[i],
            monte_carlo_dropout_flags=skip_mc_dropout_flag_by_level[i],
            use_batch_norm=use_batch_normalization,
            basic_layer_name='decoder{0:d}_skip'.format(i)
        )

    if include_penultimate_conv:
        skip_layer_by_level[0] = get_conv_block(
            input_layer_object=skip_layer_by_level[0],
            do_residual=use_residual_blocks,
            do_convnext_v1=use_convnext_v1_blocks,
            do_convnext_v2=use_convnext_v2_blocks,
            simplify_convnext=simplify_convnext_blocks,
            num_conv_layers=1,
            filter_size_px=3,
            num_filters=2 * num_output_wavelengths * ensemble_size,
            regularizer_object=regularizer_object,
            activation_function_name=inner_activ_function_name,
            activation_function_alpha=inner_activ_function_alpha,
            dropout_rates=penultimate_conv_dropout_rate,
            monte_carlo_dropout_flags=penultimate_conv_mc_dropout_flag,
            use_batch_norm=use_batch_normalization,
            basic_layer_name='penultimate_conv'
        )

    this_convnext_v1_flag = use_convnext_v1_blocks and not simplify_output_layer
    this_convnext_v2_flag = use_convnext_v2_blocks and not simplify_output_layer

    conv_output_layer_object = get_conv_block(
        input_layer_object=skip_layer_by_level[0],
        do_residual=use_residual_blocks,
        do_convnext_v1=this_convnext_v1_flag,
        do_convnext_v2=this_convnext_v2_flag,
        simplify_convnext=simplify_convnext_blocks,
        num_conv_layers=1,
        filter_size_px=1,
        num_filters=num_output_wavelengths * ensemble_size,
        regularizer_object=regularizer_object,
        activation_function_name=conv_output_activ_func_name,
        activation_function_alpha=conv_output_activ_func_alpha,
        dropout_rates=-1.,
        monte_carlo_dropout_flags=False,
        use_batch_norm=False,
        basic_layer_name='last_conv'
    )

    need_output_activ = (
        (this_convnext_v1_flag or this_convnext_v2_flag)
        and conv_output_activ_func_name is not None
    )
    need_output_activ = False
    if need_output_activ:
        # conv_output_layer_object = keras.layers.PReLU(name='last_conv_activ')(
        #     conv_output_layer_object
        # )
        conv_output_layer_object = keras.layers.Activation(
            'softplus', name='last_conv_activ'
        )(conv_output_layer_object)

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

    if heating_rate_mask_matrix is None:
        conv_output_layer_object = zero_top_heating_rate(
            input_layer_object=conv_output_layer_object,
            ensemble_size=ensemble_size,
            output_layer_name=neural_net.HEATING_RATE_TARGETS_KEY
        )
    else:
        conv_output_layer_object = HeatingRateMask(
            mask_matrix=heating_rate_mask_matrix,
            name=neural_net.HEATING_RATE_TARGETS_KEY
        )(conv_output_layer_object)

    output_layer_objects = [conv_output_layer_object]
    loss_dict = {neural_net.HEATING_RATE_TARGETS_KEY: vector_loss_function}
    metric_dict = {neural_net.HEATING_RATE_TARGETS_KEY: metric_function_list}

    if has_dense_layers:
        num_dense_layers = len(dense_layer_neuron_nums)
        dense_output_layer_object = architecture_utils.get_flattening_layer()(
            conv_layer_by_level[-1]
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
                    dense_layer_dropout_rates[j] <= 0
                    and dense_output_activ_func_name is None
                    and flux_mask_matrix is None
            ):
                this_name = neural_net.FLUX_TARGETS_KEY
            else:
                this_name = None

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
                if (
                        dense_layer_dropout_rates[j] <= 0
                        and flux_mask_matrix is None
                ):
                    this_name = neural_net.FLUX_TARGETS_KEY
                else:
                    this_name = None

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
            if j == num_dense_layers - 1 and flux_mask_matrix is None:
                this_name = neural_net.FLUX_TARGETS_KEY
            else:
                this_name = None

            this_mc_flag = bool(dense_layer_mc_dropout_flags[j])

            dense_output_layer_object = (
                architecture_utils.get_dropout_layer(
                    dropout_fraction=dense_layer_dropout_rates[j],
                    layer_name=this_name
                )(dense_output_layer_object, training=this_mc_flag)
            )

        if j == num_dense_layers - 1 and flux_mask_matrix is not None:
            dense_output_layer_object = FluxMask(
                mask_matrix=flux_mask_matrix,
                name=neural_net.FLUX_TARGETS_KEY
            )(dense_output_layer_object)

        if use_batch_normalization and j != num_dense_layers - 1:
            dense_output_layer_object = (
                architecture_utils.get_batch_norm_layer()(
                    dense_output_layer_object
                )
            )

    if has_dense_layers:
        output_layer_objects.insert(1, dense_output_layer_object)
        loss_dict[neural_net.FLUX_TARGETS_KEY] = scalar_loss_function
        metric_dict[neural_net.FLUX_TARGETS_KEY] = metric_function_list

    model_object = keras.models.Model(
        inputs=main_input_layer_object, outputs=output_layer_objects
    )
    model_object.compile(
        loss=loss_dict,
        optimizer=optimizer_function,
        metrics=metric_dict
    )

    model_object.summary()
    return model_object
