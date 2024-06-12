"""Methods for building, training, and applying neural nets."""

import os
import sys
import dill
import numpy
import keras

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import error_checking
import example_io
import example_utils
import normalization
import custom_losses
import custom_metrics

SENTINEL_VALUE = -9999.
LARGE_INTEGER = int(1e12)
LARGE_FLOAT = 1e12

METRES_TO_MICRONS = 1e6

# TODO(thunderhoser): This should become an input arg.
MAX_NUM_VALIDATION_EXAMPLES = int(5e5)
MAX_NUM_TRAINING_EXAMPLES = int(1e6)

PLATEAU_PATIENCE_EPOCHS = 10
DEFAULT_LEARNING_RATE_MULTIPLIER = 0.5
PLATEAU_COOLDOWN_EPOCHS = 0
DEFAULT_EARLY_STOPPING_PATIENCE_EPOCHS = 200
LOSS_PATIENCE = 0.

EXAMPLE_DIRECTORY_KEY = 'example_dir_name'
BATCH_SIZE_KEY = 'num_examples_per_batch'
SCALAR_PREDICTOR_NAMES_KEY = 'scalar_predictor_names'
VECTOR_PREDICTOR_NAMES_KEY = 'vector_predictor_names'
SCALAR_TARGET_NAMES_KEY = 'scalar_target_names'
VECTOR_TARGET_NAMES_KEY = 'vector_target_names'
HEIGHTS_KEY = 'heights_m_agl'
TARGET_WAVELENGTHS_KEY = 'target_wavelengths_metres'
FIRST_TIME_KEY = 'first_time_unix_sec'
LAST_TIME_KEY = 'last_time_unix_sec'
NORMALIZATION_FILE_KEY = 'normalization_file_name'
NORMALIZE_PREDICTORS_KEY = 'normalize_predictors'
NORMALIZE_SCALAR_TARGETS_KEY = 'normalize_scalar_targets'
NORMALIZE_VECTOR_TARGETS_KEY = 'normalize_vector_targets'
JOINED_OUTPUT_LAYER_KEY = 'joined_output_layer'
NUM_DEEP_SUPER_LAYERS_KEY = 'num_deep_supervision_layers'
NORMALIZATION_FILE_FOR_MASK_KEY = 'normalization_file_name_for_mask'
MIN_HEATING_RATE_FOR_MASK_KEY = 'min_heating_rate_for_mask_k_day01'
MIN_FLUX_FOR_MASK_KEY = 'min_flux_for_mask_w_m02'

DEFAULT_GENERATOR_OPTION_DICT = {
    SCALAR_PREDICTOR_NAMES_KEY: example_utils.ALL_SCALAR_PREDICTOR_NAMES,
    VECTOR_PREDICTOR_NAMES_KEY: example_utils.BASIC_VECTOR_PREDICTOR_NAMES,
    SCALAR_TARGET_NAMES_KEY: example_utils.ALL_SCALAR_TARGET_NAMES,
    VECTOR_TARGET_NAMES_KEY: example_utils.ALL_VECTOR_TARGET_NAMES,
    JOINED_OUTPUT_LAYER_KEY: False,
    NUM_DEEP_SUPER_LAYERS_KEY: 0
}

METRIC_FUNCTION_LIST = [
    custom_metrics.mean_bias, custom_metrics.mean_absolute_error,
    custom_metrics.mae_skill_score, custom_metrics.mean_squared_error,
    custom_metrics.mse_skill_score, custom_metrics.correlation
]

METRIC_FUNCTION_DICT = {
    'mean_bias': custom_metrics.mean_bias,
    'mean_absolute_error': custom_metrics.mean_absolute_error,
    'mae_skill_score': custom_metrics.mae_skill_score,
    'mean_squared_error': custom_metrics.mean_squared_error,
    'mse_skill_score': custom_metrics.mse_skill_score,
    'correlation': custom_metrics.correlation
}

NUM_EPOCHS_KEY = 'num_epochs'
NUM_TRAINING_BATCHES_KEY = 'num_training_batches_per_epoch'
TRAINING_OPTIONS_KEY = 'training_option_dict'
NUM_VALIDATION_BATCHES_KEY = 'num_validation_batches_per_epoch'
VALIDATION_OPTIONS_KEY = 'validation_option_dict'
LOSS_FUNCTION_OR_DICT_KEY = 'loss_function_or_dict'
PLATEAU_LR_MUTIPLIER_KEY = 'plateau_lr_multiplier'
EARLY_STOPPING_PATIENCE_KEY = 'early_stopping_patience_epochs'
DENSE_ARCHITECTURE_KEY = 'dense_architecture_dict'
CNN_ARCHITECTURE_KEY = 'cnn_architecture_dict'
BNN_ARCHITECTURE_KEY = 'bnn_architecture_dict'
U_NET_ARCHITECTURE_KEY = 'u_net_architecture_dict'
U_NET_PP_ARCHITECTURE_KEY = 'u_net_plusplus_architecture_dict'
U_NET_PPP_ARCHITECTURE_KEY = 'u_net_3plus_architecture_dict'

METADATA_KEYS = [
    NUM_EPOCHS_KEY, NUM_TRAINING_BATCHES_KEY, TRAINING_OPTIONS_KEY,
    NUM_VALIDATION_BATCHES_KEY, VALIDATION_OPTIONS_KEY,
    LOSS_FUNCTION_OR_DICT_KEY,
    PLATEAU_LR_MUTIPLIER_KEY, EARLY_STOPPING_PATIENCE_KEY,
    DENSE_ARCHITECTURE_KEY, CNN_ARCHITECTURE_KEY, BNN_ARCHITECTURE_KEY,
    U_NET_ARCHITECTURE_KEY, U_NET_PP_ARCHITECTURE_KEY,
    U_NET_PPP_ARCHITECTURE_KEY
]

MAIN_PREDICTORS_KEY = 'main_predictors'
HEATING_RATE_MASK_KEY = 'heating_rate_mask_1_for_in'
FLUX_MASK_KEY = 'flux_mask_1_for_in'
PREDICTOR_KEYS_IN_ORDER = [
    MAIN_PREDICTORS_KEY, HEATING_RATE_MASK_KEY, FLUX_MASK_KEY
]

HEATING_RATE_TARGETS_KEY = 'conv_output'
FLUX_TARGETS_KEY = 'dense_output'
TARGET_KEYS_IN_ORDER = [HEATING_RATE_TARGETS_KEY, FLUX_TARGETS_KEY]

VECTOR_PREDICTOR_NAMES_FOR_PETER = [
    'pressure_pascals', 'temperature_kelvins', 'specific_humidity_kg_kg01',
    'liquid_water_content_kg_m03', 'ice_water_content_kg_m03',
    'o3_mixing_ratio_kg_kg01', 'co2_concentration_ppmv',
    'ch4_concentration_ppmv', 'n2o_concentration_ppmv',
    'aerosol_extinction_metres01',
    'liquid_effective_radius_metres', 'ice_effective_radius_metres',
    'height_thickness_metres', 'pressure_thickness_pascals', 'height_m_agl',
    'liquid_water_path_kg_m02', 'ice_water_path_kg_m02', 'vapour_path_kg_m02',
    'upward_liquid_water_path_kg_m02', 'upward_ice_water_path_kg_m02',
    'upward_vapour_path_kg_m02', 'relative_humidity_unitless'
]

SCALAR_PREDICTOR_NAMES_FOR_PETER = [
    example_utils.ZENITH_ANGLE_NAME, example_utils.ALBEDO_NAME,
    example_utils.LATITUDE_NAME, example_utils.LONGITUDE_NAME,
    example_utils.COLUMN_LIQUID_WATER_PATH_NAME,
    example_utils.COLUMN_ICE_WATER_PATH_NAME,
    example_utils.AEROSOL_ALBEDO_NAME,
    example_utils.AEROSOL_ASYMMETRY_PARAM_NAME
]

VECTOR_TARGET_NAMES_FOR_PETER = [
    example_utils.SHORTWAVE_DOWN_FLUX_NAME,
    example_utils.SHORTWAVE_UP_FLUX_NAME,
    example_utils.SHORTWAVE_HEATING_RATE_NAME
]

SCALAR_TARGET_NAMES_FOR_PETER = []

TARGET_WAVELENGTHS_FOR_PETER_METRES = numpy.array([
    example_utils.DUMMY_BROADBAND_WAVELENGTH_METRES
])

HEIGHTS_FOR_PETER_M_AGL = numpy.array([
    21, 44, 68, 93, 120, 149, 179, 212, 246, 282, 321, 361, 405, 450, 499, 550,
    604, 661, 722, 785, 853, 924, 999, 1078, 1161, 1249, 1342, 1439, 1542, 1649,
    1762, 1881, 2005, 2136, 2272, 2415, 2564, 2720, 2882, 3051, 3228, 3411,
    3601, 3798, 4002, 4214, 4433, 4659, 4892, 5132, 5379, 5633, 5894, 6162,
    6436, 6716, 7003, 7296, 7594, 7899, 8208, 8523, 8842, 9166, 9494, 9827,
    10164, 10505, 10849, 11198, 11550, 11906, 12266, 12630, 12997, 13368, 13744,
    14123, 14506, 14895, 15287, 15686, 16090, 16501, 16920, 17350, 17791, 18246,
    18717, 19205, 19715, 20249, 20809, 21400, 22022, 22681, 23379, 24119, 24903,
    25736, 26619, 27558, 28556, 29616, 30743, 31940, 33211, 34566, 36012, 37560,
    39218, 40990, 42882, 44899, 47042, 49299, 51644, 54067, 56552, 59089, 61677,
    64314, 67001, 69747, 72521, 75256, 77803
], dtype=float)


def _check_generator_args(option_dict):
    """Error-checks input arguments for generator.

    :param option_dict: See doc for `data_generator`.
    :return: option_dict: Same as input, except defaults may have been added.
    """

    orig_option_dict = option_dict.copy()
    option_dict = DEFAULT_GENERATOR_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    error_checking.assert_is_integer(option_dict[BATCH_SIZE_KEY])
    error_checking.assert_is_geq(option_dict[BATCH_SIZE_KEY], 2)

    error_checking.assert_is_numpy_array(
        numpy.array(option_dict[SCALAR_PREDICTOR_NAMES_KEY]), num_dimensions=1
    )
    error_checking.assert_is_numpy_array(
        numpy.array(option_dict[VECTOR_PREDICTOR_NAMES_KEY]), num_dimensions=1
    )
    error_checking.assert_is_numpy_array(
        numpy.array(option_dict[SCALAR_TARGET_NAMES_KEY]), num_dimensions=1
    )
    error_checking.assert_is_numpy_array(
        numpy.array(option_dict[VECTOR_TARGET_NAMES_KEY]), num_dimensions=1
    )

    error_checking.assert_is_numpy_array(
        option_dict[HEIGHTS_KEY], num_dimensions=1
    )
    error_checking.assert_is_geq_numpy_array(option_dict[HEIGHTS_KEY], 0.)

    error_checking.assert_is_numpy_array(
        option_dict[TARGET_WAVELENGTHS_KEY], num_dimensions=1
    )
    error_checking.assert_is_geq_numpy_array(
        option_dict[TARGET_WAVELENGTHS_KEY], 0.
    )

    if option_dict[NORMALIZATION_FILE_KEY] is not None:
        error_checking.assert_is_string(option_dict[NORMALIZATION_FILE_KEY])

    error_checking.assert_is_boolean(option_dict[NORMALIZE_PREDICTORS_KEY])
    error_checking.assert_is_boolean(option_dict[NORMALIZE_SCALAR_TARGETS_KEY])
    error_checking.assert_is_boolean(option_dict[NORMALIZE_VECTOR_TARGETS_KEY])

    if option_dict[MIN_HEATING_RATE_FOR_MASK_KEY] is not None:
        error_checking.assert_is_greater(
            option_dict[MIN_HEATING_RATE_FOR_MASK_KEY], 0.
        )

    if option_dict[MIN_FLUX_FOR_MASK_KEY] is not None:
        error_checking.assert_is_greater(
            option_dict[MIN_FLUX_FOR_MASK_KEY], 0.
        )

    return option_dict


def _check_inference_args(predictor_matrix_or_list, num_examples_per_batch,
                          verbose):
    """Error-checks input arguments for inference.

    :param predictor_matrix: See doc for `apply_model`.
    :param num_examples_per_batch: Batch size.
    :param verbose: Boolean flag.  If True, will print progress messages during
        inference.
    :return: num_examples: Total number of examples.
    :return: num_examples_per_batch: Batch size (may be different than input).
    """

    if isinstance(predictor_matrix_or_list, list):
        for this_matrix in predictor_matrix_or_list:
            error_checking.assert_is_numpy_array_without_nan(this_matrix)
            num_examples = this_matrix.shape[0]
    else:
        error_checking.assert_is_numpy_array_without_nan(
            predictor_matrix_or_list
        )
        num_examples = predictor_matrix_or_list.shape[0]

    if num_examples_per_batch is None:
        num_examples_per_batch = num_examples + 0
    else:
        error_checking.assert_is_integer(num_examples_per_batch)
        # error_checking.assert_is_geq(num_examples_per_batch, 100)
        error_checking.assert_is_geq(num_examples_per_batch, 1)

    num_examples_per_batch = min([num_examples_per_batch, num_examples])
    error_checking.assert_is_boolean(verbose)

    return num_examples, num_examples_per_batch


def _read_file_for_generator(
        example_file_name, first_time_unix_sec, last_time_unix_sec, field_names,
        heights_m_agl, target_wavelengths_metres, normalization_file_name,
        normalize_predictors, normalize_vector_targets,
        normalize_scalar_targets):
    """Reads one file for generator.

    H = number of heights in grid
    W = number of wavelengths
    F_st = number of scalar target fields

    :param example_file_name: Path to input file (will be read by
        `example_io.read_file`).
    :param first_time_unix_sec: See doc for `data_generator`.
    :param last_time_unix_sec: Same.
    :param field_names: 1-D list of fields to keep.
    :param heights_m_agl: 1-D numpy array of heights to keep (metres above
        ground level).
    :param target_wavelengths_metres: 1-D numpy array of wavelengths to keep.
    :param normalization_file_name: See doc for `data_generator`.
    :param normalize_predictors: Same.
    :param normalize_vector_targets: Same.
    :param normalize_scalar_targets: Same.
    :return: example_dict: See doc for `example_io.read_file`.
    """

    print('\nReading data from: "{0:s}"...'.format(example_file_name))
    example_dict = example_io.read_file(netcdf_file_name=example_file_name)

    example_dict = example_utils.subset_by_time(
        example_dict=example_dict,
        first_time_unix_sec=first_time_unix_sec,
        last_time_unix_sec=last_time_unix_sec
    )[0]

    example_dict = example_utils.subset_by_wavelength(
        example_dict=example_dict,
        target_wavelengths_metres=target_wavelengths_metres
    )
    example_dict = example_utils.subset_by_field(
        example_dict=example_dict, field_names=field_names
    )
    example_dict = example_utils.subset_by_height(
        example_dict=example_dict, heights_m_agl=heights_m_agl
    )

    previous_norm_file_name = example_dict[
        example_utils.NORMALIZATION_METADATA_KEY
    ][example_io.NORMALIZATION_FILE_KEY]

    if previous_norm_file_name is not None:
        normalization_metadata_dict = {
            example_io.NORMALIZATION_FILE_KEY: normalization_file_name,
            example_io.NORMALIZE_PREDICTORS_KEY: normalize_predictors,
            example_io.NORMALIZE_SCALAR_TARGETS_KEY: normalize_scalar_targets,
            example_io.NORMALIZE_VECTOR_TARGETS_KEY: normalize_vector_targets
        }

        assert example_io.are_normalization_metadata_same(
            normalization_metadata_dict,
            example_dict[example_utils.NORMALIZATION_METADATA_KEY]
        )

        return example_dict

    if normalization_file_name is None:
        return example_dict

    print('Reading normalization params from: "{0:s}"...'.format(
        normalization_file_name
    ))
    norm_param_table_xarray = normalization.read_params(normalization_file_name)

    print('Normalizing data...')
    normalization.normalize_data(
        example_dict=example_dict,
        normalization_param_table_xarray=norm_param_table_xarray,
        apply_to_predictors=normalize_predictors,
        apply_to_scalar_targets=normalize_scalar_targets,
        apply_to_vector_targets=normalize_vector_targets
    )

    return example_dict


def _write_metafile(
        dill_file_name, num_epochs, num_training_batches_per_epoch,
        training_option_dict, num_validation_batches_per_epoch,
        validation_option_dict, loss_function_or_dict,
        plateau_lr_multiplier, early_stopping_patience_epochs,
        dense_architecture_dict, cnn_architecture_dict, bnn_architecture_dict,
        u_net_architecture_dict, u_net_plusplus_architecture_dict,
        u_net_3plus_architecture_dict):
    """Writes metadata to Dill file.

    :param dill_file_name: Path to output file.
    :param num_epochs: See doc for `train_model`.
    :param num_training_batches_per_epoch: Same.
    :param training_option_dict: Same.
    :param num_validation_batches_per_epoch: Same.
    :param validation_option_dict: Same.
    :param loss_function_or_dict: Same.
    :param plateau_lr_multiplier: Same.
    :param early_stopping_patience_epochs: Same.
    :param dense_architecture_dict: Same.
    :param cnn_architecture_dict: Same.
    :param bnn_architecture_dict: Same.
    :param u_net_architecture_dict: Same.
    :param u_net_plusplus_architecture_dict: Same.
    :param u_net_3plus_architecture_dict: Same.
    """

    metadata_dict = {
        NUM_EPOCHS_KEY: num_epochs,
        NUM_TRAINING_BATCHES_KEY: num_training_batches_per_epoch,
        TRAINING_OPTIONS_KEY: training_option_dict,
        NUM_VALIDATION_BATCHES_KEY: num_validation_batches_per_epoch,
        VALIDATION_OPTIONS_KEY: validation_option_dict,
        LOSS_FUNCTION_OR_DICT_KEY: loss_function_or_dict,
        PLATEAU_LR_MUTIPLIER_KEY: plateau_lr_multiplier,
        EARLY_STOPPING_PATIENCE_KEY: early_stopping_patience_epochs,
        DENSE_ARCHITECTURE_KEY: dense_architecture_dict,
        CNN_ARCHITECTURE_KEY: cnn_architecture_dict,
        BNN_ARCHITECTURE_KEY: bnn_architecture_dict,
        U_NET_ARCHITECTURE_KEY: u_net_architecture_dict,
        U_NET_PP_ARCHITECTURE_KEY: u_net_plusplus_architecture_dict,
        U_NET_PPP_ARCHITECTURE_KEY: u_net_3plus_architecture_dict
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=dill_file_name)

    dill_file_handle = open(dill_file_name, 'wb')
    dill.dump(metadata_dict, dill_file_handle)
    dill_file_handle.close()


def predictors_dict_to_numpy(example_dict):
    """Converts predictors from dictionary to numpy array.

    :param example_dict: Dictionary of examples (in the format returned by
        `example_io.read_file`).
    :return: predictor_matrix: See output doc for `data_generator`.
    :return: predictor_name_matrix: numpy array of predictor names (strings), in
        the same shape as predictor_matrix[0, ...].
    :return: height_matrix_m_agl: numpy array of heights (metres above ground
        level), in the same shape as predictor_matrix[0, ...].  For scalar
        variables, the matrix entry will be NaN.
    """

    heights_m_agl = example_dict[example_utils.HEIGHTS_KEY]
    vector_predictor_names = numpy.array(
        example_dict[example_utils.VECTOR_PREDICTOR_NAMES_KEY]
    )
    scalar_predictor_names = numpy.array(
        example_dict[example_utils.SCALAR_PREDICTOR_NAMES_KEY]
    )

    num_heights = len(heights_m_agl)
    num_vector_predictors = len(vector_predictor_names)
    num_scalar_predictors = len(scalar_predictor_names)

    vector_predictor_matrix = (
        example_dict[example_utils.VECTOR_PREDICTOR_VALS_KEY]
    )

    vector_height_matrix_m_agl = numpy.reshape(
        heights_m_agl, (num_heights, 1)
    )
    vector_height_matrix_m_agl = numpy.repeat(
        vector_height_matrix_m_agl, repeats=num_vector_predictors, axis=1
    )
    vector_predictor_name_matrix = numpy.reshape(
        vector_predictor_names, (1, num_vector_predictors)
    )
    vector_predictor_name_matrix = numpy.repeat(
        vector_predictor_name_matrix, repeats=num_heights, axis=0
    )

    scalar_predictor_matrix = (
        example_dict[example_utils.SCALAR_PREDICTOR_VALS_KEY]
    )

    scalar_predictor_matrix = numpy.expand_dims(
        scalar_predictor_matrix, axis=1
    )
    scalar_predictor_matrix = numpy.repeat(
        scalar_predictor_matrix, repeats=num_heights, axis=1
    )
    scalar_height_matrix_m_agl = numpy.full(
        scalar_predictor_matrix.shape[1:], numpy.nan
    )
    scalar_predictor_name_matrix = numpy.reshape(
        scalar_predictor_names, (1, num_scalar_predictors)
    )
    scalar_predictor_name_matrix = numpy.repeat(
        scalar_predictor_name_matrix, repeats=num_heights, axis=0
    )

    predictor_matrix = numpy.concatenate(
        (vector_predictor_matrix, scalar_predictor_matrix), axis=-1
    )
    height_matrix_m_agl = numpy.concatenate(
        (vector_height_matrix_m_agl, scalar_height_matrix_m_agl), axis=-1
    )
    predictor_name_matrix = numpy.concatenate((
        vector_predictor_name_matrix, scalar_predictor_name_matrix
    ), axis=-1)

    return predictor_matrix, predictor_name_matrix, height_matrix_m_agl


def predictors_numpy_to_dict(predictor_matrix, example_dict):
    """Converts predictors from numpy array to dictionary.

    This method is the inverse of `predictors_dict_to_numpy`.

    :param predictor_matrix: numpy array created by `predictors_dict_to_numpy`.
    :param example_dict: Dictionary with the following keys.  See doc for
        `example_io.read_file` for details on each key.
    example_dict['scalar_predictor_names']
    example_dict['vector_predictor_names']
    example_dict['heights_m_agl']

    :return: example_dict: Dictionary with the following keys.  See doc for
        `example_io.read_file` for details on each key.
    example_dict['scalar_predictor_matrix']
    example_dict['vector_predictor_matrix']
    """

    error_checking.assert_is_numpy_array_without_nan(predictor_matrix)
    error_checking.assert_is_numpy_array(predictor_matrix, num_dimensions=3)

    num_scalar_predictors = len(
        example_dict[example_utils.SCALAR_PREDICTOR_NAMES_KEY]
    )

    if num_scalar_predictors == 0:
        scalar_predictor_matrix = predictor_matrix[:, 0, :0]
        vector_predictor_matrix = predictor_matrix + 0.
    else:
        scalar_predictor_matrix = (
            predictor_matrix[:, 0, -num_scalar_predictors:]
        )
        vector_predictor_matrix = predictor_matrix[..., :-num_scalar_predictors]

    return {
        example_utils.SCALAR_PREDICTOR_VALS_KEY: scalar_predictor_matrix,
        example_utils.VECTOR_PREDICTOR_VALS_KEY: vector_predictor_matrix
    }


def targets_dict_to_numpy(example_dict):
    """Converts targets from dictionary to numpy array.

    E = number of examples
    H = number of heights in grid
    W = number of wavelengths
    T_v = number of vector target variables
    T_s = number of scalar target variables

    :param example_dict: Dictionary of examples (in the format returned by
        `example_io.read_file`).
    :return: vector_target_matrix: numpy array (E x H x W x T_v) of target
        values.
    :return: scalar_target_matrix: numpy array (E x W x T_s) of target values.
        If there are no scalar target variables (i.e., the NN does not predict
        fluxes), this is None.
    """

    vector_target_matrix = (
        example_dict[example_utils.VECTOR_TARGET_VALS_KEY]
    )
    scalar_target_matrix = (
        example_dict[example_utils.SCALAR_TARGET_VALS_KEY]
    )

    if scalar_target_matrix.size == 0:
        scalar_target_matrix = None

    return vector_target_matrix, scalar_target_matrix


def neuron_indices_to_target_var(neuron_indices, example_dict):
    """Converts indices of output neuron to metadata for target variable.

    :param neuron_indices: 1-D numpy array with indices of output neuron.  Must
        have length of either 1 (for scalar target variable) or 2 (for vector
        target variable).
    :param example_dict: Dictionary with the following keys.  See doc for
        `example_io.read_file` for details on each key.
    example_dict['scalar_target_names']
    example_dict['vector_target_names']
    example_dict['heights_m_agl']
    example_dict['target_wavelengths_metres']

    :return: target_name: Name of target variable.
    :return: height_m_agl: Height (metres above ground level) of target
        variable.  If target variable is scalar, this will be None.
    :return: wavelength_metres: Wavelength of target variable.
    """

    # TODO(thunderhoser): This won't work for NNs that output an ensemble.

    error_checking.assert_is_integer_numpy_array(neuron_indices)
    error_checking.assert_is_geq_numpy_array(neuron_indices, 0)
    error_checking.assert_is_numpy_array(neuron_indices, num_dimensions=1)

    error_checking.assert_is_geq(len(neuron_indices), 2)
    error_checking.assert_is_leq(len(neuron_indices), 3)

    scalar_target_names = example_dict[example_utils.SCALAR_TARGET_NAMES_KEY]
    vector_target_names = example_dict[example_utils.VECTOR_TARGET_NAMES_KEY]
    heights_m_agl = example_dict[example_utils.HEIGHTS_KEY]
    wavelengths_metres = example_dict[example_utils.TARGET_WAVELENGTHS_KEY]

    if len(neuron_indices) == 3:
        return (
            vector_target_names[neuron_indices[2]],
            heights_m_agl[neuron_indices[0]],
            wavelengths_metres[neuron_indices[1]]
        )

    return (
        scalar_target_names[neuron_indices[1]],
        None,
        wavelengths_metres[neuron_indices[0]]
    )


def target_var_to_neuron_indices(
        example_dict, target_name, wavelength_metres, height_m_agl=None):
    """Converts metadata for target variable to indices of output neuron.

    This method is the inverse of `neuron_indices_to_target_var`.

    :param example_dict: See doc for `neuron_indices_to_target_var`.
    :param target_name: Same.
    :param wavelength_metres: Same.
    :param height_m_agl: Same.
    :return: neuron_indices: Same.
    """

    # TODO(thunderhoser): This won't work for NNs that output an ensemble.

    error_checking.assert_is_string(target_name)

    scalar_target_names = example_dict[example_utils.SCALAR_TARGET_NAMES_KEY]
    vector_target_names = example_dict[example_utils.VECTOR_TARGET_NAMES_KEY]

    w = example_utils.match_wavelengths(
        wavelengths_metres=example_dict[example_utils.TARGET_WAVELENGTHS_KEY],
        desired_wavelength_metres=wavelength_metres
    )

    if height_m_agl is None:
        t = numpy.where(numpy.array(scalar_target_names) == target_name)[0][0]
        return numpy.array([w, t], dtype=int)

    h = example_utils.match_heights(
        heights_m_agl=example_dict[example_utils.HEIGHTS_KEY],
        desired_height_m_agl=height_m_agl
    )
    t = numpy.where(numpy.array(vector_target_names) == target_name)[0][0]
    return numpy.array([h, w, t], dtype=int)


def create_mask(
        normalization_file_name, min_heating_rate_k_day01, min_flux_w_m02,
        heights_m_agl, target_wavelengths_metres, vector_target_name,
        scalar_target_names, num_examples):
    """Creates mask.

    E = number of examples
    H = number of heights in grid
    W = number of wavelengths
    T_s = number of scalar target variables

    All height/wavelength and variable/wavelength pairs will be ignored by the
    NN, i.e., the NN will be forced to predict zero for all these values.

    :param normalization_file_name: Path to normalization file, readable by
        `normalization.read_params`.  For every height/wavelength and variable/
        wavelength pair, the climo-max value will be read from this file.  The
        file will be read by `example_io.read_file`.
    :param min_heating_rate_k_day01: Minimum heating rate.  Height/wavelength
        pairs with a climo-max HR below this threshold will be masked out.
    :param min_flux_w_m02: Minimum flux.  Variable/wavelength pairs with a
        climo-max flux below this threshold will be masked out.
    :param heights_m_agl: length-H numpy array of heights in grid.
    :param target_wavelengths_metres: length-W numpy array of wavelengths.
    :param vector_target_name: Name of vector target variable (heating-rate
        variable).
    :param scalar_target_names: length-T_s list with names of scalar target
        variables (fluxes).
    :param num_examples: Number of examples.
    :return: heating_rate_mask_1_for_in: E-by-H-by-W numpy array of
        Boolean flags, where 1 indicates that the given height/wavelength
        pair is masked in (the NN makes an actual prediction for this H/W pair)
        and 0 indicates that the given H/W pair is masked out (the NN always
        returns 0 K day^-1 for this H/W pair).
    :return: flux_mask_1_for_in: numpy array (E x W x T_s) of
        Boolean flags, where 1 indicates that the given variable/wavelength
        pair is masked in (the NN makes an actual prediction for this V/W pair)
        and 0 indicates that the given V/W pair is masked out (the NN always
        returns 0 W m^-2 for this V/W pair).
    """

    if normalization_file_name is None:
        return None, None

    print('Reading climo-max values for masking from: "{0:s}"...'.format(
        normalization_file_name
    ))
    norm_param_table_xarray = normalization.read_params(normalization_file_name)
    npt = norm_param_table_xarray

    if normalization.VECTOR_TARGET_DIM not in npt.coords:
        return _create_mask_old(
            normalization_file_name=normalization_file_name,
            min_heating_rate_k_day01=min_heating_rate_k_day01,
            min_flux_w_m02=min_flux_w_m02,
            heights_m_agl=heights_m_agl,
            target_wavelengths_metres=target_wavelengths_metres,
            vector_target_name=vector_target_name,
            scalar_target_names=scalar_target_names,
            num_examples=num_examples
        )

    num_heights = len(heights_m_agl)
    num_wavelengths = len(target_wavelengths_metres)
    heating_rate_mask_1_for_in = numpy.full(
        (num_heights, num_wavelengths), 0, dtype=int
    )

    j_new = numpy.where(
        npt.coords[normalization.VECTOR_TARGET_DIM].values == vector_target_name
    )[0][0]

    for h in range(num_heights):
        h_new = example_utils.match_heights(
            heights_m_agl=npt.coords[normalization.HEIGHT_DIM].values,
            desired_height_m_agl=heights_m_agl[h]
        )

        for w in range(num_wavelengths):
            w_new = example_utils.match_wavelengths(
                wavelengths_metres=
                npt.coords[normalization.WAVELENGTH_DIM].values,
                desired_wavelength_metres=target_wavelengths_metres[w]
            )

            this_max_value = numpy.max(numpy.absolute(
                npt[normalization.VECTOR_TARGET_QUANTILE_KEY].values[
                    h_new, w_new, j_new, :
                ]
            ))

            heating_rate_mask_1_for_in[h, w] = (
                this_max_value >= numpy.absolute(min_heating_rate_k_day01)
            ).astype(int)

            if heating_rate_mask_1_for_in[h, w] == 1:
                continue

            print((
                'Heating rate at {0:.0f} m AGL and {1:.2f} microns will be '
                'MASKED OUT (always zero)!'
            ).format(
                heights_m_agl[h],
                METRES_TO_MICRONS * target_wavelengths_metres[w]
            ))

    heating_rate_mask_1_for_in = numpy.expand_dims(
        heating_rate_mask_1_for_in, axis=0
    )
    heating_rate_mask_1_for_in = numpy.repeat(
        heating_rate_mask_1_for_in, axis=0, repeats=num_examples
    )

    num_flux_vars = len(scalar_target_names)
    if num_flux_vars == 0:
        return heating_rate_mask_1_for_in, None

    flux_mask_1_for_in = numpy.full(
        (num_wavelengths, num_flux_vars), 0, dtype=int
    )

    for w in range(num_wavelengths):
        w_new = example_utils.match_wavelengths(
            wavelengths_metres=
            npt.coords[normalization.WAVELENGTH_DIM].values,
            desired_wavelength_metres=target_wavelengths_metres[w]
        )

        for j in range(num_flux_vars):
            j_new = numpy.where(
                npt.coords[normalization.SCALAR_TARGET_DIM].values ==
                scalar_target_names[j]
            )[0][0]

            this_max_value = numpy.max(numpy.absolute(
                npt[normalization.SCALAR_TARGET_QUANTILE_KEY].values[
                    w_new, j_new, :
                ]
            ))

            flux_mask_1_for_in[w, j] = (
                this_max_value >= numpy.absolute(min_flux_w_m02)
            ).astype(int)

            if flux_mask_1_for_in[w, j] == 1:
                continue

            print((
                '{0:s} at {1:.2f} microns will be MASKED OUT (always zero)!'
            ).format(
                scalar_target_names[j],
                METRES_TO_MICRONS * target_wavelengths_metres[w]
            ))

    flux_mask_1_for_in = numpy.expand_dims(flux_mask_1_for_in, axis=0)
    flux_mask_1_for_in = numpy.repeat(
        flux_mask_1_for_in, axis=0, repeats=num_examples
    )

    return heating_rate_mask_1_for_in, flux_mask_1_for_in


def _create_mask_old(
        normalization_file_name, min_heating_rate_k_day01, min_flux_w_m02,
        heights_m_agl, target_wavelengths_metres, vector_target_name,
        scalar_target_names, num_examples):
    """Creates mask.

    E = number of examples
    H = number of heights in grid
    W = number of wavelengths
    T_s = number of scalar target variables

    All height/wavelength and variable/wavelength pairs will be ignored by the
    NN, i.e., the NN will be forced to predict zero for all these values.

    :param normalization_file_name: Path to normalization file, containing
        values in training data.  For every height/wavelength and variable/
        wavelength pair, the climo-max value will be read from this file.  The
        file will be read by `example_io.read_file`.
    :param min_heating_rate_k_day01: Minimum heating rate.  Height/wavelength
        pairs with a climo-max HR below this threshold will be masked out.
    :param min_flux_w_m02: Minimum flux.  Variable/wavelength pairs with a
        climo-max flux below this threshold will be masked out.
    :param heights_m_agl: length-H numpy array of heights in grid.
    :param target_wavelengths_metres: length-W numpy array of wavelengths.
    :param vector_target_name: Name of vector target variable (heating-rate
        variable).
    :param scalar_target_names: length-T_s list with names of scalar target
        variables (fluxes).
    :param num_examples: Number of examples.
    :return: heating_rate_mask_1_for_in: E-by-H-by-W numpy array of
        Boolean flags, where 1 indicates that the given height/wavelength
        pair is masked in (the NN makes an actual prediction for this H/W pair)
        and 0 indicates that the given H/W pair is masked out (the NN always
        returns 0 K day^-1 for this H/W pair).
    :return: flux_mask_1_for_in: numpy array (E x W x T_s) of
        Boolean flags, where 1 indicates that the given variable/wavelength
        pair is masked in (the NN makes an actual prediction for this V/W pair)
        and 0 indicates that the given V/W pair is masked out (the NN always
        returns 0 W m^-2 for this V/W pair).
    """

    # TODO(thunderhoser): I can delete this legacy method once I no longer have
    # models that are using the old type of mask (based on the old type of
    # normalization file, which is just 100 000 training examples with no
    # statistics).

    if normalization_file_name is None:
        return None, None

    print('Reading climo-max values for masking from: "{0:s}"...'.format(
        normalization_file_name
    ))
    training_example_dict = example_io.read_file(normalization_file_name)

    num_heights = len(heights_m_agl)
    num_wavelengths = len(target_wavelengths_metres)
    heating_rate_mask_1_for_in = numpy.full(
        (num_heights, num_wavelengths), 0, dtype=int
    )

    for i in range(num_heights):
        for j in range(num_wavelengths):
            these_values = example_utils.get_field_from_dict(
                example_dict=training_example_dict,
                field_name=vector_target_name,
                height_m_agl=heights_m_agl[i],
                target_wavelength_metres=target_wavelengths_metres[j]
            )

            heating_rate_mask_1_for_in[i, j] = (
                numpy.max(numpy.absolute(these_values)) >=
                numpy.absolute(min_heating_rate_k_day01)
            ).astype(int)

            if heating_rate_mask_1_for_in[i, j] == 1:
                continue

            print((
                'Heating rate at {0:.0f} m AGL and {1:.2f} microns will be '
                'MASKED OUT (always zero)!'
            ).format(
                heights_m_agl[i],
                METRES_TO_MICRONS * target_wavelengths_metres[j]
            ))

    heating_rate_mask_1_for_in = numpy.expand_dims(
        heating_rate_mask_1_for_in, axis=0
    )
    heating_rate_mask_1_for_in = numpy.repeat(
        heating_rate_mask_1_for_in, axis=0, repeats=num_examples
    )

    num_flux_vars = len(scalar_target_names)
    if num_flux_vars == 0:
        return heating_rate_mask_1_for_in, None

    flux_mask_1_for_in = numpy.full(
        (num_wavelengths, num_flux_vars), 0, dtype=int
    )

    for j in range(num_wavelengths):
        for k in range(num_flux_vars):
            these_values = example_utils.get_field_from_dict(
                example_dict=training_example_dict,
                field_name=scalar_target_names[k],
                target_wavelength_metres=target_wavelengths_metres[j]
            )

            flux_mask_1_for_in[j, k] = (
                numpy.max(numpy.absolute(these_values)) >=
                numpy.absolute(min_flux_w_m02)
            ).astype(int)

            if flux_mask_1_for_in[j, k] == 1:
                continue

            print((
                '{0:s} at {1:.2f} microns will be MASKED OUT (always zero)!'
            ).format(
                scalar_target_names[k],
                METRES_TO_MICRONS * target_wavelengths_metres[j]
            ))

    flux_mask_1_for_in = numpy.expand_dims(flux_mask_1_for_in, axis=0)
    flux_mask_1_for_in = numpy.repeat(
        flux_mask_1_for_in, axis=0, repeats=num_examples
    )

    return heating_rate_mask_1_for_in, flux_mask_1_for_in


def data_generator(option_dict, for_inference):
    """Generates training data for any kind of neural net.

    E = number of examples per batch (batch size)
    H = number of heights
    W = number of wavelengths
    P = number of predictor variables (channels)
    T_v = number of vector target variables (channels)
    T_s = number of scalar target variables
    T = number of target variables

    :param option_dict: Dictionary with the following keys.
    option_dict['example_dir_name']: Name of directory with example files.
        Files therein will be found by `example_io.find_file` and read by
        `example_io.read_file`.
    option_dict['num_examples_per_batch']: Batch size.
    option_dict['predictor_names']: 1-D list with names of predictor variables
        (valid names listed in example_utils.py).
    option_dict['target_names']: Same but for target variables.
    option_dict['heights_m_agl']: length-H numpy array of heights (metres above
        ground level).  Applies to both predictor and target variables.
    option_dict['target_wavelengths_metres']: length-W numpy array of
        wavelengths to model.
    option_dict['first_time_unix_sec']: Start time (will not generate examples
        before this time).
    option_dict['last_time_unix_sec']: End time (will not generate examples
        after this time).
    option_dict['normalization_file_name']: Path to file with normalization
        parameters (will be read by `normalization.read_params`).
    option_dict['normalize_predictors']: Boolean flag.  If True, will normalize
        predictor variables.
    option_dict['normalize_scalar_target']: Boolean flag.  If True, will
        normalize scalar target variables (fluxes).
    option_dict['normalize_vector_target']: Boolean flag.  If True, will
        normalize vector target variables (heating rates).
    option_dict['joined_output_layer']: Boolean flag.  If True, heating rates
        and fluxes are all joined into one output layer.
    option_dict['num_deep_supervision_layers']: Number of deep-supervision
        layers.
    option_dict['min_heating_rate_for_mask_k_day01']: Minimum heating rate for
        masking.  Every height/wavelength pair with a climo-max (based on the
        normalization file) heating rate below this threshold will be masked
        out, i.e., the NN will be forced to predict zero for this
        height/wavelength pair.  If you do not want to apply masking, make this
        None.
    option_dict['min_flux_for_mask_w_m02']: Minimum flux for masking.  Every
        variable/wavelength pair with a climo-max (based on the normalization
        file) flux below this threshold will be masked out, i.e., the NN will be
        forced to predict zero for this variable/wavelength pair.  If you do not
        want to apply masking, make this None.
    option_dict['normalization_file_name_for_mask']: Climo-max heating rates and
        fluxes will be found in this file, to be read by
        `normalization.read_params`.  If you do not want to apply masking, make
        this None.

    :param for_inference: Boolean flag.  If True, generator is being used for
        inference stage (applying trained model to new data).  If False,
        generator is being used for training or monitoring (on-the-fly
        validation).

    :return: predictor_matrix_or_dict: Without masking, this is a single array
        of predictor values, with dimensions E x H x P.  With masking, this is a
        dictionary with the following keys.

    predictor_matrix_or_dict['main_predictors']: E-by-H-by-P numpy array of
        predictor values.
    predictor_matrix_or_dict['heating_rate_mask_1_for_in']: H-by-W numpy array
        of Boolean flags, where 1 indicates that the given height/wavelength
        pair is masked in (the NN makes an actual prediction for this H/W pair)
        and 0 indicates that the given H/W pair is masked out (the NN always
        returns 0 K day^-1 for this H/W pair).
    predictor_matrix_or_dict['flux_mask_1_for_in']: numpy array (W x T_s) of
        Boolean flags, where 1 indicates that the given variable/wavelength
        pair is masked in (the NN makes an actual prediction for this V/W pair)
        and 0 indicates that the given V/W pair is masked out (the NN always
        returns 0 W m^-2 for this V/W pair).

    :return: target_matrix_or_dict: If the NN does not predict fluxes (i.e.,
        predicts only heating rates), this is a single numpy array, with
        dimensions E x H x W x T_v.  If the NN predicts fluxes, this is a
        dictionary with the following keys.
    target_matrix_or_dict['conv_output']: numpy array (E x H x W x T_v) of
        actual heating rates.
    target_matrix_or_dict['dense_output']: numpy array (E x W x T_s) of actual
        fluxes.

    :return: example_id_strings: [returned only if `for_inference == True`]
        length-E list of example IDs created by
        `example_utils.create_example_ids`.
    """

    option_dict = _check_generator_args(option_dict)
    error_checking.assert_is_boolean(for_inference)

    example_dir_name = option_dict[EXAMPLE_DIRECTORY_KEY]
    num_examples_per_batch = option_dict[BATCH_SIZE_KEY]
    scalar_predictor_names = option_dict[SCALAR_PREDICTOR_NAMES_KEY]
    vector_predictor_names = option_dict[VECTOR_PREDICTOR_NAMES_KEY]
    scalar_target_names = option_dict[SCALAR_TARGET_NAMES_KEY]
    vector_target_names = option_dict[VECTOR_TARGET_NAMES_KEY]
    heights_m_agl = option_dict[HEIGHTS_KEY]
    target_wavelengths_metres = option_dict[TARGET_WAVELENGTHS_KEY]
    first_time_unix_sec = option_dict[FIRST_TIME_KEY]
    last_time_unix_sec = option_dict[LAST_TIME_KEY]

    all_field_names = (
        scalar_predictor_names + vector_predictor_names +
        scalar_target_names + vector_target_names
    )

    normalization_file_name = option_dict[NORMALIZATION_FILE_KEY]
    normalize_predictors = option_dict[NORMALIZE_PREDICTORS_KEY]
    normalize_scalar_targets = option_dict[NORMALIZE_SCALAR_TARGETS_KEY]
    normalize_vector_targets = option_dict[NORMALIZE_VECTOR_TARGETS_KEY]
    joined_output_layer = option_dict[JOINED_OUTPUT_LAYER_KEY]
    num_deep_supervision_layers = option_dict[NUM_DEEP_SUPER_LAYERS_KEY]
    min_heating_rate_for_mask_k_day01 = option_dict[
        MIN_HEATING_RATE_FOR_MASK_KEY
    ]
    min_flux_for_mask_w_m02 = option_dict[MIN_FLUX_FOR_MASK_KEY]
    normalization_file_name_for_mask = option_dict[
        NORMALIZATION_FILE_FOR_MASK_KEY
    ]

    heating_rate_mask_matrix, flux_mask_matrix = create_mask(
        normalization_file_name=normalization_file_name_for_mask,
        min_heating_rate_k_day01=min_heating_rate_for_mask_k_day01,
        min_flux_w_m02=min_flux_for_mask_w_m02,
        heights_m_agl=heights_m_agl,
        target_wavelengths_metres=target_wavelengths_metres,
        vector_target_name=vector_target_names[0],
        scalar_target_names=scalar_target_names,
        num_examples=num_examples_per_batch
    )

    assert not (joined_output_layer and num_deep_supervision_layers > 0)

    example_file_names = example_io.find_many_files(
        directory_name=example_dir_name,
        first_time_unix_sec=first_time_unix_sec,
        last_time_unix_sec=last_time_unix_sec,
        raise_error_if_any_missing=False
    )

    example_dict = _read_file_for_generator(
        example_file_name=example_file_names[0],
        first_time_unix_sec=first_time_unix_sec,
        last_time_unix_sec=last_time_unix_sec,
        field_names=all_field_names,
        heights_m_agl=heights_m_agl,
        target_wavelengths_metres=target_wavelengths_metres,
        normalization_file_name=normalization_file_name,
        normalize_predictors=normalize_predictors,
        normalize_scalar_targets=normalize_scalar_targets,
        normalize_vector_targets=normalize_vector_targets
    )

    num_examples_in_dict = len(example_dict[example_utils.EXAMPLE_IDS_KEY])
    example_dict = example_utils.subset_by_index(
        example_dict=example_dict,
        desired_indices=numpy.random.permutation(num_examples_in_dict)
    )

    file_index = 0
    first_example_index = 0

    while True:
        num_examples_in_memory = 0
        predictor_matrix = None
        vector_target_matrix = None
        scalar_target_matrix = None
        example_id_strings = []

        if (
                for_inference and
                file_index >= len(example_file_names) - 1 and
                first_example_index >= num_examples_in_dict
        ):
            raise StopIteration

        while num_examples_in_memory < num_examples_per_batch:
            if (
                    for_inference and
                    file_index >= len(example_file_names) - 1 and
                    first_example_index >= num_examples_in_dict
            ):
                if predictor_matrix is None:
                    raise StopIteration

                break

            if first_example_index >= num_examples_in_dict:
                first_example_index = 0
                file_index += 1
                if file_index == len(example_file_names):
                    file_index = 0

                example_dict = _read_file_for_generator(
                    example_file_name=example_file_names[file_index],
                    first_time_unix_sec=first_time_unix_sec,
                    last_time_unix_sec=last_time_unix_sec,
                    field_names=all_field_names,
                    heights_m_agl=heights_m_agl,
                    target_wavelengths_metres=target_wavelengths_metres,
                    normalization_file_name=normalization_file_name,
                    normalize_predictors=normalize_predictors,
                    normalize_scalar_targets=normalize_scalar_targets,
                    normalize_vector_targets=normalize_vector_targets
                )

                num_examples_in_dict = len(
                    example_dict[example_utils.EXAMPLE_IDS_KEY]
                )
                example_dict = example_utils.subset_by_index(
                    example_dict=example_dict,
                    desired_indices=
                    numpy.random.permutation(num_examples_in_dict)
                )

            this_num_examples = num_examples_per_batch - num_examples_in_memory
            last_example_index = min([
                first_example_index + this_num_examples,
                num_examples_in_dict
            ])

            this_example_dict = dict()

            for k in [
                    example_utils.VECTOR_PREDICTOR_VALS_KEY,
                    example_utils.SCALAR_PREDICTOR_VALS_KEY,
                    example_utils.VECTOR_TARGET_VALS_KEY,
                    example_utils.SCALAR_TARGET_VALS_KEY
            ]:
                this_example_dict[k] = (
                    example_dict[k][first_example_index:last_example_index, ...]
                )

            print((
                'Mean vector-predictor value for examples {0:d}-{1:d} '
                'of {2:d} = {3:.4f}'
            ).format(
                first_example_index + 1, last_example_index + 1,
                num_examples_in_dict,
                numpy.mean(
                    this_example_dict[example_utils.VECTOR_PREDICTOR_VALS_KEY]
                )
            ))

            for k in [
                    example_utils.HEIGHTS_KEY,
                    example_utils.VECTOR_PREDICTOR_NAMES_KEY,
                    example_utils.SCALAR_PREDICTOR_NAMES_KEY
            ]:
                this_example_dict[k] = example_dict[k]

            example_id_strings += example_dict[example_utils.EXAMPLE_IDS_KEY][
                first_example_index:last_example_index
            ]
            first_example_index = last_example_index + 0

            this_predictor_matrix = predictors_dict_to_numpy(
                this_example_dict
            )[0]
            this_vector_target_matrix, this_scalar_target_matrix = (
                targets_dict_to_numpy(this_example_dict)
            )

            if predictor_matrix is None:
                predictor_matrix = this_predictor_matrix + 0.
                if this_vector_target_matrix is not None:
                    vector_target_matrix = this_vector_target_matrix + 0.
                if this_scalar_target_matrix is not None:
                    scalar_target_matrix = this_scalar_target_matrix + 0.
            else:
                predictor_matrix = numpy.concatenate(
                    (predictor_matrix, this_predictor_matrix), axis=0
                )

                if this_vector_target_matrix is not None:
                    vector_target_matrix = numpy.concatenate(
                        (vector_target_matrix, this_vector_target_matrix),
                        axis=0
                    )
                if this_scalar_target_matrix is not None:
                    scalar_target_matrix = numpy.concatenate(
                        (scalar_target_matrix, this_scalar_target_matrix),
                        axis=0
                    )

            num_examples_in_memory = predictor_matrix.shape[0]

        predictor_matrix = predictor_matrix.astype('float32')

        if joined_output_layer:
            target_matrix_or_dict = vector_target_matrix[..., 0].astype(
                'float32'
            )

            if scalar_target_matrix is not None:
                scalar_target_matrix = numpy.swapaxes(
                    scalar_target_matrix, 1, 2
                )
                target_matrix_or_dict = numpy.concatenate([
                    target_matrix_or_dict,
                    scalar_target_matrix.astype('float32')
                ], axis=-2)
        else:
            target_matrix_or_dict = {
                HEATING_RATE_TARGETS_KEY: vector_target_matrix.astype('float32')
            }

            if scalar_target_matrix is None:
                target_matrix_or_dict = target_matrix_or_dict[
                    HEATING_RATE_TARGETS_KEY
                ]
            else:
                target_matrix_or_dict[FLUX_TARGETS_KEY] = (
                    scalar_target_matrix.astype('float32')
                )

        # TODO(thunderhoser): This does not work anymore; in Keras 3 the
        # generator must return a dictionary, not a list.

        # for _ in range(num_deep_supervision_layers):
        #     target_array.append(target_array[0])

        if heating_rate_mask_matrix is None:
            predictor_matrix_or_dict = predictor_matrix
        else:
            predictor_matrix_or_dict = {
                MAIN_PREDICTORS_KEY: predictor_matrix,
                HEATING_RATE_MASK_KEY: heating_rate_mask_matrix,
                FLUX_MASK_KEY: flux_mask_matrix
            }

        if for_inference:
            yield (
                predictor_matrix_or_dict,
                target_matrix_or_dict,
                example_id_strings
            )
        else:
            yield predictor_matrix_or_dict, target_matrix_or_dict


def data_generator_for_peter(option_dict):
    """Generates training data for any kind of neural net.

    E = number of examples per batch (batch size)
    H = number of heights
    W = number of wavelengths
    P = number of predictor variables (channels)
    T_v = number of vector target variables (channels)
    T_s = number of scalar target variables
    T = number of target variables

    :param option_dict: Dictionary with the following keys.
    option_dict['example_dir_name']: Name of directory with example files.
        Files therein will be found by `example_io.find_file` and read by
        `example_io.read_file`.
    option_dict['num_examples_per_batch']: Batch size.
    option_dict['first_time_unix_sec']: Start time (will not generate examples
        before this time).
    option_dict['last_time_unix_sec']: End time (will not generate examples
        after this time).

    :return: predictor_dict: Dictionary with the following keys.
    predictor_dict['scalar_predictor_matrix']: numpy array (E x P_s) of
        predictor values.
    predictor_dict['vector_predictor_matrix']: numpy array (E x H x P_v) of
        predictor values.
    predictor_dict['toa_flux_input_matrix']: numpy array (E x 1 x 1) of TOA
        fluxes (W m^-2).

    :return: target_matrix: numpy array (E x H x T_v) of target values.
        target_matrix[:, 0] contains shortwave downward fluxes (W m^-2);
        target_matrix[:, 1] contains shortwave upward fluxes (W m^-2); and
        target_matrix[:, 2] contains shortwave heating rates (K day^-1).
    """

    # TODO(thunderhoser): Need to add input arg for_inference.
    # TODO(thunderhoser): At least I assume that the targets aren't normalized.
    example_dir_name = option_dict[EXAMPLE_DIRECTORY_KEY]
    num_examples_per_batch = option_dict[BATCH_SIZE_KEY]
    first_time_unix_sec = option_dict[FIRST_TIME_KEY]
    last_time_unix_sec = option_dict[LAST_TIME_KEY]

    all_field_names = (
        SCALAR_PREDICTOR_NAMES_FOR_PETER +
        VECTOR_PREDICTOR_NAMES_FOR_PETER +
        SCALAR_TARGET_NAMES_FOR_PETER +
        VECTOR_TARGET_NAMES_FOR_PETER
    )

    example_file_names = example_io.find_many_files(
        directory_name=example_dir_name,
        first_time_unix_sec=first_time_unix_sec,
        last_time_unix_sec=last_time_unix_sec,
        raise_error_if_any_missing=False
    )

    example_dict = _read_file_for_generator(
        example_file_name=example_file_names[0],
        first_time_unix_sec=first_time_unix_sec,
        last_time_unix_sec=last_time_unix_sec,
        field_names=all_field_names,
        heights_m_agl=HEIGHTS_FOR_PETER_M_AGL,
        target_wavelengths_metres=TARGET_WAVELENGTHS_FOR_PETER_METRES,
        normalization_file_name=None,
        normalize_predictors=False,
        normalize_scalar_targets=False,
        normalize_vector_targets=False
    )

    num_examples_in_dict = len(example_dict[example_utils.EXAMPLE_IDS_KEY])
    example_dict = example_utils.subset_by_index(
        example_dict=example_dict,
        desired_indices=numpy.random.permutation(num_examples_in_dict)
    )

    file_index = 0
    first_example_index = 0

    while True:
        num_examples_in_memory = 0
        predictor_matrix = None
        target_matrix = None

        while num_examples_in_memory < num_examples_per_batch:
            if first_example_index >= num_examples_in_dict:
                first_example_index = 0
                file_index += 1
                if file_index == len(example_file_names):
                    file_index = 0

                example_dict = _read_file_for_generator(
                    example_file_name=example_file_names[file_index],
                    first_time_unix_sec=first_time_unix_sec,
                    last_time_unix_sec=last_time_unix_sec,
                    field_names=all_field_names,
                    heights_m_agl=HEIGHTS_FOR_PETER_M_AGL,
                    target_wavelengths_metres=
                    TARGET_WAVELENGTHS_FOR_PETER_METRES,
                    normalization_file_name=None,
                    normalize_predictors=False,
                    normalize_scalar_targets=False,
                    normalize_vector_targets=False
                )

                num_examples_in_dict = len(
                    example_dict[example_utils.EXAMPLE_IDS_KEY]
                )
                example_dict = example_utils.subset_by_index(
                    example_dict=example_dict,
                    desired_indices=
                    numpy.random.permutation(num_examples_in_dict)
                )

            this_num_examples = num_examples_per_batch - num_examples_in_memory
            last_example_index = min([
                first_example_index + this_num_examples,
                num_examples_in_dict
            ])

            this_example_dict = dict()

            for k in [
                    example_utils.VECTOR_PREDICTOR_VALS_KEY,
                    example_utils.SCALAR_PREDICTOR_VALS_KEY,
                    example_utils.VECTOR_TARGET_VALS_KEY,
                    example_utils.SCALAR_TARGET_VALS_KEY
            ]:
                this_example_dict[k] = (
                    example_dict[k][first_example_index:last_example_index, ...]
                )

            print((
                'Mean vector-predictor value for examples {0:d}-{1:d} '
                'of {2:d} = {3:.4f}'
            ).format(
                first_example_index + 1, last_example_index + 1,
                num_examples_in_dict,
                numpy.mean(
                    this_example_dict[example_utils.VECTOR_PREDICTOR_VALS_KEY]
                )
            ))

            for k in [
                    example_utils.HEIGHTS_KEY,
                    example_utils.VECTOR_PREDICTOR_NAMES_KEY,
                    example_utils.SCALAR_PREDICTOR_NAMES_KEY
            ]:
                this_example_dict[k] = example_dict[k]

            first_example_index = last_example_index + 0

            this_predictor_matrix = predictors_dict_to_numpy(
                this_example_dict
            )[0]
            this_target_matrix = targets_dict_to_numpy(this_example_dict)[0]
            this_target_matrix = this_target_matrix[:, :, 0, :]

            if predictor_matrix is None:
                predictor_matrix = this_predictor_matrix + 0.
                target_matrix = this_target_matrix + 0.
            else:
                predictor_matrix = numpy.concatenate(
                    (predictor_matrix, this_predictor_matrix), axis=0
                )
                target_matrix = numpy.concatenate(
                    (target_matrix, this_target_matrix), axis=0
                )

            num_examples_in_memory = predictor_matrix.shape[0]

        predictor_matrix = predictor_matrix.astype('float32')
        predictor_dict = {
            'scalar_predictor_matrix':
                predictor_matrix[:, 0, len(VECTOR_PREDICTOR_NAMES_FOR_PETER):],
            'vector_predictor_matrix':
                predictor_matrix[..., :len(VECTOR_PREDICTOR_NAMES_FOR_PETER)],
            'toa_flux_input_matrix': target_matrix[:, -1:, :1]
        }

        yield predictor_dict, target_matrix


def create_data(option_dict):
    """Creates data for any kind of neural net.

    This method is the same as `data_generator`, except that it returns all the
    data at once, rather than generating batches on the fly.

    :param option_dict: See doc for `data_generator`.
    :return: predictor_dict: Same.
    :return: target_dict: Same.
    :return: example_id_strings: Same.
    """

    option_dict = _check_generator_args(option_dict)

    example_dir_name = option_dict[EXAMPLE_DIRECTORY_KEY]
    scalar_predictor_names = option_dict[SCALAR_PREDICTOR_NAMES_KEY]
    vector_predictor_names = option_dict[VECTOR_PREDICTOR_NAMES_KEY]
    scalar_target_names = option_dict[SCALAR_TARGET_NAMES_KEY]
    vector_target_names = option_dict[VECTOR_TARGET_NAMES_KEY]
    heights_m_agl = option_dict[HEIGHTS_KEY]
    target_wavelengths_metres = option_dict[TARGET_WAVELENGTHS_KEY]
    first_time_unix_sec = option_dict[FIRST_TIME_KEY]
    last_time_unix_sec = option_dict[LAST_TIME_KEY]

    all_field_names = (
        scalar_predictor_names + vector_predictor_names +
        scalar_target_names + vector_target_names
    )

    normalization_file_name = option_dict[NORMALIZATION_FILE_KEY]
    normalize_predictors = option_dict[NORMALIZE_PREDICTORS_KEY]
    normalize_scalar_targets = option_dict[NORMALIZE_SCALAR_TARGETS_KEY]
    normalize_vector_targets = option_dict[NORMALIZE_VECTOR_TARGETS_KEY]
    joined_output_layer = option_dict[JOINED_OUTPUT_LAYER_KEY]
    num_deep_supervision_layers = option_dict[NUM_DEEP_SUPER_LAYERS_KEY]
    min_heating_rate_for_mask_k_day01 = option_dict[
        MIN_HEATING_RATE_FOR_MASK_KEY
    ]
    min_flux_for_mask_w_m02 = option_dict[MIN_FLUX_FOR_MASK_KEY]
    normalization_file_name_for_mask = option_dict[
        NORMALIZATION_FILE_FOR_MASK_KEY
    ]

    assert not (joined_output_layer and num_deep_supervision_layers > 0)

    example_file_names = example_io.find_many_files(
        directory_name=example_dir_name,
        first_time_unix_sec=first_time_unix_sec,
        last_time_unix_sec=last_time_unix_sec,
        raise_error_if_any_missing=False
    )

    example_dicts = []

    for this_file_name in example_file_names:
        this_example_dict = _read_file_for_generator(
            example_file_name=this_file_name,
            first_time_unix_sec=first_time_unix_sec,
            last_time_unix_sec=last_time_unix_sec,
            field_names=all_field_names,
            heights_m_agl=heights_m_agl,
            target_wavelengths_metres=target_wavelengths_metres,
            normalization_file_name=normalization_file_name,
            normalize_predictors=normalize_predictors,
            normalize_scalar_targets=normalize_scalar_targets,
            normalize_vector_targets=normalize_vector_targets
        )

        example_dicts.append(this_example_dict)

    example_dict = example_utils.concat_examples(example_dicts)
    predictor_matrix = predictors_dict_to_numpy(example_dict)[0]
    predictor_matrix = predictor_matrix.astype('float32')

    vector_target_matrix, scalar_target_matrix = targets_dict_to_numpy(
        example_dict
    )

    if joined_output_layer:
        vector_target_matrix = vector_target_matrix[..., 0].astype('float32')

        if scalar_target_matrix is not None:
            scalar_target_matrix = numpy.swapaxes(scalar_target_matrix, 1, 2)
            vector_target_matrix = numpy.concatenate(
                [vector_target_matrix, scalar_target_matrix.astype('float32')],
                axis=-2
            )

        target_dict = {
            HEATING_RATE_TARGETS_KEY: vector_target_matrix.astype('float32')
        }
    else:
        target_dict = {
            HEATING_RATE_TARGETS_KEY: vector_target_matrix.astype('float32')
        }
        if scalar_target_matrix is not None:
            target_dict[FLUX_TARGETS_KEY] = scalar_target_matrix.astype('float32')

    # TODO(thunderhoser): Deep supervision is all fucked now.
    # for _ in range(num_deep_supervision_layers):
    #     target_array.append(target_array[0])

    heating_rate_mask_matrix, flux_mask_matrix = create_mask(
        normalization_file_name=normalization_file_name_for_mask,
        min_heating_rate_k_day01=min_heating_rate_for_mask_k_day01,
        min_flux_w_m02=min_flux_for_mask_w_m02,
        heights_m_agl=heights_m_agl,
        target_wavelengths_metres=target_wavelengths_metres,
        vector_target_name=vector_target_names[0],
        scalar_target_names=scalar_target_names,
        num_examples=predictor_matrix.shape[0]
    )

    predictor_dict = {MAIN_PREDICTORS_KEY: predictor_matrix}
    if heating_rate_mask_matrix is not None:
        predictor_dict[HEATING_RATE_MASK_KEY] = heating_rate_mask_matrix
    if flux_mask_matrix is not None:
        predictor_dict[FLUX_MASK_KEY] = flux_mask_matrix

    return (
        predictor_dict, target_dict, example_dict[example_utils.EXAMPLE_IDS_KEY]
    )


def create_data_specific_examples(option_dict, example_id_strings):
    """Creates data for specific examples.

    This method is the same as `create_data`, except that it creates specific
    examples.  Also, note that this method should be run only in inference mode
    (not in training mode).

    :param option_dict: See doc for `data_generator`.
    :param example_id_strings: 1-D list of example IDs.
    :return: predictor_dict: See doc for `data_generator`.
    :return: target_dict: Same.
    """

    option_dict = _check_generator_args(option_dict)

    example_times_unix_sec = example_utils.parse_example_ids(
        example_id_strings
    )[example_utils.VALID_TIMES_KEY]

    example_dir_name = option_dict[EXAMPLE_DIRECTORY_KEY]
    scalar_predictor_names = option_dict[SCALAR_PREDICTOR_NAMES_KEY]
    vector_predictor_names = option_dict[VECTOR_PREDICTOR_NAMES_KEY]
    scalar_target_names = option_dict[SCALAR_TARGET_NAMES_KEY]
    vector_target_names = option_dict[VECTOR_TARGET_NAMES_KEY]
    heights_m_agl = option_dict[HEIGHTS_KEY]
    target_wavelengths_metres = option_dict[TARGET_WAVELENGTHS_KEY]

    all_field_names = (
        scalar_predictor_names + vector_predictor_names +
        scalar_target_names + vector_target_names
    )

    normalization_file_name = option_dict[NORMALIZATION_FILE_KEY]
    normalize_predictors = option_dict[NORMALIZE_PREDICTORS_KEY]
    normalize_scalar_targets = option_dict[NORMALIZE_SCALAR_TARGETS_KEY]
    normalize_vector_targets = option_dict[NORMALIZE_VECTOR_TARGETS_KEY]
    min_heating_rate_for_mask_k_day01 = option_dict[
        MIN_HEATING_RATE_FOR_MASK_KEY
    ]
    min_flux_for_mask_w_m02 = option_dict[MIN_FLUX_FOR_MASK_KEY]
    normalization_file_name_for_mask = option_dict[
        NORMALIZATION_FILE_FOR_MASK_KEY
    ]

    num_examples = len(example_id_strings)
    heating_rate_mask_matrix, flux_mask_matrix = create_mask(
        normalization_file_name=normalization_file_name_for_mask,
        min_heating_rate_k_day01=min_heating_rate_for_mask_k_day01,
        min_flux_w_m02=min_flux_for_mask_w_m02,
        heights_m_agl=heights_m_agl,
        target_wavelengths_metres=target_wavelengths_metres,
        vector_target_name=vector_target_names[0],
        scalar_target_names=scalar_target_names,
        num_examples=num_examples
    )

    example_file_names = example_io.find_many_files(
        directory_name=example_dir_name,
        first_time_unix_sec=numpy.min(example_times_unix_sec),
        last_time_unix_sec=numpy.max(example_times_unix_sec),
        raise_error_if_any_missing=False
    )
    found_example_flags = numpy.full(num_examples, 0, dtype=bool)

    predictor_matrix = None
    vector_target_matrix = None
    scalar_target_matrix = None

    for this_file_name in example_file_names:
        missing_example_indices = numpy.where(
            numpy.invert(found_example_flags)
        )[0]

        if len(missing_example_indices) == 0:
            break

        this_example_dict = _read_file_for_generator(
            example_file_name=this_file_name,
            first_time_unix_sec=numpy.min(example_times_unix_sec),
            last_time_unix_sec=numpy.max(example_times_unix_sec),
            field_names=all_field_names,
            heights_m_agl=heights_m_agl,
            target_wavelengths_metres=target_wavelengths_metres,
            normalization_file_name=normalization_file_name,
            normalize_predictors=normalize_predictors,
            normalize_scalar_targets=normalize_scalar_targets,
            normalize_vector_targets=normalize_vector_targets
        )

        if len(this_example_dict[example_utils.EXAMPLE_IDS_KEY]) == 0:
            continue

        missing_to_dict_indices = example_utils.find_examples(
            all_id_strings=this_example_dict[example_utils.EXAMPLE_IDS_KEY],
            desired_id_strings=[
                example_id_strings[k] for k in missing_example_indices
            ],
            allow_missing=True
        )

        if numpy.all(missing_to_dict_indices < 0):
            continue

        missing_example_indices = (
            missing_example_indices[missing_to_dict_indices >= 0]
        )
        missing_to_dict_indices = (
            missing_to_dict_indices[missing_to_dict_indices >= 0]
        )
        this_example_dict = example_utils.subset_by_index(
            example_dict=this_example_dict,
            desired_indices=missing_to_dict_indices
        )

        this_predictor_matrix = predictors_dict_to_numpy(this_example_dict)[0]
        this_vector_target_matrix, this_scalar_target_matrix = (
            targets_dict_to_numpy(this_example_dict)
        )

        if predictor_matrix is None:
            predictor_matrix = numpy.full(
                (num_examples,) + this_predictor_matrix.shape[1:],
                numpy.nan
            )
            vector_target_matrix = numpy.full(
                (num_examples,) + this_vector_target_matrix.shape[1:],
                numpy.nan
            )

            if this_scalar_target_matrix is not None:
                scalar_target_matrix = numpy.full(
                    (num_examples,) + this_scalar_target_matrix.shape[1:],
                    numpy.nan
                )

        predictor_matrix[missing_example_indices, ...] = this_predictor_matrix
        vector_target_matrix[missing_example_indices, ...] = (
            this_vector_target_matrix
        )
        if this_scalar_target_matrix is not None:
            scalar_target_matrix[missing_example_indices, :] = (
                this_scalar_target_matrix
            )

        found_example_flags[missing_example_indices] = True

    assert numpy.all(found_example_flags)
    predictor_matrix = predictor_matrix.astype('float32')

    target_dict = {
        HEATING_RATE_TARGETS_KEY: vector_target_matrix.astype('float32')
    }
    if scalar_target_matrix is not None:
        target_dict[FLUX_TARGETS_KEY] = scalar_target_matrix.astype('float32')

    predictor_dict = {MAIN_PREDICTORS_KEY: predictor_matrix}
    if heating_rate_mask_matrix is not None:
        predictor_dict[HEATING_RATE_MASK_KEY] = heating_rate_mask_matrix
    if flux_mask_matrix is not None:
        predictor_dict[FLUX_MASK_KEY] = flux_mask_matrix

    return predictor_dict, target_dict


def train_model_with_generator_for_peter(
        model_object, output_dir_name, num_epochs,
        num_training_batches_per_epoch, training_option_dict,
        validation_option_dict,
        use_generator_for_validn, num_validation_batches_per_epoch,
        plateau_lr_multiplier, early_stopping_patience_epochs):
    """Trains any kind of neural net with generator.

    :param model_object: See doc for `train_model_with_generator`.
    :param output_dir_name: Same.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param training_option_dict: Same.
    :param validation_option_dict: Same.
    :param use_generator_for_validn: Same.
    :param num_validation_batches_per_epoch: Same.
    :param plateau_lr_multiplier: Same.
    :param early_stopping_patience_epochs: Same.
    """

    # TODO(thunderhoser): Might want to bring back input arg
    # loss_function_or_dict.

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    backup_dir_name = '{0:s}/backup_and_restore'.format(output_dir_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=backup_dir_name
    )

    error_checking.assert_is_integer(num_epochs)
    error_checking.assert_is_geq(num_epochs, 2)
    error_checking.assert_is_integer(num_training_batches_per_epoch)
    error_checking.assert_is_geq(num_training_batches_per_epoch, 10)
    error_checking.assert_is_boolean(use_generator_for_validn)
    error_checking.assert_is_greater(plateau_lr_multiplier, 0.)
    error_checking.assert_is_less_than(plateau_lr_multiplier, 1.)
    error_checking.assert_is_integer(early_stopping_patience_epochs)
    error_checking.assert_is_greater(early_stopping_patience_epochs, 0)

    if use_generator_for_validn:
        error_checking.assert_is_integer(num_validation_batches_per_epoch)
        error_checking.assert_is_geq(num_validation_batches_per_epoch, 10)
    else:
        num_validation_batches_per_epoch = None

    # training_option_dict = _check_generator_args(training_option_dict)

    validation_keys_to_keep = [
        EXAMPLE_DIRECTORY_KEY, BATCH_SIZE_KEY, FIRST_TIME_KEY, LAST_TIME_KEY
    ]

    for this_key in list(training_option_dict.keys()):
        if this_key in validation_keys_to_keep:
            continue

        validation_option_dict[this_key] = training_option_dict[this_key]

    # validation_option_dict = _check_generator_args(validation_option_dict)

    model_file_name = '{0:s}/model.keras'.format(output_dir_name)

    history_object = keras.callbacks.CSVLogger(
        filename='{0:s}/history.csv'.format(output_dir_name),
        separator=',', append=False
    )
    checkpoint_object = keras.callbacks.ModelCheckpoint(
        filepath=model_file_name, monitor='val_loss', verbose=1,
        save_best_only=True, save_weights_only=False, mode='min',
        save_freq='epoch'
    )
    backup_object = keras.callbacks.BackupAndRestore(
        backup_dir_name, save_freq='epoch', delete_checkpoint=True
    )
    list_of_callback_objects = [
        history_object, checkpoint_object, backup_object
    ]

    early_stopping_object = keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=LOSS_PATIENCE,
        patience=early_stopping_patience_epochs, verbose=1, mode='min'
    )
    list_of_callback_objects.append(early_stopping_object)

    plateau_object = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=plateau_lr_multiplier,
        patience=PLATEAU_PATIENCE_EPOCHS, verbose=1, mode='min',
        min_delta=LOSS_PATIENCE, cooldown=PLATEAU_COOLDOWN_EPOCHS
    )
    list_of_callback_objects.append(plateau_object)

    metafile_name = find_metafile(output_dir_name, raise_error_if_missing=False)
    print('Writing metadata to: "{0:s}"...'.format(metafile_name))

    _write_metafile(
        dill_file_name=metafile_name, num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        training_option_dict=training_option_dict,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        validation_option_dict=validation_option_dict,
        loss_function_or_dict='mse',
        plateau_lr_multiplier=plateau_lr_multiplier,
        early_stopping_patience_epochs=early_stopping_patience_epochs,
        dense_architecture_dict=None,
        cnn_architecture_dict=None,
        bnn_architecture_dict=None,
        u_net_architecture_dict=None,
        u_net_plusplus_architecture_dict=None,
        u_net_3plus_architecture_dict=None
    )

    training_generator = data_generator_for_peter(training_option_dict)

    if use_generator_for_validn:
        validation_generator = data_generator_for_peter(validation_option_dict)
        validation_data_arg = validation_generator
        validation_steps_arg = num_validation_batches_per_epoch
    else:
        validation_predictor_dict, validation_target_dict = create_data(
            validation_option_dict
        )[:2]

        validation_data_arg = (
            validation_predictor_dict, validation_target_dict
        )
        validation_steps_arg = None

    model_object.fit(
        x=training_generator,
        steps_per_epoch=num_training_batches_per_epoch,
        epochs=num_epochs, verbose=1, callbacks=list_of_callback_objects,
        validation_data=validation_data_arg,
        validation_steps=validation_steps_arg
    )


def train_model_with_generator(
        model_object, output_dir_name, num_epochs,
        num_training_batches_per_epoch, training_option_dict,
        validation_option_dict, loss_function_or_dict,
        use_generator_for_validn, num_validation_batches_per_epoch,
        plateau_lr_multiplier, early_stopping_patience_epochs,
        dense_architecture_dict, cnn_architecture_dict, bnn_architecture_dict,
        u_net_architecture_dict, u_net_plusplus_architecture_dict,
        u_net_3plus_architecture_dict):
    """Trains any kind of neural net with generator.

    :param model_object: Untrained neural net (instance of `keras.models.Model`
        or `keras.models.Sequential`).
    :param output_dir_name: Path to output directory (model and training history
        will be saved here).
    :param num_epochs: Number of training epochs.
    :param num_training_batches_per_epoch: Number of training batches per epoch.
    :param training_option_dict: See doc for `data_generator`.  This dictionary
        will be used to generate training data.
    :param validation_option_dict: See doc for `data_generator`.  For validation
        only, the following values will replace corresponding values in
        `training_option_dict`:
    validation_option_dict['example_dir_name']
    validation_option_dict['num_examples_per_batch']
    validation_option_dict['first_time_unix_sec']
    validation_option_dict['last_time_unix_sec']

    :param loss_function_or_dict: Loss function(s).  If the net has one loss
        function, this should be a function handle.  If the net has multiple
        loss functions, this should be a dictionary.
    :param use_generator_for_validn: Boolean flag.  If True, will use
        generator for validation data.  If False, will load all validation data
        into memory at once.
    :param num_validation_batches_per_epoch: Number of validation batches per
        epoch.  This is used only if `use_generator_for_validn = True`.  If
        `use_generator_for_validn = False`, all validation data will be used at
        each epoch.
    :param plateau_lr_multiplier: Multiplier for learning rate.  Learning
        rate will be multiplied by this factor upon plateau in validation
        performance.
    :param early_stopping_patience_epochs: Patience for early stopping.  Early
        stopping will be triggered if validation loss has not improved over this
        many epochs.
    :param dense_architecture_dict: Dictionary with architecture options for
        dense NN.  If the model being trained is not dense, make this None.
    :param cnn_architecture_dict: Same but for CNN.
    :param bnn_architecture_dict: Same but for Bayesian U-net++.
    :param u_net_architecture_dict: Same but for U-net.
    :param u_net_plusplus_architecture_dict: Same but for U-net++.
    :param u_net_3plus_architecture_dict: Same but for U-net 3+.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    backup_dir_name = '{0:s}/backup_and_restore'.format(output_dir_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=backup_dir_name
    )

    error_checking.assert_is_integer(num_epochs)
    error_checking.assert_is_geq(num_epochs, 2)
    error_checking.assert_is_integer(num_training_batches_per_epoch)
    error_checking.assert_is_geq(num_training_batches_per_epoch, 10)
    error_checking.assert_is_boolean(use_generator_for_validn)
    error_checking.assert_is_greater(plateau_lr_multiplier, 0.)
    error_checking.assert_is_less_than(plateau_lr_multiplier, 1.)
    error_checking.assert_is_integer(early_stopping_patience_epochs)
    error_checking.assert_is_greater(early_stopping_patience_epochs, 0)

    if use_generator_for_validn:
        error_checking.assert_is_integer(num_validation_batches_per_epoch)
        error_checking.assert_is_geq(num_validation_batches_per_epoch, 10)
    else:
        num_validation_batches_per_epoch = None

    training_option_dict = _check_generator_args(training_option_dict)

    validation_keys_to_keep = [
        EXAMPLE_DIRECTORY_KEY, BATCH_SIZE_KEY, FIRST_TIME_KEY, LAST_TIME_KEY
    ]

    for this_key in list(training_option_dict.keys()):
        if this_key in validation_keys_to_keep:
            continue

        validation_option_dict[this_key] = training_option_dict[this_key]

    validation_option_dict = _check_generator_args(validation_option_dict)

    model_file_name = '{0:s}/model.keras'.format(output_dir_name)

    history_object = keras.callbacks.CSVLogger(
        filename='{0:s}/history.csv'.format(output_dir_name),
        separator=',', append=False
    )
    checkpoint_object = keras.callbacks.ModelCheckpoint(
        filepath=model_file_name, monitor='val_loss', verbose=1,
        save_best_only=True, save_weights_only=False, mode='min',
        save_freq='epoch'
    )
    backup_object = keras.callbacks.BackupAndRestore(
        backup_dir_name, save_freq='epoch', delete_checkpoint=True
    )
    list_of_callback_objects = [
        history_object, checkpoint_object, backup_object
    ]

    early_stopping_object = keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=LOSS_PATIENCE,
        patience=early_stopping_patience_epochs, verbose=1, mode='min'
    )
    list_of_callback_objects.append(early_stopping_object)

    plateau_object = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=plateau_lr_multiplier,
        patience=PLATEAU_PATIENCE_EPOCHS, verbose=1, mode='min',
        min_delta=LOSS_PATIENCE, cooldown=PLATEAU_COOLDOWN_EPOCHS
    )
    list_of_callback_objects.append(plateau_object)

    metafile_name = find_metafile(output_dir_name, raise_error_if_missing=False)
    print('Writing metadata to: "{0:s}"...'.format(metafile_name))

    _write_metafile(
        dill_file_name=metafile_name, num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        training_option_dict=training_option_dict,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        validation_option_dict=validation_option_dict,
        loss_function_or_dict=loss_function_or_dict,
        plateau_lr_multiplier=plateau_lr_multiplier,
        early_stopping_patience_epochs=early_stopping_patience_epochs,
        dense_architecture_dict=dense_architecture_dict,
        cnn_architecture_dict=cnn_architecture_dict,
        bnn_architecture_dict=bnn_architecture_dict,
        u_net_architecture_dict=u_net_architecture_dict,
        u_net_plusplus_architecture_dict=u_net_plusplus_architecture_dict,
        u_net_3plus_architecture_dict=u_net_3plus_architecture_dict
    )

    training_generator = data_generator(
        option_dict=training_option_dict, for_inference=False
    )

    if use_generator_for_validn:
        validation_generator = data_generator(
            option_dict=validation_option_dict, for_inference=False
        )

        validation_data_arg = validation_generator
        validation_steps_arg = num_validation_batches_per_epoch
    else:
        validation_predictor_dict, validation_target_dict = create_data(
            validation_option_dict
        )[:2]

        validation_data_arg = (
            validation_predictor_dict, validation_target_dict
        )
        validation_steps_arg = None

    model_object.fit(
        x=training_generator,
        steps_per_epoch=num_training_batches_per_epoch,
        epochs=num_epochs, verbose=1, callbacks=list_of_callback_objects,
        validation_data=validation_data_arg,
        validation_steps=validation_steps_arg
    )


def train_model_sans_generator(
        model_object, output_dir_name, num_epochs, training_option_dict,
        validation_option_dict, loss_function_or_dict,
        num_training_batches_per_epoch, num_validation_batches_per_epoch,
        plateau_lr_multiplier, early_stopping_patience_epochs,
        dense_architecture_dict, cnn_architecture_dict, bnn_architecture_dict,
        u_net_architecture_dict, u_net_plusplus_architecture_dict,
        u_net_3plus_architecture_dict):
    """Trains any kind of neural net without generator.

    :param model_object: See doc for `train_model_with_generator`.
    :param output_dir_name: Same.
    :param num_epochs: Same.
    :param training_option_dict: Same.
    :param validation_option_dict: Same.
    :param loss_function_or_dict: Same.
    :param num_training_batches_per_epoch: Number of training batches per epoch.
        If None, each training example will be used once per epoch.
    :param num_validation_batches_per_epoch: Number of validation batches per
        epoch.  If None, each validation example will be used once per epoch.
    :param plateau_lr_multiplier: See doc for `train_model_with_generator`.
    :param early_stopping_patience_epochs: Same.
    :param dense_architecture_dict: Same.
    :param cnn_architecture_dict: Same.
    :param bnn_architecture_dict: Same.
    :param u_net_architecture_dict: Same.
    :param u_net_plusplus_architecture_dict: Same.
    :param u_net_3plus_architecture_dict: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    backup_dir_name = '{0:s}/backup_and_restore'.format(output_dir_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=backup_dir_name
    )

    error_checking.assert_is_integer(num_epochs)
    error_checking.assert_is_geq(num_epochs, 2)
    error_checking.assert_is_greater(plateau_lr_multiplier, 0.)
    error_checking.assert_is_less_than(plateau_lr_multiplier, 1.)
    error_checking.assert_is_integer(early_stopping_patience_epochs)
    error_checking.assert_is_greater(early_stopping_patience_epochs, 0)

    training_option_dict = _check_generator_args(training_option_dict)

    validation_keys_to_keep = [
        EXAMPLE_DIRECTORY_KEY, BATCH_SIZE_KEY, FIRST_TIME_KEY, LAST_TIME_KEY
    ]

    for this_key in list(training_option_dict.keys()):
        if this_key in validation_keys_to_keep:
            continue

        validation_option_dict[this_key] = training_option_dict[this_key]

    validation_option_dict = _check_generator_args(validation_option_dict)

    model_file_name = '{0:s}/model.keras'.format(output_dir_name)

    history_object = keras.callbacks.CSVLogger(
        filename='{0:s}/history.csv'.format(output_dir_name),
        separator=',', append=False
    )
    checkpoint_object = keras.callbacks.ModelCheckpoint(
        filepath=model_file_name, monitor='val_loss', verbose=1,
        save_best_only=True, save_weights_only=False, mode='min',
        save_freq='epoch'
    )
    backup_object = keras.callbacks.BackupAndRestore(
        backup_dir_name, save_freq='epoch', delete_checkpoint=True
    )
    list_of_callback_objects = [
        history_object, checkpoint_object, backup_object
    ]

    early_stopping_object = keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=LOSS_PATIENCE,
        patience=early_stopping_patience_epochs, verbose=1, mode='min'
    )
    list_of_callback_objects.append(early_stopping_object)

    plateau_object = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=plateau_lr_multiplier,
        patience=PLATEAU_PATIENCE_EPOCHS, verbose=1, mode='min',
        min_delta=LOSS_PATIENCE, cooldown=PLATEAU_COOLDOWN_EPOCHS
    )
    list_of_callback_objects.append(plateau_object)

    metafile_name = find_metafile(output_dir_name, raise_error_if_missing=False)
    print('Writing metadata to: "{0:s}"...'.format(metafile_name))

    _write_metafile(
        dill_file_name=metafile_name, num_epochs=num_epochs,
        num_training_batches_per_epoch=None,
        training_option_dict=training_option_dict,
        num_validation_batches_per_epoch=None,
        validation_option_dict=validation_option_dict,
        loss_function_or_dict=loss_function_or_dict,
        plateau_lr_multiplier=plateau_lr_multiplier,
        early_stopping_patience_epochs=early_stopping_patience_epochs,
        dense_architecture_dict=dense_architecture_dict,
        cnn_architecture_dict=cnn_architecture_dict,
        bnn_architecture_dict=bnn_architecture_dict,
        u_net_architecture_dict=u_net_architecture_dict,
        u_net_plusplus_architecture_dict=u_net_plusplus_architecture_dict,
        u_net_3plus_architecture_dict=u_net_3plus_architecture_dict
    )

    training_predictor_dict, training_target_dict = create_data(
        training_option_dict
    )[:2]

    validation_predictor_dict, validation_target_dict = create_data(
        validation_option_dict
    )[:2]

    # TODO(thunderhoser): HACK to deal with out-of-memory errors.
    num_validation_examples = (
        validation_predictor_dict[MAIN_PREDICTORS_KEY].shape[0]
    )
    print('Number of validation examples = {0:d}'.format(
        num_validation_examples
    ))

    if num_validation_examples > MAX_NUM_VALIDATION_EXAMPLES:
        print((
            'POTENTIAL ERROR: Reducing number of validation examples '
            'from {0:d} to {1:d}.'
        ).format(
            num_validation_examples, MAX_NUM_VALIDATION_EXAMPLES
        ))

        random_indices = numpy.linspace(
            0, num_validation_examples - 1, num=num_validation_examples,
            dtype=int
        )
        random_indices = numpy.random.choice(
            random_indices, size=MAX_NUM_VALIDATION_EXAMPLES, replace=False
        )

        for this_key in validation_predictor_dict:
            validation_predictor_dict[this_key] = (
                validation_predictor_dict[this_key][random_indices, ...]
            )
        for this_key in validation_target_dict:
            validation_target_dict[this_key] = (
                validation_target_dict[this_key][random_indices, ...]
            )

    num_training_examples = (
        training_predictor_dict[MAIN_PREDICTORS_KEY].shape[0]
    )
    print('Number of training examples = {0:d}'.format(
        num_training_examples
    ))

    if num_training_examples > MAX_NUM_TRAINING_EXAMPLES:
        print((
            'POTENTIAL ERROR: Reducing number of training examples '
            'from {0:d} to {1:d}.'
        ).format(
            num_training_examples, MAX_NUM_TRAINING_EXAMPLES
        ))

        random_indices = numpy.linspace(
            0, num_training_examples - 1, num=num_training_examples,
            dtype=int
        )
        random_indices = numpy.random.choice(
            random_indices, size=MAX_NUM_TRAINING_EXAMPLES, replace=False
        )

        for this_key in training_predictor_dict:
            training_predictor_dict[this_key] = (
                training_predictor_dict[this_key][random_indices, ...]
            )
        for this_key in training_target_dict:
            training_target_dict[this_key] = (
                training_target_dict[this_key][random_indices, ...]
            )

    model_object.fit(
        x=training_predictor_dict,
        y=training_target_dict,
        batch_size=training_option_dict[BATCH_SIZE_KEY],
        epochs=num_epochs,
        steps_per_epoch=num_training_batches_per_epoch,
        shuffle=True,
        verbose=1,
        callbacks=list_of_callback_objects,
        validation_data=(validation_predictor_dict, validation_target_dict),
        # validation_batch_size=validation_option_dict[BATCH_SIZE_KEY],
        validation_steps=num_validation_batches_per_epoch
    )


def read_model(hdf5_file_name):
    """Reads model from HDF5 file.

    :param hdf5_file_name: Path to input file.
    :return: model_object: Instance of `keras.models.Model` or
        `keras.models.Sequential`.
    """

    error_checking.assert_file_exists(hdf5_file_name)

    metafile_name = find_metafile(
        model_dir_name=os.path.split(hdf5_file_name)[0],
        raise_error_if_missing=True
    )
    metadata_dict = read_metafile(metafile_name)

    if (
            isinstance(metadata_dict[LOSS_FUNCTION_OR_DICT_KEY], str)
            and metadata_dict[LOSS_FUNCTION_OR_DICT_KEY] == 'mse'
    ):
        import peter_brnn_architecture

        model_object = peter_brnn_architecture.rnn_sw(
            nneur=64, lstm=True,
            activ_last='sigmoid', activ_surface='linear',
            add_dense=False, add_scalars_to_levels=True, simpler_inputs=True
        )
        model_object.load_weights(hdf5_file_name)
        return model_object

    dense_architecture_dict = metadata_dict[DENSE_ARCHITECTURE_KEY]

    if dense_architecture_dict is not None:
        import dense_net_architecture

        for this_key in [
                dense_net_architecture.VECTOR_LOSS_FUNCTION_KEY,
                dense_net_architecture.SCALAR_LOSS_FUNCTION_KEY
        ]:
            dense_architecture_dict[this_key] = eval(
                dense_architecture_dict[this_key]
            )

        model_object = dense_net_architecture.create_model(
            dense_architecture_dict
        )
        model_object.load_weights(hdf5_file_name)
        return model_object

    cnn_architecture_dict = metadata_dict[CNN_ARCHITECTURE_KEY]

    if cnn_architecture_dict is not None:
        import cnn_architecture

        for this_key in [
                cnn_architecture.VECTOR_LOSS_FUNCTION_KEY,
                cnn_architecture.SCALAR_LOSS_FUNCTION_KEY
        ]:
            cnn_architecture_dict[this_key] = eval(
                cnn_architecture_dict[this_key]
            )

        model_object = cnn_architecture.create_model(cnn_architecture_dict)
        model_object.load_weights(hdf5_file_name)
        return model_object

    bnn_architecture_dict = metadata_dict[BNN_ARCHITECTURE_KEY]

    if bnn_architecture_dict is not None:
        import u_net_pp_architecture_bayesian

        for this_key in [
                u_net_pp_architecture_bayesian.VECTOR_LOSS_FUNCTION_KEY,
                u_net_pp_architecture_bayesian.SCALAR_LOSS_FUNCTION_KEY
        ]:
            bnn_architecture_dict[this_key] = eval(
                bnn_architecture_dict[this_key]
            )

        model_object = u_net_pp_architecture_bayesian.create_bayesian_model(
            bnn_architecture_dict
        )
        model_object.load_weights(hdf5_file_name)
        return model_object

    u_net_architecture_dict = metadata_dict[U_NET_ARCHITECTURE_KEY]

    if u_net_architecture_dict is not None:
        import u_net_architecture

        for this_key in [
                u_net_architecture.VECTOR_LOSS_FUNCTION_KEY,
                u_net_architecture.SCALAR_LOSS_FUNCTION_KEY
        ]:
            u_net_architecture_dict[this_key] = eval(
                u_net_architecture_dict[this_key]
            )

        for this_key in [u_net_architecture.OPTIMIZER_FUNCTION_KEY]:
            if this_key not in u_net_architecture_dict:
                continue

            u_net_architecture_dict[this_key] = eval(
                u_net_architecture_dict[this_key]
            )

        model_object = u_net_architecture.create_model(u_net_architecture_dict)
        model_object.load_weights(hdf5_file_name)
        return model_object

    u_net_plusplus_architecture_dict = metadata_dict[U_NET_PP_ARCHITECTURE_KEY]
    joined_output_layer = (
        metadata_dict[TRAINING_OPTIONS_KEY][JOINED_OUTPUT_LAYER_KEY]
    )

    if u_net_plusplus_architecture_dict is not None:
        import u_net_pp_architecture

        if joined_output_layer:
            for this_key in [u_net_pp_architecture.JOINED_LOSS_FUNCTION_KEY]:
                u_net_plusplus_architecture_dict[this_key] = eval(
                    u_net_plusplus_architecture_dict[this_key]
                )

            model_object = u_net_pp_architecture.create_model_1output_layer(
                u_net_plusplus_architecture_dict
            )
        else:
            for this_key in [
                    u_net_pp_architecture.VECTOR_LOSS_FUNCTION_KEY,
                    u_net_pp_architecture.SCALAR_LOSS_FUNCTION_KEY
            ]:
                u_net_plusplus_architecture_dict[this_key] = eval(
                    u_net_plusplus_architecture_dict[this_key]
                )

            for this_key in [u_net_pp_architecture.OPTIMIZER_FUNCTION_KEY]:
                if this_key not in u_net_plusplus_architecture_dict:
                    continue

                u_net_plusplus_architecture_dict[this_key] = eval(
                    u_net_plusplus_architecture_dict[this_key]
                )

            model_object = u_net_pp_architecture.create_model(
                u_net_plusplus_architecture_dict
            )

        model_object.load_weights(hdf5_file_name)
        return model_object

    u_net_3plus_architecture_dict = metadata_dict[U_NET_PPP_ARCHITECTURE_KEY]
    assert u_net_3plus_architecture_dict is not None

    import u_net_ppp_architecture

    if joined_output_layer:
        for this_key in [u_net_ppp_architecture.JOINED_LOSS_FUNCTION_KEY]:
            u_net_3plus_architecture_dict[this_key] = eval(
                u_net_3plus_architecture_dict[this_key]
            )

        model_object = u_net_ppp_architecture.create_model_1output_layer(
            u_net_3plus_architecture_dict
        )
    else:
        for this_key in [
                u_net_ppp_architecture.VECTOR_LOSS_FUNCTION_KEY,
                u_net_ppp_architecture.SCALAR_LOSS_FUNCTION_KEY
        ]:
            u_net_3plus_architecture_dict[this_key] = eval(
                u_net_3plus_architecture_dict[this_key]
            )

        model_object = u_net_ppp_architecture.create_model(
            u_net_3plus_architecture_dict
        )

    model_object.load_weights(hdf5_file_name)
    return model_object


def find_metafile(model_dir_name, raise_error_if_missing=True):
    """Finds metafile for neural net.

    :param model_dir_name: Name of model directory.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: metafile_name: Path to metafile.
    """

    error_checking.assert_is_string(model_dir_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    metafile_name = '{0:s}/model_metadata.dill'.format(model_dir_name)

    if raise_error_if_missing and not os.path.isfile(metafile_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            metafile_name
        )
        raise ValueError(error_string)

    return metafile_name


def read_metafile(dill_file_name):
    """Reads metadata for neural net from Dill file.

    :param dill_file_name: Path to input file.
    :return: metadata_dict: Dictionary with the following keys.
    metadata_dict['num_epochs']: See doc for `train_model`.
    metadata_dict['num_training_batches_per_epoch']: Same.
    metadata_dict['training_option_dict']: Same.
    metadata_dict['num_validation_batches_per_epoch']: Same.
    metadata_dict['validation_option_dict']: Same.
    metadata_dict['loss_function_or_dict']: Same.
    metadata_dict['dense_architecture_dict']: Same.
    metadata_dict['cnn_architecture_dict']: Same.
    metadata_dict['bnn_architecture_dict']: Same.
    metadata_dict['u_net_architecture_dict']: Same.
    metadata_dict['u_net_plusplus_architecture_dict']: Same.
    metadata_dict['u_net_3plus_architecture_dict']: Same.

    :raises: ValueError: if any expected key is not found in dictionary.
    """

    error_checking.assert_file_exists(dill_file_name)

    dill_file_handle = open(dill_file_name, 'rb')
    metadata_dict = dill.load(dill_file_handle)
    dill_file_handle.close()

    if EARLY_STOPPING_PATIENCE_KEY not in metadata_dict:
        metadata_dict[EARLY_STOPPING_PATIENCE_KEY] = 200

    t = metadata_dict[TRAINING_OPTIONS_KEY]
    v = metadata_dict[VALIDATION_OPTIONS_KEY]

    if MIN_FLUX_FOR_MASK_KEY not in metadata_dict[TRAINING_OPTIONS_KEY]:
        t[NORMALIZATION_FILE_FOR_MASK_KEY] = None
        t[MIN_HEATING_RATE_FOR_MASK_KEY] = None
        t[MIN_FLUX_FOR_MASK_KEY] = None

        v[NORMALIZATION_FILE_FOR_MASK_KEY] = None
        v[MIN_HEATING_RATE_FOR_MASK_KEY] = None
        v[MIN_FLUX_FOR_MASK_KEY] = None

    if NORMALIZE_PREDICTORS_KEY not in metadata_dict[TRAINING_OPTIONS_KEY]:
        t[NORMALIZE_PREDICTORS_KEY] = (
            t['predictor_norm_type_string'] is not None
        )
        t[NORMALIZE_SCALAR_TARGETS_KEY] = (
            t['scalar_target_norm_type_string'] is not None
        )
        t[NORMALIZE_VECTOR_TARGETS_KEY] = (
            t['vector_target_norm_type_string'] is not None
        )

        v[NORMALIZE_PREDICTORS_KEY] = t[NORMALIZE_PREDICTORS_KEY]
        v[NORMALIZE_SCALAR_TARGETS_KEY] = t[NORMALIZE_SCALAR_TARGETS_KEY]
        v[NORMALIZE_VECTOR_TARGETS_KEY] = t[NORMALIZE_VECTOR_TARGETS_KEY]

    if JOINED_OUTPUT_LAYER_KEY not in metadata_dict[TRAINING_OPTIONS_KEY]:
        t[JOINED_OUTPUT_LAYER_KEY] = False
        v[JOINED_OUTPUT_LAYER_KEY] = False

    metadata_dict[TRAINING_OPTIONS_KEY] = t
    metadata_dict[VALIDATION_OPTIONS_KEY] = v

    if DENSE_ARCHITECTURE_KEY not in metadata_dict:
        metadata_dict[DENSE_ARCHITECTURE_KEY] = None
    if CNN_ARCHITECTURE_KEY not in metadata_dict:
        metadata_dict[CNN_ARCHITECTURE_KEY] = None
    if BNN_ARCHITECTURE_KEY not in metadata_dict:
        metadata_dict[BNN_ARCHITECTURE_KEY] = None
    if U_NET_PP_ARCHITECTURE_KEY not in metadata_dict:
        metadata_dict[U_NET_PP_ARCHITECTURE_KEY] = None
    if U_NET_ARCHITECTURE_KEY not in metadata_dict:
        metadata_dict[U_NET_ARCHITECTURE_KEY] = None
    if U_NET_PPP_ARCHITECTURE_KEY not in metadata_dict:
        metadata_dict[U_NET_PPP_ARCHITECTURE_KEY] = None

    missing_keys = list(set(METADATA_KEYS) - set(metadata_dict.keys()))
    if len(missing_keys) == 0:
        return metadata_dict

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), dill_file_name)

    raise ValueError(error_string)


def apply_model(
        model_object, predictor_matrix_or_list, num_examples_per_batch,
        use_dropout=False, verbose=False):
    """Applies trained neural net (of any kind) to new data.

    E = number of examples
    H = number of heights
    W = number of wavelengths
    T_v = number of vector target variables (channels)
    T_s = number of scalar target variables
    S = ensemble size

    :param model_object: Trained neural net (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param predictor_matrix_or_list: Same as output from `data_generator`,
        except this is a single numpy array or list -- rather than a single
        numpy array or dict.
    :param num_examples_per_batch: Batch size.
    :param use_dropout: Boolean flag.  If True, will keep dropout in all layers
        turned on.  Using dropout at inference time is called "Monte Carlo
        dropout".
    :param verbose: Boolean flag.  If True, will print progress messages.

    :return: prediction_dict: Dictionary with the following keys.
    prediction_dict['conv_output']: numpy array (E x H x W x T_v x S)
        of predicted heating rates.
    prediction_dict['dense_output']: numpy array (E x W x T_s x S) of
        predicted fluxes.
    """

    # Check input args.
    num_examples, num_examples_per_batch = _check_inference_args(
        predictor_matrix_or_list=predictor_matrix_or_list,
        num_examples_per_batch=num_examples_per_batch,
        verbose=verbose
    )

    error_checking.assert_is_boolean(use_dropout)
    if use_dropout:
        for layer_object in model_object.layers:
            if 'batch' in layer_object.name.lower():
                print('Layer "{0:s}" set to NON-TRAINABLE!'.format(
                    layer_object.name
                ))
                layer_object.trainable = False

    # Do actual stuff.
    vector_prediction_matrix = None
    scalar_prediction_matrix = None

    for i in range(0, num_examples, num_examples_per_batch):
        this_first_index = i
        this_last_index = min(
            [i + num_examples_per_batch - 1, num_examples - 1]
        )
        these_indices = numpy.linspace(
            this_first_index, this_last_index,
            num=this_last_index - this_first_index + 1, dtype=int
        )

        if verbose:
            print('Applying NN to examples {0:d}-{1:d} of {2:d}...'.format(
                this_first_index + 1, this_last_index + 1, num_examples
            ))

        if use_dropout:
            if isinstance(predictor_matrix_or_list, list):
                this_output = model_object(
                    [p[these_indices, ...] for p in predictor_matrix_or_list],
                    training=True
                )
            else:
                this_output = model_object(
                    predictor_matrix_or_list[these_indices, ...],
                    training=True
                )

            if isinstance(this_output, list):
                this_output = [a.numpy() for a in this_output]
            else:
                this_output = this_output.numpy()
        else:
            if isinstance(predictor_matrix_or_list, list):
                this_output = model_object.predict_on_batch(
                    [p[these_indices, ...] for p in predictor_matrix_or_list]
                )
            else:
                this_output = model_object.predict_on_batch(
                    predictor_matrix_or_list[these_indices, ...]
                )

        if not isinstance(this_output, list):
            this_output = [this_output]

        # Add ensemble dimension if necessary.
        if len(this_output[0].shape) == 4:
            this_output[0] = numpy.expand_dims(this_output[0], axis=-1)
        if len(this_output) > 1 and len(this_output[1].shape) == 3:
            this_output[1] = numpy.expand_dims(this_output[1], axis=-1)

        if vector_prediction_matrix is None:
            vector_prediction_matrix = numpy.full(
                (num_examples,) + this_output[0].shape[1:], numpy.nan
            )

            if len(this_output) > 1:
                scalar_prediction_matrix = numpy.full(
                    (num_examples,) + this_output[1].shape[1:], numpy.nan
                )

        vector_prediction_matrix[
            this_first_index:(this_last_index + 1), ...
        ] = this_output[0]

        if len(this_output) > 1:
            scalar_prediction_matrix[
                this_first_index:(this_last_index + 1), ...
            ] = this_output[1]

    if verbose:
        print('Have applied NN to all {0:d} examples!'.format(num_examples))

    if scalar_prediction_matrix is None:
        num_examples = vector_prediction_matrix.shape[0]
        num_wavelengths = vector_prediction_matrix.shape[2]
        ensemble_size = vector_prediction_matrix.shape[4]
        scalar_prediction_matrix = numpy.full(
            (num_examples, num_wavelengths, 0, ensemble_size), 0.
        )

    return {
        HEATING_RATE_TARGETS_KEY: vector_prediction_matrix,
        FLUX_TARGETS_KEY: scalar_prediction_matrix
    }
