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

# TODO(thunderhoser): This should become an input arg.
MAX_NUM_VALIDATION_EXAMPLES = int(5e5)
MAX_NUM_TRAINING_EXAMPLES = int(1e6)

PLATEAU_PATIENCE_EPOCHS = 10
DEFAULT_LEARNING_RATE_MULTIPLIER = 0.5
PLATEAU_COOLDOWN_EPOCHS = 0
EARLY_STOPPING_PATIENCE_EPOCHS = 200
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
UNIFORMIZE_FLAG_KEY = 'uniformize'
PREDICTOR_NORM_TYPE_KEY = 'predictor_norm_type_string'
PREDICTOR_MIN_NORM_VALUE_KEY = 'predictor_min_norm_value'
PREDICTOR_MAX_NORM_VALUE_KEY = 'predictor_max_norm_value'
VECTOR_TARGET_NORM_TYPE_KEY = 'vector_target_norm_type_string'
VECTOR_TARGET_MIN_VALUE_KEY = 'vector_target_min_norm_value'
VECTOR_TARGET_MAX_VALUE_KEY = 'vector_target_max_norm_value'
SCALAR_TARGET_NORM_TYPE_KEY = 'scalar_target_norm_type_string'
SCALAR_TARGET_MIN_VALUE_KEY = 'scalar_target_min_norm_value'
SCALAR_TARGET_MAX_VALUE_KEY = 'scalar_target_max_norm_value'
JOINED_OUTPUT_LAYER_KEY = 'joined_output_layer'
NUM_DEEP_SUPER_LAYERS_KEY = 'num_deep_supervision_layers'

DEFAULT_GENERATOR_OPTION_DICT = {
    SCALAR_PREDICTOR_NAMES_KEY: example_utils.ALL_SCALAR_PREDICTOR_NAMES,
    VECTOR_PREDICTOR_NAMES_KEY: example_utils.BASIC_VECTOR_PREDICTOR_NAMES,
    SCALAR_TARGET_NAMES_KEY: example_utils.ALL_SCALAR_TARGET_NAMES,
    VECTOR_TARGET_NAMES_KEY: example_utils.ALL_VECTOR_TARGET_NAMES,
    PREDICTOR_NORM_TYPE_KEY: normalization.Z_SCORE_NORM_STRING,
    PREDICTOR_MIN_NORM_VALUE_KEY: None,
    PREDICTOR_MAX_NORM_VALUE_KEY: None,
    VECTOR_TARGET_NORM_TYPE_KEY: normalization.MINMAX_NORM_STRING,
    VECTOR_TARGET_MIN_VALUE_KEY: 0.,
    VECTOR_TARGET_MAX_VALUE_KEY: 1.,
    SCALAR_TARGET_NORM_TYPE_KEY: normalization.MINMAX_NORM_STRING,
    SCALAR_TARGET_MIN_VALUE_KEY: 0.,
    SCALAR_TARGET_MAX_VALUE_KEY: 1.,
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
EARLY_STOPPING_KEY = 'do_early_stopping'
PLATEAU_LR_MUTIPLIER_KEY = 'plateau_lr_multiplier'
DENSE_ARCHITECTURE_KEY = 'dense_architecture_dict'
CNN_ARCHITECTURE_KEY = 'cnn_architecture_dict'
BNN_ARCHITECTURE_KEY = 'bnn_architecture_dict'
U_NET_ARCHITECTURE_KEY = 'u_net_architecture_dict'
U_NET_PP_ARCHITECTURE_KEY = 'u_net_plusplus_architecture_dict'
U_NET_PPP_ARCHITECTURE_KEY = 'u_net_3plus_architecture_dict'

METADATA_KEYS = [
    NUM_EPOCHS_KEY, NUM_TRAINING_BATCHES_KEY, TRAINING_OPTIONS_KEY,
    NUM_VALIDATION_BATCHES_KEY, VALIDATION_OPTIONS_KEY,
    LOSS_FUNCTION_OR_DICT_KEY, EARLY_STOPPING_KEY, PLATEAU_LR_MUTIPLIER_KEY,
    DENSE_ARCHITECTURE_KEY, CNN_ARCHITECTURE_KEY, BNN_ARCHITECTURE_KEY,
    U_NET_ARCHITECTURE_KEY, U_NET_PP_ARCHITECTURE_KEY,
    U_NET_PPP_ARCHITECTURE_KEY
]


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

    if option_dict[PREDICTOR_NORM_TYPE_KEY] is not None:
        error_checking.assert_is_string(option_dict[PREDICTOR_NORM_TYPE_KEY])
    if option_dict[VECTOR_TARGET_NORM_TYPE_KEY] is not None:
        error_checking.assert_is_string(
            option_dict[VECTOR_TARGET_NORM_TYPE_KEY]
        )
    if option_dict[SCALAR_TARGET_NORM_TYPE_KEY] is not None:
        error_checking.assert_is_string(
            option_dict[SCALAR_TARGET_NORM_TYPE_KEY]
        )

    return option_dict


def _check_inference_args(predictor_matrix, num_examples_per_batch, verbose):
    """Error-checks input arguments for inference.

    :param predictor_matrix: See doc for `apply_model`.
    :param num_examples_per_batch: Batch size.
    :param verbose: Boolean flag.  If True, will print progress messages during
        inference.
    :return: num_examples_per_batch: Batch size (may be different than input).
    """

    error_checking.assert_is_numpy_array_without_nan(predictor_matrix)
    num_examples = predictor_matrix.shape[0]

    if num_examples_per_batch is None:
        num_examples_per_batch = num_examples + 0
    else:
        error_checking.assert_is_integer(num_examples_per_batch)
        # error_checking.assert_is_geq(num_examples_per_batch, 100)
        error_checking.assert_is_geq(num_examples_per_batch, 1)

    num_examples_per_batch = min([num_examples_per_batch, num_examples])
    error_checking.assert_is_boolean(verbose)

    return num_examples_per_batch


def _read_file_for_generator(
        example_file_name, first_time_unix_sec, last_time_unix_sec, field_names,
        heights_m_agl, target_wavelengths_metres, normalization_file_name,
        uniformize, predictor_norm_type_string, predictor_min_norm_value,
        predictor_max_norm_value, vector_target_norm_type_string,
        vector_target_min_norm_value, vector_target_max_norm_value,
        scalar_target_norm_type_string, scalar_target_min_norm_value,
        scalar_target_max_norm_value):
    """Reads one file for generator.

    :param example_file_name: Path to input file (will be read by
        `example_io.read_file`).
    :param first_time_unix_sec: See doc for `data_generator`.
    :param last_time_unix_sec: Same.
    :param field_names: 1-D list of fields to keep.
    :param heights_m_agl: 1-D numpy array of heights to keep (metres above
        ground level).
    :param target_wavelengths_metres: 1-D numpy array of wavelengths to keep.
    :param normalization_file_name: See doc for `data_generator`.
    :param uniformize: Same.
    :param predictor_norm_type_string: Same.
    :param predictor_min_norm_value: Same.
    :param predictor_max_norm_value: Same.
    :param vector_target_norm_type_string: Same.
    :param vector_target_min_norm_value: Same.
    :param vector_target_max_norm_value: Same.
    :param scalar_target_norm_type_string: Same.
    :param scalar_target_min_norm_value: Same.
    :param scalar_target_max_norm_value: Same.
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
            example_io.UNIFORMIZE_FLAG_KEY: uniformize,
            example_io.PREDICTOR_NORM_TYPE_KEY: predictor_norm_type_string,
            example_io.PREDICTOR_MIN_VALUE_KEY: predictor_min_norm_value,
            example_io.PREDICTOR_MAX_VALUE_KEY: predictor_max_norm_value,
            example_io.VECTOR_TARGET_NORM_TYPE_KEY:
                vector_target_norm_type_string,
            example_io.VECTOR_TARGET_MIN_VALUE_KEY:
                vector_target_min_norm_value,
            example_io.VECTOR_TARGET_MAX_VALUE_KEY:
                vector_target_max_norm_value,
            example_io.SCALAR_TARGET_NORM_TYPE_KEY:
                scalar_target_norm_type_string,
            example_io.SCALAR_TARGET_MIN_VALUE_KEY:
                scalar_target_min_norm_value,
            example_io.SCALAR_TARGET_MAX_VALUE_KEY:
                scalar_target_max_norm_value
        }

        assert example_io.are_normalization_metadata_same(
            normalization_metadata_dict,
            example_dict[example_utils.NORMALIZATION_METADATA_KEY]
        )

        return example_dict

    if normalization_file_name is None:
        return example_dict

    print((
        'Reading training examples (for normalization) from: "{0:s}"...'
    ).format(
        normalization_file_name
    ))
    training_example_dict = example_io.read_file(normalization_file_name)
    training_example_dict = example_utils.subset_by_height(
        example_dict=training_example_dict, heights_m_agl=heights_m_agl
    )
    training_example_dict = example_utils.subset_by_wavelength(
        example_dict=training_example_dict,
        target_wavelengths_metres=target_wavelengths_metres
    )

    if predictor_norm_type_string is not None:
        print('Applying {0:s} normalization to predictors...'.format(
            predictor_norm_type_string.upper()
        ))
        # example_dict = normalization.normalize_data(
        #     new_example_dict=example_dict,
        #     training_example_dict=training_example_dict,
        #     normalization_type_string=predictor_norm_type_string,
        #     uniformize=uniformize,
        #     min_normalized_value=predictor_min_norm_value,
        #     max_normalized_value=predictor_max_norm_value,
        #     separate_heights=True, apply_to_predictors=True,
        #     apply_to_vector_targets=False, apply_to_scalar_targets=False
        # )

        if example_utils.AEROSOL_EXTINCTION_NAME in field_names:
            pw_linear_model_file_name = (
                '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4rt_project/'
                'gfs_data/examples_with_correct_vertical_coords/shortwave/'
                'training/piecewise_linear_models_for_uniformization.nc'
            )
        else:
            pw_linear_model_file_name = (
                '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4rt_project/'
                'gfs_data/examples_with_correct_vertical_coords/longwave/'
                'training/piecewise_linear_models_for_uniformization.nc'
            )

        print((
            'Reading piecewise-linear model for uniformization from: "{0:s}"...'
        ).format(
            pw_linear_model_file_name
        ))

        pw_linear_model_table_xarray = (
            normalization.read_piecewise_linear_models_for_unif(
                pw_linear_model_file_name
            )
        )

        example_dict = (
            normalization.normalize_data_with_pw_linear_models_for_unif(
                new_example_dict=example_dict,
                pw_linear_model_table_xarray=pw_linear_model_table_xarray
            )
        )

    if vector_target_norm_type_string is not None:
        print('Applying {0:s} normalization to vector targets...'.format(
            vector_target_norm_type_string.upper()
        ))
        example_dict = normalization.normalize_data(
            new_example_dict=example_dict,
            training_example_dict=training_example_dict,
            normalization_type_string=vector_target_norm_type_string,
            uniformize=uniformize,
            min_normalized_value=vector_target_min_norm_value,
            max_normalized_value=vector_target_max_norm_value,
            separate_heights=True, apply_to_predictors=False,
            apply_to_vector_targets=True, apply_to_scalar_targets=False
        )

    if scalar_target_norm_type_string is not None:
        print('Applying {0:s} normalization to scalar targets...'.format(
            scalar_target_norm_type_string.upper()
        ))
        example_dict = normalization.normalize_data(
            new_example_dict=example_dict,
            training_example_dict=training_example_dict,
            normalization_type_string=scalar_target_norm_type_string,
            uniformize=uniformize,
            min_normalized_value=scalar_target_min_norm_value,
            max_normalized_value=scalar_target_max_norm_value,
            separate_heights=True, apply_to_predictors=False,
            apply_to_vector_targets=False, apply_to_scalar_targets=True
        )

    return example_dict


def _write_metafile(
        dill_file_name, num_epochs, num_training_batches_per_epoch,
        training_option_dict, num_validation_batches_per_epoch,
        validation_option_dict, loss_function_or_dict,
        do_early_stopping, plateau_lr_multiplier, dense_architecture_dict,
        cnn_architecture_dict, bnn_architecture_dict, u_net_architecture_dict,
        u_net_plusplus_architecture_dict, u_net_3plus_architecture_dict):
    """Writes metadata to Dill file.

    :param dill_file_name: Path to output file.
    :param num_epochs: See doc for `train_model`.
    :param num_training_batches_per_epoch: Same.
    :param training_option_dict: Same.
    :param num_validation_batches_per_epoch: Same.
    :param validation_option_dict: Same.
    :param loss_function_or_dict: Same.
    :param do_early_stopping: Same.
    :param plateau_lr_multiplier: Same.
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
        EARLY_STOPPING_KEY: do_early_stopping,
        PLATEAU_LR_MUTIPLIER_KEY: plateau_lr_multiplier,
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

    :param example_dict: Dictionary of examples (in the format returned by
        `example_io.read_file`).
    :return: target_matrices: Same as output from `data_generator`.
    """

    vector_target_matrix = (
        example_dict[example_utils.VECTOR_TARGET_VALS_KEY]
    )
    scalar_target_matrix = (
        example_dict[example_utils.SCALAR_TARGET_VALS_KEY]
    )

    if scalar_target_matrix.size == 0:
        return [vector_target_matrix]

    return [vector_target_matrix, scalar_target_matrix]


def targets_numpy_to_dict(target_matrices):
    """Converts targets from numpy array to dictionary.

    This method is the inverse of `targets_dict_to_numpy`.

    :param target_matrices: List created by `targets_dict_to_numpy`.
    :return: example_dict: Dictionary with the following keys.  See doc for
        `example_io.read_file` for details on each key.
    example_dict['scalar_target_matrix']
    example_dict['vector_target_matrix']
    """

    vector_target_matrix = target_matrices[0]
    error_checking.assert_is_numpy_array_without_nan(vector_target_matrix)
    error_checking.assert_is_numpy_array(vector_target_matrix, num_dimensions=4)

    num_examples = vector_target_matrix.shape[0]
    num_wavelengths = vector_target_matrix.shape[2]

    if len(target_matrices) == 1:
        scalar_target_matrix = numpy.full(
            (num_examples, num_wavelengths, 0), 0.
        )
    else:
        scalar_target_matrix = target_matrices[1]

    error_checking.assert_is_numpy_array_without_nan(scalar_target_matrix)
    error_checking.assert_is_numpy_array(
        scalar_target_matrix, num_dimensions=3
    )

    return {
        example_utils.SCALAR_TARGET_VALS_KEY: scalar_target_matrix,
        example_utils.VECTOR_TARGET_VALS_KEY: vector_target_matrix
    }


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
    option_dict['normalization_file_name']: File with training examples to use
        for normalization (will be read by `example_io.read_file`).
    option_dict['uniformize']: Boolean flag, used only for z-score
        normalization.  If True, will convert each variable to uniform
        distribution and then z-scores; if False, will convert directly to
        z-scores.
    option_dict['predictor_norm_type_string']: Normalization type for predictors
        (must be accepted by `normalization.check_normalization_type`).  If you
        do not want to normalize predictors, make this None.
    option_dict['predictor_min_norm_value']: Minimum normalized value for
        predictors (used only if normalization type is min-max).
    option_dict['predictor_max_norm_value']: Same but max value.
    option_dict['vector_target_norm_type_string']: Normalization type for vector
        targets (must be accepted by `normalization.check_normalization_type`).
        If you do not want to normalize vector targets, make this None.
    option_dict['vector_target_min_norm_value']: Minimum normalized value for
        vector targets (used only if normalization type is min-max).
    option_dict['vector_target_max_norm_value']: Same but max value.
    option_dict['scalar_target_norm_type_string']: Same as
        "vector_target_norm_type_string" but for scalar targets.
    option_dict['scalar_target_min_norm_value']: Same as
        "vector_target_min_norm_value" but for scalar targets.
    option_dict['scalar_target_max_norm_value']: Same as
        "vector_target_max_norm_value" but for scalar targets.
    option_dict['joined_output_layer']: Boolean flag.  If True, heating rates
        and fluxes are all joined into one output layer.
    option_dict['num_deep_supervision_layers']: Number of deep-supervision
        layers.

    :param for_inference: Boolean flag.  If True, generator is being used for
        inference stage (applying trained model to new data).  If False,
        generator is being used for training or monitoring (on-the-fly
        validation).

    :return: predictor_matrix: numpy array of predictor values.  If net type is
        dense, the array will be E x P.  Otherwise, will be E x H x P.
    :return: target_array: If net type is dense, this is a numpy array (E x T)
        of target values.  Otherwise, a list with two elements:

    target_array[0] = vector_target_matrix: numpy array (E x H x T_v) of target
        values.
    target_array[1] = scalar_target_matrix: numpy array (E x T_s) of target
        values.

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
    uniformize = option_dict[UNIFORMIZE_FLAG_KEY]
    predictor_norm_type_string = option_dict[PREDICTOR_NORM_TYPE_KEY]
    predictor_min_norm_value = option_dict[PREDICTOR_MIN_NORM_VALUE_KEY]
    predictor_max_norm_value = option_dict[PREDICTOR_MAX_NORM_VALUE_KEY]
    vector_target_norm_type_string = option_dict[VECTOR_TARGET_NORM_TYPE_KEY]
    vector_target_min_norm_value = option_dict[VECTOR_TARGET_MIN_VALUE_KEY]
    vector_target_max_norm_value = option_dict[VECTOR_TARGET_MAX_VALUE_KEY]
    scalar_target_norm_type_string = option_dict[SCALAR_TARGET_NORM_TYPE_KEY]
    scalar_target_min_norm_value = option_dict[SCALAR_TARGET_MIN_VALUE_KEY]
    scalar_target_max_norm_value = option_dict[SCALAR_TARGET_MAX_VALUE_KEY]
    joined_output_layer = option_dict[JOINED_OUTPUT_LAYER_KEY]
    num_deep_supervision_layers = option_dict[NUM_DEEP_SUPER_LAYERS_KEY]

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
        uniformize=uniformize,
        predictor_norm_type_string=predictor_norm_type_string,
        predictor_min_norm_value=predictor_min_norm_value,
        predictor_max_norm_value=predictor_max_norm_value,
        vector_target_norm_type_string=vector_target_norm_type_string,
        vector_target_min_norm_value=vector_target_min_norm_value,
        vector_target_max_norm_value=vector_target_max_norm_value,
        scalar_target_norm_type_string=scalar_target_norm_type_string,
        scalar_target_min_norm_value=scalar_target_min_norm_value,
        scalar_target_max_norm_value=scalar_target_max_norm_value
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
                    uniformize=uniformize,
                    predictor_norm_type_string=predictor_norm_type_string,
                    predictor_min_norm_value=predictor_min_norm_value,
                    predictor_max_norm_value=predictor_max_norm_value,
                    vector_target_norm_type_string=
                    vector_target_norm_type_string,
                    vector_target_min_norm_value=vector_target_min_norm_value,
                    vector_target_max_norm_value=vector_target_max_norm_value,
                    scalar_target_norm_type_string=
                    scalar_target_norm_type_string,
                    scalar_target_min_norm_value=scalar_target_min_norm_value,
                    scalar_target_max_norm_value=scalar_target_max_norm_value
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
            this_target_list = targets_dict_to_numpy(this_example_dict)

            this_vector_target_matrix = this_target_list[0]
            if len(this_target_list) == 1:
                this_scalar_target_matrix = None
            else:
                this_scalar_target_matrix = this_target_list[1]

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

        if numpy.max(numpy.absolute(predictor_matrix)) > 2**16 - 1:
            predictor_matrix = predictor_matrix.astype('float32')
        else:
            predictor_matrix = predictor_matrix.astype('float16')

        if joined_output_layer:
            target_array = vector_target_matrix[..., 0].astype('float32')

            if scalar_target_matrix is not None:
                scalar_target_matrix = numpy.swapaxes(scalar_target_matrix, 1, 2)
                target_array = numpy.concatenate(
                    [target_array, scalar_target_matrix.astype('float32')],
                    axis=-2
                )
        else:
            target_array = [vector_target_matrix.astype('float32')]
            if scalar_target_matrix is not None:
                target_array.append(scalar_target_matrix.astype('float32'))

        for _ in range(num_deep_supervision_layers):
            target_array.append(target_array[0])

        if for_inference:
            if len(target_array) > 1:
                target_dict = {
                    'conv_output': target_array[0],
                    'dense_output': target_array[1]
                }
                yield predictor_matrix, target_dict, example_id_strings
            else:
                yield predictor_matrix, target_array[0], example_id_strings
        else:
            if len(target_array) > 1:
                target_dict = {
                    'conv_output': target_array[0],
                    'dense_output': target_array[1]
                }
                yield predictor_matrix, target_dict
            else:
                yield predictor_matrix, target_array[0]


def create_data(option_dict):
    """Creates data for any kind of neural net.

    This method is the same as `data_generator`, except that it returns all the
    data at once, rather than generating batches on the fly.

    :param option_dict: See doc for `data_generator`.
    :return: predictor_matrix: Same.
    :return: target_array: Same.
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
    uniformize = option_dict[UNIFORMIZE_FLAG_KEY]
    predictor_norm_type_string = option_dict[PREDICTOR_NORM_TYPE_KEY]
    predictor_min_norm_value = option_dict[PREDICTOR_MIN_NORM_VALUE_KEY]
    predictor_max_norm_value = option_dict[PREDICTOR_MAX_NORM_VALUE_KEY]
    vector_target_norm_type_string = option_dict[VECTOR_TARGET_NORM_TYPE_KEY]
    vector_target_min_norm_value = option_dict[VECTOR_TARGET_MIN_VALUE_KEY]
    vector_target_max_norm_value = option_dict[VECTOR_TARGET_MAX_VALUE_KEY]
    scalar_target_norm_type_string = option_dict[SCALAR_TARGET_NORM_TYPE_KEY]
    scalar_target_min_norm_value = option_dict[SCALAR_TARGET_MIN_VALUE_KEY]
    scalar_target_max_norm_value = option_dict[SCALAR_TARGET_MAX_VALUE_KEY]
    joined_output_layer = option_dict[JOINED_OUTPUT_LAYER_KEY]
    num_deep_supervision_layers = option_dict[NUM_DEEP_SUPER_LAYERS_KEY]

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
            uniformize=uniformize,
            predictor_norm_type_string=predictor_norm_type_string,
            predictor_min_norm_value=predictor_min_norm_value,
            predictor_max_norm_value=predictor_max_norm_value,
            vector_target_norm_type_string=vector_target_norm_type_string,
            vector_target_min_norm_value=vector_target_min_norm_value,
            vector_target_max_norm_value=vector_target_max_norm_value,
            scalar_target_norm_type_string=scalar_target_norm_type_string,
            scalar_target_min_norm_value=scalar_target_min_norm_value,
            scalar_target_max_norm_value=scalar_target_max_norm_value
        )

        example_dicts.append(this_example_dict)

    example_dict = example_utils.concat_examples(example_dicts)
    predictor_matrix = predictors_dict_to_numpy(example_dict)[0]

    if numpy.max(numpy.absolute(predictor_matrix)) > 2**16 - 1:
        predictor_matrix = predictor_matrix.astype('float32')
    else:
        predictor_matrix = predictor_matrix.astype('float16')

    prelim_target_list = targets_dict_to_numpy(example_dict)
    vector_target_matrix = prelim_target_list[0]

    if joined_output_layer:
        target_array = vector_target_matrix[..., 0].astype('float32')

        if len(prelim_target_list) > 1:
            scalar_target_matrix = numpy.swapaxes(prelim_target_list[1], 1, 2)
            target_array = numpy.concatenate(
                [target_array, scalar_target_matrix.astype('float32')],
                axis=-2
            )
    else:
        target_array = [vector_target_matrix.astype('float32')]

        if len(prelim_target_list) > 1:
            scalar_target_matrix = prelim_target_list[1]
            target_array.append(scalar_target_matrix.astype('float32'))

    for _ in range(num_deep_supervision_layers):
        target_array.append(target_array[0])

    return (
        predictor_matrix,
        target_array,
        example_dict[example_utils.EXAMPLE_IDS_KEY]
    )


def create_data_specific_examples(option_dict, example_id_strings):
    """Creates data for specific examples.

    This method is the same as `create_data`, except that it creates specific
    examples.  Also, note that this method should be run only in inference mode
    (not in training mode).

    :param option_dict: See doc for `data_generator`.
    :param example_id_strings: 1-D list of example IDs.
    :return: predictor_matrix: See doc for `data_generator`.
    :return: target_array: Same.
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
    uniformize = option_dict[UNIFORMIZE_FLAG_KEY]
    predictor_norm_type_string = option_dict[PREDICTOR_NORM_TYPE_KEY]
    predictor_min_norm_value = option_dict[PREDICTOR_MIN_NORM_VALUE_KEY]
    predictor_max_norm_value = option_dict[PREDICTOR_MAX_NORM_VALUE_KEY]
    vector_target_norm_type_string = option_dict[VECTOR_TARGET_NORM_TYPE_KEY]
    vector_target_min_norm_value = option_dict[VECTOR_TARGET_MIN_VALUE_KEY]
    vector_target_max_norm_value = option_dict[VECTOR_TARGET_MAX_VALUE_KEY]
    scalar_target_norm_type_string = option_dict[SCALAR_TARGET_NORM_TYPE_KEY]
    scalar_target_min_norm_value = option_dict[SCALAR_TARGET_MIN_VALUE_KEY]
    scalar_target_max_norm_value = option_dict[SCALAR_TARGET_MAX_VALUE_KEY]

    example_file_names = example_io.find_many_files(
        directory_name=example_dir_name,
        first_time_unix_sec=numpy.min(example_times_unix_sec),
        last_time_unix_sec=numpy.max(example_times_unix_sec),
        raise_error_if_any_missing=False
    )

    num_examples = len(example_id_strings)
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
            uniformize=uniformize,
            predictor_norm_type_string=predictor_norm_type_string,
            predictor_min_norm_value=predictor_min_norm_value,
            predictor_max_norm_value=predictor_max_norm_value,
            vector_target_norm_type_string=vector_target_norm_type_string,
            vector_target_min_norm_value=vector_target_min_norm_value,
            vector_target_max_norm_value=vector_target_max_norm_value,
            scalar_target_norm_type_string=scalar_target_norm_type_string,
            scalar_target_min_norm_value=scalar_target_min_norm_value,
            scalar_target_max_norm_value=scalar_target_max_norm_value
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
        prelim_target_list = targets_dict_to_numpy(this_example_dict)

        if predictor_matrix is None:
            predictor_matrix = numpy.full(
                (num_examples,) + this_predictor_matrix.shape[1:],
                numpy.nan
            )
            vector_target_matrix = numpy.full(
                (num_examples,) + prelim_target_list[0].shape[1:],
                numpy.nan
            )

            if len(prelim_target_list) > 1:
                scalar_target_matrix = numpy.full(
                    (num_examples,) + prelim_target_list[1].shape[1:],
                    numpy.nan
                )

        predictor_matrix[missing_example_indices, ...] = this_predictor_matrix
        vector_target_matrix[missing_example_indices, ...] = (
            prelim_target_list[0]
        )
        if len(prelim_target_list) > 1:
            scalar_target_matrix[missing_example_indices, :] = (
                prelim_target_list[1]
            )

        found_example_flags[missing_example_indices] = True

    assert numpy.all(found_example_flags)

    if numpy.max(numpy.absolute(predictor_matrix)) > 2**16 - 1:
        predictor_matrix = predictor_matrix.astype('float32')
    else:
        predictor_matrix = predictor_matrix.astype('float16')

    target_array = [vector_target_matrix.astype('float32')]
    if scalar_target_matrix is not None:
        target_array.append(scalar_target_matrix.astype('float32'))

    return predictor_matrix, target_array


def train_model_with_generator(
        model_object, output_dir_name, num_epochs,
        num_training_batches_per_epoch, training_option_dict,
        validation_option_dict, loss_function_or_dict,
        use_generator_for_validn, num_validation_batches_per_epoch,
        do_early_stopping, plateau_lr_multiplier, dense_architecture_dict,
        cnn_architecture_dict, bnn_architecture_dict, u_net_architecture_dict,
        u_net_plusplus_architecture_dict, u_net_3plus_architecture_dict):
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
    :param do_early_stopping: Boolean flag.  If True, will stop training early
        if validation loss has not improved over last several epochs (see
        constants at top of file for what exactly this means).
    :param plateau_lr_multiplier: Multiplier for learning rate.  Learning
        rate will be multiplied by this factor upon plateau in validation
        performance.
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
    error_checking.assert_is_boolean(do_early_stopping)

    if do_early_stopping:
        error_checking.assert_is_greater(plateau_lr_multiplier, 0.)
        error_checking.assert_is_less_than(plateau_lr_multiplier, 1.)

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

    if do_early_stopping:
        early_stopping_object = keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=LOSS_PATIENCE,
            patience=EARLY_STOPPING_PATIENCE_EPOCHS, verbose=1, mode='min'
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
        do_early_stopping=do_early_stopping,
        plateau_lr_multiplier=plateau_lr_multiplier,
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
        validation_predictor_matrix, validation_target_array = create_data(
            validation_option_dict
        )[:2]

        validation_data_arg = (
            validation_predictor_matrix, validation_target_array
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
        do_early_stopping, num_training_batches_per_epoch,
        num_validation_batches_per_epoch, plateau_lr_multiplier,
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
    :param do_early_stopping: Same.
    :param num_training_batches_per_epoch: Number of training batches per epoch.
        If None, each training example will be used once per epoch.
    :param num_validation_batches_per_epoch: Number of validation batches per
        epoch.  If None, each validation example will be used once per epoch.
    :param plateau_lr_multiplier: See doc for `train_model_with_generator`.
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
    error_checking.assert_is_boolean(do_early_stopping)

    if do_early_stopping:
        error_checking.assert_is_greater(plateau_lr_multiplier, 0.)
        error_checking.assert_is_less_than(plateau_lr_multiplier, 1.)

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

    if do_early_stopping:
        early_stopping_object = keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=LOSS_PATIENCE,
            patience=EARLY_STOPPING_PATIENCE_EPOCHS, verbose=1, mode='min'
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
        do_early_stopping=do_early_stopping,
        plateau_lr_multiplier=plateau_lr_multiplier,
        dense_architecture_dict=dense_architecture_dict,
        cnn_architecture_dict=cnn_architecture_dict,
        bnn_architecture_dict=bnn_architecture_dict,
        u_net_architecture_dict=u_net_architecture_dict,
        u_net_plusplus_architecture_dict=u_net_plusplus_architecture_dict,
        u_net_3plus_architecture_dict=u_net_3plus_architecture_dict
    )

    training_predictor_matrix, training_target_array = create_data(
        training_option_dict
    )[:2]

    validation_predictor_matrix, validation_target_array = create_data(
        validation_option_dict
    )[:2]

    # TODO(thunderhoser): HACK to deal with out-of-memory errors.
    num_validation_examples = validation_predictor_matrix.shape[0]
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

        validation_predictor_matrix = validation_predictor_matrix[
            random_indices, ...
        ]

        if isinstance(validation_target_array, list):
            for k in range(len(validation_target_array)):
                validation_target_array[k] = validation_target_array[k][
                    random_indices, ...
                ]
        else:
            validation_target_array = validation_target_array[
                random_indices, ...
            ]

    num_training_examples = training_predictor_matrix.shape[0]
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

        training_predictor_matrix = training_predictor_matrix[
            random_indices, ...
        ]

        if isinstance(training_target_array, list):
            for k in range(len(training_target_array)):
                training_target_array[k] = training_target_array[k][
                    random_indices, ...
                ]
        else:
            training_target_array = training_target_array[
                random_indices, ...
            ]

    model_object.fit(
        x=training_predictor_matrix, y=training_target_array,
        batch_size=training_option_dict[BATCH_SIZE_KEY],
        epochs=num_epochs, steps_per_epoch=num_training_batches_per_epoch,
        shuffle=True, verbose=1, callbacks=list_of_callback_objects,
        validation_data=(validation_predictor_matrix, validation_target_array),
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

    t = metadata_dict[TRAINING_OPTIONS_KEY]
    v = metadata_dict[VALIDATION_OPTIONS_KEY]

    if VECTOR_TARGET_NORM_TYPE_KEY not in metadata_dict[TRAINING_OPTIONS_KEY]:
        target_norm_type_string = t['target_norm_type_string']
        target_min_norm_value = t['target_min_norm_value']
        target_max_norm_value = t['target_max_norm_value']

        t[VECTOR_TARGET_NORM_TYPE_KEY] = target_norm_type_string
        t[VECTOR_TARGET_MIN_VALUE_KEY] = target_min_norm_value
        t[VECTOR_TARGET_MAX_VALUE_KEY] = target_max_norm_value
        t[SCALAR_TARGET_NORM_TYPE_KEY] = target_norm_type_string
        t[SCALAR_TARGET_MIN_VALUE_KEY] = target_min_norm_value
        t[SCALAR_TARGET_MAX_VALUE_KEY] = target_max_norm_value

        v[VECTOR_TARGET_NORM_TYPE_KEY] = target_norm_type_string
        v[VECTOR_TARGET_MIN_VALUE_KEY] = target_min_norm_value
        v[VECTOR_TARGET_MAX_VALUE_KEY] = target_max_norm_value
        v[SCALAR_TARGET_NORM_TYPE_KEY] = target_norm_type_string
        v[SCALAR_TARGET_MIN_VALUE_KEY] = target_min_norm_value
        v[SCALAR_TARGET_MAX_VALUE_KEY] = target_max_norm_value

    if TARGET_WAVELENGTHS_KEY not in metadata_dict[TRAINING_OPTIONS_KEY]:
        t[TARGET_WAVELENGTHS_KEY] = numpy.array([
            example_utils.DUMMY_BROADBAND_WAVELENGTH_METRES
        ])
        v[TARGET_WAVELENGTHS_KEY] = numpy.array([
            example_utils.DUMMY_BROADBAND_WAVELENGTH_METRES
        ])

    if UNIFORMIZE_FLAG_KEY not in metadata_dict[TRAINING_OPTIONS_KEY]:
        t[UNIFORMIZE_FLAG_KEY] = True
        v[UNIFORMIZE_FLAG_KEY] = True

    if JOINED_OUTPUT_LAYER_KEY not in metadata_dict[TRAINING_OPTIONS_KEY]:
        t[JOINED_OUTPUT_LAYER_KEY] = False
        v[JOINED_OUTPUT_LAYER_KEY] = False

    if NUM_DEEP_SUPER_LAYERS_KEY not in metadata_dict[TRAINING_OPTIONS_KEY]:
        t[NUM_DEEP_SUPER_LAYERS_KEY] = 0
        v[NUM_DEEP_SUPER_LAYERS_KEY] = 0

    metadata_dict[TRAINING_OPTIONS_KEY] = t
    metadata_dict[VALIDATION_OPTIONS_KEY] = v

    if EARLY_STOPPING_KEY not in metadata_dict:
        metadata_dict[EARLY_STOPPING_KEY] = True

    if PLATEAU_LR_MUTIPLIER_KEY not in metadata_dict:
        metadata_dict[PLATEAU_LR_MUTIPLIER_KEY] = 0.5

    if LOSS_FUNCTION_OR_DICT_KEY not in metadata_dict:
        metadata_dict[LOSS_FUNCTION_OR_DICT_KEY] = (
            metadata_dict['loss_function']
        )

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

    # TODO(thunderhoser): HACK!
    if (
            metadata_dict[BNN_ARCHITECTURE_KEY] is None and
            'shortwave_bnn_experiment01/' in dill_file_name
    ):
        new_file_name = dill_file_name.replace(
            'shortwave_bnn_experiment01/',
            'shortwave_bnn_experiment01/templates/'
        )

        new_file_handle = open(new_file_name, 'rb')
        new_metadata_dict = dill.load(new_file_handle)
        new_file_handle.close()

        metadata_dict[BNN_ARCHITECTURE_KEY] = (
            new_metadata_dict[BNN_ARCHITECTURE_KEY]
        )

    missing_keys = list(set(METADATA_KEYS) - set(metadata_dict.keys()))
    if len(missing_keys) == 0:
        return metadata_dict

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), dill_file_name)

    raise ValueError(error_string)


def apply_model(
        model_object, predictor_matrix, num_examples_per_batch,
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
    :param predictor_matrix: See output doc for `data_generator`.
    :param num_examples_per_batch: Batch size.
    :param use_dropout: Boolean flag.  If True, will keep dropout in all layers
        turned on.  Using dropout at inference time is called "Monte Carlo
        dropout".
    :param verbose: Boolean flag.  If True, will print progress messages.

    :return: prediction_list: See below.
    prediction_list[0] = vector_prediction_matrix: numpy array
        (E x H x W x T_v x S) of predicted values.
    prediction_list[1] = scalar_prediction_matrix: numpy array (E x W x T_s x S)
        of predicted values.
    """

    # Check input args.
    num_examples_per_batch = _check_inference_args(
        predictor_matrix=predictor_matrix,
        num_examples_per_batch=num_examples_per_batch, verbose=verbose
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
    num_examples = predictor_matrix.shape[0]

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
            this_output = model_object(
                predictor_matrix[these_indices, ...], training=True
            )

            if isinstance(this_output, list):
                this_output = [a.numpy() for a in this_output]
            else:
                this_output = this_output.numpy()
        else:
            this_output = model_object.predict_on_batch(
                predictor_matrix[these_indices, ...]
            )

        if not isinstance(this_output, list):
            this_output = [this_output]

        # Add ensemble dimension if necessary.
        if len(this_output[0].shape) == 4:
            this_output[0] = numpy.expand_dims(this_output[0], axis=-1)

        # Add ensemble dimension if necessary.
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

    return [vector_prediction_matrix, scalar_prediction_matrix]
