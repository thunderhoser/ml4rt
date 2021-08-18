"""Methods for building, training, and applying neural nets."""

import copy
import random
import os.path
import dill
import numpy
import keras
import tensorflow.keras as tf_keras
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from ml4rt.io import example_io
from ml4rt.utils import example_utils
from ml4rt.utils import normalization
from ml4rt.machine_learning import keras_metrics as custom_metrics
from ml4rt.machine_learning import keras_losses as custom_losses

SENTINEL_VALUE = -9999.

LARGE_INTEGER = int(1e12)
LARGE_FLOAT = 1e12

PLATEAU_PATIENCE_EPOCHS = 10
DEFAULT_LEARNING_RATE_MULTIPLIER = 0.5
PLATEAU_COOLDOWN_EPOCHS = 0
EARLY_STOPPING_PATIENCE_EPOCHS = 30
LOSS_PATIENCE = 0.

CNN_TYPE_STRING = 'cnn'
DENSE_NET_TYPE_STRING = 'dense_net'
U_NET_TYPE_STRING = 'u_net'
VALID_NET_TYPE_STRINGS = [
    CNN_TYPE_STRING, DENSE_NET_TYPE_STRING, U_NET_TYPE_STRING
]

USE_MSE_SKILL_KEY = 'use_mse_skill_score'
USE_WEIGHTED_MSE_KEY = 'use_weighted_mse'
USE_DUAL_WEIGHTED_MSE_KEY = 'use_dual_weighted_mse'

TOA_UP_FLUX_INDEX_KEY = 'toa_up_flux_index'
TOA_UP_FLUX_WEIGHT_KEY = 'toa_up_flux_weight'
SURFACE_DOWN_FLUX_INDEX_KEY = 'surface_down_flux_index'
SURFACE_DOWN_FLUX_WEIGHT_KEY = 'surface_down_flux_weight'
UP_FLUX_CHANNEL_INDEX_KEY = 'up_flux_channel_index'
DOWN_FLUX_CHANNEL_INDEX_KEY = 'down_flux_channel_index'
HIGHEST_UP_FLUX_INDEX_KEY = 'highest_up_flux_index'
LOWEST_DOWN_FLUX_INDEX_KEY = 'lowest_down_flux_index'
NET_FLUX_WEIGHT_KEY = 'net_flux_weight'

EXAMPLE_DIRECTORY_KEY = 'example_dir_name'
BATCH_SIZE_KEY = 'num_examples_per_batch'
SCALAR_PREDICTOR_NAMES_KEY = 'scalar_predictor_names'
VECTOR_PREDICTOR_NAMES_KEY = 'vector_predictor_names'
SCALAR_TARGET_NAMES_KEY = 'scalar_target_names'
VECTOR_TARGET_NAMES_KEY = 'vector_target_names'
HEIGHTS_KEY = 'heights_m_agl'
FIRST_TIME_KEY = 'first_time_unix_sec'
LAST_TIME_KEY = 'last_time_unix_sec'
MIN_COLUMN_LWP_KEY = 'min_column_lwp_kg_m02'
MAX_COLUMN_LWP_KEY = 'max_column_lwp_kg_m02'
OMIT_HEATING_RATE_KEY = 'omit_heating_rate'
NORMALIZATION_FILE_KEY = 'normalization_file_name'
PREDICTOR_NORM_TYPE_KEY = 'predictor_norm_type_string'
PREDICTOR_MIN_NORM_VALUE_KEY = 'predictor_min_norm_value'
PREDICTOR_MAX_NORM_VALUE_KEY = 'predictor_max_norm_value'
VECTOR_TARGET_NORM_TYPE_KEY = 'vector_target_norm_type_string'
VECTOR_TARGET_MIN_VALUE_KEY = 'vector_target_min_norm_value'
VECTOR_TARGET_MAX_VALUE_KEY = 'vector_target_max_norm_value'
SCALAR_TARGET_NORM_TYPE_KEY = 'scalar_target_norm_type_string'
SCALAR_TARGET_MIN_VALUE_KEY = 'scalar_target_min_norm_value'
SCALAR_TARGET_MAX_VALUE_KEY = 'scalar_target_max_norm_value'

DEFAULT_GENERATOR_OPTION_DICT = {
    SCALAR_PREDICTOR_NAMES_KEY: example_utils.ALL_SCALAR_PREDICTOR_NAMES,
    VECTOR_PREDICTOR_NAMES_KEY: example_utils.BASIC_VECTOR_PREDICTOR_NAMES,
    SCALAR_TARGET_NAMES_KEY: example_utils.ALL_SCALAR_TARGET_NAMES,
    VECTOR_TARGET_NAMES_KEY: example_utils.ALL_VECTOR_TARGET_NAMES,
    HEIGHTS_KEY: example_utils.DEFAULT_HEIGHTS_M_AGL,
    MIN_COLUMN_LWP_KEY: 0.,
    MAX_COLUMN_LWP_KEY: LARGE_FLOAT,
    OMIT_HEATING_RATE_KEY: False,
    PREDICTOR_NORM_TYPE_KEY: normalization.Z_SCORE_NORM_STRING,
    PREDICTOR_MIN_NORM_VALUE_KEY: None,
    PREDICTOR_MAX_NORM_VALUE_KEY: None,
    VECTOR_TARGET_NORM_TYPE_KEY: normalization.MINMAX_NORM_STRING,
    VECTOR_TARGET_MIN_VALUE_KEY: 0.,
    VECTOR_TARGET_MAX_VALUE_KEY: 1.,
    SCALAR_TARGET_NORM_TYPE_KEY: normalization.MINMAX_NORM_STRING,
    SCALAR_TARGET_MIN_VALUE_KEY: 0.,
    SCALAR_TARGET_MAX_VALUE_KEY: 1.
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
NET_TYPE_KEY = 'net_type_string'
LOSS_FUNCTION_OR_DICT_KEY = 'loss_function_or_dict'
EARLY_STOPPING_KEY = 'do_early_stopping'
PLATEAU_LR_MUTIPLIER_KEY = 'plateau_lr_multiplier'

METADATA_KEYS = [
    NUM_EPOCHS_KEY, NUM_TRAINING_BATCHES_KEY, TRAINING_OPTIONS_KEY,
    NUM_VALIDATION_BATCHES_KEY, VALIDATION_OPTIONS_KEY, NET_TYPE_KEY,
    LOSS_FUNCTION_OR_DICT_KEY, EARLY_STOPPING_KEY, PLATEAU_LR_MUTIPLIER_KEY
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

    omit_heating_rate = option_dict[OMIT_HEATING_RATE_KEY]
    error_checking.assert_is_boolean(omit_heating_rate)

    required_vector_target_names = [
        example_utils.SHORTWAVE_UP_FLUX_INC_NAME,
        example_utils.SHORTWAVE_DOWN_FLUX_INC_NAME,
        example_utils.SHORTWAVE_HEATING_RATE_NAME
    ]
    this_flag = all([
        n in option_dict[VECTOR_TARGET_NAMES_KEY]
        for n in required_vector_target_names
    ])

    option_dict[OMIT_HEATING_RATE_KEY] = omit_heating_rate and this_flag
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
        heights_m_agl, min_column_lwp_kg_m02, max_column_lwp_kg_m02,
        training_example_dict, predictor_norm_type_string,
        predictor_min_norm_value, predictor_max_norm_value,
        vector_target_norm_type_string, vector_target_min_norm_value,
        vector_target_max_norm_value, scalar_target_norm_type_string,
        scalar_target_min_norm_value, scalar_target_max_norm_value,
        exclude_summit_greenland=False):
    """Reads one file for generator.

    :param example_file_name: Path to input file (will be read by
        `example_io.read_file`).
    :param first_time_unix_sec: See doc for `data_generator`.
    :param last_time_unix_sec: Same.
    :param field_names: 1-D list of fields to keep.
    :param heights_m_agl: 1-D numpy array of heights to keep (metres above
        ground level).
    :param min_column_lwp_kg_m02: See doc for `data_generator`.
    :param max_column_lwp_kg_m02: Same.
    :param training_example_dict: Dictionary with training examples (in format
        specified by `example_io.read_file`), which will be used for
        normalization.
    :param predictor_norm_type_string: See doc for `data_generator`.
    :param predictor_min_norm_value: Same.
    :param predictor_max_norm_value: Same.
    :param vector_target_norm_type_string: Same.
    :param vector_target_min_norm_value: Same.
    :param vector_target_max_norm_value: Same.
    :param scalar_target_norm_type_string: Same.
    :param scalar_target_min_norm_value: Same.
    :param scalar_target_max_norm_value: Same.
    :param exclude_summit_greenland: Boolean flag.  If True, will exclude
        examples from Summit.
    :return: example_dict: See doc for `example_io.read_file`.
    """

    print('\nReading data from: "{0:s}"...'.format(example_file_name))
    example_dict = example_io.read_file(
        netcdf_file_name=example_file_name,
        exclude_summit_greenland=exclude_summit_greenland
    )

    example_dict = example_utils.subset_by_time(
        example_dict=example_dict,
        first_time_unix_sec=first_time_unix_sec,
        last_time_unix_sec=last_time_unix_sec
    )[0]

    example_dict = example_utils.subset_by_column_lwp(
        example_dict=example_dict, min_lwp_kg_m02=min_column_lwp_kg_m02,
        max_lwp_kg_m02=max_column_lwp_kg_m02
    )[0]

    if any([f in field_names for f in example_utils.TRACE_GAS_NAMES]):
        example_dict = example_utils.add_trace_gases(
            example_dict=example_dict, noise_stdev_fractional=0.05
        )

    if any([f in field_names for f in example_utils.EFFECTIVE_RADIUS_NAMES]):
        example_dict = example_utils.add_effective_radii(
            example_dict=example_dict, ice_noise_stdev_fractional=0.05
        )

    if any([f in field_names for f in example_utils.AEROSOL_NAMES]):
        example_dict = example_utils.add_aerosols(example_dict)

    example_dict = example_utils.subset_by_field(
        example_dict=example_dict, field_names=field_names
    )
    example_dict = example_utils.subset_by_height(
        example_dict=example_dict, heights_m_agl=heights_m_agl
    )

    if predictor_norm_type_string is not None:
        print('Applying {0:s} normalization to predictors...'.format(
            predictor_norm_type_string.upper()
        ))
        example_dict = normalization.normalize_data(
            new_example_dict=example_dict,
            training_example_dict=training_example_dict,
            normalization_type_string=predictor_norm_type_string,
            min_normalized_value=predictor_min_norm_value,
            max_normalized_value=predictor_max_norm_value,
            separate_heights=True, apply_to_predictors=True,
            apply_to_vector_targets=False, apply_to_scalar_targets=False
        )

    if vector_target_norm_type_string is not None:
        print('Applying {0:s} normalization to vector targets...'.format(
            vector_target_norm_type_string.upper()
        ))
        example_dict = normalization.normalize_data(
            new_example_dict=example_dict,
            training_example_dict=training_example_dict,
            normalization_type_string=vector_target_norm_type_string,
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
            min_normalized_value=scalar_target_min_norm_value,
            max_normalized_value=scalar_target_max_norm_value,
            separate_heights=True, apply_to_predictors=False,
            apply_to_vector_targets=False, apply_to_scalar_targets=True
        )

    return example_dict


def _read_specific_examples(
        example_file_names, example_id_strings, field_names, heights_m_agl,
        training_example_dict, predictor_norm_type_string,
        predictor_min_norm_value, predictor_max_norm_value,
        vector_target_norm_type_string, vector_target_min_norm_value,
        vector_target_max_norm_value, scalar_target_norm_type_string,
        scalar_target_min_norm_value, scalar_target_max_norm_value):
    """Reads specific examples for generator.

    :param example_file_names: 1-D list of paths to input files (will be read by
        `example_io.read_file`).
    :param example_id_strings: Same.
    :param field_names: Same.
    :param heights_m_agl: Same.
    :param training_example_dict: Same.
    :param predictor_norm_type_string: Same.
    :param predictor_min_norm_value: Same.
    :param predictor_max_norm_value: Same.
    :param vector_target_norm_type_string: Same.
    :param vector_target_min_norm_value: Same.
    :param vector_target_max_norm_value: Same.
    :param scalar_target_norm_type_string: Same.
    :param scalar_target_min_norm_value: Same.
    :param scalar_target_max_norm_value: Same.
    :return: example_dict: Same.
    """

    file_years = numpy.array(
        [example_io.file_name_to_year(f) for f in example_file_names], dtype=int
    )

    example_times_unix_sec = example_utils.parse_example_ids(
        example_id_strings
    )[example_utils.VALID_TIMES_KEY]

    example_years = numpy.array([
        int(time_conversion.unix_sec_to_string(t, '%Y'))
        for t in example_times_unix_sec
    ], dtype=int)

    unique_example_years = numpy.unique(example_years)
    example_dicts = []

    for this_year in unique_example_years:
        these_indices = numpy.where(example_years == this_year)[0]
        these_example_id_strings = [
            example_id_strings[k] for k in these_indices
        ]
        these_file_indices = numpy.where(file_years == this_year)[0]

        print('\n')
        these_example_dicts = []

        for k in these_file_indices:
            print('Reading data from: "{0:s}"...'.format(example_file_names[k]))
            these_example_dicts.append(
                example_io.read_file(example_file_names[k])
            )

        this_example_dict = example_utils.concat_examples(these_example_dicts)

        these_indices = example_utils.find_examples(
            all_id_strings=this_example_dict[example_utils.EXAMPLE_IDS_KEY],
            desired_id_strings=these_example_id_strings,
            allow_missing=False
        )
        this_example_dict = example_utils.subset_by_index(
            example_dict=this_example_dict, desired_indices=these_indices
        )

        if any([f in field_names for f in example_utils.TRACE_GAS_NAMES]):
            this_example_dict = example_utils.add_trace_gases(
                example_dict=this_example_dict, noise_stdev_fractional=0.05
            )

        if any([
                f in field_names for f in example_utils.EFFECTIVE_RADIUS_NAMES
        ]):
            this_example_dict = example_utils.add_effective_radii(
                example_dict=this_example_dict, ice_noise_stdev_fractional=0.05
            )

        if any([f in field_names for f in example_utils.AEROSOL_NAMES]):
            this_example_dict = example_utils.add_aerosols(this_example_dict)

        this_example_dict = example_utils.subset_by_field(
            example_dict=this_example_dict, field_names=field_names
        )
        this_example_dict = example_utils.subset_by_height(
            example_dict=this_example_dict, heights_m_agl=heights_m_agl
        )

        example_dicts.append(this_example_dict)

    example_dict = example_utils.concat_examples(example_dicts)

    if predictor_norm_type_string is not None:
        print('Applying {0:s} normalization to predictors...'.format(
            predictor_norm_type_string.upper()
        ))
        example_dict = normalization.normalize_data(
            new_example_dict=example_dict,
            training_example_dict=training_example_dict,
            normalization_type_string=predictor_norm_type_string,
            min_normalized_value=predictor_min_norm_value,
            max_normalized_value=predictor_max_norm_value,
            separate_heights=True, apply_to_predictors=True,
            apply_to_vector_targets=False, apply_to_scalar_targets=False
        )

    if vector_target_norm_type_string is not None:
        print('Applying {0:s} normalization to vector targets...'.format(
            vector_target_norm_type_string.upper()
        ))
        example_dict = normalization.normalize_data(
            new_example_dict=example_dict,
            training_example_dict=training_example_dict,
            normalization_type_string=vector_target_norm_type_string,
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
            min_normalized_value=scalar_target_min_norm_value,
            max_normalized_value=scalar_target_max_norm_value,
            separate_heights=True, apply_to_predictors=False,
            apply_to_vector_targets=False, apply_to_scalar_targets=True
        )

    return example_dict


def _write_metafile(
        dill_file_name, num_epochs, num_training_batches_per_epoch,
        training_option_dict, num_validation_batches_per_epoch,
        validation_option_dict, net_type_string, loss_function_or_dict,
        do_early_stopping, plateau_lr_multiplier):
    """Writes metadata to Dill file.

    :param dill_file_name: Path to output file.
    :param num_epochs: See doc for `train_model`.
    :param num_training_batches_per_epoch: Same.
    :param training_option_dict: Same.
    :param num_validation_batches_per_epoch: Same.
    :param validation_option_dict: Same.
    :param net_type_string: Same.
    :param loss_function_or_dict: Same.
    :param do_early_stopping: Same.
    :param plateau_lr_multiplier: Same.
    """

    metadata_dict = {
        NUM_EPOCHS_KEY: num_epochs,
        NUM_TRAINING_BATCHES_KEY: num_training_batches_per_epoch,
        TRAINING_OPTIONS_KEY: training_option_dict,
        NUM_VALIDATION_BATCHES_KEY: num_validation_batches_per_epoch,
        VALIDATION_OPTIONS_KEY: validation_option_dict,
        NET_TYPE_KEY: net_type_string,
        LOSS_FUNCTION_OR_DICT_KEY: loss_function_or_dict,
        EARLY_STOPPING_KEY: do_early_stopping,
        PLATEAU_LR_MUTIPLIER_KEY: plateau_lr_multiplier
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=dill_file_name)

    dill_file_handle = open(dill_file_name, 'wb')
    dill.dump(metadata_dict, dill_file_handle)
    dill_file_handle.close()


def check_net_type(net_type_string):
    """Ensures that neural-net type is valid.

    :param net_type_string: Neural-net type.
    :raises: ValueError: if `net_type_string not in VALID_NET_TYPE_STRINGS`.
    """

    error_checking.assert_is_string(net_type_string)
    if net_type_string in VALID_NET_TYPE_STRINGS:
        return

    error_string = (
        '\nField "{0:s}" is not valid neural-net type.  Valid options listed '
        'below:\n{1:s}'
    ).format(net_type_string, str(VALID_NET_TYPE_STRINGS))

    raise ValueError(error_string)


def predictors_dict_to_numpy(example_dict, net_type_string):
    """Converts predictors from dictionary to numpy array.

    :param example_dict: Dictionary of examples (in the format returned by
        `example_io.read_file`).
    :param net_type_string: Type of neural net (must be accepted by
        `check_net_type`).
    :return: predictor_matrix: See output doc for `data_generator`.
    :return: predictor_name_matrix: numpy array of predictor names (strings), in
        the same shape as predictor_matrix[0, ...].
    :return: height_matrix_m_agl: numpy array of heights (metres above ground
        level), in the same shape as predictor_matrix[0, ...].  For scalar
        variables, the matrix entry will be NaN.
    """

    check_net_type(net_type_string)

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

    if net_type_string != DENSE_NET_TYPE_STRING:
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

    num_examples = vector_predictor_matrix.shape[0]

    vector_predictor_matrix = numpy.reshape(
        vector_predictor_matrix,
        (num_examples, num_heights * num_vector_predictors),
        order='F'
    )
    vector_predictor_name_matrix = numpy.reshape(
        vector_predictor_name_matrix, num_heights * num_vector_predictors,
        order='F'
    )
    vector_height_matrix_m_agl = numpy.reshape(
        vector_height_matrix_m_agl, num_heights * num_vector_predictors,
        order='F'
    )

    predictor_matrix = numpy.concatenate(
        (vector_predictor_matrix, scalar_predictor_matrix), axis=-1
    )
    scalar_height_matrix_m_agl = numpy.full(num_scalar_predictors, numpy.nan)
    height_matrix_m_agl = numpy.concatenate(
        (vector_height_matrix_m_agl, scalar_height_matrix_m_agl), axis=0
    )
    predictor_name_matrix = numpy.concatenate(
        (vector_predictor_name_matrix, scalar_predictor_names), axis=0
    )

    return predictor_matrix, predictor_name_matrix, height_matrix_m_agl


def predictors_numpy_to_dict(predictor_matrix, example_dict, net_type_string):
    """Converts predictors from numpy array to dictionary.

    This method is the inverse of `predictors_dict_to_numpy`.

    :param predictor_matrix: numpy array created by `predictors_dict_to_numpy`.
    :param example_dict: Dictionary with the following keys.  See doc for
        `example_io.read_file` for details on each key.
    example_dict['scalar_predictor_names']
    example_dict['vector_predictor_names']
    example_dict['heights_m_agl']

    :param net_type_string: Type of neural net (must be accepted by
        `check_net_type`).

    :return: example_dict: Dictionary with the following keys.  See doc for
        `example_io.read_file` for details on each key.
    example_dict['scalar_predictor_matrix']
    example_dict['vector_predictor_matrix']
    """

    error_checking.assert_is_numpy_array_without_nan(predictor_matrix)
    check_net_type(net_type_string)

    num_scalar_predictors = len(
        example_dict[example_utils.SCALAR_PREDICTOR_NAMES_KEY]
    )

    if net_type_string == DENSE_NET_TYPE_STRING:
        error_checking.assert_is_numpy_array(predictor_matrix, num_dimensions=2)

        if num_scalar_predictors == 0:
            scalar_predictor_matrix = predictor_matrix[:, :0]
            vector_predictor_matrix = predictor_matrix + 0.
        else:
            scalar_predictor_matrix = (
                predictor_matrix[:, -num_scalar_predictors:]
            )
            vector_predictor_matrix = (
                predictor_matrix[:, :-num_scalar_predictors]
            )

        num_heights = len(example_dict[example_utils.HEIGHTS_KEY])
        num_vector_predictors = len(
            example_dict[example_utils.VECTOR_PREDICTOR_NAMES_KEY]
        )
        num_examples = vector_predictor_matrix.shape[0]

        vector_predictor_matrix = numpy.reshape(
            vector_predictor_matrix,
            (num_examples, num_heights, num_vector_predictors),
            order='F'
        )

        return {
            example_utils.SCALAR_PREDICTOR_VALS_KEY: scalar_predictor_matrix,
            example_utils.VECTOR_PREDICTOR_VALS_KEY: vector_predictor_matrix
        }

    error_checking.assert_is_numpy_array(predictor_matrix, num_dimensions=3)

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


def targets_dict_to_numpy(example_dict, net_type_string):
    """Converts targets from dictionary to numpy array.

    :param example_dict: Dictionary of examples (in the format returned by
        `example_io.read_file`).
    :param net_type_string: Type of neural net (must be accepted by
        `check_net_type`).
    :return: target_matrices: If net type is CNN, same as output from
        `data_generator`.  Otherwise, same as output from `data_generator` but
        in a one-element list.
    """

    check_net_type(net_type_string)

    if net_type_string == DENSE_NET_TYPE_STRING:
        vector_target_matrix = (
            example_dict[example_utils.VECTOR_TARGET_VALS_KEY]
        )
        scalar_target_matrix = (
            example_dict[example_utils.SCALAR_TARGET_VALS_KEY]
        )

        num_examples = vector_target_matrix.shape[0]
        num_heights = vector_target_matrix.shape[1]
        num_fields = vector_target_matrix.shape[2]
        vector_target_matrix = numpy.reshape(
            vector_target_matrix, (num_examples, num_heights * num_fields),
            order='F'
        )

        if scalar_target_matrix.size == 0:
            return [vector_target_matrix]

        return [vector_target_matrix, scalar_target_matrix]

    vector_target_matrix = (
        example_dict[example_utils.VECTOR_TARGET_VALS_KEY]
    )
    scalar_target_matrix = (
        example_dict[example_utils.SCALAR_TARGET_VALS_KEY]
    )

    if scalar_target_matrix.size == 0:
        return [vector_target_matrix]

    return [vector_target_matrix, scalar_target_matrix]


def targets_numpy_to_dict(target_matrices, example_dict, net_type_string):
    """Converts targets from numpy array to dictionary.

    This method is the inverse of `targets_dict_to_numpy`.

    :param target_matrices: List created by `targets_dict_to_numpy`.
    :param example_dict: Dictionary with the following keys.  See doc for
        `example_io.read_file` for details on each key.
    example_dict['scalar_target_names']
    example_dict['vector_target_names']
    example_dict['heights_m_agl']

    :param net_type_string: Type of neural net (must be accepted by
        `check_net_type`).

    :return: example_dict: Dictionary with the following keys.  See doc for
        `example_io.read_file` for details on each key.
    example_dict['scalar_target_matrix']
    example_dict['vector_target_matrix']
    """

    check_net_type(net_type_string)

    vector_target_matrix = target_matrices[0]
    error_checking.assert_is_numpy_array_without_nan(vector_target_matrix)
    error_checking.assert_is_numpy_array(
        vector_target_matrix,
        num_dimensions=3 - int(net_type_string == DENSE_NET_TYPE_STRING)
    )

    if len(target_matrices) == 1:
        scalar_target_matrix = numpy.full(
            (vector_target_matrix.shape[0], 0), 0.
        )
    else:
        scalar_target_matrix = target_matrices[1]

    error_checking.assert_is_numpy_array_without_nan(scalar_target_matrix)
    error_checking.assert_is_numpy_array(
        scalar_target_matrix, num_dimensions=2
    )

    if net_type_string != DENSE_NET_TYPE_STRING:
        return {
            example_utils.SCALAR_TARGET_VALS_KEY: scalar_target_matrix,
            example_utils.VECTOR_TARGET_VALS_KEY: vector_target_matrix
        }

    num_heights = len(example_dict[example_utils.HEIGHTS_KEY])
    num_vector_targets = len(
        example_dict[example_utils.VECTOR_TARGET_NAMES_KEY]
    )
    num_examples = vector_target_matrix.shape[0]

    try:
        vector_target_matrix = numpy.reshape(
            vector_target_matrix,
            (num_examples, num_heights, num_vector_targets),
            order='F'
        )
    except ValueError:

        # TODO(thunderhoser): This is kind of a HACK.
        vector_target_matrix = numpy.reshape(
            vector_target_matrix,
            (num_examples, num_heights, num_vector_targets - 1),
            order='F'
        )
        heating_rate_index = (
            example_dict[example_utils.VECTOR_TARGET_NAMES_KEY].index(
                example_utils.SHORTWAVE_HEATING_RATE_NAME
            )
        )
        vector_target_matrix = numpy.insert(
            vector_target_matrix, obj=heating_rate_index,
            values=0., axis=-1
        )

    return {
        example_utils.SCALAR_TARGET_VALS_KEY: scalar_target_matrix,
        example_utils.VECTOR_TARGET_VALS_KEY: vector_target_matrix
    }


def neuron_indices_to_target_var(neuron_indices, example_dict, net_type_string,
                                 for_scalar_output=False):
    """Converts indices of output neuron to metadata for target variable.

    :param neuron_indices: 1-D numpy array with indices of output neuron.  Must
        have length of either 1 (for scalar target variable) or 2 (for vector
        target variable).
    :param example_dict: See doc for `targets_numpy_to_dict`.
    :param net_type_string: Same.
    :param for_scalar_output: Boolean flag, used only for dense nets.
    :return: target_name: Name of target variable.
    :return: height_m_agl: Height (metres above ground level) of target
        variable.  If target variable is scalar, this will be None.
    """

    error_checking.assert_is_integer_numpy_array(neuron_indices)
    error_checking.assert_is_geq_numpy_array(neuron_indices, 0)
    check_net_type(net_type_string)

    if net_type_string == DENSE_NET_TYPE_STRING:
        min_num_indices = 1
        max_num_indices = 1
        error_checking.assert_is_boolean(for_scalar_output)
    else:
        min_num_indices = 1
        max_num_indices = 2
        for_scalar_output = False

    num_indices = len(neuron_indices)
    error_checking.assert_is_geq(num_indices, min_num_indices)
    error_checking.assert_is_leq(num_indices, max_num_indices)

    vector_target_names = example_dict[example_utils.VECTOR_TARGET_NAMES_KEY]
    heights_m_agl = example_dict[example_utils.HEIGHTS_KEY]

    if num_indices == 2:
        return (
            vector_target_names[neuron_indices[1]],
            heights_m_agl[neuron_indices[0]]
        )

    scalar_target_names = example_dict[example_utils.SCALAR_TARGET_NAMES_KEY]
    num_scalar_targets = len(scalar_target_names)
    num_vector_targets = len(vector_target_names)
    num_heights = len(heights_m_agl)

    if net_type_string == DENSE_NET_TYPE_STRING:
        vector_target_matrix_keras = numpy.full(
            (1, num_vector_targets * num_heights), 0.
        )

        if for_scalar_output:
            scalar_target_matrix_keras = numpy.full((1, num_scalar_targets), 0.)
            scalar_target_matrix_keras[0, neuron_indices[0]] = SENTINEL_VALUE
            target_matrices_keras = [
                vector_target_matrix_keras, scalar_target_matrix_keras
            ]
        else:
            vector_target_matrix_keras[0, neuron_indices[0]] = SENTINEL_VALUE
            target_matrices_keras = [vector_target_matrix_keras]

        example_dict = targets_numpy_to_dict(
            target_matrices=target_matrices_keras,
            example_dict=example_dict, net_type_string=net_type_string
        )

        if for_scalar_output:
            scalar_target_matrix_orig = (
                example_dict[example_utils.SCALAR_TARGET_VALS_KEY][0, ...]
            )
            this_index = numpy.where(
                scalar_target_matrix_orig < SENTINEL_VALUE + 1
            )[0][0]

            return scalar_target_names[this_index], None

        vector_target_matrix_orig = (
            example_dict[example_utils.VECTOR_TARGET_VALS_KEY][0, ...]
        )
        these_height_indices, these_field_indices = numpy.where(
            vector_target_matrix_orig < SENTINEL_VALUE + 1
        )
        return (
            vector_target_names[these_field_indices[0]],
            heights_m_agl[these_height_indices[0]]
        )

    # If execution reaches this point, the net is a CNN or U-net.
    vector_target_matrix_keras = numpy.full(
        (1, num_heights, num_vector_targets), 0.
    )
    scalar_target_matrix_keras = numpy.full((1, num_scalar_targets), 0.)
    scalar_target_matrix_keras[0, neuron_indices[0]] = SENTINEL_VALUE

    example_dict = targets_numpy_to_dict(
        target_matrices=
        [vector_target_matrix_keras, scalar_target_matrix_keras],
        example_dict=example_dict, net_type_string=net_type_string
    )
    scalar_target_matrix_orig = (
        example_dict[example_utils.SCALAR_TARGET_VALS_KEY][0, ...]
    )

    these_indices = numpy.where(
        scalar_target_matrix_orig < SENTINEL_VALUE + 1
    )[0]
    return scalar_target_names[these_indices[0]], None


def target_var_to_neuron_indices(example_dict, net_type_string, target_name,
                                 height_m_agl=None):
    """Converts metadata for target variable to indices of output neuron.

    This method is the inverse of `neuron_indices_to_target_var`.

    :param example_dict: See doc for `neuron_indices_to_target_var`.
    :param net_type_string: Same.
    :param target_name: Same.
    :param height_m_agl: Same.
    :return: neuron_indices: Same.
    """

    check_net_type(net_type_string)
    error_checking.assert_is_string(target_name)

    scalar_target_names = example_dict[example_utils.SCALAR_TARGET_NAMES_KEY]
    vector_target_names = example_dict[example_utils.VECTOR_TARGET_NAMES_KEY]
    heights_m_agl = example_dict[example_utils.HEIGHTS_KEY]

    num_scalar_targets = len(scalar_target_names)
    num_vector_targets = len(vector_target_names)
    num_heights = len(heights_m_agl)

    vector_target_matrix_orig = numpy.full(
        (1, num_heights, num_vector_targets), 0.
    )
    scalar_target_matrix_orig = numpy.full((1, num_scalar_targets), 0.)

    if height_m_agl is None:
        channel_index = scalar_target_names.index(target_name)
        scalar_target_matrix_orig[:, channel_index] = SENTINEL_VALUE

        new_example_dict = {
            example_utils.SCALAR_TARGET_VALS_KEY: scalar_target_matrix_orig,
            example_utils.VECTOR_TARGET_VALS_KEY: vector_target_matrix_orig
        }

        scalar_target_matrix_keras = targets_dict_to_numpy(
            example_dict=new_example_dict, net_type_string=net_type_string
        )[-1][0, ...]

        neuron_index = numpy.where(
            scalar_target_matrix_keras < SENTINEL_VALUE + 1
        )[0][0]

        return numpy.array([neuron_index], dtype=int)

    channel_index = vector_target_names.index(target_name)
    height_index = example_utils.match_heights(
        heights_m_agl=heights_m_agl, desired_height_m_agl=height_m_agl
    )
    vector_target_matrix_orig[:, height_index, channel_index] = SENTINEL_VALUE

    new_example_dict = {
        example_utils.SCALAR_TARGET_VALS_KEY: scalar_target_matrix_orig,
        example_utils.VECTOR_TARGET_VALS_KEY: vector_target_matrix_orig
    }

    target_matrices_keras = targets_dict_to_numpy(
        example_dict=new_example_dict, net_type_string=net_type_string
    )
    vector_target_matrix_keras = target_matrices_keras[0][0, ...]

    if net_type_string == DENSE_NET_TYPE_STRING:
        neuron_index = numpy.where(
            vector_target_matrix_keras < SENTINEL_VALUE + 1
        )[0][0]

        return numpy.array([neuron_index], dtype=int)

    height_indices, field_indices = numpy.where(
        vector_target_matrix_keras < SENTINEL_VALUE + 1
    )

    return numpy.array([height_indices[0], field_indices[0]], dtype=int)


def data_generator(option_dict, for_inference, net_type_string):
    """Generates training data for any kind of neural net.

    E = number of examples per batch (batch size)
    H = number of heights
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
    option_dict['first_time_unix_sec']: Start time (will not generate examples
        before this time).
    option_dict['last_time_unix_sec']: End time (will not generate examples
        after this time).
    option_dict['min_column_lwp_kg_m02']: Minimum full-column liquid-water path
        (LWP; kg m^-2).
    option_dict['max_column_lwp_kg_m02']: Max full-column LWP (kg m^-2).
    option_dict['omit_heating_rate']: Boolean flag.  If True, the net will not
        directly predict heating rate, even if it is in the list of target
        variables.
    option_dict['normalization_file_name']: File with training examples to use
        for normalization (will be read by `example_io.read_file`).
    option_dict['predictor_norm_type_string']: Normalization type for predictors
        (must be accepted by `normalization._check_normalization_type`).  If you
        do not want to normalize predictors, make this None.
    option_dict['predictor_min_norm_value']: Minimum normalized value for
        predictors (used only if normalization type is min-max).
    option_dict['predictor_max_norm_value']: Same but max value.
    option_dict['vector_target_norm_type_string']: Normalization type for vector
        targets (must be accepted by `normalization._check_normalization_type`).
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

    :param for_inference: Boolean flag.  If True, generator is being used for
        inference stage (applying trained model to new data).  If False,
        generator is being used for training or monitoring (on-the-fly
        validation).
    :param net_type_string: Type of neural net (must be accepted by
        `check_net_type`).

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
    check_net_type(net_type_string)

    example_dir_name = option_dict[EXAMPLE_DIRECTORY_KEY]
    num_examples_per_batch = option_dict[BATCH_SIZE_KEY]
    scalar_predictor_names = option_dict[SCALAR_PREDICTOR_NAMES_KEY]
    vector_predictor_names = option_dict[VECTOR_PREDICTOR_NAMES_KEY]
    scalar_target_names = option_dict[SCALAR_TARGET_NAMES_KEY]
    vector_target_names = option_dict[VECTOR_TARGET_NAMES_KEY]
    heights_m_agl = option_dict[HEIGHTS_KEY]
    first_time_unix_sec = option_dict[FIRST_TIME_KEY]
    last_time_unix_sec = option_dict[LAST_TIME_KEY]
    min_column_lwp_kg_m02 = option_dict[MIN_COLUMN_LWP_KEY]
    max_column_lwp_kg_m02 = option_dict[MAX_COLUMN_LWP_KEY]
    omit_heating_rate = option_dict[OMIT_HEATING_RATE_KEY] and not for_inference

    all_field_names = (
        scalar_predictor_names + vector_predictor_names +
        scalar_target_names + vector_target_names
    )
    if omit_heating_rate:
        all_field_names = [
            f for f in all_field_names
            if f != example_utils.SHORTWAVE_HEATING_RATE_NAME
        ]

    normalization_file_name = option_dict[NORMALIZATION_FILE_KEY]
    predictor_norm_type_string = option_dict[PREDICTOR_NORM_TYPE_KEY]
    predictor_min_norm_value = option_dict[PREDICTOR_MIN_NORM_VALUE_KEY]
    predictor_max_norm_value = option_dict[PREDICTOR_MAX_NORM_VALUE_KEY]
    vector_target_norm_type_string = option_dict[VECTOR_TARGET_NORM_TYPE_KEY]
    vector_target_min_norm_value = option_dict[VECTOR_TARGET_MIN_VALUE_KEY]
    vector_target_max_norm_value = option_dict[VECTOR_TARGET_MAX_VALUE_KEY]
    scalar_target_norm_type_string = option_dict[SCALAR_TARGET_NORM_TYPE_KEY]
    scalar_target_min_norm_value = option_dict[SCALAR_TARGET_MIN_VALUE_KEY]
    scalar_target_max_norm_value = option_dict[SCALAR_TARGET_MAX_VALUE_KEY]

    print((
        'Reading training examples (for normalization) from: "{0:s}"...'
    ).format(
        normalization_file_name
    ))
    training_example_dict = example_io.read_file(normalization_file_name)
    training_example_dict = example_utils.subset_by_height(
        example_dict=training_example_dict, heights_m_agl=heights_m_agl
    )

    example_file_names = example_io.find_many_files(
        directory_name=example_dir_name,
        first_time_unix_sec=first_time_unix_sec,
        last_time_unix_sec=last_time_unix_sec,
        raise_error_if_any_missing=False
    )

    all_desired_id_strings = []

    for this_file_name in example_file_names:
        this_example_dict = _read_file_for_generator(
            example_file_name=this_file_name,
            first_time_unix_sec=first_time_unix_sec,
            last_time_unix_sec=last_time_unix_sec,
            field_names=all_field_names, heights_m_agl=heights_m_agl,
            min_column_lwp_kg_m02=min_column_lwp_kg_m02,
            max_column_lwp_kg_m02=max_column_lwp_kg_m02,
            training_example_dict=training_example_dict,
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

        these_id_strings = this_example_dict[example_utils.EXAMPLE_IDS_KEY]
        random.shuffle(these_id_strings)
        all_desired_id_strings += these_id_strings

    example_index = 0

    while True:
        if for_inference and example_index >= len(all_desired_id_strings):
            raise StopIteration

        num_examples_in_memory = 0
        predictor_matrix = None
        vector_target_matrix = None
        scalar_target_matrix = None
        example_id_strings = []

        while num_examples_in_memory < num_examples_per_batch:
            if example_index == len(all_desired_id_strings):
                if for_inference:
                    if predictor_matrix is None:
                        raise StopIteration

                    break

                example_index = 0

            this_num_examples = num_examples_per_batch - num_examples_in_memory

            this_example_dict = _read_specific_examples(
                example_file_names=example_file_names,
                example_id_strings=
                all_desired_id_strings[example_index:][:this_num_examples],
                field_names=all_field_names, heights_m_agl=heights_m_agl,
                training_example_dict=training_example_dict,
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

            these_id_strings = this_example_dict[example_utils.EXAMPLE_IDS_KEY]
            example_id_strings += these_id_strings
            example_index += len(these_id_strings)

            this_predictor_matrix = predictors_dict_to_numpy(
                example_dict=this_example_dict, net_type_string=net_type_string
            )[0]
            this_target_list = targets_dict_to_numpy(
                example_dict=this_example_dict, net_type_string=net_type_string
            )

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

        predictor_matrix = predictor_matrix.astype('float32')
        target_array = [vector_target_matrix.astype('float32')]
        if scalar_target_matrix is not None:
            target_array.append(scalar_target_matrix.astype('float32'))

        if for_inference:
            yield predictor_matrix, target_array, example_id_strings
        else:
            yield predictor_matrix, target_array


def create_data(option_dict, for_inference, net_type_string,
                exclude_summit_greenland=False):
    """Creates data for any kind of neural net.

    This method is the same as `data_generator`, except that it returns all the
    data at once, rather than generating batches on the fly.

    :param option_dict: See doc for `data_generator`.
    :param for_inference: Same.
    :param net_type_string: Same.
    :param exclude_summit_greenland: Boolean flag.  If True, will exclude
        examples from Summit.
    :return: predictor_matrix: Same.
    :return: target_array: Same.
    :return: example_id_strings: Same.
    """

    option_dict = _check_generator_args(option_dict)
    check_net_type(net_type_string)
    error_checking.assert_is_boolean(exclude_summit_greenland)

    example_dir_name = option_dict[EXAMPLE_DIRECTORY_KEY]
    scalar_predictor_names = option_dict[SCALAR_PREDICTOR_NAMES_KEY]
    vector_predictor_names = option_dict[VECTOR_PREDICTOR_NAMES_KEY]
    scalar_target_names = option_dict[SCALAR_TARGET_NAMES_KEY]
    vector_target_names = option_dict[VECTOR_TARGET_NAMES_KEY]
    heights_m_agl = option_dict[HEIGHTS_KEY]
    first_time_unix_sec = option_dict[FIRST_TIME_KEY]
    last_time_unix_sec = option_dict[LAST_TIME_KEY]
    min_column_lwp_kg_m02 = option_dict[MIN_COLUMN_LWP_KEY]
    max_column_lwp_kg_m02 = option_dict[MAX_COLUMN_LWP_KEY]
    omit_heating_rate = option_dict[OMIT_HEATING_RATE_KEY] and not for_inference

    all_field_names = (
        scalar_predictor_names + vector_predictor_names +
        scalar_target_names + vector_target_names
    )
    if omit_heating_rate:
        all_field_names = [
            f for f in all_field_names
            if f != example_utils.SHORTWAVE_HEATING_RATE_NAME
        ]

    normalization_file_name = option_dict[NORMALIZATION_FILE_KEY]
    predictor_norm_type_string = option_dict[PREDICTOR_NORM_TYPE_KEY]
    predictor_min_norm_value = option_dict[PREDICTOR_MIN_NORM_VALUE_KEY]
    predictor_max_norm_value = option_dict[PREDICTOR_MAX_NORM_VALUE_KEY]
    vector_target_norm_type_string = option_dict[VECTOR_TARGET_NORM_TYPE_KEY]
    vector_target_min_norm_value = option_dict[VECTOR_TARGET_MIN_VALUE_KEY]
    vector_target_max_norm_value = option_dict[VECTOR_TARGET_MAX_VALUE_KEY]
    scalar_target_norm_type_string = option_dict[SCALAR_TARGET_NORM_TYPE_KEY]
    scalar_target_min_norm_value = option_dict[SCALAR_TARGET_MIN_VALUE_KEY]
    scalar_target_max_norm_value = option_dict[SCALAR_TARGET_MAX_VALUE_KEY]

    print((
        'Reading training examples (for normalization) from: "{0:s}"...'
    ).format(
        normalization_file_name
    ))
    training_example_dict = example_io.read_file(normalization_file_name)
    training_example_dict = example_utils.subset_by_height(
        example_dict=training_example_dict, heights_m_agl=heights_m_agl
    )

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
            field_names=all_field_names, heights_m_agl=heights_m_agl,
            min_column_lwp_kg_m02=min_column_lwp_kg_m02,
            max_column_lwp_kg_m02=max_column_lwp_kg_m02,
            training_example_dict=training_example_dict,
            predictor_norm_type_string=predictor_norm_type_string,
            predictor_min_norm_value=predictor_min_norm_value,
            predictor_max_norm_value=predictor_max_norm_value,
            vector_target_norm_type_string=vector_target_norm_type_string,
            vector_target_min_norm_value=vector_target_min_norm_value,
            vector_target_max_norm_value=vector_target_max_norm_value,
            scalar_target_norm_type_string=scalar_target_norm_type_string,
            scalar_target_min_norm_value=scalar_target_min_norm_value,
            scalar_target_max_norm_value=scalar_target_max_norm_value,
            exclude_summit_greenland=exclude_summit_greenland
        )

        example_dicts.append(this_example_dict)

    example_dict = example_utils.concat_examples(example_dicts)

    num_examples = len(example_dict[example_utils.VALID_TIMES_KEY])
    desired_indices = numpy.linspace(
        0, num_examples - 1, num=num_examples, dtype=int
    )
    numpy.random.shuffle(desired_indices)

    example_dict = example_utils.subset_by_index(
        example_dict=example_dict, desired_indices=desired_indices
    )

    predictor_matrix = predictors_dict_to_numpy(
        example_dict=example_dict, net_type_string=net_type_string
    )[0]
    predictor_matrix = predictor_matrix.astype('float32')

    prelim_target_list = targets_dict_to_numpy(
        example_dict=example_dict, net_type_string=net_type_string
    )

    vector_target_matrix = prelim_target_list[0]
    target_array = [vector_target_matrix.astype('float32')]

    if len(prelim_target_list) > 1:
        scalar_target_matrix = prelim_target_list[1]
        target_array.append(scalar_target_matrix.astype('float32'))

    return (
        predictor_matrix,
        target_array,
        example_dict[example_utils.EXAMPLE_IDS_KEY]
    )


def data_generator_specific_examples(option_dict, net_type_string,
                                     example_id_strings):
    """Generates training data for specific examples.

    This method is the same as `data_generator`, except that it generates
    specific examples.  Also, note that this method should be run only in
    inference mode (not in training mode).

    :param option_dict: See doc for `data_generator`.
    :param net_type_string: Same.
    :param example_id_strings: 1-D list of example IDs.
    :return: Same output variables as `data_generator`, except without
        `example_id_strings`.
    """

    option_dict = _check_generator_args(option_dict)
    check_net_type(net_type_string)

    example_times_unix_sec = example_utils.parse_example_ids(
        example_id_strings
    )[example_utils.VALID_TIMES_KEY]

    example_dir_name = option_dict[EXAMPLE_DIRECTORY_KEY]
    num_examples_per_batch = option_dict[BATCH_SIZE_KEY]
    scalar_predictor_names = option_dict[SCALAR_PREDICTOR_NAMES_KEY]
    vector_predictor_names = option_dict[VECTOR_PREDICTOR_NAMES_KEY]
    scalar_target_names = option_dict[SCALAR_TARGET_NAMES_KEY]
    vector_target_names = option_dict[VECTOR_TARGET_NAMES_KEY]
    heights_m_agl = option_dict[HEIGHTS_KEY]

    all_field_names = (
        scalar_predictor_names + vector_predictor_names +
        scalar_target_names + vector_target_names
    )

    normalization_file_name = option_dict[NORMALIZATION_FILE_KEY]
    predictor_norm_type_string = option_dict[PREDICTOR_NORM_TYPE_KEY]
    predictor_min_norm_value = option_dict[PREDICTOR_MIN_NORM_VALUE_KEY]
    predictor_max_norm_value = option_dict[PREDICTOR_MAX_NORM_VALUE_KEY]
    vector_target_norm_type_string = option_dict[VECTOR_TARGET_NORM_TYPE_KEY]
    vector_target_min_norm_value = option_dict[VECTOR_TARGET_MIN_VALUE_KEY]
    vector_target_max_norm_value = option_dict[VECTOR_TARGET_MAX_VALUE_KEY]
    scalar_target_norm_type_string = option_dict[SCALAR_TARGET_NORM_TYPE_KEY]
    scalar_target_min_norm_value = option_dict[SCALAR_TARGET_MIN_VALUE_KEY]
    scalar_target_max_norm_value = option_dict[SCALAR_TARGET_MAX_VALUE_KEY]

    print((
        'Reading training examples (for normalization) from: "{0:s}"...'
    ).format(
        normalization_file_name
    ))
    training_example_dict = example_io.read_file(normalization_file_name)
    training_example_dict = example_utils.subset_by_height(
        example_dict=training_example_dict, heights_m_agl=heights_m_agl
    )

    example_file_names = example_io.find_many_files(
        directory_name=example_dir_name,
        first_time_unix_sec=numpy.min(example_times_unix_sec),
        last_time_unix_sec=numpy.max(example_times_unix_sec),
        raise_error_if_any_missing=False
    )

    example_index = 0

    while True:
        if example_index >= len(example_id_strings):
            raise StopIteration

        num_examples_in_memory = 0
        predictor_matrix = None
        vector_target_matrix = None
        scalar_target_matrix = None

        while num_examples_in_memory < num_examples_per_batch:
            if example_index >= len(example_id_strings):
                if predictor_matrix is None:
                    raise StopIteration

                break

            this_num_examples = num_examples_per_batch - num_examples_in_memory

            this_example_dict = _read_specific_examples(
                example_file_names=example_file_names,
                example_id_strings=
                example_id_strings[example_index:][:this_num_examples],
                field_names=all_field_names, heights_m_agl=heights_m_agl,
                training_example_dict=training_example_dict,
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

            example_index += len(
                this_example_dict[example_utils.EXAMPLE_IDS_KEY]
            )

            this_predictor_matrix = predictors_dict_to_numpy(
                example_dict=this_example_dict, net_type_string=net_type_string
            )[0]
            this_target_list = targets_dict_to_numpy(
                example_dict=this_example_dict, net_type_string=net_type_string
            )

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

        predictor_matrix = predictor_matrix.astype('float32')
        target_array = [vector_target_matrix.astype('float32')]
        if scalar_target_matrix is not None:
            target_array.append(scalar_target_matrix.astype('float32'))

        yield predictor_matrix, target_array


def create_data_specific_examples(
        option_dict, net_type_string, example_id_strings):
    """Creates data for specific examples.

    This method is the same as `create_data`, except that it creates specific
    examples.  Also, note that this method should be run only in inference mode
    (not in training mode).

    :param option_dict: See doc for `data_generator`.
    :param net_type_string: Same.
    :param example_id_strings: 1-D list of example IDs.
    :return: predictor_matrix: See doc for `data_generator`.
    :return: target_array: Same.
    """

    option_dict = _check_generator_args(option_dict)
    check_net_type(net_type_string)

    example_times_unix_sec = example_utils.parse_example_ids(
        example_id_strings
    )[example_utils.VALID_TIMES_KEY]

    example_dir_name = option_dict[EXAMPLE_DIRECTORY_KEY]
    scalar_predictor_names = option_dict[SCALAR_PREDICTOR_NAMES_KEY]
    vector_predictor_names = option_dict[VECTOR_PREDICTOR_NAMES_KEY]
    scalar_target_names = option_dict[SCALAR_TARGET_NAMES_KEY]
    vector_target_names = option_dict[VECTOR_TARGET_NAMES_KEY]
    heights_m_agl = option_dict[HEIGHTS_KEY]

    all_field_names = (
        scalar_predictor_names + vector_predictor_names +
        scalar_target_names + vector_target_names
    )

    normalization_file_name = option_dict[NORMALIZATION_FILE_KEY]
    predictor_norm_type_string = option_dict[PREDICTOR_NORM_TYPE_KEY]
    predictor_min_norm_value = option_dict[PREDICTOR_MIN_NORM_VALUE_KEY]
    predictor_max_norm_value = option_dict[PREDICTOR_MAX_NORM_VALUE_KEY]
    vector_target_norm_type_string = option_dict[VECTOR_TARGET_NORM_TYPE_KEY]
    vector_target_min_norm_value = option_dict[VECTOR_TARGET_MIN_VALUE_KEY]
    vector_target_max_norm_value = option_dict[VECTOR_TARGET_MAX_VALUE_KEY]
    scalar_target_norm_type_string = option_dict[SCALAR_TARGET_NORM_TYPE_KEY]
    scalar_target_min_norm_value = option_dict[SCALAR_TARGET_MIN_VALUE_KEY]
    scalar_target_max_norm_value = option_dict[SCALAR_TARGET_MAX_VALUE_KEY]

    print((
        'Reading training examples (for normalization) from: "{0:s}"...'
    ).format(
        normalization_file_name
    ))
    training_example_dict = example_io.read_file(normalization_file_name)
    training_example_dict = example_utils.subset_by_height(
        example_dict=training_example_dict, heights_m_agl=heights_m_agl
    )

    example_file_names = example_io.find_many_files(
        directory_name=example_dir_name,
        first_time_unix_sec=numpy.min(example_times_unix_sec),
        last_time_unix_sec=numpy.max(example_times_unix_sec),
        raise_error_if_any_missing=False
    )

    example_dict = _read_specific_examples(
        example_file_names=example_file_names,
        example_id_strings=example_id_strings,
        field_names=all_field_names, heights_m_agl=heights_m_agl,
        training_example_dict=training_example_dict,
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

    predictor_matrix = predictors_dict_to_numpy(
        example_dict=example_dict, net_type_string=net_type_string
    )[0]
    predictor_matrix = predictor_matrix.astype('float32')

    prelim_target_list = targets_dict_to_numpy(
        example_dict=example_dict, net_type_string=net_type_string
    )

    vector_target_matrix = prelim_target_list[0]
    target_array = [vector_target_matrix.astype('float32')]

    if len(prelim_target_list) > 1:
        scalar_target_matrix = prelim_target_list[1]
        target_array.append(scalar_target_matrix.astype('float32'))

    return predictor_matrix, target_array


def train_model_with_generator(
        model_object, output_dir_name, num_epochs,
        num_training_batches_per_epoch, training_option_dict,
        validation_option_dict, net_type_string, loss_function_or_dict,
        use_generator_for_validn=False, num_validation_batches_per_epoch=None,
        do_early_stopping=True,
        plateau_lr_multiplier=DEFAULT_LEARNING_RATE_MULTIPLIER):
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

    :param net_type_string: Neural-net type (must be accepted by
        `check_net_type`).
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
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    error_checking.assert_is_integer(num_epochs)
    error_checking.assert_is_geq(num_epochs, 2)
    error_checking.assert_is_integer(num_training_batches_per_epoch)
    error_checking.assert_is_geq(num_training_batches_per_epoch, 10)
    check_net_type(net_type_string)
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

    # model_file_name = (
    #     output_dir_name + '/model_epoch={epoch:03d}_val-loss={val_loss:.6f}.h5'
    # )
    model_file_name = output_dir_name + '/model.h5'

    history_object = keras.callbacks.CSVLogger(
        filename='{0:s}/history.csv'.format(output_dir_name),
        separator=',', append=False
    )
    checkpoint_object = keras.callbacks.ModelCheckpoint(
        filepath=model_file_name, monitor='val_loss', verbose=1,
        save_best_only=do_early_stopping, save_weights_only=False, mode='min',
        period=1
    )
    list_of_callback_objects = [history_object, checkpoint_object]

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
        net_type_string=net_type_string,
        loss_function_or_dict=loss_function_or_dict,
        do_early_stopping=do_early_stopping,
        plateau_lr_multiplier=plateau_lr_multiplier
    )

    training_generator = data_generator(
        option_dict=training_option_dict, for_inference=False,
        net_type_string=net_type_string
    )

    if use_generator_for_validn:
        validation_generator = data_generator(
            option_dict=validation_option_dict, for_inference=False,
            net_type_string=net_type_string
        )

        validation_data_arg = validation_generator
        validation_steps_arg = num_validation_batches_per_epoch
    else:
        validation_predictor_matrix, validation_target_array = create_data(
            option_dict=validation_option_dict, for_inference=False,
            net_type_string=net_type_string
        )[:2]

        validation_data_arg = (
            validation_predictor_matrix, validation_target_array
        )
        validation_steps_arg = None

    model_object.fit_generator(
        generator=training_generator,
        steps_per_epoch=num_training_batches_per_epoch,
        epochs=num_epochs, verbose=1, callbacks=list_of_callback_objects,
        validation_data=validation_data_arg,
        validation_steps=validation_steps_arg
    )


def train_model_sans_generator(
        model_object, output_dir_name, num_epochs, training_option_dict,
        validation_option_dict, net_type_string, loss_function_or_dict,
        do_early_stopping=True,
        plateau_lr_multiplier=DEFAULT_LEARNING_RATE_MULTIPLIER):
    """Trains any kind of neural net without generator.

    :param model_object: See doc for `train_model_with_generator`.
    :param output_dir_name: Same.
    :param num_epochs: Same.
    :param training_option_dict: Same.
    :param validation_option_dict: Same.
    :param net_type_string: Same.
    :param loss_function_or_dict: Same.
    :param do_early_stopping: Same.
    :param plateau_lr_multiplier: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    error_checking.assert_is_integer(num_epochs)
    error_checking.assert_is_geq(num_epochs, 2)
    check_net_type(net_type_string)
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

    # model_file_name = (
    #     output_dir_name + '/model_epoch={epoch:03d}_val-loss={val_loss:.6f}.h5'
    # )
    model_file_name = output_dir_name + '/model.h5'

    history_object = keras.callbacks.CSVLogger(
        filename='{0:s}/history.csv'.format(output_dir_name),
        separator=',', append=False
    )
    checkpoint_object = keras.callbacks.ModelCheckpoint(
        filepath=model_file_name, monitor='val_loss', verbose=1,
        save_best_only=do_early_stopping, save_weights_only=False, mode='min',
        period=1
    )
    list_of_callback_objects = [history_object, checkpoint_object]

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
        net_type_string=net_type_string,
        loss_function_or_dict=loss_function_or_dict,
        do_early_stopping=do_early_stopping,
        plateau_lr_multiplier=plateau_lr_multiplier
    )

    training_predictor_matrix, training_target_array = create_data(
        option_dict=training_option_dict, for_inference=False,
        net_type_string=net_type_string
    )[:2]

    validation_predictor_matrix, validation_target_array = create_data(
        option_dict=validation_option_dict, for_inference=False,
        net_type_string=net_type_string
    )[:2]

    model_object.fit(
        x=training_predictor_matrix, y=training_target_array,
        batch_size=training_option_dict[BATCH_SIZE_KEY],
        epochs=num_epochs, steps_per_epoch=None, shuffle=True, verbose=1,
        callbacks=list_of_callback_objects,
        validation_data=(validation_predictor_matrix, validation_target_array),
        # validation_batch_size=validation_option_dict[BATCH_SIZE_KEY],
        validation_steps=None
    )


def read_model(hdf5_file_name):
    """Reads model from HDF5 file.

    :param hdf5_file_name: Path to input file.
    :return: model_object: Instance of `keras.models.Model` or
        `keras.models.Sequential`.
    """

    error_checking.assert_file_exists(hdf5_file_name)

    try:
        return tf_keras.models.load_model(
            hdf5_file_name, custom_objects=METRIC_FUNCTION_DICT
        )
    except ValueError:
        pass

    metafile_name = find_metafile(
        model_dir_name=os.path.split(hdf5_file_name)[0],
        raise_error_if_missing=True
    )

    metadata_dict = read_metafile(metafile_name)
    custom_object_dict = copy.deepcopy(METRIC_FUNCTION_DICT)
    loss_function_or_dict = metadata_dict[LOSS_FUNCTION_OR_DICT_KEY]

    if isinstance(loss_function_or_dict, dict):
        for this_key in loss_function_or_dict:
            custom_object_dict[this_key + '_loss'] = eval(
                loss_function_or_dict[this_key]
            )

        custom_object_dict['loss'] = eval(loss_function_or_dict['conv_output'])
    else:
        custom_object_dict['loss'] = eval(loss_function_or_dict)

    return tf_keras.models.load_model(
        hdf5_file_name, custom_objects=custom_object_dict
    )


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
    metadata_dict['net_type_string']: Same.
    metadata_dict['loss_function_or_dict']: Same.

    :raises: ValueError: if any expected key is not found in dictionary.
    """

    error_checking.assert_file_exists(dill_file_name)

    dill_file_handle = open(dill_file_name, 'rb')
    metadata_dict = dill.load(dill_file_handle)
    dill_file_handle.close()

    if OMIT_HEATING_RATE_KEY not in metadata_dict[TRAINING_OPTIONS_KEY]:
        metadata_dict[TRAINING_OPTIONS_KEY][OMIT_HEATING_RATE_KEY] = False
        metadata_dict[VALIDATION_OPTIONS_KEY][OMIT_HEATING_RATE_KEY] = False

    if VECTOR_TARGET_NORM_TYPE_KEY not in metadata_dict[TRAINING_OPTIONS_KEY]:
        t = metadata_dict[TRAINING_OPTIONS_KEY]
        v = metadata_dict[VALIDATION_OPTIONS_KEY]

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

    # TODO(thunderhoser): This is a HACK to deal with Hera not letting me
    # install Dill.
    if isinstance(metadata_dict[LOSS_FUNCTION_OR_DICT_KEY], str):
        if 'dual_weighted_mse' in metadata_dict[LOSS_FUNCTION_OR_DICT_KEY]:
            metadata_dict[LOSS_FUNCTION_OR_DICT_KEY] = (
                custom_losses.dual_weighted_mse()
            )
        else:
            metadata_dict[LOSS_FUNCTION_OR_DICT_KEY] = keras.losses.mse

    missing_keys = list(set(METADATA_KEYS) - set(metadata_dict.keys()))
    if len(missing_keys) == 0:
        return metadata_dict

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), dill_file_name)

    raise ValueError(error_string)


def apply_model(
        model_object, predictor_matrix, num_examples_per_batch, net_type_string,
        verbose=False):
    """Applies trained neural net (of any kind) to new data.

    E = number of examples
    H = number of heights
    T_v = number of vector target variables (channels)
    T_s = number of scalar target variables
    T = number of target variables

    :param model_object: Trained neural net (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param predictor_matrix: See output doc for `data_generator`.
    :param num_examples_per_batch: Batch size.
    :param net_type_string: Type of neural net (must be accepted by
        `check_net_type`).
    :param verbose: Boolean flag.  If True, will print progress messages.

    :return: prediction_list: See below.
    prediction_list[0] = vector_prediction_matrix: numpy array (E x H x T_v) of
        predicted values.
    prediction_list[1] = scalar_prediction_matrix: numpy array (E x T_s) of
        predicted values.
    """

    check_net_type(net_type_string)

    num_examples_per_batch = _check_inference_args(
        predictor_matrix=predictor_matrix,
        num_examples_per_batch=num_examples_per_batch, verbose=verbose
    )

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
            print((
                'Applying {0:s} to examples {1:d}-{2:d} of {3:d}...'
            ).format(
                net_type_string.upper(),
                this_first_index + 1, this_last_index + 1, num_examples
            ))

        this_output = model_object.predict(
            predictor_matrix[these_indices, ...], batch_size=len(these_indices)
        )

        if not isinstance(this_output, list):
            this_output = [this_output]

        if vector_prediction_matrix is None:
            vector_prediction_matrix = this_output[0] + 0.

            if len(this_output) == 2:
                scalar_prediction_matrix = this_output[1] + 0.
        else:
            vector_prediction_matrix = numpy.concatenate(
                (vector_prediction_matrix, this_output[0]), axis=0
            )

            if len(this_output) == 2:
                scalar_prediction_matrix = numpy.concatenate(
                    (scalar_prediction_matrix, this_output[1]), axis=0
                )

    if verbose:
        print('Have applied {0:s} to all {1:d} examples!'.format(
            net_type_string.upper(), num_examples
        ))

    if scalar_prediction_matrix is None:
        scalar_prediction_matrix = numpy.full(
            (vector_prediction_matrix.shape[0], 0), 0.
        )

    return [vector_prediction_matrix, scalar_prediction_matrix]


def get_feature_maps(
        model_object, predictor_matrix, num_examples_per_batch,
        feature_layer_name, verbose=False):
    """Uses trained neural net (of any kind) to create feature maps.

    :param model_object: See doc for `apply_model`.
    :param predictor_matrix: Same.
    :param num_examples_per_batch: Same.
    :param feature_layer_name: Feature maps will be returned for this layer.
    :param verbose: See doc for `apply_model`.
    :return: feature_matrix: numpy array of feature maps.
    """

    num_examples_per_batch = _check_inference_args(
        predictor_matrix=predictor_matrix,
        num_examples_per_batch=num_examples_per_batch, verbose=verbose
    )

    partial_model_object = cnn.model_to_feature_generator(
        model_object=model_object, feature_layer_name=feature_layer_name
    )

    feature_matrix = None
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
            print((
                'Creating feature maps for examples {0:d}-{1:d} of {2:d}...'
            ).format(
                this_first_index + 1, this_last_index + 1, num_examples
            ))

        this_feature_matrix = partial_model_object.predict(
            predictor_matrix[these_indices, ...], batch_size=len(these_indices)
        )

        if feature_matrix is None:
            feature_matrix = this_feature_matrix + 0.
        else:
            feature_matrix = numpy.concatenate(
                (feature_matrix, this_feature_matrix), axis=0
            )

    if verbose:
        print('Have created feature maps for all {0:d} examples!'.format(
            num_examples
        ))

    return feature_matrix
