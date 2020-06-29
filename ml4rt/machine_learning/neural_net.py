"""Methods for building, training, and applying neural nets."""

import copy
import pickle
import os.path
import numpy
import keras
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from ml4rt.io import example_io
from ml4rt.utils import normalization
from ml4rt.machine_learning import keras_metrics as custom_metrics
from ml4rt.machine_learning import keras_losses as custom_losses

PLATEAU_PATIENCE_EPOCHS = 3
PLATEAU_LEARNING_RATE_MULTIPLIER = 0.5
PLATEAU_COOLDOWN_EPOCHS = 0
EARLY_STOPPING_PATIENCE_EPOCHS = 15
LOSS_PATIENCE = 0.

CNN_TYPE_STRING = 'cnn'
DENSE_NET_TYPE_STRING = 'dense_net'
U_NET_TYPE_STRING = 'u_net'
VALID_NET_TYPE_STRINGS = [
    CNN_TYPE_STRING, DENSE_NET_TYPE_STRING, U_NET_TYPE_STRING
]

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
NORMALIZATION_FILE_KEY = 'normalization_file_name'
PREDICTOR_NORM_TYPE_KEY = 'predictor_norm_type_string'
PREDICTOR_MIN_NORM_VALUE_KEY = 'predictor_min_norm_value'
PREDICTOR_MAX_NORM_VALUE_KEY = 'predictor_max_norm_value'
TARGET_NORM_TYPE_KEY = 'target_norm_type_string'
TARGET_MIN_NORM_VALUE_KEY = 'target_min_norm_value'
TARGET_MAX_NORM_VALUE_KEY = 'target_max_norm_value'

DEFAULT_GENERATOR_OPTION_DICT = {
    SCALAR_PREDICTOR_NAMES_KEY: example_io.SCALAR_PREDICTOR_NAMES,
    VECTOR_PREDICTOR_NAMES_KEY: example_io.VECTOR_PREDICTOR_NAMES,
    SCALAR_TARGET_NAMES_KEY: example_io.SCALAR_TARGET_NAMES,
    VECTOR_TARGET_NAMES_KEY: example_io.VECTOR_TARGET_NAMES,
    HEIGHTS_KEY: example_io.DEFAULT_HEIGHTS_M_AGL,
    PREDICTOR_NORM_TYPE_KEY: normalization.Z_SCORE_NORM_STRING,
    PREDICTOR_MIN_NORM_VALUE_KEY: None,
    PREDICTOR_MAX_NORM_VALUE_KEY: None,
    TARGET_NORM_TYPE_KEY: normalization.MINMAX_NORM_STRING,
    TARGET_MIN_NORM_VALUE_KEY: 0.,
    TARGET_MAX_NORM_VALUE_KEY: 1.
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
USE_MSESS_LOSS_KEY = 'use_msess_loss'
CUSTOM_LOSS_KEY = 'custom_loss_dict'

METADATA_KEYS = [
    NUM_EPOCHS_KEY, NUM_TRAINING_BATCHES_KEY, TRAINING_OPTIONS_KEY,
    NUM_VALIDATION_BATCHES_KEY, VALIDATION_OPTIONS_KEY, NET_TYPE_KEY,
    USE_MSESS_LOSS_KEY, CUSTOM_LOSS_KEY
]


def _check_net_type(net_type_string):
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


def _check_generator_args(option_dict):
    """Error-checks input arguments for generator.

    :param option_dict: See doc for `data_generator`.
    :return: option_dict: Same as input, except defaults may have been added.
    """

    orig_option_dict = option_dict.copy()
    option_dict = DEFAULT_GENERATOR_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    error_checking.assert_is_integer(option_dict[BATCH_SIZE_KEY])
    error_checking.assert_is_geq(option_dict[BATCH_SIZE_KEY], 32)

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

    error_checking.assert_is_string(option_dict[PREDICTOR_NORM_TYPE_KEY])
    error_checking.assert_is_string(option_dict[TARGET_NORM_TYPE_KEY])

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
        example_file_name, num_examples_to_keep, for_inference,
        first_time_unix_sec, last_time_unix_sec, field_names, heights_m_agl,
        training_example_dict, predictor_norm_type_string,
        predictor_min_norm_value, predictor_max_norm_value,
        target_norm_type_string, target_min_norm_value, target_max_norm_value,
        first_example_to_keep=None):
    """Reads one file for generator.

    :param example_file_name: Path to input file (will be read by
        `example_io.read_file`).
    :param num_examples_to_keep: Number of examples to keep.
    :param for_inference: Boolean flag.  If True, data are being used for
        inference stage (applying trained model to new data).  If False, data
        are being used for training or monitoring (on-the-fly validation).
    :param first_time_unix_sec: See doc for `data_generator`.
    :param last_time_unix_sec: Same.
    :param field_names: 1-D list of fields to keep.
    :param heights_m_agl: 1-D numpy array of heights to keep (metres above
        ground level).
    :param training_example_dict: Dictionary with training examples (in format
        specified by `example_io.read_file`), which will be used for
        normalization.
    :param predictor_norm_type_string: Same.
    :param predictor_min_norm_value: Same.
    :param predictor_max_norm_value: Same.
    :param target_norm_type_string: Same.
    :param target_min_norm_value: Same.
    :param target_max_norm_value: Same.
    :param first_example_to_keep: Index of first example to keep.  If specified,
        this method will return examples i through i + N - 1, where
        i = `first_example_to_keep` and N = `num_examples_to_keep`.  If None,
        this method will return N random examples.
    :return: example_dict: See doc for `example_io.read_file`.
    :return: example_id_strings: 1-D list of IDs created by
        `example_io.create_example_ids`.  If `for_inference == False`, this is
        None.
    """

    print('\nReading data from: "{0:s}"...'.format(example_file_name))
    example_dict = example_io.read_file(example_file_name)

    example_dict = example_io.reduce_sample_size(
        example_dict=example_dict, num_examples_to_keep=num_examples_to_keep,
        first_example_to_keep=first_example_to_keep
    )
    example_dict = example_io.subset_by_time(
        example_dict=example_dict,
        first_time_unix_sec=first_time_unix_sec,
        last_time_unix_sec=last_time_unix_sec
    )
    example_dict = example_io.subset_by_field(
        example_dict=example_dict, field_names=field_names
    )
    example_dict = example_io.subset_by_height(
        example_dict=example_dict, heights_m_agl=heights_m_agl
    )

    if for_inference:
        example_id_strings = example_io.create_example_ids(example_dict)
    else:
        example_id_strings = None

    print('Applying {0:s} normalization to predictors...'.format(
        predictor_norm_type_string.upper()
    ))
    example_dict = normalization.normalize_data(
        new_example_dict=example_dict,
        training_example_dict=training_example_dict,
        normalization_type_string=predictor_norm_type_string,
        min_normalized_value=predictor_min_norm_value,
        max_normalized_value=predictor_max_norm_value,
        separate_heights=True,
        apply_to_predictors=True, apply_to_targets=False
    )

    print('Applying {0:s} normalization to targets...'.format(
        target_norm_type_string.upper()
    ))
    example_dict = normalization.normalize_data(
        new_example_dict=example_dict,
        training_example_dict=training_example_dict,
        normalization_type_string=target_norm_type_string,
        min_normalized_value=target_min_norm_value,
        max_normalized_value=target_max_norm_value,
        separate_heights=True,
        apply_to_predictors=False, apply_to_targets=True
    )

    return example_dict, example_id_strings


def _write_metafile(
        pickle_file_name, num_epochs, num_training_batches_per_epoch,
        training_option_dict, num_validation_batches_per_epoch,
        validation_option_dict, net_type_string, use_msess_loss,
        custom_loss_dict):
    """Writes metadata to Pickle file.

    :param pickle_file_name: Path to output file.
    :param num_epochs: See doc for `train_model`.
    :param num_training_batches_per_epoch: Same.
    :param training_option_dict: Same.
    :param num_validation_batches_per_epoch: Same.
    :param validation_option_dict: Same.
    :param net_type_string: Same.
    :param use_msess_loss: Same.
    :param custom_loss_dict: Same.
    """

    metadata_dict = {
        NUM_EPOCHS_KEY: num_epochs,
        NUM_TRAINING_BATCHES_KEY: num_training_batches_per_epoch,
        TRAINING_OPTIONS_KEY: training_option_dict,
        NUM_VALIDATION_BATCHES_KEY: num_validation_batches_per_epoch,
        VALIDATION_OPTIONS_KEY: validation_option_dict,
        NET_TYPE_KEY: net_type_string,
        USE_MSESS_LOSS_KEY: use_msess_loss,
        CUSTOM_LOSS_KEY: custom_loss_dict
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(metadata_dict, pickle_file_handle)
    pickle_file_handle.close()


def get_custom_loss_function(custom_loss_dict, net_type_string):
    """Creates custom loss function.

    TOA = top of atmosphere

    :param custom_loss_dict: Dictionary with the following keys.
    custom_loss_dict['toa_up_flux_index']: Variable index (in scalar output
        matrix) for TOA upwelling flux.
    custom_loss_dict['toa_up_flux_weight']: Weight used to penalize difference
        between TOA upwelling flux and upwelling flux at max height in profile.
    custom_loss_dict['surface_down_flux_index']: Variable index (in scalar
        output matrix) for surface downwelling flux.
    custom_loss_dict['surface_down_flux_weight']: Weight used to penalize
        difference between surface downwelling flux and downwelling flux at
        lowest height in profile.
    custom_loss_dict['net_flux_weight']: Weight used to penalize difference
        between actual and predicted net flux (surface downwelling minus TOA
        upwelling).

    If net type is CNN, expect the following keys.

    custom_loss_dict['up_flux_channel_index']: Channel index (in vector target
        matrix) for upwelling flux.
    custom_loss_dict['down_flux_channel_index']: Channel index (in vector target
        matrix) for downwelling flux.

    If net type is dense net, expect the following keys.

    custom_loss_dict['highest_up_flux_index']: Variable index (in scalar output
        matrix) for upwelling flux at max height in profile.
    custom_loss_dict['lowest_down_flux_index']: Variable index (in scalar output
        matrix) for downwelling flux at lowest height in profile.

    :param net_type_string: Type of neural net (must be accepted by
        `_check_net_type`).
    :return: loss_function: The function itself.
    :raises: ValueError: if net type is neither CNN nor dense net.
    """

    error_checking.assert_is_string(net_type_string)
    valid_net_type_strings = [CNN_TYPE_STRING, DENSE_NET_TYPE_STRING]

    if net_type_string not in valid_net_type_strings:
        error_string = (
            '\nThis method does not work for net type "{0:s}".  Works only for '
            'those listed below:\n{1:s}'
        ).format(net_type_string, str(valid_net_type_strings))

        raise ValueError(error_string)

    toa_up_flux_index = custom_loss_dict[TOA_UP_FLUX_INDEX_KEY]
    toa_up_flux_weight = custom_loss_dict[TOA_UP_FLUX_WEIGHT_KEY]
    surface_down_flux_index = custom_loss_dict[SURFACE_DOWN_FLUX_INDEX_KEY]
    surface_down_flux_weight = (
        custom_loss_dict[SURFACE_DOWN_FLUX_WEIGHT_KEY]
    )
    net_flux_weight = custom_loss_dict[NET_FLUX_WEIGHT_KEY]

    error_checking.assert_is_integer(toa_up_flux_index)
    error_checking.assert_is_geq(toa_up_flux_index, 0)
    error_checking.assert_is_integer(surface_down_flux_index)
    error_checking.assert_is_geq(surface_down_flux_index, 0)

    if net_type_string == CNN_TYPE_STRING:
        up_flux_channel_index = custom_loss_dict[UP_FLUX_CHANNEL_INDEX_KEY]
        down_flux_channel_index = custom_loss_dict[DOWN_FLUX_CHANNEL_INDEX_KEY]

        error_checking.assert_is_integer(up_flux_channel_index)
        error_checking.assert_is_geq(up_flux_channel_index, 0)
        error_checking.assert_is_integer(down_flux_channel_index)
        error_checking.assert_is_geq(down_flux_channel_index, 0)

        return custom_losses.constrained_mse(
            toa_up_flux_index=toa_up_flux_index + 2,
            toa_up_flux_weight=toa_up_flux_weight,
            surface_down_flux_index=surface_down_flux_index + 2,
            surface_down_flux_weight=surface_down_flux_weight,
            highest_up_flux_index=0, lowest_down_flux_index=1,
            net_flux_weight=net_flux_weight, for_cnn=True
        )

    highest_up_flux_index = custom_loss_dict[HIGHEST_UP_FLUX_INDEX_KEY]
    lowest_down_flux_index = custom_loss_dict[LOWEST_DOWN_FLUX_INDEX_KEY]

    error_checking.assert_is_integer(highest_up_flux_index)
    error_checking.assert_is_geq(highest_up_flux_index, 0)
    error_checking.assert_is_integer(lowest_down_flux_index)
    error_checking.assert_is_geq(lowest_down_flux_index, 0)

    return custom_losses.constrained_mse(
        toa_up_flux_index=toa_up_flux_index,
        toa_up_flux_weight=toa_up_flux_weight,
        surface_down_flux_index=surface_down_flux_index,
        surface_down_flux_weight=surface_down_flux_weight,
        highest_up_flux_index=highest_up_flux_index,
        lowest_down_flux_index=lowest_down_flux_index,
        net_flux_weight=net_flux_weight, for_cnn=False
    )


def predictors_dict_to_numpy(example_dict, net_type_string):
    """Converts predictors from dictionary to numpy array.

    :param example_dict: Dictionary of examples (in the format returned by
        `example_io.read_file`).
    :param net_type_string: Type of neural net (must be accepted by
        `_check_net_type`).
    :return: predictor_matrix: See output doc for `data_generator`.
    """

    _check_net_type(net_type_string)

    if net_type_string != DENSE_NET_TYPE_STRING:
        num_heights = len(example_dict[example_io.HEIGHTS_KEY])

        scalar_predictor_matrix = (
            example_dict[example_io.SCALAR_PREDICTOR_VALS_KEY]
        )
        scalar_predictor_matrix = numpy.expand_dims(
            scalar_predictor_matrix, axis=1
        )
        scalar_predictor_matrix = numpy.repeat(
            scalar_predictor_matrix, repeats=num_heights, axis=1
        )

        return numpy.concatenate((
            example_dict[example_io.VECTOR_PREDICTOR_VALS_KEY],
            scalar_predictor_matrix
        ), axis=-1)

    vector_predictor_matrix = (
        example_dict[example_io.VECTOR_PREDICTOR_VALS_KEY]
    )
    num_examples = vector_predictor_matrix.shape[0]
    num_heights = vector_predictor_matrix.shape[1]
    num_fields = vector_predictor_matrix.shape[2]

    vector_predictor_matrix = numpy.reshape(
        vector_predictor_matrix, (num_examples, num_heights * num_fields),
        order='F'
    )

    return numpy.concatenate((
        vector_predictor_matrix,
        example_dict[example_io.SCALAR_PREDICTOR_VALS_KEY]
    ), axis=-1)


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
        `_check_net_type`).

    :return: example_dict: Dictionary with the following keys.  See doc for
        `example_io.read_file` for details on each key.
    example_dict['scalar_predictor_matrix']
    example_dict['vector_predictor_matrix']
    """

    error_checking.assert_is_numpy_array_without_nan(predictor_matrix)
    _check_net_type(net_type_string)

    num_scalar_predictors = len(
        example_dict[example_io.SCALAR_PREDICTOR_NAMES_KEY]
    )

    if net_type_string == DENSE_NET_TYPE_STRING:
        error_checking.assert_is_numpy_array(predictor_matrix, num_dimensions=2)

        scalar_predictor_matrix = predictor_matrix[:, -num_scalar_predictors:]
        vector_predictor_matrix = predictor_matrix[:, :-num_scalar_predictors]

        num_heights = len(example_dict[example_io.HEIGHTS_KEY])
        num_vector_predictors = len(
            example_dict[example_io.VECTOR_PREDICTOR_NAMES_KEY]
        )
        num_examples = vector_predictor_matrix.shape[0]

        vector_predictor_matrix = numpy.reshape(
            vector_predictor_matrix,
            (num_examples, num_heights, num_vector_predictors),
            order='F'
        )

        return {
            example_io.SCALAR_PREDICTOR_VALS_KEY: scalar_predictor_matrix,
            example_io.VECTOR_PREDICTOR_VALS_KEY: vector_predictor_matrix
        }

    error_checking.assert_is_numpy_array(predictor_matrix, num_dimensions=3)

    scalar_predictor_matrix = (
        predictor_matrix[:, 0, -num_scalar_predictors:]
    )
    vector_predictor_matrix = predictor_matrix[..., :-num_scalar_predictors]

    return {
        example_io.SCALAR_PREDICTOR_VALS_KEY: scalar_predictor_matrix,
        example_io.VECTOR_PREDICTOR_VALS_KEY: vector_predictor_matrix
    }


def targets_dict_to_numpy(example_dict, net_type_string,
                          use_custom_cnn_loss=None):
    """Converts targets from dictionary to numpy array.

    :param example_dict: Dictionary of examples (in the format returned by
        `example_io.read_file`).
    :param net_type_string: Type of neural net (must be accepted by
        `_check_net_type`).
    :param use_custom_cnn_loss: Boolean flag.  If True, using custom loss for
        CNN.
    :return: target_matrices: If net type is CNN, same as output from
        `data_generator`.  Otherwise, same as output from `data_generator` but
        in a one-element list.
    """

    _check_net_type(net_type_string)

    if net_type_string == U_NET_TYPE_STRING:
        return [example_dict[example_io.VECTOR_TARGET_VALS_KEY]]

    if net_type_string == CNN_TYPE_STRING:
        error_checking.assert_is_boolean(use_custom_cnn_loss)

        if not use_custom_cnn_loss:
            return [
                example_dict[example_io.VECTOR_TARGET_VALS_KEY],
                example_dict[example_io.SCALAR_TARGET_VALS_KEY]
            ]

        up_flux_channel_index = (
            example_dict[example_io.VECTOR_TARGET_NAMES_KEY].index(
                example_io.SHORTWAVE_UP_FLUX_NAME
            )
        )
        down_flux_channel_index = (
            example_dict[example_io.VECTOR_TARGET_NAMES_KEY].index(
                example_io.SHORTWAVE_DOWN_FLUX_NAME
            )
        )

        vector_target_matrix = example_dict[example_io.VECTOR_TARGET_VALS_KEY]
        scalar_target_matrix = example_dict[example_io.SCALAR_TARGET_VALS_KEY]

        this_vector_target_matrix = numpy.stack((
            vector_target_matrix[:, -1, up_flux_channel_index],
            vector_target_matrix[:, 0, down_flux_channel_index],
        ), axis=-1)

        scalar_target_matrix = numpy.concatenate((
            this_vector_target_matrix, scalar_target_matrix
        ), axis=-1)

        return [vector_target_matrix, scalar_target_matrix]

    vector_target_matrix = example_dict[example_io.VECTOR_TARGET_VALS_KEY]
    num_examples = vector_target_matrix.shape[0]
    num_heights = vector_target_matrix.shape[1]
    num_fields = vector_target_matrix.shape[2]

    vector_target_matrix = numpy.reshape(
        vector_target_matrix, (num_examples, num_heights * num_fields),
        order='F'
    )

    target_matrix = numpy.concatenate((
        vector_target_matrix,
        example_dict[example_io.SCALAR_TARGET_VALS_KEY]
    ), axis=-1)

    return [target_matrix]


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
        `_check_net_type`).

    :return: example_dict: Dictionary with the following keys.  See doc for
        `example_io.read_file` for details on each key.
    example_dict['scalar_target_matrix']
    example_dict['vector_target_matrix']
    """

    _check_net_type(net_type_string)

    if net_type_string == U_NET_TYPE_STRING:
        vector_target_matrix = target_matrices[0]

        error_checking.assert_is_numpy_array_without_nan(vector_target_matrix)
        error_checking.assert_is_numpy_array(
            vector_target_matrix, num_dimensions=3
        )

        scalar_target_matrix = numpy.full(
            (vector_target_matrix.shape[0], 0), 0.
        )

        return {
            example_io.SCALAR_TARGET_VALS_KEY: scalar_target_matrix,
            example_io.VECTOR_TARGET_VALS_KEY: vector_target_matrix
        }

    if net_type_string == CNN_TYPE_STRING:
        vector_target_matrix = target_matrices[0]
        scalar_target_matrix = target_matrices[1]

        error_checking.assert_is_numpy_array_without_nan(vector_target_matrix)
        error_checking.assert_is_numpy_array(
            vector_target_matrix, num_dimensions=3
        )

        error_checking.assert_is_numpy_array_without_nan(scalar_target_matrix)
        error_checking.assert_is_numpy_array(
            scalar_target_matrix, num_dimensions=2
        )

        return {
            example_io.SCALAR_TARGET_VALS_KEY: scalar_target_matrix,
            example_io.VECTOR_TARGET_VALS_KEY: vector_target_matrix
        }

    target_matrix = target_matrices[0]

    error_checking.assert_is_numpy_array_without_nan(target_matrix)
    error_checking.assert_is_numpy_array(target_matrix, num_dimensions=2)

    num_scalar_targets = len(example_dict[example_io.SCALAR_TARGET_NAMES_KEY])
    scalar_target_matrix = target_matrix[:, -num_scalar_targets:]
    vector_target_matrix = target_matrix[:, :-num_scalar_targets]

    num_heights = len(example_dict[example_io.HEIGHTS_KEY])
    num_vector_targets = len(
        example_dict[example_io.VECTOR_TARGET_NAMES_KEY]
    )
    num_examples = vector_target_matrix.shape[0]

    vector_target_matrix = numpy.reshape(
        vector_target_matrix, (num_examples, num_heights, num_vector_targets),
        order='F'
    )

    return {
        example_io.SCALAR_TARGET_VALS_KEY: scalar_target_matrix,
        example_io.VECTOR_TARGET_VALS_KEY: vector_target_matrix
    }


def data_generator(option_dict, for_inference, net_type_string,
                   use_custom_cnn_loss=None):
    """Generates examples for any kind of neural net.

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
        (valid names listed in example_io.py).
    option_dict['target_names']: Same but for target variables.
    option_dict['first_time_unix_sec']: Start time (will not generate examples
        before this time).
    option_dict['last_time_unix_sec']: End time (will not generate examples after
        this time).
    option_dict['normalization_file_name']: File with training examples to use
        for normalization (will be read by `example_io.read_file`).
    option_dict['predictor_norm_type_string']: Normalization type for predictors
        (must be accepted by `normalization._check_normalization_type`).
    option_dict['predictor_min_norm_value']: Minimum normalized value for
        predictors (used only if normalization type is min-max).
    option_dict['predictor_max_norm_value']: Same but max value.
    option_dict['target_norm_type_string']: Normalization type for targets (must
        be accepted by `normalization._check_normalization_type`).
    option_dict['target_min_norm_value']: Minimum normalized value for targets
        (used only if normalization type is min-max).
    option_dict['target_max_norm_value']: Same but max value.

    :param for_inference: Boolean flag.  If True, generator is being used for
        inference stage (applying trained model to new data).  If False,
        generator is being used for training or monitoring (on-the-fly
        validation).
    :param net_type_string: Type of neural net (must be accepted by
        `_check_net_type`).
    :param use_custom_cnn_loss: Boolean flag.  If True, using CNN with custom
        loss function.

    If net type is CNN...

    :return: predictor_matrix: E-by-H-by-P numpy array of predictor values.
    :return: target_list: List with 2 items.
    target_list[0] = vector_target_matrix: numpy array (E x H x T_v) of target
        values.
    target_list[1] = scalar_target_matrix: numpy array (E x T_s) of target
        values.

    :return: example_id_strings: [returned only if `for_inference == True`]
        length-E list of example IDs created by `example_io.create_example_ids`.

    If net type is dense net...

    :return: predictor_matrix: E-by-P numpy array of predictor values.
    :return: target_matrix: E-by-T numpy array of target values.
    :return: example_id_strings: Same as for CNN.

    If net type is U-net...

    :return: predictor_matrix: E-by-H-by-P numpy array of predictor values.
    :return: target_matrix: numpy array (E x H x T_v) of target values.
    :return: example_id_strings: Same as for CNN.
    """

    option_dict = _check_generator_args(option_dict)
    error_checking.assert_is_boolean(for_inference)
    _check_net_type(net_type_string)

    if net_type_string != CNN_TYPE_STRING:
        use_custom_cnn_loss = False

    error_checking.assert_is_boolean(use_custom_cnn_loss)

    example_dir_name = option_dict[EXAMPLE_DIRECTORY_KEY]
    num_examples_per_batch = option_dict[BATCH_SIZE_KEY]
    scalar_predictor_names = option_dict[SCALAR_PREDICTOR_NAMES_KEY]
    vector_predictor_names = option_dict[VECTOR_PREDICTOR_NAMES_KEY]
    scalar_target_names = option_dict[SCALAR_TARGET_NAMES_KEY]
    vector_target_names = option_dict[VECTOR_TARGET_NAMES_KEY]
    heights_m_agl = option_dict[HEIGHTS_KEY]
    first_time_unix_sec = option_dict[FIRST_TIME_KEY]
    last_time_unix_sec = option_dict[LAST_TIME_KEY]

    all_field_names = (
        scalar_predictor_names + vector_predictor_names +
        scalar_target_names + vector_target_names
    )

    normalization_file_name = option_dict[NORMALIZATION_FILE_KEY]
    predictor_norm_type_string = option_dict[PREDICTOR_NORM_TYPE_KEY]
    predictor_min_norm_value = option_dict[PREDICTOR_MIN_NORM_VALUE_KEY]
    predictor_max_norm_value = option_dict[PREDICTOR_MAX_NORM_VALUE_KEY]
    target_norm_type_string = option_dict[TARGET_NORM_TYPE_KEY]
    target_min_norm_value = option_dict[TARGET_MIN_NORM_VALUE_KEY]
    target_max_norm_value = option_dict[TARGET_MAX_NORM_VALUE_KEY]

    print((
        'Reading training examples (for normalization) from: "{0:s}"...'
    ).format(
        normalization_file_name
    ))
    training_example_dict = example_io.read_file(normalization_file_name)

    example_file_names = example_io.find_many_files(
        example_dir_name=example_dir_name,
        first_time_unix_sec=first_time_unix_sec,
        last_time_unix_sec=last_time_unix_sec,
        raise_error_if_any_missing=False
    )

    file_index = 0

    if for_inference:
        example_index = 0
    else:
        example_index = None

    while True:
        if for_inference and file_index >= len(example_file_names):
            raise StopIteration

        num_examples_in_memory = 0
        predictor_matrix = None
        target_matrix = None
        vector_target_matrix = None
        scalar_target_matrix = None
        example_id_strings = []

        while num_examples_in_memory < num_examples_per_batch:
            if file_index == len(example_file_names):
                if for_inference:
                    if predictor_matrix is None:
                        raise StopIteration

                    break

                file_index = 0

            this_example_dict, these_id_strings = _read_file_for_generator(
                example_file_name=example_file_names[file_index],
                num_examples_to_keep=
                num_examples_per_batch - num_examples_in_memory,
                for_inference=for_inference,
                first_time_unix_sec=first_time_unix_sec,
                last_time_unix_sec=last_time_unix_sec,
                field_names=all_field_names, heights_m_agl=heights_m_agl,
                training_example_dict=training_example_dict,
                predictor_norm_type_string=predictor_norm_type_string,
                predictor_min_norm_value=predictor_min_norm_value,
                predictor_max_norm_value=predictor_max_norm_value,
                target_norm_type_string=target_norm_type_string,
                target_min_norm_value=target_min_norm_value,
                target_max_norm_value=target_max_norm_value,
                first_example_to_keep=example_index
            )

            if for_inference:
                this_num_examples = len(
                    this_example_dict[example_io.VALID_TIMES_KEY]
                )

                if this_num_examples == 0:
                    file_index += 1
                    example_index = 0
                else:
                    example_index += this_num_examples

                example_id_strings += these_id_strings
            else:
                file_index += 1

            this_predictor_matrix = predictors_dict_to_numpy(
                example_dict=this_example_dict, net_type_string=net_type_string
            )
            this_list = targets_dict_to_numpy(
                example_dict=this_example_dict, net_type_string=net_type_string,
                use_custom_cnn_loss=use_custom_cnn_loss
            )

            if net_type_string == CNN_TYPE_STRING:
                this_vector_target_matrix = this_list[0]
                this_scalar_target_matrix = this_list[1]
                this_target_matrix = None
            else:
                this_vector_target_matrix = None
                this_scalar_target_matrix = None
                this_target_matrix = this_list[0]

            if predictor_matrix is None:
                predictor_matrix = this_predictor_matrix + 0.

                if net_type_string == CNN_TYPE_STRING:
                    vector_target_matrix = this_vector_target_matrix + 0.
                    scalar_target_matrix = this_scalar_target_matrix + 0.
                else:
                    target_matrix = this_target_matrix + 0.
            else:
                predictor_matrix = numpy.concatenate(
                    (predictor_matrix, this_predictor_matrix), axis=0
                )

                if net_type_string == CNN_TYPE_STRING:
                    vector_target_matrix = numpy.concatenate((
                        vector_target_matrix, this_vector_target_matrix
                    ), axis=0)

                    scalar_target_matrix = numpy.concatenate((
                        scalar_target_matrix, this_scalar_target_matrix
                    ), axis=0)
                else:
                    target_matrix = numpy.concatenate(
                        (target_matrix, this_target_matrix), axis=0
                    )

            num_examples_in_memory = predictor_matrix.shape[0]

        predictor_matrix = predictor_matrix.astype('float32')

        if net_type_string == CNN_TYPE_STRING:
            second_output = [
                vector_target_matrix.astype('float32'),
                scalar_target_matrix.astype('float32')
            ]
        else:
            second_output = target_matrix.astype('float32')

        if for_inference:
            yield predictor_matrix, second_output, example_id_strings
        else:
            yield predictor_matrix, second_output


def train_model(
        model_object, output_dir_name, num_epochs,
        num_training_batches_per_epoch, training_option_dict,
        num_validation_batches_per_epoch, validation_option_dict,
        net_type_string, use_msess_loss, custom_loss_dict):
    """Trains any kind of neural net.

    :param model_object: Untrained neural net (instance of `keras.models.Model`
        or `keras.models.Sequential`).
    :param output_dir_name: Path to output directory (model and training history
        will be saved here).
    :param num_epochs: Number of training epochs.
    :param num_training_batches_per_epoch: Number of training batches per epoch.
    :param training_option_dict: See doc for `data_generator`.  This dictionary
        will be used to generate training data.
    :param num_validation_batches_per_epoch: Number of validation batches per
        epoch.
    :param validation_option_dict: See doc for `data_generator`.  For validation
        only, the following values will replace corresponding values in
        `training_option_dict`:
    validation_option_dict['example_dir_name']
    validation_option_dict['num_examples_per_batch']
    validation_option_dict['first_time_unix_sec']
    validation_option_dict['last_time_unix_sec']

    :param net_type_string: Neural-net type (must be accepted by
        `_check_net_type`).
    :param use_msess_loss: Boolean flag.  If True, model uses MSE (mean squared
        error) skill score as loss function.
    :param custom_loss_dict: See doc for `get_custom_loss_function`.  If the
        neural net does not have a custom loss function, make this None.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    error_checking.assert_is_integer(num_epochs)
    # error_checking.assert_is_geq(num_epochs, 10)
    error_checking.assert_is_geq(num_epochs, 2)
    error_checking.assert_is_integer(num_training_batches_per_epoch)
    error_checking.assert_is_geq(num_training_batches_per_epoch, 10)
    error_checking.assert_is_integer(num_validation_batches_per_epoch)
    error_checking.assert_is_geq(num_validation_batches_per_epoch, 10)
    _check_net_type(net_type_string)
    error_checking.assert_is_boolean(use_msess_loss)

    if net_type_string == U_NET_TYPE_STRING or use_msess_loss:
        custom_loss_dict = None

    training_option_dict = _check_generator_args(training_option_dict)

    validation_keys_to_keep = [
        EXAMPLE_DIRECTORY_KEY, BATCH_SIZE_KEY, FIRST_TIME_KEY, LAST_TIME_KEY
    ]

    for this_key in list(training_option_dict.keys()):
        if this_key in validation_keys_to_keep:
            continue

        validation_option_dict[this_key] = training_option_dict[this_key]

    validation_option_dict = _check_generator_args(validation_option_dict)

    history_object = keras.callbacks.CSVLogger(
        filename='{0:s}/history.csv'.format(output_dir_name),
        separator=',', append=False
    )

    model_file_name = (
        output_dir_name + '/model_epoch={epoch:02d}_val-loss={val_loss:.6f}.h5'
    )

    checkpoint_object = keras.callbacks.ModelCheckpoint(
        filepath=model_file_name, monitor='val_loss', verbose=1,
        save_best_only=True, save_weights_only=False, mode='min', period=1
    )

    early_stopping_object = keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=LOSS_PATIENCE,
        patience=EARLY_STOPPING_PATIENCE_EPOCHS, verbose=1, mode='min'
    )

    plateau_object = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=PLATEAU_LEARNING_RATE_MULTIPLIER,
        patience=PLATEAU_PATIENCE_EPOCHS, verbose=1, mode='min',
        min_delta=LOSS_PATIENCE, cooldown=PLATEAU_COOLDOWN_EPOCHS
    )

    list_of_callback_objects = [
        history_object, checkpoint_object, early_stopping_object, plateau_object
    ]

    metafile_name = find_metafile(output_dir_name, raise_error_if_missing=False)
    print('Writing metadata to: "{0:s}"...'.format(metafile_name))

    _write_metafile(
        pickle_file_name=metafile_name, num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        training_option_dict=training_option_dict,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        validation_option_dict=validation_option_dict,
        net_type_string=net_type_string,
        use_msess_loss=use_msess_loss, custom_loss_dict=custom_loss_dict
    )

    use_custom_cnn_loss = (
        net_type_string == CNN_TYPE_STRING and custom_loss_dict is not None
    )
    training_generator = data_generator(
        option_dict=training_option_dict, for_inference=False,
        net_type_string=net_type_string, use_custom_cnn_loss=use_custom_cnn_loss
    )
    validation_generator = data_generator(
        option_dict=validation_option_dict, for_inference=False,
        net_type_string=net_type_string, use_custom_cnn_loss=use_custom_cnn_loss
    )

    model_object.fit_generator(
        generator=training_generator,
        steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
        verbose=1, callbacks=list_of_callback_objects,
        validation_data=validation_generator,
        validation_steps=num_validation_batches_per_epoch
    )


def read_model(hdf5_file_name):
    """Reads model from HDF5 file.

    :param hdf5_file_name: Path to input file.
    :return: model_object: Instance of `keras.models.Model` or
        `keras.models.Sequential`.
    """

    error_checking.assert_file_exists(hdf5_file_name)

    try:
        return keras.models.load_model(
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

    # TODO(thunderhoser): This code is hacky.
    if metadata_dict[USE_MSESS_LOSS_KEY]:
        custom_object_dict['loss'] = custom_losses.negative_mse_skill_score()
    else:
        dense_loss_function = get_custom_loss_function(
            custom_loss_dict=metadata_dict[CUSTOM_LOSS_KEY],
            net_type_string=metadata_dict[NET_TYPE_KEY]
        )

        custom_object_dict['loss'] = dense_loss_function

    return keras.models.load_model(
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

    metafile_name = '{0:s}/model_metadata.p'.format(model_dir_name)

    if raise_error_if_missing and not os.path.isfile(metafile_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            metafile_name
        )
        raise ValueError(error_string)

    return metafile_name


def read_metafile(pickle_file_name):
    """Reads metadata for neural net from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: metadata_dict: Dictionary with the following keys.
    metadata_dict['num_epochs']: See doc for `train_model`.
    metadata_dict['num_training_batches_per_epoch']: Same.
    metadata_dict['training_option_dict']: Same.
    metadata_dict['num_validation_batches_per_epoch']: Same.
    metadata_dict['validation_option_dict']: Same.
    metadata_dict['is_cnn']: Same.
    """

    error_checking.assert_file_exists(pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'rb')
    metadata_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    if NET_TYPE_KEY not in metadata_dict:
        metadata_dict[NET_TYPE_KEY] = (
            CNN_TYPE_STRING if metadata_dict['is_cnn']
            else DENSE_NET_TYPE_STRING
        )

    if USE_MSESS_LOSS_KEY not in metadata_dict:
        metadata_dict[USE_MSESS_LOSS_KEY] = False
    if CUSTOM_LOSS_KEY not in metadata_dict:
        metadata_dict[CUSTOM_LOSS_KEY] = None

    missing_keys = list(set(METADATA_KEYS) - set(metadata_dict.keys()))
    if len(missing_keys) == 0:
        return metadata_dict

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), pickle_file_name)

    raise ValueError(error_string)


def apply_model(
        model_object, predictor_matrix, num_examples_per_batch, net_type_string,
        used_custom_cnn_loss=None, verbose=False):
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
        `_check_net_type`).
    :param used_custom_cnn_loss: Boolean flag.  If True, used custom loss for
        CNN.
    :param verbose: Boolean flag.  If True, will print progress messages.

    If net type is CNN...

    :return: prediction_list: List with the following elements.
    prediction_list[0] = vector_prediction_matrix: numpy array (E x H x T_v) of
        predicted values.
    prediction_list[1] = scalar_prediction_matrix: numpy array (E x T_s) of
        predicted values.

    If net type is U-net...

    :return: prediction_list: List with the following elements.
    prediction_list[0] = prediction_matrix: numpy array (E x H x T_v) of
        predicted values.

    If net type is dense net...

    :return: prediction_list: List with the following elements.
    prediction_list[0] = prediction_matrix: E-by-T numpy array of predicted
        values.
    """

    _check_net_type(net_type_string)
    if net_type_string != CNN_TYPE_STRING:
        used_custom_cnn_loss = False

    error_checking.assert_is_boolean(used_custom_cnn_loss)

    num_examples_per_batch = _check_inference_args(
        predictor_matrix=predictor_matrix,
        num_examples_per_batch=num_examples_per_batch, verbose=verbose
    )

    vector_prediction_matrix = None
    scalar_prediction_matrix = None
    prediction_matrix = None
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

        if net_type_string == CNN_TYPE_STRING:
            if vector_prediction_matrix is None:
                vector_prediction_matrix = this_output[0] + 0.
                scalar_prediction_matrix = this_output[1] + 0.
            else:
                vector_prediction_matrix = numpy.concatenate(
                    (vector_prediction_matrix, this_output[0]), axis=0
                )
                scalar_prediction_matrix = numpy.concatenate(
                    (scalar_prediction_matrix, this_output[1]), axis=0
                )
        else:
            if prediction_matrix is None:
                prediction_matrix = this_output + 0.
            else:
                prediction_matrix = numpy.concatenate(
                    (prediction_matrix, this_output), axis=0
                )

    if verbose:
        print('Have applied {0:s} to all {1:d} examples!'.format(
            net_type_string.upper(), num_examples
        ))

    if used_custom_cnn_loss:
        scalar_prediction_matrix = scalar_prediction_matrix[:, 2:]

    if net_type_string == CNN_TYPE_STRING:
        return [vector_prediction_matrix, scalar_prediction_matrix]

    return [prediction_matrix]


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
