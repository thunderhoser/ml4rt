"""Applies trained neural net and interpolates heating rates to new grid."""

import os
import sys
import copy
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import longitude_conversion as lng_conversion
import example_io
import prediction_io
import example_utils
import normalization
import heating_rate_interp
import neural_net
import apply_neural_net as apply_nn

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
NUM_EXAMPLES_PER_BATCH = 500
EXAMPLE_MATCHING_TIME_SEC = 180

MODEL_FILE_ARG_NAME = 'input_model_file_name'
ORIG_EXAMPLE_DIR_ARG_NAME = 'input_orig_unnorm_example_dir_name'
NEW_EXAMPLE_DIR_ARG_NAME = 'input_new_unnorm_example_dir_name'
NEW_NORM_FILE_ARG_NAME = 'input_new_norm_file_name'
HALF_WINDOW_SIZE_ARG_NAME = 'half_window_size_for_interp_px'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

MODEL_FILE_HELP_STRING = (
    'Path to trained model.  Will be read by `neural_net.read_model`.'
)
ORIG_EXAMPLE_DIR_HELP_STRING = (
    'Name of directory with unnormalized data examples on original grid.  '
    'Files therein will be found by `example_io.find_file` and read by '
    '`example_io.read_file`.'
)
NEW_EXAMPLE_DIR_HELP_STRING = (
    'Name of directory with unnormalized data examples on new grid.  Files '
    'therein will be found by `example_io.find_file` and read by '
    '`example_io.read_file`.'
)
NEW_NORM_FILE_HELP_STRING = (
    'Path to normalization file for new grid.  Will be read by '
    '`example_io.read_file`.'
)
HALF_WINDOW_SIZE_HELP_STRING = (
    'Half-window size (pixels) for maximum filter used during interpolation.'
)
TIME_HELP_STRING = (
    'Time (format "yyyy-mm-dd-HHMMSS").  The neural net will be applied only to'
    ' examples from `{0:s}` to `{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

OUTPUT_FILE_HELP_STRING = (
    'Path to output file (will be written by `prediction_io.write_file`).'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + ORIG_EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=ORIG_EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NEW_EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=NEW_EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NEW_NORM_FILE_ARG_NAME, type=str, required=True,
    help=NEW_NORM_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + HALF_WINDOW_SIZE_ARG_NAME, type=int, required=True,
    help=HALF_WINDOW_SIZE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _denorm_predictions(prediction_example_dict, model_metadata_dict):
    """Denormalizes predictions.

    :param prediction_example_dict: Dictionary with predictions, in format
        specified by `example_io.read_file`.
    :param model_metadata_dict: Dictionary with metadata, in format returned by
        `neural_net.read_metafile`.
    :return: prediction_example_dict: Same as input but with dummy predictors
        added *and* denormalized predictions.
    """

    d = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    normalization_file_name = d[neural_net.NORMALIZATION_FILE_KEY]
    vector_target_norm_type_string = d[neural_net.VECTOR_TARGET_NORM_TYPE_KEY]
    scalar_target_norm_type_string = d[neural_net.SCALAR_TARGET_NORM_TYPE_KEY]

    print('Reading normalization params from: "{0:s}"...'.format(
        normalization_file_name
    ))
    norm_example_dict = example_io.read_file(normalization_file_name)
    norm_example_dict = example_utils.subset_by_height(
        example_dict=norm_example_dict, heights_m_agl=d[neural_net.HEIGHTS_KEY]
    )

    this_matrix = example_utils.get_field_from_dict(
        example_dict=prediction_example_dict,
        field_name=example_utils.SHORTWAVE_HEATING_RATE_NAME
    )
    num_examples = this_matrix.shape[0]
    num_heights = this_matrix.shape[1]

    this_dict = {
        example_utils.VECTOR_PREDICTOR_NAMES_KEY: [],
        example_utils.VECTOR_PREDICTOR_VALS_KEY:
            numpy.full((num_examples, num_heights, 0), 0.),
        example_utils.SCALAR_PREDICTOR_NAMES_KEY: [],
        example_utils.SCALAR_PREDICTOR_VALS_KEY:
            numpy.full((num_examples, 0), 0.)
    }
    prediction_example_dict.update(this_dict)

    if vector_target_norm_type_string is not None:
        print('Denormalizing predicted vectors...')
        prediction_example_dict = normalization.denormalize_data(
            new_example_dict=prediction_example_dict,
            training_example_dict=norm_example_dict,
            normalization_type_string=vector_target_norm_type_string,
            min_normalized_value=d[neural_net.VECTOR_TARGET_MIN_VALUE_KEY],
            max_normalized_value=d[neural_net.VECTOR_TARGET_MAX_VALUE_KEY],
            separate_heights=True, apply_to_predictors=False,
            apply_to_vector_targets=True, apply_to_scalar_targets=False
        )

    if scalar_target_norm_type_string is not None:
        print('Denormalizing predicted scalars...')
        prediction_example_dict = normalization.denormalize_data(
            new_example_dict=prediction_example_dict,
            training_example_dict=norm_example_dict,
            normalization_type_string=scalar_target_norm_type_string,
            min_normalized_value=d[neural_net.SCALAR_TARGET_MIN_VALUE_KEY],
            max_normalized_value=d[neural_net.SCALAR_TARGET_MAX_VALUE_KEY],
            separate_heights=True, apply_to_predictors=False,
            apply_to_vector_targets=False, apply_to_scalar_targets=True
        )

    return prediction_example_dict


def _get_data_on_orig_grid(
        model_object, model_metadata_dict, example_dir_name,
        first_time_unix_sec, last_time_unix_sec):
    """Returns data on original grid.

    E = number of examples
    H = number of heights in grid
    S = number of scalar target variables

    :param model_object: Trained neural net (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param model_metadata_dict: Dictionary with metadata, in format returned by
        `neural_net.read_metafile`.
    :param example_dir_name: See documentation at top of file.
    :param first_time_unix_sec: Same.
    :param last_time_unix_sec: Same.
    :return: predicted_hr_matrix_w_m02: E-by-H numpy array of heating rates.
    :return: scalar_prediction_matrix: E-by-S numpy array of scalar predictions
        (or None if there are no scalar target variables).
    :return: example_dict: Dictionary in format specified by
        `example_io.write_file`, containing example IDs and variables required
        to compute air density.
    """

    option_dict = copy.deepcopy(
        model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    )
    option_dict[neural_net.EXAMPLE_DIRECTORY_KEY] = example_dir_name
    option_dict[neural_net.FIRST_TIME_KEY] = first_time_unix_sec
    option_dict[neural_net.LAST_TIME_KEY] = last_time_unix_sec

    predictor_matrix, _, example_id_strings = neural_net.create_data(
        option_dict=option_dict,
        net_type_string=model_metadata_dict[neural_net.NET_TYPE_KEY],
        exclude_summit_greenland=True
    )
    print(SEPARATOR_STRING)

    variable_names = [
        example_utils.SPECIFIC_HUMIDITY_NAME, example_utils.TEMPERATURE_NAME,
        example_utils.PRESSURE_NAME
    ]

    option_dict[neural_net.SCALAR_PREDICTOR_NAMES_KEY] = []
    option_dict[neural_net.VECTOR_PREDICTOR_NAMES_KEY] = variable_names
    option_dict[neural_net.NORMALIZATION_FILE_KEY] = None
    option_dict[neural_net.PREDICTOR_NORM_TYPE_KEY] = None
    option_dict[neural_net.VECTOR_TARGET_NORM_TYPE_KEY] = None
    option_dict[neural_net.SCALAR_TARGET_NORM_TYPE_KEY] = None

    data_matrix, _, new_example_id_strings = neural_net.create_data(
        option_dict=option_dict,
        net_type_string=model_metadata_dict[neural_net.NET_TYPE_KEY],
        exclude_summit_greenland=True
    )
    print(SEPARATOR_STRING)

    assert example_id_strings == new_example_id_strings

    prediction_array = neural_net.apply_model(
        model_object=model_object, predictor_matrix=predictor_matrix,
        num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
        net_type_string=model_metadata_dict[neural_net.NET_TYPE_KEY],
        verbose=True
    )

    this_example_dict = apply_nn._targets_numpy_to_dict(
        scalar_target_matrix=(
            None if len(prediction_array) == 1 else prediction_array[1]
        ),
        vector_target_matrix=prediction_array[0],
        model_metadata_dict=model_metadata_dict
    )
    this_example_dict = _denorm_predictions(
        prediction_example_dict=this_example_dict,
        model_metadata_dict=model_metadata_dict
    )

    predicted_hr_matrix_k_day01 = (
        this_example_dict[example_utils.VECTOR_TARGET_VALS_KEY][..., 0]
    )
    num_examples = predicted_hr_matrix_k_day01.shape[0]

    if len(prediction_array) == 2:
        scalar_prediction_matrix = (
            this_example_dict[example_utils.SCALAR_TARGET_VALS_KEY]
        )
    else:
        scalar_prediction_matrix = None

    example_dict = {
        example_utils.HEIGHTS_KEY: option_dict[neural_net.HEIGHTS_KEY],
        example_utils.EXAMPLE_IDS_KEY: example_id_strings,
        example_utils.VALID_TIMES_KEY: numpy.full(num_examples, 0, dtype=int),
        example_utils.STANDARD_ATMO_FLAGS_KEY:
            numpy.full(num_examples, 0, dtype=int),
        example_utils.SCALAR_PREDICTOR_NAMES_KEY: [],
        example_utils.SCALAR_PREDICTOR_VALS_KEY:
            numpy.full((num_examples, 0), 0.),
        example_utils.SCALAR_TARGET_NAMES_KEY: [],
        example_utils.SCALAR_TARGET_VALS_KEY:
            numpy.full((num_examples, 0), 0.),
        example_utils.VECTOR_PREDICTOR_NAMES_KEY: variable_names,
        example_utils.VECTOR_PREDICTOR_VALS_KEY: data_matrix,
        example_utils.VECTOR_TARGET_NAMES_KEY:
            [example_utils.SHORTWAVE_HEATING_RATE_NAME],
        example_utils.VECTOR_TARGET_VALS_KEY:
            numpy.expand_dims(predicted_hr_matrix_k_day01, axis=-1)
    }

    predicted_hr_matrix_w_m02 = example_utils.heating_rate_to_w_m02(
        example_dict
    )

    return (
        predicted_hr_matrix_k_day01, scalar_prediction_matrix, example_dict
    )


def _get_data_on_new_grid(
        model_metadata_dict, example_dir_name, normalization_file_name,
        first_time_unix_sec, last_time_unix_sec):
    """Returns predictors on new grid.

    E = number of examples
    H = number of heights in grid
    S = number of scalar target variables

    :param model_metadata_dict: Dictionary with metadata, in format returned by
        `neural_net.read_metafile`.
    :param example_dir_name: See documentation at top of file.
    :param normalization_file_name: Same.
    :param first_time_unix_sec: Same.
    :param last_time_unix_sec: Same.
    :return: actual_hr_matrix_w_m02: E-by-H numpy array of heating rates.
    :return: scalar_target_matrix: E-by-S numpy array of scalar target values
        (or None if there are no scalar target variables).
    :return: example_dict: Dictionary in format specified by
        `example_io.write_file`, containing example IDs and variables required
        to compute air density.
    """

    print('Reading normalization params from: "{0:s}"...'.format(
        normalization_file_name
    ))
    training_example_dict = example_io.read_file(normalization_file_name)

    predictor_names = [
        example_utils.SPECIFIC_HUMIDITY_NAME, example_utils.TEMPERATURE_NAME,
        example_utils.PRESSURE_NAME, example_utils.HEIGHT_NAME
    ]

    option_dict = copy.deepcopy(
        model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    )
    option_dict[neural_net.EXAMPLE_DIRECTORY_KEY] = example_dir_name
    option_dict[neural_net.FIRST_TIME_KEY] = first_time_unix_sec
    option_dict[neural_net.LAST_TIME_KEY] = last_time_unix_sec
    option_dict[neural_net.HEIGHTS_KEY] = (
        training_example_dict[example_utils.HEIGHTS_KEY]
    )
    option_dict[neural_net.SCALAR_PREDICTOR_NAMES_KEY] = []
    option_dict[neural_net.VECTOR_PREDICTOR_NAMES_KEY] = predictor_names
    option_dict[neural_net.NORMALIZATION_FILE_KEY] = None
    option_dict[neural_net.PREDICTOR_NORM_TYPE_KEY] = None
    option_dict[neural_net.VECTOR_TARGET_NORM_TYPE_KEY] = None
    option_dict[neural_net.SCALAR_TARGET_NORM_TYPE_KEY] = None

    try:
        predictor_matrix, target_array, example_id_strings = (
            neural_net.create_data(
                option_dict=option_dict,
                net_type_string=model_metadata_dict[neural_net.NET_TYPE_KEY],
                exclude_summit_greenland=True
            )
        )
    except:
        option_dict[neural_net.VECTOR_PREDICTOR_NAMES_KEY] = (
            predictor_names[:-1]
        )

        predictor_matrix, target_array, example_id_strings = (
            neural_net.create_data(
                option_dict=option_dict,
                net_type_string=model_metadata_dict[neural_net.NET_TYPE_KEY],
                exclude_summit_greenland=True
            )
        )

    actual_hr_matrix_k_day01 = target_array[0][..., 0]
    num_examples = actual_hr_matrix_k_day01.shape[0]

    example_dict = {
        example_utils.HEIGHTS_KEY:
            training_example_dict[example_utils.HEIGHTS_KEY],
        example_utils.EXAMPLE_IDS_KEY: example_id_strings,
        example_utils.VALID_TIMES_KEY:
            numpy.full(num_examples, 0, dtype=int),
        example_utils.STANDARD_ATMO_FLAGS_KEY:
            numpy.full(num_examples, 0, dtype=int),
        example_utils.SCALAR_PREDICTOR_NAMES_KEY: [],
        example_utils.SCALAR_PREDICTOR_VALS_KEY:
            numpy.full((num_examples, 0), 0.),
        example_utils.SCALAR_TARGET_NAMES_KEY: [],
        example_utils.SCALAR_TARGET_VALS_KEY:
            numpy.full((num_examples, 0), 0.),
        example_utils.VECTOR_PREDICTOR_NAMES_KEY: predictor_names,
        example_utils.VECTOR_PREDICTOR_VALS_KEY: predictor_matrix,
        example_utils.VECTOR_TARGET_NAMES_KEY:
            [example_utils.SHORTWAVE_HEATING_RATE_NAME],
        example_utils.VECTOR_TARGET_VALS_KEY:
            numpy.expand_dims(actual_hr_matrix_k_day01, axis=-1)
    }

    if (
            example_utils.HEIGHT_NAME in
            option_dict[neural_net.VECTOR_PREDICTOR_NAMES_KEY]
    ):
        actual_hr_matrix_w_m02 = numpy.full(
            actual_hr_matrix_k_day01.shape, numpy.nan
        )

        for i in range(num_examples):
            this_example_dict = {
                example_utils.HEIGHTS_KEY:
                    training_example_dict[example_utils.HEIGHTS_KEY],
                example_utils.EXAMPLE_IDS_KEY: [example_id_strings[i]],
                example_utils.VALID_TIMES_KEY: numpy.full(1, 0, dtype=int),
                example_utils.STANDARD_ATMO_FLAGS_KEY:
                    numpy.full(1, 0, dtype=int),
                example_utils.SCALAR_PREDICTOR_NAMES_KEY: [],
                example_utils.SCALAR_PREDICTOR_VALS_KEY: numpy.full((1, 0), 0.),
                example_utils.SCALAR_TARGET_NAMES_KEY: [],
                example_utils.SCALAR_TARGET_VALS_KEY: numpy.full((1, 0), 0.),
                example_utils.VECTOR_PREDICTOR_NAMES_KEY: predictor_names,
                example_utils.VECTOR_PREDICTOR_VALS_KEY:
                    predictor_matrix[[i], ...],
                example_utils.VECTOR_TARGET_NAMES_KEY:
                    [example_utils.SHORTWAVE_HEATING_RATE_NAME],
                example_utils.VECTOR_TARGET_VALS_KEY:
                    numpy.expand_dims(actual_hr_matrix_k_day01[[0], :], axis=-1)
            }

            actual_hr_matrix_w_m02[i, :] = example_utils.heating_rate_to_w_m02(
                this_example_dict
            )[[0], :]
    else:
        actual_hr_matrix_w_m02 = example_utils.heating_rate_to_w_m02(
            example_dict
        )

    if len(target_array) == 2:
        scalar_target_matrix = target_array[1]
    else:
        scalar_target_matrix = None

    return actual_hr_matrix_k_day01, scalar_target_matrix, example_dict


def _match_examples(orig_example_id_strings, new_example_id_strings):
    """Matches examples between original and new grid.

    :param orig_example_id_strings: 1-D numpy array of example IDs.
    :param new_example_id_strings: 1-D numpy array of example IDs.
    :return: desired_indices: See output doc for
        `example_utils.find_examples_with_time_tolerance`.
    """

    new_metadata_dict = example_utils.parse_example_ids(new_example_id_strings)
    new_metadata_dict[example_utils.LONGITUDES_KEY] = (
        lng_conversion.convert_lng_positive_in_west(
            new_metadata_dict[example_utils.LONGITUDES_KEY]
        )
    )
    new_id_strings_dummy = [
        'lat={0:07.4f}_long={1:08.4f}_zenith-angle-rad={2:.6f}_' \
        'time={3:010d}_atmo={4:1d}_albedo={5:.6f}_' \
        'temp-10m-kelvins={6:010.6f}'.format(
            lat, long, theta, t, f, alpha, t10
        )
        for lat, long, theta, t, f, alpha, t10 in
        zip(
            new_metadata_dict[example_utils.LATITUDES_KEY],
            new_metadata_dict[example_utils.LONGITUDES_KEY],
            new_metadata_dict[example_utils.ZENITH_ANGLES_KEY],
            new_metadata_dict[example_utils.VALID_TIMES_KEY],
            new_metadata_dict[example_utils.STANDARD_ATMO_FLAGS_KEY],
            new_metadata_dict[example_utils.ALBEDOS_KEY],
            new_metadata_dict[example_utils.TEMPERATURES_10M_KEY]
        )
    ]

    orig_metadata_dict = example_utils.parse_example_ids(
        orig_example_id_strings
    )
    orig_metadata_dict[example_utils.LONGITUDES_KEY] = (
        lng_conversion.convert_lng_positive_in_west(
            orig_metadata_dict[example_utils.LONGITUDES_KEY]
        )
    )
    orig_id_strings_dummy = [
        'lat={0:07.4f}_long={1:08.4f}_zenith-angle-rad={2:.6f}_' \
        'time={3:010d}_atmo={4:1d}_albedo={5:.6f}_' \
        'temp-10m-kelvins={6:010.6f}'.format(
            lat, long, theta, t, f, alpha, t10
        )
        for lat, long, theta, t, f, alpha, t10 in
        zip(
            orig_metadata_dict[example_utils.LATITUDES_KEY],
            orig_metadata_dict[example_utils.LONGITUDES_KEY],
            orig_metadata_dict[example_utils.ZENITH_ANGLES_KEY],
            orig_metadata_dict[example_utils.VALID_TIMES_KEY],
            orig_metadata_dict[example_utils.STANDARD_ATMO_FLAGS_KEY],
            orig_metadata_dict[example_utils.ALBEDOS_KEY],
            orig_metadata_dict[example_utils.TEMPERATURES_10M_KEY]
        )
    ]

    return example_utils.find_examples_with_time_tolerance(
        all_id_strings=orig_id_strings_dummy,
        desired_id_strings=new_id_strings_dummy,
        time_tolerance_sec=EXAMPLE_MATCHING_TIME_SEC,
        allow_missing=True, verbose=True, allow_non_unique_matches=True
    )


def _run(model_file_name, orig_example_dir_name, new_example_dir_name,
         new_norm_file_name, half_window_size_for_interp_px,
         first_time_string, last_time_string, output_file_name):
    """Applies trained neural net and interpolates heating rates to new grid.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param orig_example_dir_name: Same.
    :param new_example_dir_name: Same.
    :param new_norm_file_name: Same.
    :param half_window_size_for_interp_px: Same.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param output_file_name: Same.
    """

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, TIME_FORMAT
    )
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, TIME_FORMAT
    )

    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = neural_net.read_model(model_file_name)
    metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(metafile_name))
    model_metadata_dict = neural_net.read_metafile(metafile_name)

    generator_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    assert (
        generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY] ==
        [example_utils.SHORTWAVE_HEATING_RATE_NAME]
    )

    print(SEPARATOR_STRING)

    (
        orig_predicted_hr_matrix_w_m02, orig_scalar_prediction_matrix,
        orig_example_dict
    ) = _get_data_on_orig_grid(
        model_object=model_object, model_metadata_dict=model_metadata_dict,
        example_dir_name=orig_example_dir_name,
        first_time_unix_sec=first_time_unix_sec,
        last_time_unix_sec=last_time_unix_sec
    )
    print(SEPARATOR_STRING)

    new_actual_hr_matrix_w_m02, new_scalar_target_matrix, new_example_dict = (
        _get_data_on_new_grid(
            model_metadata_dict=model_metadata_dict,
            example_dir_name=new_example_dir_name,
            normalization_file_name=new_norm_file_name,
            first_time_unix_sec=first_time_unix_sec,
            last_time_unix_sec=last_time_unix_sec
        )
    )
    print(SEPARATOR_STRING)

    desired_indices = _match_examples(
        orig_example_id_strings=
        orig_example_dict[example_utils.EXAMPLE_IDS_KEY],
        new_example_id_strings=new_example_dict[example_utils.EXAMPLE_IDS_KEY]
    )
    print(SEPARATOR_STRING)

    new_indices_to_keep = numpy.where(desired_indices >= 0)[0]
    orig_indices_to_keep = desired_indices[new_indices_to_keep]

    orig_example_dict = example_utils.subset_by_index(
        example_dict=orig_example_dict, desired_indices=orig_indices_to_keep
    )
    orig_predicted_hr_matrix_w_m02 = (
        orig_predicted_hr_matrix_w_m02[orig_indices_to_keep, :]
    )
    if orig_scalar_prediction_matrix is not None:
        orig_scalar_prediction_matrix = (
            orig_scalar_prediction_matrix[orig_indices_to_keep, :]
        )

    new_example_dict = example_utils.subset_by_index(
        example_dict=new_example_dict, desired_indices=new_indices_to_keep
    )
    new_actual_hr_matrix_w_m02 = (
        new_actual_hr_matrix_w_m02[new_indices_to_keep, :]
    )
    if new_scalar_target_matrix is not None:
        new_scalar_target_matrix = (
            new_scalar_target_matrix[new_indices_to_keep, :]
        )

    if (
            example_utils.HEIGHT_NAME in
            new_example_dict[example_utils.VECTOR_PREDICTOR_NAMES_KEY]
    ):
        new_height_matrix_m_agl = example_utils.get_field_from_dict(
            example_dict=new_example_dict, field_name=example_utils.HEIGHT_NAME
        )

        new_predicted_hr_matrix_w_m02 = numpy.full(
            new_height_matrix_m_agl.shape, numpy.nan
        )
        num_examples = new_predicted_hr_matrix_w_m02.shape[0]

        for i in range(num_examples):
            new_predicted_hr_matrix_w_m02[i, :] = (
                heating_rate_interp.interpolate(
                    orig_heating_rate_matrix_k_day01=
                    orig_predicted_hr_matrix_w_m02[[i], :],
                    orig_heights_m_agl=
                    orig_example_dict[example_utils.HEIGHTS_KEY],
                    new_heights_m_agl=new_height_matrix_m_agl[i, :],
                    half_window_size_for_filter_px=
                    half_window_size_for_interp_px
                )[0, :]
            )

            top_indices = numpy.where(
                new_height_matrix_m_agl[i, :] >
                orig_example_dict[example_utils.HEIGHTS_KEY][-1]
            )[0]

            new_predicted_hr_matrix_w_m02[i, top_indices] = 0.
            new_actual_hr_matrix_w_m02[i, top_indices] = 0.
    else:
        new_predicted_hr_matrix_w_m02 = heating_rate_interp.interpolate(
            orig_heating_rate_matrix_k_day01=orig_predicted_hr_matrix_w_m02,
            orig_heights_m_agl=orig_example_dict[example_utils.HEIGHTS_KEY],
            new_heights_m_agl=new_example_dict[example_utils.HEIGHTS_KEY],
            half_window_size_for_filter_px=half_window_size_for_interp_px
        )

        top_indices = numpy.where(
            new_example_dict[example_utils.HEIGHTS_KEY] >
            orig_example_dict[example_utils.HEIGHTS_KEY][-1]
        )[0]
        new_predicted_hr_matrix_w_m02[:, top_indices] = 0.
        new_actual_hr_matrix_w_m02[:, top_indices] = 0.

    new_predicted_hr_matrix_w_m02[:, -1] = 0.
    new_actual_hr_matrix_w_m02[:, -1] = 0.

    # if (
    #         example_utils.HEIGHT_NAME in
    #         new_example_dict[example_utils.VECTOR_PREDICTOR_NAMES_KEY]
    # ):
    #     new_height_matrix_m_agl = example_utils.get_field_from_dict(
    #         example_dict=new_example_dict, field_name=example_utils.HEIGHT_NAME
    #     )
    #
    #     new_predicted_hr_matrix_k_day01 = numpy.full(
    #         new_predicted_hr_matrix_w_m02.shape, numpy.nan
    #     )
    #     new_actual_hr_matrix_k_day01 = numpy.full(
    #         new_actual_hr_matrix_w_m02.shape, numpy.nan
    #     )
    #     num_examples = new_predicted_hr_matrix_w_m02.shape[0]
    #
    #     for i in range(num_examples):
    #         this_example_dict = example_utils.subset_by_index(
    #             example_dict=copy.deepcopy(new_example_dict),
    #             desired_indices=numpy.array([i], dtype=int)
    #         )
    #         this_example_dict[example_utils.HEIGHTS_KEY] = (
    #             new_height_matrix_m_agl[i, :]
    #         )
    #
    #         this_example_dict = example_utils.heating_rate_to_k_day01(
    #             example_dict=this_example_dict,
    #             heating_rate_matrix_w_m02=new_predicted_hr_matrix_w_m02[[i], :]
    #         )
    #         new_predicted_hr_matrix_k_day01[i, :] = (
    #             example_utils.get_field_from_dict(
    #                 example_dict=this_example_dict,
    #                 field_name=example_utils.SHORTWAVE_HEATING_RATE_NAME
    #             )[0, :]
    #         )
    #
    #         this_example_dict = example_utils.heating_rate_to_k_day01(
    #             example_dict=this_example_dict,
    #             heating_rate_matrix_w_m02=new_actual_hr_matrix_w_m02[[i], :]
    #         )
    #         new_actual_hr_matrix_k_day01[i, :] = (
    #             example_utils.get_field_from_dict(
    #                 example_dict=this_example_dict,
    #                 field_name=example_utils.SHORTWAVE_HEATING_RATE_NAME
    #             )[0, :]
    #         )
    # else:
    #     new_example_dict = example_utils.heating_rate_to_k_day01(
    #         example_dict=new_example_dict,
    #         heating_rate_matrix_w_m02=new_predicted_hr_matrix_w_m02
    #     )
    #     new_predicted_hr_matrix_k_day01 = example_utils.get_field_from_dict(
    #         example_dict=new_example_dict,
    #         field_name=example_utils.SHORTWAVE_HEATING_RATE_NAME
    #     )
    #
    #     new_example_dict = example_utils.heating_rate_to_k_day01(
    #         example_dict=new_example_dict,
    #         heating_rate_matrix_w_m02=new_actual_hr_matrix_w_m02
    #     )
    #     new_actual_hr_matrix_k_day01 = example_utils.get_field_from_dict(
    #         example_dict=new_example_dict,
    #         field_name=example_utils.SHORTWAVE_HEATING_RATE_NAME
    #     )

    new_predicted_hr_matrix_k_day01 = new_predicted_hr_matrix_w_m02 + 0.
    new_actual_hr_matrix_k_day01 = new_actual_hr_matrix_w_m02 + 0.

    print(numpy.mean(new_predicted_hr_matrix_w_m02, axis=0))
    print('\n')
    print(numpy.mean(new_actual_hr_matrix_w_m02, axis=0))
    print('\n\n\n')
    print(numpy.mean(new_predicted_hr_matrix_k_day01, axis=0))
    print('\n')
    print(numpy.mean(new_actual_hr_matrix_k_day01, axis=0))
    print('\n\n\n***********************\n\n\n')

    print('Writing target (actual) and predicted values to: "{0:s}"...'.format(
        output_file_name
    ))
    prediction_io.write_file(
        netcdf_file_name=output_file_name,
        scalar_target_matrix=new_scalar_target_matrix,
        vector_target_matrix=
        numpy.expand_dims(new_actual_hr_matrix_k_day01, axis=-1),
        scalar_prediction_matrix=orig_scalar_prediction_matrix,
        vector_prediction_matrix=
        numpy.expand_dims(new_predicted_hr_matrix_k_day01, axis=-1),
        heights_m_agl=new_example_dict[example_utils.HEIGHTS_KEY],
        example_id_strings=new_example_dict[example_utils.EXAMPLE_IDS_KEY],
        model_file_name=model_file_name,
        normalization_file_name=new_norm_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        orig_example_dir_name=getattr(
            INPUT_ARG_OBJECT, ORIG_EXAMPLE_DIR_ARG_NAME
        ),
        new_example_dir_name=getattr(
            INPUT_ARG_OBJECT, NEW_EXAMPLE_DIR_ARG_NAME
        ),
        new_norm_file_name=getattr(
            INPUT_ARG_OBJECT, NEW_NORM_FILE_ARG_NAME
        ),
        half_window_size_for_interp_px=getattr(
            INPUT_ARG_OBJECT, HALF_WINDOW_SIZE_ARG_NAME
        ),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
