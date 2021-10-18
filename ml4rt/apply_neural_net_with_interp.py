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
ORIG_GRID_EXAMPLE_DIR_ARG_NAME = 'input_orig_grid_example_dir_name'
NEW_GRID_EXAMPLE_DIR_ARG_NAME = 'input_new_grid_example_dir_name'
NEW_GRID_NORM_FILE_ARG_NAME = 'input_new_grid_norm_file_name'
HALF_WINDOW_SIZE_ARG_NAME = 'half_window_size_for_interp_px'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

MODEL_FILE_HELP_STRING = (
    'Path to trained model.  Will be read by `neural_net.read_model`.'
)
ORIG_GRID_EXAMPLE_DIR_HELP_STRING = (
    'Name of directory with data examples on original grid.  Files therein will'
    ' be found by `example_io.find_file` and read by `example_io.read_file`.'
)
NEW_GRID_EXAMPLE_DIR_HELP_STRING = (
    'Name of directory with data examples on new grid.  Files therein will'
    ' be found by `example_io.find_file` and read by `example_io.read_file`.'
)
NEW_GRID_NORM_FILE_HELP_STRING = (
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
    '--' + ORIG_GRID_EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=ORIG_GRID_EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NEW_GRID_EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=NEW_GRID_EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NEW_GRID_NORM_FILE_ARG_NAME, type=str, required=True,
    help=NEW_GRID_NORM_FILE_HELP_STRING
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


def _get_predictions_and_targets(
        model_object, model_metadata_dict, orig_grid_example_dir_name,
        new_grid_example_dir_name, new_grid_norm_file_name, first_time_unix_sec,
        last_time_unix_sec):
    """Returns predicted and target values on both grids.

    :param model_object: Trained neural net (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param model_metadata_dict: Dictionary with metadata, in format returned by
        `neural_net.read_metafile`.
    :param orig_grid_example_dir_name: See documentation at top of file.
    :param new_grid_example_dir_name: Same.
    :param new_grid_norm_file_name: Same.
    :param first_time_unix_sec: Same.
    :param last_time_unix_sec: Same.
    :return: orig_grid_prediction_example_dict: Dictionary with predictions on
        original grid, in format specified by `example_io.read_file`.
    :return: new_grid_target_example_dict: Dictionary with target values on new
        grid, in format specified by `example_io.read_file`.
    :return: example_id_strings: 1-D list of example IDs.
    """

    print('Reading normalization params from: "{0:s}"...'.format(
        new_grid_norm_file_name
    ))
    new_grid_norm_example_dict = example_io.read_file(new_grid_norm_file_name)

    net_type_string = model_metadata_dict[neural_net.NET_TYPE_KEY]
    d = copy.deepcopy(model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY])

    d[neural_net.EXAMPLE_DIRECTORY_KEY] = orig_grid_example_dir_name
    d[neural_net.FIRST_TIME_KEY] = first_time_unix_sec
    d[neural_net.LAST_TIME_KEY] = last_time_unix_sec

    predictor_matrix, _, example_id_strings = neural_net.create_data(
        option_dict=d, net_type_string=net_type_string,
        exclude_summit_greenland=True
    )
    print(SEPARATOR_STRING)

    d[neural_net.EXAMPLE_DIRECTORY_KEY] = new_grid_example_dir_name
    d[neural_net.HEIGHTS_KEY] = (
        new_grid_norm_example_dict[example_utils.HEIGHTS_KEY]
    )
    d[neural_net.SCALAR_PREDICTOR_NAMES_KEY] = None
    d[neural_net.VECTOR_PREDICTOR_NAMES_KEY] = [example_utils.HEIGHT_NAME]
    d[neural_net.NORMALIZATION_FILE_KEY] = None
    d[neural_net.PREDICTOR_NORM_TYPE_KEY] = None
    d[neural_net.VECTOR_TARGET_NORM_TYPE_KEY] = None
    d[neural_net.SCALAR_TARGET_NORM_TYPE_KEY] = None

    height_matrix_m_agl = None

    try:
        height_matrix_m_agl, new_grid_target_array, new_grid_id_strings = (
            neural_net.create_data(
                option_dict=d, net_type_string=net_type_string,
                exclude_summit_greenland=True
            )
        )

        height_matrix_m_agl = height_matrix_m_agl[..., 0]
    except:
        d[neural_net.VECTOR_PREDICTOR_NAMES_KEY] = None
        _, new_grid_target_array, new_grid_id_strings = neural_net.create_data(
            option_dict=d, net_type_string=net_type_string,
            exclude_summit_greenland=True
        )

    print(SEPARATOR_STRING)

    new_grid_metadata_dict = example_utils.parse_example_ids(
        new_grid_id_strings
    )
    new_grid_metadata_dict[example_utils.LONGITUDES_KEY] = (
        lng_conversion.convert_lng_positive_in_west(
            new_grid_metadata_dict[example_utils.LONGITUDES_KEY]
        )
    )
    new_grid_id_strings = [
        'lat={0:07.4f}_long={1:08.4f}_zenith-angle-rad={2:.6f}_' \
        'time={3:010d}_atmo={4:1d}_albedo={5:.6f}_' \
        'temp-10m-kelvins={6:010.6f}'.format(
            lat, long, theta, t, f, alpha, t10
        )
        for lat, long, theta, t, f, alpha, t10 in
        zip(
            new_grid_metadata_dict[example_utils.LATITUDES_KEY],
            new_grid_metadata_dict[example_utils.LONGITUDES_KEY],
            new_grid_metadata_dict[example_utils.ZENITH_ANGLES_KEY],
            new_grid_metadata_dict[example_utils.VALID_TIMES_KEY],
            new_grid_metadata_dict[example_utils.STANDARD_ATMO_FLAGS_KEY],
            new_grid_metadata_dict[example_utils.ALBEDOS_KEY],
            new_grid_metadata_dict[example_utils.TEMPERATURES_10M_KEY]
        )
    ]

    orig_grid_metadata_dict = example_utils.parse_example_ids(
        example_id_strings
    )
    orig_grid_metadata_dict[example_utils.LONGITUDES_KEY] = (
        lng_conversion.convert_lng_positive_in_west(
            orig_grid_metadata_dict[example_utils.LONGITUDES_KEY]
        )
    )
    orig_grid_id_strings = [
        'lat={0:07.4f}_long={1:08.4f}_zenith-angle-rad={2:.6f}_' \
        'time={3:010d}_atmo={4:1d}_albedo={5:.6f}_' \
        'temp-10m-kelvins={6:010.6f}'.format(
            lat, long, theta, t, f, alpha, t10
        )
        for lat, long, theta, t, f, alpha, t10 in
        zip(
            orig_grid_metadata_dict[example_utils.LATITUDES_KEY],
            orig_grid_metadata_dict[example_utils.LONGITUDES_KEY],
            orig_grid_metadata_dict[example_utils.ZENITH_ANGLES_KEY],
            orig_grid_metadata_dict[example_utils.VALID_TIMES_KEY],
            orig_grid_metadata_dict[example_utils.STANDARD_ATMO_FLAGS_KEY],
            orig_grid_metadata_dict[example_utils.ALBEDOS_KEY],
            orig_grid_metadata_dict[example_utils.TEMPERATURES_10M_KEY]
        )
    ]

    desired_indices = example_utils.find_examples_with_time_tolerance(
        all_id_strings=orig_grid_id_strings,
        desired_id_strings=new_grid_id_strings,
        time_tolerance_sec=EXAMPLE_MATCHING_TIME_SEC,
        allow_missing=True, verbose=True, allow_non_unique_matches=True
    )
    del new_grid_id_strings
    print(SEPARATOR_STRING)

    these_indices = numpy.where(desired_indices >= 0)[0]
    desired_indices = desired_indices[these_indices]
    new_grid_target_array = [
        a[these_indices, ...] for a in new_grid_target_array
    ]

    predictor_matrix = predictor_matrix[desired_indices, ...]
    example_id_strings = [example_id_strings[k] for k in desired_indices]

    orig_prediction_array = neural_net.apply_model(
        model_object=model_object, predictor_matrix=predictor_matrix,
        num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
        net_type_string=net_type_string, verbose=True
    )

    new_grid_vector_target_matrix = new_grid_target_array[0]
    orig_grid_vector_prediction_matrix = orig_prediction_array[0]

    if len(new_grid_target_array) == 2:
        scalar_target_matrix = new_grid_target_array[1]
        scalar_prediction_matrix = orig_prediction_array[1]
    else:
        scalar_target_matrix = None
        scalar_prediction_matrix = None

    new_grid_metadata_dict = copy.deepcopy(model_metadata_dict)
    d = new_grid_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    d[neural_net.HEIGHTS_KEY] = (
        new_grid_norm_example_dict[example_utils.HEIGHTS_KEY]
    )
    new_grid_metadata_dict[neural_net.TRAINING_OPTIONS_KEY] = d

    new_grid_target_example_dict = apply_nn._targets_numpy_to_dict(
        scalar_target_matrix=scalar_target_matrix,
        vector_target_matrix=new_grid_vector_target_matrix,
        model_metadata_dict=new_grid_metadata_dict
    )

    num_examples = new_grid_vector_target_matrix.shape[0]
    new_grid_target_example_dict.update({
        example_utils.SCALAR_PREDICTOR_NAMES_KEY: [],
        example_utils.SCALAR_PREDICTOR_VALS_KEY:
            numpy.full((num_examples, 0), numpy.nan)
    })

    if height_matrix_m_agl is None:
        num_heights = len(d[neural_net.HEIGHTS_KEY])

        new_grid_target_example_dict.update({
            example_utils.VECTOR_PREDICTOR_NAMES_KEY: [],
            example_utils.VECTOR_PREDICTOR_VALS_KEY:
                numpy.full((num_examples, num_heights, 0), numpy.nan)
        })
    else:
        new_grid_target_example_dict.update({
            example_utils.VECTOR_PREDICTOR_NAMES_KEY:
                [example_utils.HEIGHT_NAME],
            example_utils.VECTOR_PREDICTOR_VALS_KEY:
                numpy.expand_dims(height_matrix_m_agl, axis=-1)
        })

    orig_grid_prediction_example_dict = apply_nn._targets_numpy_to_dict(
        scalar_target_matrix=scalar_prediction_matrix,
        vector_target_matrix=orig_grid_vector_prediction_matrix,
        model_metadata_dict=model_metadata_dict
    )

    return (
        orig_grid_prediction_example_dict, new_grid_target_example_dict,
        example_id_strings
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


def _run(model_file_name, orig_grid_example_dir_name, new_grid_example_dir_name,
         new_grid_norm_file_name, half_window_size_for_interp_px,
         first_time_string, last_time_string, output_file_name):
    """Applies trained neural net and interpolates heating rates to new grid.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param orig_grid_example_dir_name: Same.
    :param new_grid_example_dir_name: Same.
    :param new_grid_norm_file_name: Same.
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
        orig_grid_prediction_example_dict,
        new_grid_target_example_dict,
        example_id_strings
    ) = _get_predictions_and_targets(
        model_object=model_object, model_metadata_dict=model_metadata_dict,
        orig_grid_example_dir_name=orig_grid_example_dir_name,
        new_grid_example_dir_name=new_grid_example_dir_name,
        new_grid_norm_file_name=new_grid_norm_file_name,
        first_time_unix_sec=first_time_unix_sec,
        last_time_unix_sec=last_time_unix_sec
    )
    print(SEPARATOR_STRING)

    orig_grid_prediction_example_dict = _denorm_predictions(
        prediction_example_dict=orig_grid_prediction_example_dict,
        model_metadata_dict=model_metadata_dict
    )
    orig_heating_rate_matrix_k_day01 = example_utils.get_field_from_dict(
        example_dict=orig_grid_prediction_example_dict,
        field_name=example_utils.SHORTWAVE_HEATING_RATE_NAME
    )

    these_names = new_grid_target_example_dict[
        example_utils.VECTOR_PREDICTOR_NAMES_KEY
    ]

    if example_utils.HEIGHT_NAME in these_names:
        height_matrix_m_agl = example_utils.get_field_from_dict(
            example_dict=new_grid_target_example_dict,
            field_name=example_utils.HEIGHT_NAME
        )

        new_heating_rate_matrix_k_day01 = numpy.full(
            height_matrix_m_agl.shape, numpy.nan
        )
        num_examples = new_heating_rate_matrix_k_day01.shape[0]

        for i in range(num_examples):
            new_heating_rate_matrix_k_day01[i, :] = (
                heating_rate_interp.interpolate(
                    orig_heating_rate_matrix_k_day01=
                    orig_heating_rate_matrix_k_day01[[i], :],
                    orig_heights_m_agl=orig_grid_prediction_example_dict[
                        example_utils.HEIGHTS_KEY
                    ],
                    new_heights_m_agl=height_matrix_m_agl[i, :],
                    half_window_size_for_filter_px=
                    half_window_size_for_interp_px
                )[0, :]
            )
    else:
        new_heating_rate_matrix_k_day01 = heating_rate_interp.interpolate(
            orig_heating_rate_matrix_k_day01=orig_heating_rate_matrix_k_day01,
            orig_heights_m_agl=
            orig_grid_prediction_example_dict[example_utils.HEIGHTS_KEY],
            new_heights_m_agl=
            new_grid_target_example_dict[example_utils.HEIGHTS_KEY],
            half_window_size_for_filter_px=half_window_size_for_interp_px
        )

    new_heating_rate_matrix_k_day01[:, -1] = 0.

    print('Writing target (actual) and predicted values to: "{0:s}"...'.format(
        output_file_name
    ))
    prediction_io.write_file(
        netcdf_file_name=output_file_name,
        scalar_target_matrix=
        new_grid_target_example_dict[example_utils.SCALAR_TARGET_VALS_KEY],
        vector_target_matrix=
        new_grid_target_example_dict[example_utils.VECTOR_TARGET_VALS_KEY],
        scalar_prediction_matrix=
        orig_grid_prediction_example_dict[example_utils.SCALAR_TARGET_VALS_KEY],
        vector_prediction_matrix=
        numpy.expand_dims(new_heating_rate_matrix_k_day01, axis=-1),
        heights_m_agl=new_grid_target_example_dict[example_utils.HEIGHTS_KEY],
        example_id_strings=example_id_strings,
        model_file_name=model_file_name,
        normalization_file_name=new_grid_norm_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        orig_grid_example_dir_name=getattr(
            INPUT_ARG_OBJECT, ORIG_GRID_EXAMPLE_DIR_ARG_NAME
        ),
        new_grid_example_dir_name=getattr(
            INPUT_ARG_OBJECT, NEW_GRID_EXAMPLE_DIR_ARG_NAME
        ),
        new_grid_norm_file_name=getattr(
            INPUT_ARG_OBJECT, NEW_GRID_NORM_FILE_ARG_NAME
        ),
        half_window_size_for_interp_px=getattr(
            INPUT_ARG_OBJECT, HALF_WINDOW_SIZE_ARG_NAME
        ),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
