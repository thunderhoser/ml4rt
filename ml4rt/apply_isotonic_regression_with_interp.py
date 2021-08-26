"""Applies trained iso-reg model and interpolates heating rates to new grid."""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import longitude_conversion as lng_conversion
import prediction_io
import example_io
import example_utils
import heating_rate_interp
import isotonic_regression
import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

EXAMPLE_MATCHING_TIME_SEC = 180
ZERO_HEATING_HEIGHT_M_AGL = 49999.

INPUT_PREDICTION_FILE_ARG_NAME = 'input_prediction_file_name'
MODEL_FILE_ARG_NAME = 'input_model_file_name'
NEW_GRID_EXAMPLE_DIR_ARG_NAME = 'input_new_grid_example_dir_name'
HALF_WINDOW_SIZE_ARG_NAME = 'half_window_size_for_interp_px'
OUTPUT_PREDICTION_FILE_ARG_NAME = 'output_prediction_file_name'

INPUT_PREDICTION_FILE_HELP_STRING = (
    'Path to file containing model predictions before isotonic regression.  '
    'Will be read by `prediction_io.read_file`.'
)
MODEL_FILE_HELP_STRING = (
    'Path to file with set of trained isotonic-regression models.  Will be read'
    ' by `isotonic_regression.read_file`.'
)
NEW_GRID_EXAMPLE_DIR_HELP_STRING = (
    'Name of directory with data examples on new grid.  Files therein will'
    ' be found by `example_io.find_file` and read by `example_io.read_file`.'
)
HALF_WINDOW_SIZE_HELP_STRING = (
    'Half-window size (pixels) for maximum filter used during interpolation.'
)
OUTPUT_PREDICTION_FILE_HELP_STRING = (
    'Path to output file, containing model predictions after isotonic '
    'regression.  Will be written by `prediction_io.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_PREDICTION_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_PREDICTION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NEW_GRID_EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=NEW_GRID_EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + HALF_WINDOW_SIZE_ARG_NAME, type=int, required=True,
    help=HALF_WINDOW_SIZE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_PREDICTION_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_PREDICTION_FILE_HELP_STRING
)


def _match_examples(prediction_dict, new_grid_example_dir_name):
    """Matches examples on original (model) grid with those on new grid.

    :param prediction_dict: Dictionary with predicted and actual (target)
        values, in format specified by `prediction_io.read_file`.
    :param new_grid_example_dir_name: See documentation at top of file.
    :return: prediction_dict: Same as input, except (a) with only matched
        examples and (b) with target values on new grid.
    """

    # Read metadata for base model (neural net).
    model_file_name = prediction_dict[prediction_io.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)

    training_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    scalar_target_names = (
        training_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY]
    )
    vector_target_names = (
        training_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY]
    )
    assert not training_option_dict[neural_net.OMIT_HEATING_RATE_KEY]
    assert vector_target_names == [example_utils.SHORTWAVE_HEATING_RATE_NAME]

    # Find example IDs for original (model) grid.
    orig_grid_metadata_dict = example_utils.parse_example_ids(
        prediction_dict[prediction_io.EXAMPLE_IDS_KEY]
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

    # Read actual (target) values for new grid.
    new_grid_example_file_names = example_io.find_many_files(
        directory_name=new_grid_example_dir_name,
        first_time_unix_sec=numpy.min(
            orig_grid_metadata_dict[example_utils.VALID_TIMES_KEY]
        ),
        last_time_unix_sec=numpy.max(
            orig_grid_metadata_dict[example_utils.VALID_TIMES_KEY]
        ),
        raise_error_if_all_missing=True
    )

    new_grid_id_strings = []
    new_heights_m_agl = numpy.array([])
    scalar_target_matrix = None
    vector_target_matrix = None

    for this_file_name in new_grid_example_file_names:
        print('Reading data on new grid from: "{0:s}"...'.format(
            this_file_name
        ))
        this_example_dict = example_io.read_file(
            netcdf_file_name=this_file_name,
            exclude_summit_greenland=False, max_heating_rate_k_day=numpy.inf
        )

        new_grid_id_strings += this_example_dict[example_utils.EXAMPLE_IDS_KEY]
        new_heights_m_agl = this_example_dict[example_utils.HEIGHTS_KEY]

        this_scalar_target_matrix = numpy.full(
            (len(new_grid_id_strings), len(scalar_target_names)),
            numpy.nan
        )
        these_dim = (
            len(new_grid_id_strings), len(new_heights_m_agl),
            len(vector_target_names)
        )
        this_vector_target_matrix = numpy.full(these_dim, numpy.nan)

        for j in range(len(scalar_target_names)):
            this_scalar_target_matrix[:, j] = example_utils.get_field_from_dict(
                example_dict=this_example_dict,
                field_name=scalar_target_names[j]
            )

        for j in range(len(vector_target_names)):
            this_vector_target_matrix[..., j] = (
                example_utils.get_field_from_dict(
                    example_dict=this_example_dict,
                    field_name=vector_target_names[j]
                )
            )

        if scalar_target_matrix is None:
            scalar_target_matrix = this_scalar_target_matrix + 0.
            vector_target_matrix = this_vector_target_matrix + 0.
        else:
            scalar_target_matrix = numpy.concatenate(
                (scalar_target_matrix, this_scalar_target_matrix), axis=-1
            )
            vector_target_matrix = numpy.concatenate(
                (vector_target_matrix, this_vector_target_matrix), axis=-1
            )

    # Find example IDs foe new grid.
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

    # Match example IDs between the two grids.
    print(SEPARATOR_STRING)
    desired_indices_new = example_utils.find_examples_with_time_tolerance(
        all_id_strings=new_grid_id_strings,
        desired_id_strings=orig_grid_id_strings,
        time_tolerance_sec=EXAMPLE_MATCHING_TIME_SEC,
        allow_missing=True, verbose=True
    )
    del new_grid_id_strings
    print(SEPARATOR_STRING)

    desired_indices_orig = numpy.where(desired_indices_new >= 0)[0]
    desired_indices_new = desired_indices_new[desired_indices_orig]
    prediction_dict = prediction_io.subset_by_index(
        prediction_dict=prediction_dict, desired_indices=desired_indices_orig
    )

    prediction_dict[prediction_io.SCALAR_TARGETS_KEY] = (
        scalar_target_matrix[desired_indices_new, ...]
    )
    prediction_dict[prediction_io.VECTOR_TARGETS_KEY] = (
        vector_target_matrix[desired_indices_new, ...]
    )
    prediction_dict[prediction_io.HEIGHTS_KEY] = new_heights_m_agl

    return prediction_dict


def _run(input_prediction_file_name, model_file_name, new_grid_example_dir_name,
         half_window_size_for_interp_px, output_prediction_file_name):
    """Applies trained iso-reg model and interpolates heating rates to new grid.

    This is effectively the main method.

    :param input_prediction_file_name: See documentation at top of file.
    :param model_file_name: Same.
    :param new_grid_example_dir_name: Same.
    :param half_window_size_for_interp_px: Same.
    :param output_prediction_file_name: Same.
    :raises: ValueError: if predictions in `input_prediction_file_name` were
        made with isotonic regression.
    """

    print('Reading original predictions from: "{0:s}"...'.format(
        input_prediction_file_name
    ))
    prediction_dict = prediction_io.read_file(input_prediction_file_name)
    prediction_dict = prediction_io.subset_by_index(prediction_dict=prediction_dict, desired_indices=numpy.linspace(0, 999, num=1000, dtype=int))

    if prediction_dict[prediction_io.ISOTONIC_MODEL_FILE_KEY] is not None:
        raise ValueError(
            'Input predictions must be made with base model only (i.e., must '
            'not already include isotonic regression).'
        )

    print('Reading isotonic-regression models from: "{0:s}"...'.format(
        model_file_name
    ))
    scalar_model_objects, vector_model_object_matrix = (
        isotonic_regression.read_file(model_file_name)
    )

    num_scalar_targets = (
        prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY].shape[1]
    )
    this_matrix = (
        None if num_scalar_targets == 0
        else prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY]
    )

    print(SEPARATOR_STRING)
    prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY], this_matrix = (
        isotonic_regression.apply_models(
            orig_vector_prediction_matrix=
            prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY],
            orig_scalar_prediction_matrix=this_matrix,
            scalar_model_objects=scalar_model_objects,
            vector_model_object_matrix=vector_model_object_matrix
        )
    )
    print(SEPARATOR_STRING)

    if num_scalar_targets > 0:
        prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY] = this_matrix

    orig_heights_m_agl = prediction_dict[prediction_io.HEIGHTS_KEY] + 0.
    prediction_dict = _match_examples(
        prediction_dict=prediction_dict,
        new_grid_example_dir_name=new_grid_example_dir_name
    )

    prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY] = (
        heating_rate_interp.interpolate(
            orig_heating_rate_matrix_k_day01=
            prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY][..., 0],
            orig_heights_m_agl=orig_heights_m_agl,
            new_heights_m_agl=prediction_dict[prediction_io.HEIGHTS_KEY],
            half_window_size_for_filter_px=half_window_size_for_interp_px
        )
    )
    prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY] = numpy.expand_dims(
        prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY], axis=-1
    )

    these_indices = numpy.where(
        prediction_dict[prediction_io.HEIGHTS_KEY] >= ZERO_HEATING_HEIGHT_M_AGL
    )[0]
    prediction_dict[example_utils.VECTOR_TARGET_VALS_KEY][..., 0][
        ..., these_indices
    ] = 0.

    print('Writing new predictions to: "{0:s}"...'.format(
        output_prediction_file_name
    ))
    prediction_io.write_file(
        netcdf_file_name=output_prediction_file_name,
        scalar_target_matrix=prediction_dict[prediction_io.SCALAR_TARGETS_KEY],
        vector_target_matrix=prediction_dict[prediction_io.VECTOR_TARGETS_KEY],
        scalar_prediction_matrix=
        prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY],
        vector_prediction_matrix=
        prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY],
        heights_m_agl=prediction_dict[prediction_io.HEIGHTS_KEY],
        example_id_strings=prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
        model_file_name=prediction_dict[prediction_io.MODEL_FILE_KEY],
        isotonic_model_file_name=model_file_name,
        normalization_file_name=
        prediction_dict[prediction_io.NORMALIZATION_FILE_KEY]
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_prediction_file_name=getattr(
            INPUT_ARG_OBJECT, INPUT_PREDICTION_FILE_ARG_NAME
        ),
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        new_grid_example_dir_name=getattr(
            INPUT_ARG_OBJECT, NEW_GRID_EXAMPLE_DIR_ARG_NAME
        ),
        half_window_size_for_interp_px=getattr(
            INPUT_ARG_OBJECT, HALF_WINDOW_SIZE_ARG_NAME
        ),
        output_prediction_file_name=getattr(
            INPUT_ARG_OBJECT, OUTPUT_PREDICTION_FILE_ARG_NAME
        )
    )
