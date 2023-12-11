"""Finds examples in AOD/SZA regime.

AOD = aerosol optical depth
SZA = solar zenith angle
"""

import argparse
import numpy
from gewittergefahr.gg_utils import error_checking
from ml4rt.io import prediction_io
from ml4rt.utils import example_utils
from ml4rt.jtech2022 import plot_errors_by_aod_and_sza

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

RADIANS_TO_DEGREES = 180. / numpy.pi

INPUT_FILE_ARG_NAME = 'input_prediction_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
MIN_AOD_ARG_NAME = 'min_aerosol_optical_depth'
MAX_AOD_ARG_NAME = 'max_aerosol_optical_depth'
MIN_ZENITH_ANGLE_ARG_NAME = 'min_zenith_angle_deg'
MAX_ZENITH_ANGLE_ARG_NAME = 'max_zenith_angle_deg'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing actual and predicted values.  Will be read '
    'by `prediction_io.write_file`.'
)
EXAMPLE_DIR_HELP_STRING = (
    'Name of directory with example files.  These files will be read to '
    'compute AOD for each example.'
)
MIN_AOD_HELP_STRING = 'Minimum AOD desired.'
MAX_AOD_HELP_STRING = 'Max AOD desired.'
MIN_ZENITH_ANGLE_HELP_STRING = 'Minimum zenith angle desired.'
MAX_ZENITH_ANGLE_HELP_STRING = 'Max zenith angle desired.'
NUM_EXAMPLES_HELP_STRING = (
    'Number of examples to select from AOD/SZA regime.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  The selected examples will be written here by '
    '`prediction_io.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_AOD_ARG_NAME, type=float, required=True, help=MIN_AOD_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_AOD_ARG_NAME, type=float, required=True, help=MAX_AOD_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_ZENITH_ANGLE_ARG_NAME, type=float, required=True,
    help=MIN_ZENITH_ANGLE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_ZENITH_ANGLE_ARG_NAME, type=float, required=True,
    help=MAX_ZENITH_ANGLE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=True,
    help=NUM_EXAMPLES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(input_prediction_file_name, example_dir_name,
         min_aerosol_optical_depth, max_aerosol_optical_depth,
         min_zenith_angle_deg, max_zenith_angle_deg,
         num_examples, output_file_name):
    """Finds examples in AOD/SZA regime.

    This is effectively the main method.

    :param input_prediction_file_name: See documentation at top of file.
    :param example_dir_name: Same.
    :param min_aerosol_optical_depth: Same.
    :param max_aerosol_optical_depth: Same.
    :param min_zenith_angle_deg: Same.
    :param max_zenith_angle_deg: Same.
    :param num_examples: Same.
    :param output_file_name: Same.
    """

    error_checking.assert_is_geq(min_aerosol_optical_depth, 0.)
    error_checking.assert_is_greater(
        max_aerosol_optical_depth, min_aerosol_optical_depth
    )

    error_checking.assert_is_geq(min_zenith_angle_deg, 0.)
    error_checking.assert_is_leq(max_zenith_angle_deg, 90.)
    error_checking.assert_is_greater(
        max_zenith_angle_deg, min_zenith_angle_deg
    )

    error_checking.assert_is_greater(num_examples, 0)

    print('Reading data from: "{0:s}"...'.format(input_prediction_file_name))
    prediction_dict = prediction_io.read_file(input_prediction_file_name)

    aod_values = plot_errors_by_aod_and_sza._get_aerosol_optical_depths(
        prediction_dict=prediction_dict,
        example_dir_name=example_dir_name
    )
    zenith_angles_rad = example_utils.parse_example_ids(
        prediction_dict[prediction_io.EXAMPLE_IDS_KEY]
    )[example_utils.ZENITH_ANGLES_KEY]

    zenith_angles_deg = RADIANS_TO_DEGREES * zenith_angles_rad

    good_aod_flags = numpy.logical_and(
        aod_values >= min_aerosol_optical_depth,
        aod_values <= max_aerosol_optical_depth
    )
    good_zenith_angle_flags = numpy.logical_and(
        zenith_angles_deg >= min_zenith_angle_deg,
        zenith_angles_deg <= max_zenith_angle_deg
    )
    good_indices = numpy.where(
        numpy.logical_and(good_aod_flags, good_zenith_angle_flags)
    )[0]

    if len(good_indices) > num_examples:
        good_indices = numpy.random.choice(
            good_indices, size=num_examples, replace=False
        )

    prediction_dict = prediction_io.subset_by_index(
        prediction_dict=prediction_dict, desired_indices=good_indices
    )

    print((
        'Writing examples with AOD [{0:.2f}, {1:.2f}] and zenith angle '
        '[{2:.2f}, {3:.2f}] deg to: "{4:s}"...'
    ).format(
        min_aerosol_optical_depth, max_aerosol_optical_depth,
        min_zenith_angle_deg, max_zenith_angle_deg,
        output_file_name
    ))
    prediction_io.write_file(
        netcdf_file_name=output_file_name,
        scalar_target_matrix=prediction_dict[prediction_io.SCALAR_TARGETS_KEY],
        vector_target_matrix=prediction_dict[prediction_io.VECTOR_TARGETS_KEY],
        scalar_prediction_matrix=
        prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY],
        vector_prediction_matrix=
        prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY],
        heights_m_agl=prediction_dict[prediction_io.HEIGHTS_KEY],
        target_wavelengths_metres=
        prediction_dict[prediction_io.TARGET_WAVELENGTHS_KEY],
        example_id_strings=prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
        model_file_name=prediction_dict[prediction_io.MODEL_FILE_KEY],
        isotonic_model_file_name=
        prediction_dict[prediction_io.ISOTONIC_MODEL_FILE_KEY],
        uncertainty_calib_model_file_name=
        prediction_dict[prediction_io.UNCERTAINTY_CALIB_MODEL_FILE_KEY],
        normalization_file_name=
        prediction_dict[prediction_io.NORMALIZATION_FILE_KEY]
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_prediction_file_name=getattr(
            INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME
        ),
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        min_aerosol_optical_depth=getattr(INPUT_ARG_OBJECT, MIN_AOD_ARG_NAME),
        max_aerosol_optical_depth=getattr(INPUT_ARG_OBJECT, MAX_AOD_ARG_NAME),
        min_zenith_angle_deg=getattr(
            INPUT_ARG_OBJECT, MIN_ZENITH_ANGLE_ARG_NAME
        ),
        max_zenith_angle_deg=getattr(
            INPUT_ARG_OBJECT, MAX_ZENITH_ANGLE_ARG_NAME
        ),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME),
    )
