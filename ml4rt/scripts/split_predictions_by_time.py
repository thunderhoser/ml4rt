"""Splits predictions by time of day and time of year.

Time of year is quantified by month, and time of day is quantified by solar
zenith angle.
"""

import argparse
import numpy
from gewittergefahr.gg_utils import error_checking
from ml4rt.io import prediction_io

MIN_ZENITH_ANGLE_RAD = 0.
MAX_ZENITH_ANGLE_RAD = 90.
DEGREES_TO_RADIANS = numpy.pi / 180

INPUT_FILE_ARG_NAME = 'input_prediction_file_name'
ANGLE_SPACING_ARG_NAME = 'zenith_angle_spacing_deg'
OUTPUT_DIR_ARG_NAME = 'output_prediction_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing predictions for all times of day/year.  '
    'Will be read by `prediction_io.read_file`.'
)
ANGLE_SPACING_HELP_STRING = (
    'Bin width for solar zenith angle (degrees).  Keep in mind that solar '
    'zenith angle varies from 0-90 deg.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Temporally split predictions will be written '
    'here by `prediction_io.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + ANGLE_SPACING_ARG_NAME, type=float, required=False, default=10.,
    help=ANGLE_SPACING_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_file_name, zenith_angle_spacing_deg, output_dir_name):
    """Splits predictions by time of day and time of year.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param zenith_angle_spacing_deg: Same.
    :param output_dir_name: Same.
    """

    # Process input args.
    error_checking.assert_is_greater(zenith_angle_spacing_deg, 0.)
    error_checking.assert_is_leq(zenith_angle_spacing_deg, 45.)
    zenith_angle_spacing_rad = zenith_angle_spacing_deg * DEGREES_TO_RADIANS

    num_angle_bins = int(numpy.ceil(
        (MAX_ZENITH_ANGLE_RAD - MIN_ZENITH_ANGLE_RAD) / zenith_angle_spacing_rad
    ))
    bin_edge_angles_rad = numpy.linspace(
        MIN_ZENITH_ANGLE_RAD, MAX_ZENITH_ANGLE_RAD, num=num_angle_bins + 1,
        dtype=float
    )
    bin_min_angles_rad = bin_edge_angles_rad[:-1]
    bin_max_angles_rad = bin_edge_angles_rad[1:]

    # Read data.
    print('Reading data from: "{0:s}"...\n'.format(input_file_name))
    prediction_dict = prediction_io.read_file(input_file_name)

    # Split by solar zenith angle.
    for k in range(num_angle_bins):
        this_prediction_dict = prediction_io.subset_by_zenith_angle(
            prediction_dict=prediction_dict,
            min_zenith_angle_rad=bin_min_angles_rad[k],
            max_zenith_angle_rad=bin_max_angles_rad[k]
        )

        this_output_file_name = prediction_io.find_file(
            directory_name=output_dir_name, zenith_angle_bin=k,
            raise_error_if_missing=False
        )
        print((
            'Writing data for zenith angles [{0:.4f}, {1:.4f}] rad to: '
            '"{2:s}"...'
        ).format(
            bin_min_angles_rad[k], bin_max_angles_rad[k], this_output_file_name
        ))

        prediction_io.write_file(
            netcdf_file_name=this_output_file_name,
            scalar_target_matrix=
            this_prediction_dict[prediction_io.SCALAR_TARGETS_KEY],
            vector_target_matrix=
            this_prediction_dict[prediction_io.VECTOR_TARGETS_KEY],
            scalar_prediction_matrix=
            this_prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY],
            vector_prediction_matrix=
            this_prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY],
            example_id_strings=
            this_prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
            model_file_name=this_prediction_dict[prediction_io.MODEL_FILE_KEY]
        )

    print('\n')

    # Split by month.
    for k in range(1, 13):
        this_prediction_dict = prediction_io.subset_by_month(
            prediction_dict=prediction_dict, desired_month=k
        )

        this_output_file_name = prediction_io.find_file(
            directory_name=output_dir_name, month=k,
            raise_error_if_missing=False
        )
        print('Writing data to: "{0:s}"...'.format(this_output_file_name))

        prediction_io.write_file(
            netcdf_file_name=this_output_file_name,
            scalar_target_matrix=
            this_prediction_dict[prediction_io.SCALAR_TARGETS_KEY],
            vector_target_matrix=
            this_prediction_dict[prediction_io.VECTOR_TARGETS_KEY],
            scalar_prediction_matrix=
            this_prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY],
            vector_prediction_matrix=
            this_prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY],
            example_id_strings=
            this_prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
            model_file_name=this_prediction_dict[prediction_io.MODEL_FILE_KEY]
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        zenith_angle_spacing_deg=getattr(
            INPUT_ARG_OBJECT, ANGLE_SPACING_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
