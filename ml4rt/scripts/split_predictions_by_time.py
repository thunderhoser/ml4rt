"""Splits predictions by time of day and time of year.

Time of year is quantified by month, and time of day is quantified by solar
zenith angle.
"""

import argparse
import numpy
from gewittergefahr.gg_utils import error_checking
from ml4rt.io import prediction_io

MIN_ZENITH_ANGLE_RAD = 0.
MAX_ZENITH_ANGLE_RAD = numpy.pi / 2

INPUT_FILE_ARG_NAME = 'input_prediction_file_name'
NUM_ANGLE_BINS_ARG_NAME = 'num_zenith_angle_bins'
OUTPUT_DIR_ARG_NAME = 'output_prediction_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing predictions for all times of day/year.  '
    'Will be read by `prediction_io.read_file`.'
)
NUM_ANGLE_BINS_HELP_STRING = 'Number of bins for zenith angle.'
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
    '--' + NUM_ANGLE_BINS_ARG_NAME, type=int, required=False, default=9,
    help=NUM_ANGLE_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_file_name, num_zenith_angle_bins, output_dir_name):
    """Splits predictions by time of day and time of year.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param num_zenith_angle_bins: Same.
    :param output_dir_name: Same.
    """

    # Process input args.
    error_checking.assert_is_geq(num_zenith_angle_bins, 2)
    bin_edge_angles_rad = numpy.linspace(
        MIN_ZENITH_ANGLE_RAD, MAX_ZENITH_ANGLE_RAD,
        num=num_zenith_angle_bins + 1, dtype=float
    )
    bin_min_angles_rad = bin_edge_angles_rad[:-1]
    bin_max_angles_rad = bin_edge_angles_rad[1:]

    # Read data.
    print('Reading data from: "{0:s}"...\n'.format(input_file_name))
    prediction_dict = prediction_io.read_file(input_file_name)

    # Split by solar zenith angle.
    for k in range(num_zenith_angle_bins):
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
            'Writing {0:d} examples (with zenith angles {1:.4f}...{2:.4f} rad) '
            'to: "{3:s}"...'
        ).format(
            len(this_prediction_dict[prediction_io.EXAMPLE_IDS_KEY]),
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
        print('Writing {0:d} examples to: "{1:s}"...'.format(
            len(this_prediction_dict[prediction_io.EXAMPLE_IDS_KEY]),
            this_output_file_name
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


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        num_zenith_angle_bins=getattr(
            INPUT_ARG_OBJECT, NUM_ANGLE_BINS_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
