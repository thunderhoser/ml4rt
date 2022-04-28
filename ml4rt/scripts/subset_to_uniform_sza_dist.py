"""Subsets examples to uniform distribution over solar zenith angles."""

import argparse
import numpy
from ml4rt.io import example_io
from ml4rt.utils import example_utils

MIN_ZENITH_ANGLE_RAD = 0.
MAX_ZENITH_ANGLE_RAD = (85. / 180) * numpy.pi
RADIANS_TO_DEGREES = 180. / numpy.pi

INPUT_DIR_ARG_NAME = 'input_example_dir_name'
YEARS_ARG_NAME = 'years'
NUM_BINS_ARG_NAME = 'num_zenith_angle_bins'
OUTPUT_DIR_ARG_NAME = 'output_example_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory, containing examples with pre-existing '
    'distribution of zenith angles.  Files therein will be found '
    '`example_io.find_file` and read by `example_io.read_file`.'
)
YEARS_HELP_STRING = 'Will process examples in these years.'
NUM_BINS_HELP_STRING = (
    'Number of bins for solar zenith angle, ranging from 0 to 85 deg.  This '
    'script will ensure an equal number of examples in each bin.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory, where examples with uniform zenith-angle '
    'distribution will be written.  Files will be written by '
    '`example_io.write_file`, to exact locations in this directory determined '
    'by `example_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + YEARS_ARG_NAME, type=int, nargs='+', required=True,
    help=YEARS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_BINS_ARG_NAME, type=int, required=False, default=9,
    help=NUM_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_example_dir_name, years, num_zenith_angle_bins,
         output_example_dir_name):
    """Subsets examples to uniform distribution over solar zenith angles.

    This is effectively the main method.

    :param input_example_dir_name: See documentation at top of file.
    :param years: Same.
    :param num_zenith_angle_bins: Same.
    :param output_example_dir_name: Same.
    """

    years = numpy.unique(years)
    input_example_file_names = []

    for this_year in years:
        input_example_file_names += example_io.find_files_one_year(
            directory_name=input_example_dir_name, year=this_year,
            raise_error_if_missing=True
        )

    edge_zenith_angles_rad = numpy.linspace(
        MIN_ZENITH_ANGLE_RAD, MAX_ZENITH_ANGLE_RAD,
        num=num_zenith_angle_bins + 1, dtype=float
    )
    edge_zenith_angles_rad[-1] = numpy.inf
    min_zenith_angles_rad = edge_zenith_angles_rad[:-1]
    max_zenith_angles_rad = edge_zenith_angles_rad[1:]

    num_files = len(input_example_file_names)
    zenith_angles_by_file_rad = [numpy.array([])] * num_files

    for i in range(num_files):
        print('Reading data from: "{0:s}"...'.format(
            input_example_file_names[i]
        ))
        this_example_dict = example_io.read_file(
            netcdf_file_name=input_example_file_names[i],
            exclude_summit_greenland=False,
            max_shortwave_heating_k_day01=numpy.inf,
            min_longwave_heating_k_day01=-1 * numpy.inf,
            max_longwave_heating_k_day01=numpy.inf
        )

        zenith_angles_by_file_rad[i] = example_utils.parse_example_ids(
            this_example_dict[example_utils.EXAMPLE_IDS_KEY]
        )[example_utils.ZENITH_ANGLES_KEY]

    all_zenith_angles_rad = numpy.concatenate(zenith_angles_by_file_rad)
    num_examples_by_bin = numpy.full(num_zenith_angle_bins, 0, dtype=int)
    num_examples_by_file_by_bin = num_zenith_angle_bins * [
        numpy.full(num_files, 0, dtype=int)
    ]

    for k in range(num_zenith_angle_bins):
        num_examples_by_bin[k] = numpy.sum(numpy.logical_and(
            all_zenith_angles_rad >= min_zenith_angles_rad[k],
            all_zenith_angles_rad <= max_zenith_angles_rad[k]
        ))

        for i in range(num_files):
            num_examples_by_file_by_bin[k][i] = numpy.sum(numpy.logical_and(
                zenith_angles_by_file_rad[i] >= min_zenith_angles_rad[k],
                zenith_angles_by_file_rad[i] <= max_zenith_angles_rad[k]
            ))

    indices_to_keep_by_file = [numpy.array([], dtype=int)] * num_files

    for k in range(num_zenith_angle_bins):
        these_ratios = (
            float(numpy.min(num_examples_by_bin)) /
            num_examples_by_file_by_bin[k]
        )
        this_num_examples_by_file = numpy.ceil(
            num_examples_by_file_by_bin[k] * these_ratios
        ).astype(int)

        assert (
            numpy.sum(this_num_examples_by_file) >=
            numpy.min(num_examples_by_bin)
        )

        while (
                numpy.sum(this_num_examples_by_file) >
                numpy.min(num_examples_by_bin)
        ):
            this_num_examples_by_file[
                numpy.argmax(this_num_examples_by_file)
            ] -= 1

        for i in range(num_files):
            if this_num_examples_by_file[i] == 0:
                continue

            these_indices = numpy.where(numpy.logical_and(
                zenith_angles_by_file_rad[i] >= min_zenith_angles_rad[k],
                zenith_angles_by_file_rad[i] <= max_zenith_angles_rad[k]
            ))[0]

            numpy.random.shuffle(these_indices)
            these_indices = these_indices[:this_num_examples_by_file[i]]
            indices_to_keep_by_file[i] = numpy.concatenate((
                indices_to_keep_by_file[i], these_indices
            ))

    for i in range(num_files):
        print('\nReading data from: "{0:s}"...'.format(
            input_example_file_names[i]
        ))
        this_example_dict = example_io.read_file(
            netcdf_file_name=input_example_file_names[i],
            exclude_summit_greenland=False,
            max_shortwave_heating_k_day01=numpy.inf,
            min_longwave_heating_k_day01=-1 * numpy.inf,
            max_longwave_heating_k_day01=numpy.inf
        )
        this_example_dict = example_utils.subset_by_index(
            example_dict=this_example_dict,
            desired_indices=indices_to_keep_by_file[i]
        )

        these_zenith_angles_rad = example_utils.parse_example_ids(
            this_example_dict[example_utils.EXAMPLE_IDS_KEY]
        )[example_utils.ZENITH_ANGLES_KEY]

        these_percentile_levels = numpy.linspace(
            0, 100, num=num_zenith_angle_bins + 1, dtype=float
        )
        these_percentiles_rad = numpy.percentile(
            these_zenith_angles_rad, these_percentile_levels
        )

        for k in range(num_zenith_angle_bins + 1):
            print((
                '{0:.4f}th percentile of zenith angles being kept = {1:.4f} deg'
            ).format(
                these_percentile_levels[k],
                these_percentiles_rad[k] * RADIANS_TO_DEGREES
            ))

        this_output_file_name = example_io.find_file(
            directory_name=output_example_dir_name,
            year=example_io.file_name_to_year(input_example_file_names[i]),
            year_part_number=example_io.file_name_to_year_part(
                input_example_file_names[i]
            ),
            raise_error_if_missing=False
        )

        print('Writing {0:d} examples to: "{1:s}"...'.format(
            len(this_example_dict[example_utils.EXAMPLE_IDS_KEY]),
            this_output_file_name
        ))
        example_io.write_file(
            example_dict=this_example_dict,
            netcdf_file_name=this_output_file_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_example_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        years=numpy.array(getattr(INPUT_ARG_OBJECT, YEARS_ARG_NAME), dtype=int),
        num_zenith_angle_bins=getattr(INPUT_ARG_OBJECT, NUM_BINS_ARG_NAME),
        output_example_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
