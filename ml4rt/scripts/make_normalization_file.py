"""Creates normalization file."""

import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking
from ml4rt.io import example_io
from ml4rt.utils import example_utils

TIME_FORMAT = '%Y-%m-%d-%H%M%S'

EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
FIRST_TIME_ARG_NAME = 'first_training_time_string'
LAST_TIME_ARG_NAME = 'last_training_time_string'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
TRACE_GAS_NOISE_ARG_NAME = 'trace_gas_noise_stdev_fractional'
ICE_RADIUS_NOISE_ARG_NAME = 'ice_radius_noise_stdev_fractional'
OUTPUT_FILE_ARG_NAME = 'output_norm_file_name'

EXAMPLE_DIR_HELP_STRING = (
    'Name of input directory.  Example files therein will be found by '
    '`example_io.find_file` and read by `example_io.read_file`.'
)
TIME_HELP_STRING = (
    'Training time (format "yyyy-mm-dd-HHMMSS").  Only examples in the period '
    '`{0:s}`...`{1:s}` will be used to compute normalization params.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

NUM_EXAMPLES_HELP_STRING = (
    'Number of examples to use for computing normalization params.  These '
    'examples will be selected randomly.'
)
TRACE_GAS_NOISE_HELP_STRING = (
    'Standard deviation of Gaussian noise for trace-gas concentrations, as a '
    'fraction from 0...1.'
)
ICE_RADIUS_NOISE_HELP_STRING = (
    'Standard deviation of Gaussian noise for effective ice radii, as a '
    'fraction from 0...1.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file, containing normalization params.  Will be written by '
    '`example_io.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True,
    help=TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True,
    help=TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=True,
    help=NUM_EXAMPLES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TRACE_GAS_NOISE_ARG_NAME, type=float, required=True,
    help=TRACE_GAS_NOISE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + ICE_RADIUS_NOISE_ARG_NAME, type=float, required=True,
    help=ICE_RADIUS_NOISE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(example_dir_name, first_training_time_string,
         last_training_time_string, num_examples_to_use,
         trace_gas_noise_stdev_fractional, ice_radius_noise_stdev_fractional,
         output_file_name):
    """Creates normalization file.

    This is effectively the main method.

    :param example_dir_name: See documentation at top of file.
    :param first_training_time_string: Same.
    :param last_training_time_string: Same.
    :param num_examples_to_use: Same.
    :param trace_gas_noise_stdev_fractional: Same.
    :param ice_radius_noise_stdev_fractional: Same.
    :param output_file_name: Same.
    """

    error_checking.assert_is_greater(num_examples_to_use, 0)

    first_training_time_unix_sec = time_conversion.string_to_unix_sec(
        first_training_time_string, TIME_FORMAT
    )
    last_training_time_unix_sec = time_conversion.string_to_unix_sec(
        last_training_time_string, TIME_FORMAT
    )

    example_file_names = example_io.find_many_files(
        directory_name=example_dir_name,
        first_time_unix_sec=first_training_time_unix_sec,
        last_time_unix_sec=last_training_time_unix_sec,
        raise_error_if_any_missing=True
    )

    example_dicts = []

    for this_file_name in example_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        example_dicts.append(
            example_io.read_file(this_file_name)
        )

    example_dict = example_utils.concat_examples(example_dicts)
    del example_dicts

    num_examples = len(example_dict[example_utils.EXAMPLE_IDS_KEY])
    print('Number of examples read = {0:d}'.format(num_examples))

    example_dict = example_utils.subset_by_time(
        example_dict=example_dict,
        first_time_unix_sec=first_training_time_unix_sec,
        last_time_unix_sec=last_training_time_unix_sec
    )[0]

    num_examples = len(example_dict[example_utils.EXAMPLE_IDS_KEY])
    print('Number of examples from {0:s} to {1:s} = {2:d}'.format(
        first_training_time_string, last_training_time_string, num_examples
    ))

    if num_examples_to_use < num_examples:
        desired_indices = numpy.linspace(
            0, num_examples - 1, num=num_examples, dtype=int
        )
        desired_indices = numpy.random.choice(
            desired_indices, size=num_examples_to_use, replace=False
        )
        example_dict = example_utils.subset_by_index(
            example_dict=example_dict, desired_indices=desired_indices
        )

    num_examples = len(example_dict[example_utils.EXAMPLE_IDS_KEY])
    print('Number of examples to use for normalization params = {0:d}'.format(
        num_examples
    ))

    print('Writing normalization params to: "{0:s}"...'.format(
        output_file_name
    ))
    example_io.write_file(
        example_dict=example_dict, netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        first_training_time_string=getattr(
            INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME
        ),
        last_training_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        num_examples_to_use=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        trace_gas_noise_stdev_fractional=getattr(
            INPUT_ARG_OBJECT, TRACE_GAS_NOISE_ARG_NAME
        ),
        ice_radius_noise_stdev_fractional=getattr(
            INPUT_ARG_OBJECT, ICE_RADIUS_NOISE_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
