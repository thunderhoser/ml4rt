"""Averages many examples."""

import pickle
import argparse
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from ml4rt.io import example_io

TIME_FORMAT = '%Y-%m-%d-%H%M%S'

INPUT_DIR_ARG_NAME = 'input_example_dir_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
USE_PMM_ARG_NAME = 'use_pmm'
MAX_PERCENTILE_ARG_NAME = 'max_pmm_percentile_level'
STANDARD_ATMO_TYPE_ARG_NAME = 'standard_atmo_enum'
OUTPUT_FILE_ARG_NAME = 'output_pickle_file_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory (with unaveraged examples).  Files therein will be'
    ' found by `example_io.find_file` and read by `example_io.read_file`.'
)
TIME_HELP_STRING = (
    'Time (format "yyyy-mm-dd-HHMMSS").  Only examples between `{0:s}` and '
    '`{1:s}` will be averaged.'
)
USE_PMM_HELP_STRING = (
    'Boolean flag.  If 1 (0), will use probability-matched (arithmetic) means '
    'for vertical profiles.'
)
MAX_PERCENTILE_HELP_STRING = (
    '[used only if `{0:s}` = 1] Max percentile level for probability-matched '
    'means.'
)
STANDARD_ATMO_TYPE_HELP_STRING = (
    'Will average examples in this type of standard atmosphere (must be in list'
    ' `example_io.STANDARD_ATMO_ENUMS`).'
)
OUTPUT_FILE_HELP_STRING = 'Path to output (Pickle) file.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + USE_PMM_ARG_NAME, type=int, required=True, help=USE_PMM_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=False, default=99.,
    help=MAX_PERCENTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + STANDARD_ATMO_TYPE_ARG_NAME, type=int, required=True,
    help=STANDARD_ATMO_TYPE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(example_dir_name, first_time_string, last_time_string, use_pmm,
         max_pmm_percentile_level, standard_atmo_enum, output_file_name):
    """Averages many examples.

    This is effectively the main method.

    :param example_dir_name: See documentation at top of file.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param use_pmm: Same.
    :param max_pmm_percentile_level: Same.
    :param standard_atmo_enum: Same.
    :param output_file_name: Same.
    """

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, TIME_FORMAT
    )
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, TIME_FORMAT
    )
    example_file_names = example_io.find_many_files(
        example_dir_name=example_dir_name,
        first_time_unix_sec=first_time_unix_sec,
        last_time_unix_sec=last_time_unix_sec
    )

    num_files = len(example_file_names)
    example_dicts = [dict()] * num_files

    for i in range(num_files):
        print('Reading data from: "{0:s}"...'.format(example_file_names[i]))
        example_dicts[i] = example_io.read_file(example_file_names[i])

        example_dicts[i] = example_io.subset_by_time(
            example_dict=example_dicts[i],
            first_time_unix_sec=first_time_unix_sec,
            last_time_unix_sec=last_time_unix_sec
        )[0]

        this_orig_num_examples = len(
            example_dicts[i][example_io.VALID_TIMES_KEY]
        )
        example_dicts[i] = example_io.subset_by_standard_atmo(
            example_dict=example_dicts[i], standard_atmo_enum=standard_atmo_enum
        )[0]
        this_num_examples = len(
            example_dicts[i][example_io.VALID_TIMES_KEY]
        )

        print((
            '{0:d} of {1:d} examples have standard-atmosphere type {2:d}.\n'
        ).format(
            this_num_examples, this_orig_num_examples, standard_atmo_enum
        ))

    example_dict = example_io.concat_examples(example_dicts)
    num_examples = len(
        example_dict[example_io.VALID_TIMES_KEY]
    )

    print('Averaging {0:d} examples...'.format(num_examples))
    mean_example_dict = example_io.average_examples(
        example_dict=example_dict, use_pmm=use_pmm,
        max_pmm_percentile_level=max_pmm_percentile_level
    )

    print('Writing mean example to: "{0:s}"...'.format(output_file_name))
    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    pickle_file_handle = open(output_file_name, 'wb')
    pickle.dump(mean_example_dict, pickle_file_handle)
    pickle_file_handle.close()


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        example_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        use_pmm=bool(getattr(INPUT_ARG_OBJECT, USE_PMM_ARG_NAME)),
        max_pmm_percentile_level=getattr(
            INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME
        ),
        standard_atmo_enum=getattr(
            INPUT_ARG_OBJECT, STANDARD_ATMO_TYPE_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
