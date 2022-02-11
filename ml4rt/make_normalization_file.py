"""Creates normalization file."""

import os
import sys
import argparse
import copy
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import error_checking
import example_io
import example_utils

TIME_FORMAT = '%Y-%m-%d-%H%M%S'

EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
FIRST_TIME_ARG_NAME = 'first_training_time_string'
LAST_TIME_ARG_NAME = 'last_training_time_string'
FRACTION_TO_DELETE_ARG_NAME = 'fraction_to_delete_while_reading'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
OUTPUT_FILE_ARG_NAME = 'output_norm_file_name'

EXAMPLE_DIR_HELP_STRING = (
    'Name of input directory.  Example files therein will be found by '
    '`example_io.find_file` and read by `example_io.read_file`.'
)
TIME_HELP_STRING = (
    'Training time (format "yyyy-mm-dd-HHMMSS").  Only examples in the period '
    '`{0:s}`...`{1:s}` will be used to compute normalization params.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

FRACTION_TO_DELETE_HELP_STRING = (
    'Will delete this fraction of examples while reading, to avoid memory '
    'overload.'
)
NUM_EXAMPLES_HELP_STRING = (
    'Number of examples to use for computing normalization params.  These '
    'examples will be selected randomly.'
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
    '--' + FRACTION_TO_DELETE_ARG_NAME, type=float, required=False, default=0.,
    help=FRACTION_TO_DELETE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=True,
    help=NUM_EXAMPLES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(example_dir_name, first_training_time_string,
         last_training_time_string, fraction_to_delete_while_reading,
         num_examples_to_use, output_file_name):
    """Creates normalization file.

    This is effectively the main method.

    :param example_dir_name: See documentation at top of file.
    :param first_training_time_string: Same.
    :param last_training_time_string: Same.
    :param fraction_to_delete_while_reading: Same.
    :param num_examples_to_use: Same.
    :param output_file_name: Same.
    """

    error_checking.assert_is_greater(num_examples_to_use, 0)

    fraction_to_delete_while_reading = max([
        fraction_to_delete_while_reading, 0.
    ])
    error_checking.assert_is_leq(fraction_to_delete_while_reading, 0.9)

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

    example_dict = dict()

    for this_file_name in example_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_example_dict = example_io.read_file(this_file_name)
        this_example_dict = example_utils.subset_by_time(
            example_dict=this_example_dict,
            first_time_unix_sec=first_training_time_unix_sec,
            last_time_unix_sec=last_training_time_unix_sec
        )[0]

        this_num_examples = len(
            this_example_dict[example_utils.EXAMPLE_IDS_KEY]
        )

        if this_num_examples > 0 and fraction_to_delete_while_reading > 0:
            num_examples_to_keep = int(numpy.ceil(
                fraction_to_delete_while_reading * this_num_examples
            ))
            indices_to_keep = numpy.linspace(
                0, this_num_examples - 1, num=this_num_examples, dtype=int
            )
            indices_to_keep = numpy.random.choice(
                indices_to_keep, size=num_examples_to_keep, replace=False
            )
            this_example_dict = example_utils.subset_by_index(
                example_dict=this_example_dict, desired_indices=indices_to_keep
            )

        if len(example_dict) > 0:
            example_dict = example_utils.concat_examples(
                [example_dict, this_example_dict]
            )
        else:
            example_dict = copy.deepcopy(this_example_dict)

    del this_example_dict

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
        fraction_to_delete_while_reading=getattr(
            INPUT_ARG_OBJECT, FRACTION_TO_DELETE_ARG_NAME
        ),
        num_examples_to_use=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
