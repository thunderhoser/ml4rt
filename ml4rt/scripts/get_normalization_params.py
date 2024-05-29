"""Computes normalization parameters."""

import argparse
from gewittergefahr.gg_utils import time_conversion
from ml4rt.io import example_io
from ml4rt.utils import example_utils
from ml4rt.utils import normalization

TIME_FORMAT = '%Y-%m-%d-%H%M%S'

EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
FIRST_TIME_ARG_NAME = 'first_training_time_string'
LAST_TIME_ARG_NAME = 'last_training_time_string'
NUM_QUANTILES_ARG_NAME = 'num_quantiles'
OUTPUT_FILE_ARG_NAME = 'output_norm_file_name'

EXAMPLE_DIR_HELP_STRING = (
    'Path to input directory.  Training examples therein will be found by '
    '`example_io.find_file` and read by `example_io.read_file`.'
)
FIRST_TIME_HELP_STRING = (
    'First time (format "yyyy-mm-dd-HHMMSS") in training period.  '
    'Normalization params will be based only on training period.'
)
LAST_TIME_HELP_STRING = (
    'Last time (format "yyyy-mm-dd-HHMMSS") in training period.  '
    'Normalization params will be based only on training period.'
)
NUM_QUANTILES_HELP_STRING = (
    'Number of quantiles to compute for each atomic variable, where one atomic '
    'variable = one variable at one height (if applicable) at one wavelength '
    '(if applicable).'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Normalization params will be written here by '
    '`normalization.write_params`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True,
    help=FIRST_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True,
    help=LAST_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_QUANTILES_ARG_NAME, type=int, required=True,
    help=NUM_QUANTILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(example_dir_name, first_training_time_string,
         last_training_time_string, num_quantiles, output_file_name):
    """Computes normalization parameters.

    This is effectively the main method.

    :param example_dir_name: See documentation at top of this script.
    :param first_training_time_string: Same.
    :param last_training_time_string: Same.
    :param num_quantiles: Same.
    :param output_file_name: Same.
    """

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
        this_example_dict = example_io.read_file(this_file_name)

        this_example_dict = example_utils.subset_by_time(
            example_dict=this_example_dict,
            first_time_unix_sec=first_training_time_unix_sec,
            last_time_unix_sec=last_training_time_unix_sec
        )[0]

        example_dicts.append(this_example_dict)

    del this_example_dict
    example_dict = example_utils.concat_examples(example_dicts)
    del example_dicts

    num_examples = len(example_dict[example_utils.EXAMPLE_IDS_KEY])
    print('Number of training examples = {0:d}'.format(num_examples))

    norm_param_table_xarray = normalization.get_normalization_params(
        example_dict=example_dict, num_quantiles=num_quantiles
    )

    print('Writing normalization params to: "{0:s}"...'.format(
        output_file_name
    ))
    normalization.write_params(
        normalization_param_table_xarray=norm_param_table_xarray,
        netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        first_training_time_string=getattr(
            INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME
        ),
        last_training_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        num_quantiles=getattr(INPUT_ARG_OBJECT, NUM_QUANTILES_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
