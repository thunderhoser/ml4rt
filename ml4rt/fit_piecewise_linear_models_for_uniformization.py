"""For every predictor, fits pcwise-linear model to approx uniformization."""

import os
import sys
import argparse

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import example_io
import normalization

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_FILE_ARG_NAME = 'input_normalization_file_name'
NUM_PIECES_ARG_NAME = 'num_linear_pieces'
MAX_ACCEPTABLE_ERROR_ARG_NAME = 'max_acceptable_error'
OUTPUT_FILE_ARG_NAME = 'output_model_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to normalization file, containing a large number of reference values '
    'for each field/height pair.  Will be read by `example_io.read_file`.'
)
NUM_PIECES_HELP_STRING = 'Number of linear pieces in each model.'
MAX_ACCEPTABLE_ERROR_HELP_STRING = (
    'Max acceptable absolute error (for a single transformed value for any '
    'predictor).'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  The fitted models will be written here by '
    '`normalization.write_pw_linear_unif_models`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_PIECES_ARG_NAME, type=int, required=True,
    help=NUM_PIECES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_ACCEPTABLE_ERROR_ARG_NAME, type=float, required=True,
    help=MAX_ACCEPTABLE_ERROR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(normalization_file_name, num_linear_pieces, max_acceptable_error,
         output_file_name):
    """For every predictor, fits pcwise-linear model to approx uniformization.

    This is effectively the main method.

    :param normalization_file_name: See documentation at top of file.
    :param num_linear_pieces: Same.
    :param max_acceptable_error: Same.
    :param output_file_name: Same.
    """

    print('Reading reference values from: "{0:s}"...'.format(
        normalization_file_name
    ))
    training_example_dict = example_io.read_file(normalization_file_name)
    print(SEPARATOR_STRING)

    pw_linear_model_table_xarray = (
        normalization.fit_piecewise_linear_models_for_unif(
            training_example_dict=training_example_dict,
            num_linear_pieces=num_linear_pieces,
            max_acceptable_error=max_acceptable_error
        )
    )
    print(SEPARATOR_STRING)

    print('Writing piecewise-linear models to: "{0:s}"...'.format(
        output_file_name
    ))
    normalization.write_piecewise_linear_models_for_unif(
        pw_linear_model_table_xarray=pw_linear_model_table_xarray,
        netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        normalization_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        num_linear_pieces=getattr(INPUT_ARG_OBJECT, NUM_PIECES_ARG_NAME),
        max_acceptable_error=getattr(
            INPUT_ARG_OBJECT, MAX_ACCEPTABLE_ERROR_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
