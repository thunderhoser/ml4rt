"""Finds examples in lapse-rate regime."""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import prediction_io
import plot_errors_by_sfc_temp_and_moisture as plot_errors_by_lapse_rates

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

KG_TO_GRAMS = 1000.

INPUT_FILE_ARG_NAME = 'input_prediction_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
MIN_TEMP_LAPSE_RATE_ARG_NAME = 'min_temp_lapse_rate_k_km01'
MAX_TEMP_LAPSE_RATE_ARG_NAME = 'max_temp_lapse_rate_k_km01'
MIN_HUMIDITY_LAPSE_RATE_ARG_NAME = 'min_humidity_lapse_rate_km01'
MAX_HUMIDITY_LAPSE_RATE_ARG_NAME = 'max_humidity_lapse_rate_km01'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing actual and predicted values.  Will be read '
    'by `prediction_io.write_file`.'
)
EXAMPLE_DIR_HELP_STRING = (
    'Name of directory with example files.  These files will be read to '
    'compute lapse rates for each example.'
)
MIN_TEMP_LAPSE_RATE_HELP_STRING = (
    'Minimum temperature lapse rate (Kelvins per km) desired.'
)
MAX_TEMP_LAPSE_RATE_HELP_STRING = (
    'Max temperature lapse rate (Kelvins per km) desired.'
)
MIN_HUMIDITY_LAPSE_RATE_HELP_STRING = (
    'Minimum humidity lapse rate (kg kg^-1 km^-1) desired.'
)
MAX_HUMIDITY_LAPSE_RATE_HELP_STRING = (
    'Max humidity lapse rate (kg kg^-1 km^-1) desired.'
)
NUM_EXAMPLES_HELP_STRING = (
    'Number of examples to select from lapse-rate regime.'
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
    '--' + MIN_TEMP_LAPSE_RATE_ARG_NAME, type=float, required=True,
    help=MIN_TEMP_LAPSE_RATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_TEMP_LAPSE_RATE_ARG_NAME, type=float, required=True,
    help=MAX_TEMP_LAPSE_RATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_HUMIDITY_LAPSE_RATE_ARG_NAME, type=float, required=True,
    help=MIN_HUMIDITY_LAPSE_RATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_HUMIDITY_LAPSE_RATE_ARG_NAME, type=float, required=True,
    help=MAX_HUMIDITY_LAPSE_RATE_HELP_STRING
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
         min_temp_lapse_rate_k_km01, max_temp_lapse_rate_k_km01,
         min_humidity_lapse_rate_km01, max_humidity_lapse_rate_km01,
         num_examples, output_file_name):
    """Finds examples in lapse-rate regime.

    This is effectively the main method.

    :param input_prediction_file_name: See documentation at top of file.
    :param example_dir_name: Same.
    :param min_temp_lapse_rate_k_km01: Same.
    :param max_temp_lapse_rate_k_km01: Same.
    :param min_humidity_lapse_rate_km01: Same.
    :param max_humidity_lapse_rate_km01: Same.
    :param num_examples: Same.
    :param output_file_name: Same.
    """

    error_checking.assert_is_greater(
        max_temp_lapse_rate_k_km01, min_temp_lapse_rate_k_km01
    )
    error_checking.assert_is_greater(
        max_humidity_lapse_rate_km01, min_humidity_lapse_rate_km01
    )
    error_checking.assert_is_greater(num_examples, 0)

    print('Reading data from: "{0:s}"...'.format(input_prediction_file_name))
    prediction_dict = prediction_io.read_file(input_prediction_file_name)

    temperature_lapse_rates_k_km01 = (
        plot_errors_by_lapse_rates._get_temperature_values(
            prediction_dict=prediction_dict,
            example_dir_name=example_dir_name,
            get_lapse_rates=True
        )
    )

    humidity_lapse_rates_km01 = plot_errors_by_lapse_rates._get_humidity_values(
        prediction_dict=prediction_dict,
        example_dir_name=example_dir_name,
        get_lapse_rates=True
    )

    good_temperature_flags = numpy.logical_and(
        temperature_lapse_rates_k_km01 >= min_temp_lapse_rate_k_km01,
        temperature_lapse_rates_k_km01 <= max_temp_lapse_rate_k_km01
    )
    good_humidity_flags = numpy.logical_and(
        humidity_lapse_rates_km01 >= min_humidity_lapse_rate_km01,
        humidity_lapse_rates_km01 <= max_humidity_lapse_rate_km01
    )
    good_indices = numpy.where(
        numpy.logical_and(good_temperature_flags, good_humidity_flags)
    )[0]

    if len(good_indices) > num_examples:
        good_indices = numpy.random.choice(
            good_indices, size=num_examples, replace=False
        )

    prediction_dict = prediction_io.subset_by_index(
        prediction_dict=prediction_dict, desired_indices=good_indices
    )

    print((
        'Writing examples with temperature lapse rate [{0:.2f}, {1:.2f}] '
        'K km^-1 and humidity lapse rate [{2:.2f}, {3:.2f}] g kg^-1 km^-1 to: '
        '"{4:s}"...'
    ).format(
        min_temp_lapse_rate_k_km01, max_temp_lapse_rate_k_km01,
        KG_TO_GRAMS * min_humidity_lapse_rate_km01,
        KG_TO_GRAMS * max_humidity_lapse_rate_km01,
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
        example_id_strings=prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
        model_file_name=prediction_dict[prediction_io.MODEL_FILE_KEY],
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
        min_temp_lapse_rate_k_km01=getattr(
            INPUT_ARG_OBJECT, MIN_TEMP_LAPSE_RATE_ARG_NAME
        ),
        max_temp_lapse_rate_k_km01=getattr(
            INPUT_ARG_OBJECT, MAX_TEMP_LAPSE_RATE_ARG_NAME
        ),
        min_humidity_lapse_rate_km01=getattr(
            INPUT_ARG_OBJECT, MIN_HUMIDITY_LAPSE_RATE_ARG_NAME
        ),
        max_humidity_lapse_rate_km01=getattr(
            INPUT_ARG_OBJECT, MAX_HUMIDITY_LAPSE_RATE_ARG_NAME
        ),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME),
    )
