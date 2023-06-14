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

SPECIAL_EXAMPLE_ID_STRINGS = [
    'lat=35.203381_long=065.507812_zenith-angle-rad=2.434978_time=1583863200_atmo=3_albedo=0.000000_temp-10m-kelvins=258.838654',
    'lat=42.115192_long=-112.617188_zenith-angle-rad=2.608163_time=1579413600_atmo=3_albedo=0.000000_temp-10m-kelvins=261.172363',
    'lat=06.033192_long=-72.421875_zenith-angle-rad=0.331678_time=1583258400_atmo=3_albedo=0.000000_temp-10m-kelvins=289.203400',
    'lat=18.451023_long=-103.242188_zenith-angle-rad=0.284675_time=1586541600_atmo=2_albedo=0.000000_temp-10m-kelvins=303.837952',
    'lat=27.588675_long=094.218750_zenith-angle-rad=1.347401_time=1593907200_atmo=2_albedo=0.000000_temp-10m-kelvins=295.067261',
    'lat=27.705824_long=080.742188_zenith-angle-rad=2.265906_time=1590688800_atmo=2_albedo=0.000000_temp-10m-kelvins=298.199463',
    'lat=52.541485_long=-132.773438_zenith-angle-rad=1.105001_time=1600884000_atmo=2_albedo=0.000000_temp-10m-kelvins=287.662720',
    'lat=-14.467946_long=-71.601562_zenith-angle-rad=1.285730_time=1583668800_atmo=3_albedo=0.000000_temp-10m-kelvins=273.746643',
    'lat=12.124959_long=-90.468750_zenith-angle-rad=0.186318_time=1593885600_atmo=2_albedo=0.060321_temp-10m-kelvins=298.856415',
    'lat=-12.827855_long=-178.710938_zenith-angle-rad=0.132107_time=1579392000_atmo=3_albedo=0.060000_temp-10m-kelvins=299.789368',
    'lat=02.987309_long=094.335938_zenith-angle-rad=0.138643_time=1598421600_atmo=2_albedo=0.060106_temp-10m-kelvins=300.415741',
    'lat=14.350797_long=004.101562_zenith-angle-rad=0.095796_time=1587038400_atmo=2_albedo=0.303678_temp-10m-kelvins=313.561981',
    'lat=28.760168_long=097.382812_zenith-angle-rad=0.289731_time=1597557600_atmo=2_albedo=0.114587_temp-10m-kelvins=284.148834',
    'lat=31.103155_long=082.734375_zenith-angle-rad=0.196130_time=1590559200_atmo=2_albedo=0.243435_temp-10m-kelvins=272.624695',
    'lat=-72.925468_long=069.726562_zenith-angle-rad=1.180891_time=1602050400_atmo=5_albedo=0.831088_temp-10m-kelvins=256.176880',
    'lat=-73.628365_long=093.281250_zenith-angle-rad=1.234365_time=1601359200_atmo=4_albedo=0.838981_temp-10m-kelvins=236.605591'
]

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

    example_id_strings = prediction_dict[prediction_io.EXAMPLE_IDS_KEY]
    special_indices = []

    for this_id_string in SPECIAL_EXAMPLE_ID_STRINGS:
        if this_id_string not in example_id_strings:
            continue

        special_indices.append(example_id_strings.index(this_id_string))

    special_indices = numpy.array(special_indices, dtype=int)

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

    num_examples_remaining = num_examples - len(special_indices)
    if num_examples_remaining > 0:
        if len(good_indices) > num_examples_remaining:
            good_indices = numpy.random.choice(
                good_indices, size=num_examples_remaining, replace=False
            )
    else:
        good_indices = numpy.array([], dtype=int)

    good_indices = numpy.unique(numpy.concatenate((
        good_indices, special_indices
    )))

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
