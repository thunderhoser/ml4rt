"""Plots predictor comparison for Dustin Swales.

Specifically, this script plots two sets of spaghetti lines: one for predictors
in the training data, one for Dustin's predictors.  The goal is to figure out
what's going wrong with the longwave NN in Fortran land.
"""

import os
import sys
import argparse
import numpy
import pandas
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
import matplotlib.colors

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import file_system_utils
import example_io
import example_utils
import profile_plotting

MICROGRAMS_TO_KG = 1e-9

GRID_HEIGHTS_M_AGL = numpy.array([
    21, 44, 68, 93, 120, 149, 179, 212, 246, 282, 321, 361, 405, 450, 499, 550,
    604, 661, 722, 785, 853, 924, 999, 1078, 1161, 1249, 1342, 1439, 1542, 1649,
    1762, 1881, 2005, 2136, 2272, 2415, 2564, 2720, 2882, 3051, 3228, 3411,
    3601, 3798, 4002, 4214, 4433, 4659, 4892, 5132, 5379, 5633, 5894, 6162,
    6436, 6716, 7003, 7296, 7594, 7899, 8208, 8523, 8842, 9166, 9494, 9827,
    10164, 10505, 10849, 11198, 11550, 11906, 12266, 12630, 12997, 13368, 13744,
    14123, 14506, 14895, 15287, 15686, 16090, 16501, 16920, 17350, 17791, 18246,
    18717, 19205, 19715, 20249, 20809, 21400, 22022, 22681, 23379, 24119, 24903,
    25736, 26619, 27558, 28556, 29616, 30743, 31940, 33211, 34566, 36012, 37560,
    39218, 40990, 42882, 44899, 47042, 49299, 51644, 54067, 56552, 59089, 61677,
    64314, 67001, 69747, 72521, 75256, 77803
], dtype=float)

DUSTIN_FIELD_NAMES = [
    example_utils.ZENITH_ANGLE_NAME,
    example_utils.SURFACE_TEMPERATURE_NAME,
    example_utils.SURFACE_EMISSIVITY_NAME,
    example_utils.PRESSURE_NAME,
    example_utils.TEMPERATURE_NAME,
    example_utils.SPECIFIC_HUMIDITY_NAME,
    example_utils.RELATIVE_HUMIDITY_NAME,
    example_utils.LIQUID_WATER_CONTENT_NAME,
    example_utils.ICE_WATER_CONTENT_NAME,
    example_utils.LIQUID_WATER_PATH_NAME,
    example_utils.ICE_WATER_PATH_NAME,
    example_utils.WATER_VAPOUR_PATH_NAME,
    example_utils.UPWARD_LIQUID_WATER_PATH_NAME,
    example_utils.UPWARD_ICE_WATER_PATH_NAME,
    example_utils.UPWARD_WATER_VAPOUR_PATH_NAME,
    example_utils.LIQUID_EFF_RADIUS_NAME,
    example_utils.ICE_EFF_RADIUS_NAME,
    example_utils.O3_MIXING_RATIO_NAME,
    example_utils.CO2_CONCENTRATION_NAME,
    example_utils.CH4_CONCENTRATION_NAME,
    example_utils.N2O_CONCENTRATION_NAME,
    example_utils.HEIGHT_NAME,
    example_utils.HEIGHT_THICKNESS_NAME,
    example_utils.PRESSURE_THICKNESS_NAME
]

DUSTIN_FIRST_COLUMN_INDICES = numpy.concatenate([
    numpy.array([1], dtype=int),
    numpy.linspace(6, 236, num=24, dtype=int)
]) - 1

DUSTIN_LAST_COLUMN_INDICES = numpy.linspace(6, 246, num=25, dtype=int) - 1

TRAINING_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
TRAINING_COLOUR = matplotlib.colors.to_rgba(c=TRAINING_COLOUR, alpha=0.5)
DUSTIN_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255

FIGURE_RESOLUTION_DPI = 300

TRAINING_DIR_ARG_NAME = 'input_training_example_dir_name'
NUM_TRAINING_EXAMPLES_ARG_NAME = 'num_training_examples'
DUSTIN_FILE_ARG_NAME = 'input_dustin_file_name'
PREDICTORS_ARG_NAME = 'predictor_names'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

TRAINING_DIR_HELP_STRING = (
    'Path to directory with training examples.  Files therein will be found by '
    '`example_io.find_file` and read by `example_io.read_file`.'
)
NUM_TRAINING_EXAMPLES_HELP_STRING = (
    'Number of training examples to plot.  We don''t want too many spaghetti '
    'lines.'
)
DUSTIN_FILE_HELP_STRING = (
    'Path to file with predictors from Dustin.  This will be read by the '
    'method `_read_dustin_file` found in this script.'
)
PREDICTORS_HELP_STRING = (
    'List of predictor names -- this script will create one plot for every '
    'predictor.  Each name must be accepted by '
    '`example_utils.check_field_name`.'
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + TRAINING_DIR_ARG_NAME, type=str, required=True,
    help=TRAINING_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_TRAINING_EXAMPLES_ARG_NAME, type=int, required=True,
    help=NUM_TRAINING_EXAMPLES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DUSTIN_FILE_ARG_NAME, type=str, required=True,
    help=DUSTIN_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTORS_ARG_NAME, type=str, nargs='+', required=True,
    help=PREDICTORS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _read_dustin_file(dustin_file_name):
    """Reads predictors from file provided by Dustin.

    :param dustin_file_name: Path to input file.
    :return: example_dict: Dictionary in format returned by
        `example_io.read_file`.
    """

    column_specs_arg = [
        (f, l) for f, l in
        zip(DUSTIN_FIRST_COLUMN_INDICES, DUSTIN_LAST_COLUMN_INDICES)
    ]

    predictor_table_pandas = pandas.read_fwf(
        filepath_or_buffer=dustin_file_name, colspecs=column_specs_arg,
        skiprows=1
    )
    predictor_matrix = predictor_table_pandas.to_numpy()

    good_row_indices = numpy.where(
        numpy.all(predictor_matrix != 'Layer', axis=1)
    )[0]
    predictor_matrix = predictor_matrix[good_row_indices, :]

    good_row_indices = numpy.where(
        numpy.all(predictor_matrix != '(K)', axis=1)
    )[0]
    predictor_matrix = predictor_matrix[good_row_indices, :]
    predictor_matrix = predictor_matrix.astype(float)[:, 1:]

    o3_index = DUSTIN_FIELD_NAMES.index(example_utils.O3_MIXING_RATIO_NAME)
    predictor_matrix[:, o3_index] *= MICROGRAMS_TO_KG
    predictor_matrix = predictor_matrix.reshape(
        -1, len(GRID_HEIGHTS_M_AGL), len(DUSTIN_FIELD_NAMES)
    )

    scalar_indices = numpy.where(numpy.isin(
        element=numpy.array(DUSTIN_FIELD_NAMES),
        test_elements=numpy.array(example_utils.ALL_SCALAR_PREDICTOR_NAMES)
    ))[0]
    vector_indices = numpy.where(numpy.isin(
        element=numpy.array(DUSTIN_FIELD_NAMES),
        test_elements=numpy.array(example_utils.ALL_VECTOR_PREDICTOR_NAMES)
    ))[0]
    num_examples = predictor_matrix.shape[0]

    example_dict = {
        example_utils.SCALAR_PREDICTOR_NAMES_KEY:
            [DUSTIN_FIELD_NAMES[k] for k in scalar_indices],
        example_utils.SCALAR_PREDICTOR_VALS_KEY:
            predictor_matrix[..., 0, scalar_indices],
        example_utils.VECTOR_PREDICTOR_NAMES_KEY:
            [DUSTIN_FIELD_NAMES[k] for k in vector_indices],
        example_utils.VECTOR_PREDICTOR_VALS_KEY:
            predictor_matrix[..., 0, vector_indices],
        example_utils.VALID_TIMES_KEY:
            numpy.linspace(0, num_examples - 1, num=num_examples, dtype=int),
        example_utils.HEIGHTS_KEY: GRID_HEIGHTS_M_AGL
    }

    for this_predictor_name in DUSTIN_FIELD_NAMES:
        these_values = example_utils.get_field_from_dict(
            example_dict=example_dict, field_name=this_predictor_name
        )
        print((
            'Min/median/mean/max {0:s} values from Dustin file = '
            '{1:.8f}, {2:.8f}, {3:.8f}, {4:.8f}'
        ).format(
            this_predictor_name,
            numpy.min(these_values),
            numpy.median(these_values),
            numpy.mean(these_values),
            numpy.max(these_values)
        ))

    return example_dict


def _plot_spaghetti_1dataset_1predictor(
        example_dict, predictor_name, line_colour, figure_object, axes_object):
    """Plots spaghetti lines for one predictor variable in one dataset.

    :param example_dict: Dictionary in format returned by
        `example_io.read_file`.
    :param predictor_name: Name of predictor variable.
    :param line_colour: Line colour.
    :param figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).  If None, this method will create a new
        figure.
    :param axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).  If None, this method will
        create a new figure.
    :return: figure_object: See input documentation.
    :return: axes_object: See input documentation.
    """

    predictor_matrix = example_utils.get_field_from_dict(
        example_dict=example_dict, field_name=predictor_name
    )
    if len(predictor_matrix.shape) == 1:
        predictor_matrix = numpy.expand_dims(predictor_matrix, axis=-1)
        predictor_matrix = numpy.repeat(
            predictor_matrix, axis=-1, repeats=len(GRID_HEIGHTS_M_AGL)
        )

    num_examples = predictor_matrix.shape[0]

    for i in range(num_examples):
        # this_colour = numpy.random.uniform(low=0., high=1., size=3)

        figure_object, axes_object = profile_plotting.plot_one_variable(
            values=predictor_matrix[i, :],
            heights_m_agl=example_dict[example_utils.HEIGHTS_KEY],
            use_log_scale=True,
            line_colour=line_colour,
            figure_object=figure_object
        )

    axes_object.set_xlabel(predictor_name)
    return figure_object, axes_object


def _run(training_example_dir_name, num_training_examples, dustin_file_name,
         predictor_names, output_dir_name):
    """Plots predictor comparison for Dustin Swales.

    This is effectively the main method.

    :param training_example_dir_name: See documentation at top of this script.
    :param num_training_examples: Same.
    :param dustin_file_name: Same.
    :param predictor_names: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    training_example_file_names = example_io.find_many_files(
        directory_name=training_example_dir_name,
        first_time_unix_sec=
        time_conversion.string_to_unix_sec('2015-01-01', '%Y-%m-%d'),
        last_time_unix_sec=
        time_conversion.string_to_unix_sec('2025-01-01', '%Y-%m-%d'),
        raise_error_if_any_missing=False,
        raise_error_if_all_missing=True
    )

    training_example_dicts = []

    for this_file_name in training_example_file_names:
        print('Reading data from file: "{0:s}"...'.format(this_file_name))
        training_example_dicts.append(
            example_io.read_file(this_file_name)
        )

    training_example_dict = example_utils.concat_examples(
        training_example_dicts
    )

    del training_example_dicts
    num_examples_found = len(
        training_example_dict[example_utils.EXAMPLE_IDS_KEY]
    )

    if num_examples_found > num_training_examples:
        all_indices = numpy.linspace(
            0, num_examples_found - 1, num=num_examples_found, dtype=int
        )
        desired_indices = numpy.random.choice(
            all_indices, size=num_training_examples, replace=False
        )
        training_example_dict = example_utils.subset_by_index(
            example_dict=training_example_dict, desired_indices=desired_indices
        )

    print('Reading data from: "{0:s}"...'.format(dustin_file_name))
    dustin_example_dict = _read_dustin_file(dustin_file_name)

    for this_predictor_name in predictor_names:
        figure_object, axes_object = _plot_spaghetti_1dataset_1predictor(
            example_dict=training_example_dict,
            predictor_name=this_predictor_name,
            line_colour=TRAINING_COLOUR,
            figure_object=None,
            axes_object=None
        )

        figure_object, axes_object = _plot_spaghetti_1dataset_1predictor(
            example_dict=dustin_example_dict,
            predictor_name=this_predictor_name,
            line_colour=DUSTIN_COLOUR,
            figure_object=figure_object,
            axes_object=axes_object
        )

        output_file_name = '{0:s}/{1:s}.jpg'.format(
            output_dir_name, this_predictor_name
        )

        print('Saving figure to: "{0:s}"...'.format(output_file_name))
        figure_object.savefig(
            output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
            bbox_inches='tight'
        )
        pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        training_example_dir_name=getattr(
            INPUT_ARG_OBJECT, TRAINING_DIR_ARG_NAME
        ),
        num_training_examples=getattr(
            INPUT_ARG_OBJECT, NUM_TRAINING_EXAMPLES_ARG_NAME
        ),
        dustin_file_name=getattr(INPUT_ARG_OBJECT, DUSTIN_FILE_ARG_NAME),
        predictor_names=getattr(INPUT_ARG_OBJECT, PREDICTORS_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
