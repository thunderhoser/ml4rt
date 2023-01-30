"""Makes schematic to show isotonic regression and uncertainty calibration."""

import os
import sys
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import error_checking
import prediction_io
import example_utils

TARGET_VAR_NAME = example_utils.SHORTWAVE_HEATING_RATE_NAME
APPROX_TARGET_HEIGHT_M_AGL = 2000.

BEFORE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
AFTER_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

BASE_FILE_ARG_NAME = 'input_base_prediction_file_name'
ISOTONIC_FILE_ARG_NAME = 'input_isotonic_prediction_file_name'
UNCTY_CALIBRATED_FILE_ARG_NAME = 'input_uncty_calibrated_prediction_file_name'
NUM_EXAMPLES_ARG_NAME = 'num_examples_to_plot'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

BASE_FILE_HELP_STRING = (
    'Path to file with predictions from base model, using no calibration.  '
    'This file will be read by `prediction_io.read_file`.'
)
ISOTONIC_FILE_HELP_STRING = (
    'Path to file with isotonic-regression-corrected predictions from the same '
    'base model, for the same examples.  This file will also be read by '
    '`prediction_io.read_file`.'
)
UNCTY_CALIBRATED_FILE_HELP_STRING = (
    'Path to file with uncertainty-calibration-corrected predictions from the '
    'same base model, for the same examples.  This file will also be read by '
    '`prediction_io.read_file`.'
)
NUM_EXAMPLES_HELP_STRING = (
    'Number of examples to plot.  These will be randomly selected.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + BASE_FILE_ARG_NAME, type=str, required=True,
    help=BASE_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + ISOTONIC_FILE_ARG_NAME, type=str, required=True,
    help=ISOTONIC_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + UNCTY_CALIBRATED_FILE_ARG_NAME, type=str, required=True,
    help=UNCTY_CALIBRATED_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=True,
    help=NUM_EXAMPLES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(base_prediction_file_name, isotonic_prediction_file_name,
         uncty_calibrated_prediction_file_name, num_examples_to_plot,
         output_dir_name):
    """Makes schematic to show isotonic regression and uncertainty calibration.

    This is effectively the main method.

    :param base_prediction_file_name: See documentation at top of file.
    :param isotonic_prediction_file_name: Same.
    :param uncty_calibrated_prediction_file_name: Same.
    :param num_examples_to_plot: Same.
    :param output_dir_name: Same.
    """

    # TODO(thunderhoser): I could do a lot more error-checking to make sure
    # the 3 files are compatible, but this may not be worthwhile, since I will\
    # not use this script often.

    error_checking.assert_is_geq(num_examples_to_plot, 10)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    print('Reading data from: "{0:s}"...'.format(base_prediction_file_name))
    base_prediction_dict = prediction_io.read_file(base_prediction_file_name)

    num_examples_total = len(
        base_prediction_dict[prediction_io.EXAMPLE_IDS_KEY]
    )

    if num_examples_total > num_examples_to_plot:
        desired_indices = numpy.linspace(
            0, num_examples_total - 1, num=num_examples_total, dtype=int
        )
        desired_indices = numpy.random.choice(
            desired_indices, size=num_examples_to_plot, replace=False
        )
        base_prediction_dict = prediction_io.subset_by_index(
            prediction_dict=base_prediction_dict,
            desired_indices=desired_indices
        )

    print('Reading data from: "{0:s}"...'.format(isotonic_prediction_file_name))
    isotonic_prediction_dict = prediction_io.read_file(
        isotonic_prediction_file_name
    )

    desired_indices = example_utils.find_examples(
        all_id_strings=isotonic_prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
        desired_id_strings=base_prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
        allow_missing=False
    )
    isotonic_prediction_dict = prediction_io.subset_by_index(
        prediction_dict=isotonic_prediction_dict,
        desired_indices=desired_indices
    )

    print('Reading data from: "{0:s}"...'.format(
        uncty_calibrated_prediction_file_name
    ))
    uncty_calibrated_prediction_dict = prediction_io.read_file(
        uncty_calibrated_prediction_file_name
    )

    desired_indices = example_utils.find_examples(
        all_id_strings=
        uncty_calibrated_prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
        desired_id_strings=base_prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
        allow_missing=False
    )
    uncty_calibrated_prediction_dict = prediction_io.subset_by_index(
        prediction_dict=uncty_calibrated_prediction_dict,
        desired_indices=desired_indices
    )

    height_index = numpy.argmin(numpy.absolute(
        base_prediction_dict[prediction_io.HEIGHTS_KEY] -
        APPROX_TARGET_HEIGHT_M_AGL
    ))
    var_index = base_prediction_dict[prediction_io.VECTOR_TARGETS_KEY].index(
        TARGET_VAR_NAME
    )
    target_height_m_agl = (
        base_prediction_dict[prediction_io.HEIGHTS_KEY][height_index]
    )

    base_prediction_matrix = base_prediction_dict[
        prediction_io.VECTOR_PREDICTIONS_KEY
    ][:, height_index, var_index, :]

    height_index = example_utils.match_heights(
        heights_m_agl=isotonic_prediction_dict[prediction_io.HEIGHTS_KEY],
        desired_height_m_agl=target_height_m_agl
    )
    var_index = isotonic_prediction_dict[
        prediction_io.VECTOR_TARGETS_KEY
    ].index(TARGET_VAR_NAME)

    isotonic_prediction_matrix = isotonic_prediction_dict[
        prediction_io.VECTOR_PREDICTIONS_KEY
    ][:, height_index, var_index, :]

    height_index = example_utils.match_heights(
        heights_m_agl=
        uncty_calibrated_prediction_dict[prediction_io.HEIGHTS_KEY],
        desired_height_m_agl=target_height_m_agl
    )
    var_index = uncty_calibrated_prediction_dict[
        prediction_io.VECTOR_TARGETS_KEY
    ].index(TARGET_VAR_NAME)

    uncty_calibrated_prediction_matrix = uncty_calibrated_prediction_dict[
        prediction_io.VECTOR_PREDICTIONS_KEY
    ][:, height_index, var_index, :]

    mean_base_predictions = numpy.mean(base_prediction_matrix, axis=-1)
    sort_indices = numpy.argsort(mean_base_predictions)
    mean_base_predictions = mean_base_predictions[sort_indices]
    mean_isotonic_predictions = numpy.mean(isotonic_prediction_matrix, axis=-1)
    mean_isotonic_predictions = mean_isotonic_predictions[sort_indices]

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    num_examples_to_plot = len(mean_base_predictions)
    dummy_indices = numpy.linspace(
        1, num_examples_to_plot, num=num_examples_to_plot, dtype=float
    )

    first_legend_handle = axes_object.plot(
        dummy_indices, mean_base_predictions,
        color=BEFORE_COLOUR, linestyle='solid', linewidth=3,
        marker='o', markersize=12, markerfacecolor=BEFORE_COLOUR,
        markeredgecolor=BEFORE_COLOUR, markeredgewidth=0
    )[0]

    second_legend_handle = axes_object.plot(
        dummy_indices, mean_isotonic_predictions,
        color=BEFORE_COLOUR, linestyle='solid', linewidth=3,
        marker='o', markersize=12, markerfacecolor=BEFORE_COLOUR,
        markeredgecolor=BEFORE_COLOUR, markeredgewidth=0
    )[0]

    legend_handles = [first_legend_handle, second_legend_handle]
    legend_strings = ['Before IR', 'After IR']
    axes_object.legend(
        legend_handles, legend_strings, loc='upper left',
        bbox_to_anchor=(0, 1), fancybox=True, shadow=False,
        facecolor='white', edgecolor='k', framealpha=1., ncol=1
    )

    axes_object.set_ylabel(r'Mean forecast HR (K day$^{-1}$)')
    axes_object.set_xticks([], [])
    axes_object.setxlabel('Data sample')
    axes_object.set_title('Isotonic regression')

    figure_file_name = '{0:s}/isotonic_regression_schematic.jpg'.format(
        output_dir_name
    )
    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # TODO(thunderhoser): This assumes that uncertainty calibration is applied
    # on top of isotonic regression.
    isotonic_predictions_stdev = numpy.stdev(
        isotonic_prediction_matrix, axis=-1, ddof=1
    )
    sort_indices = numpy.argsort(isotonic_predictions_stdev)
    isotonic_predictions_stdev = isotonic_predictions_stdev[sort_indices]
    uncty_calibrated_predictions_stdev = numpy.mean(
        uncty_calibrated_prediction_matrix, axis=-1
    )
    uncty_calibrated_predictions_stdev = (
        uncty_calibrated_predictions_stdev[sort_indices]
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    first_legend_handle = axes_object.plot(
        dummy_indices, isotonic_predictions_stdev,
        color=BEFORE_COLOUR, linestyle='solid', linewidth=3,
        marker='o', markersize=12, markerfacecolor=BEFORE_COLOUR,
        markeredgecolor=BEFORE_COLOUR, markeredgewidth=0
    )[0]

    second_legend_handle = axes_object.plot(
        dummy_indices, uncty_calibrated_predictions_stdev,
        color=BEFORE_COLOUR, linestyle='solid', linewidth=3,
        marker='o', markersize=12, markerfacecolor=BEFORE_COLOUR,
        markeredgecolor=BEFORE_COLOUR, markeredgewidth=0
    )[0]

    legend_handles = [first_legend_handle, second_legend_handle]
    legend_strings = ['Before UC', 'After UC']
    axes_object.legend(
        legend_handles, legend_strings, loc='upper left',
        bbox_to_anchor=(0, 1), fancybox=True, shadow=False,
        facecolor='white', edgecolor='k', framealpha=1., ncol=1
    )

    axes_object.set_ylabel(r'Stdev of forecast HR (K day$^{-1}$)')
    axes_object.set_xticks([], [])
    axes_object.setxlabel('Data sample')
    axes_object.set_title('Uncertainty calibration')

    figure_file_name = '{0:s}/uncertainty_calibration_schematic.jpg'.format(
        output_dir_name
    )
    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        base_prediction_file_name=getattr(INPUT_ARG_OBJECT, BASE_FILE_ARG_NAME),
        isotonic_prediction_file_name=getattr(
            INPUT_ARG_OBJECT, ISOTONIC_FILE_ARG_NAME
        ),
        uncty_calibrated_prediction_file_name=getattr(
            INPUT_ARG_OBJECT, UNCTY_CALIBRATED_FILE_ARG_NAME
        ),
        num_examples_to_plot=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
