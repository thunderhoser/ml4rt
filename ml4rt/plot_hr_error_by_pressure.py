"""Plots heating-rate error as a fcn of pressure, using scatter plot."""

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
import prediction_io
import example_io
import example_utils

TOLERANCE = 1e-6
PASCALS_TO_MB = 0.01

MARKER_SIZE = 1
MARKER_TYPE = 'o'
MARKER_FACE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
MARKER_EDGE_COLOUR = MARKER_FACE_COLOUR
REFERENCE_LINE_COLOUR = numpy.full(3, 152. / 255)

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

FONT_SIZE = 30
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

PREDICTION_FILE_ARG_NAME = 'input_prediction_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_predictor_dir_name'
HEIGHT_ARG_NAME = 'height_m_agl'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

PREDICTION_FILE_HELP_STRING = (
    'Path to file with predicted and actual target values.  Will be read by '
    '`prediction_io.read_file`.'
)
EXAMPLE_DIR_HELP_STRING = (
    'Name of directory with learning examples.  Files therein will be found by '
    '`example_io.find_file` and read by `example_io.read_file`.  Only pressure '
    'will be read from these files.'
)
HEIGHT_HELP_STRING = (
    'Will plot errors for heating rate at this height (metres above ground '
    'level).'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_FILE_ARG_NAME, type=str, required=True,
    help=PREDICTION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + HEIGHT_ARG_NAME, type=int, required=True, help=HEIGHT_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(prediction_file_name, example_dir_name, height_m_agl,
         output_dir_name):
    """Plots heating-rate error as a fcn of pressure, using scatter plot.

    This is effectively the main method.

    :param prediction_file_name: See documentation at top of file.
    :param example_dir_name: Same.
    :param height_m_agl: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(prediction_file_name)
    example_id_strings = prediction_dict[prediction_io.EXAMPLE_IDS_KEY]
    height_diffs_metres = numpy.absolute(
        prediction_dict[prediction_io.HEIGHTS_KEY] - height_m_agl
    )
    height_index = numpy.argmin(height_diffs_metres)

    assert numpy.min(height_diffs_metres) <= TOLERANCE
    assert prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY].shape[-1] == 1

    heating_rate_errors_k_day01 = (
        prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY][
            :, height_index, 0
        ]
        - prediction_dict[prediction_io.VECTOR_TARGETS_KEY][:, height_index, 0]
    )

    valid_times_unix_sec = example_utils.parse_example_ids(example_id_strings)[
        example_utils.VALID_TIMES_KEY
    ]
    example_file_names = example_io.find_many_files(
        directory_name=example_dir_name,
        first_time_unix_sec=numpy.min(valid_times_unix_sec),
        last_time_unix_sec=numpy.max(valid_times_unix_sec),
        raise_error_if_any_missing=False
    )

    all_example_id_strings = []
    all_pressures_pa = numpy.array([])
    all_surface_pressures_pa = numpy.array([])

    for this_file_name in example_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_example_dict = example_io.read_file(
            netcdf_file_name=this_file_name, exclude_summit_greenland=False,
            max_heating_rate_k_day=numpy.inf
        )
        this_example_dict = example_utils.subset_by_field(
            example_dict=this_example_dict,
            field_names=[example_utils.PRESSURE_NAME]
        )

        all_example_id_strings += (
            this_example_dict[example_utils.EXAMPLE_IDS_KEY]
        )
        these_pressures_pa = example_utils.get_field_from_dict(
            example_dict=this_example_dict,
            field_name=example_utils.PRESSURE_NAME,
            height_m_agl=height_m_agl
        )
        these_surface_pressures_pa = example_utils.get_field_from_dict(
            example_dict=this_example_dict,
            field_name=example_utils.PRESSURE_NAME
        )[:, 0]

        all_pressures_pa = numpy.concatenate(
            (all_pressures_pa, these_pressures_pa), axis=0
        )
        all_surface_pressures_pa = numpy.concatenate(
            (all_surface_pressures_pa, these_surface_pressures_pa), axis=0
        )

    desired_indices = example_utils.find_examples(
        all_id_strings=all_example_id_strings,
        desired_id_strings=example_id_strings, allow_missing=False
    )
    pressures_pa = all_pressures_pa[desired_indices]
    surface_pressures_pa = all_surface_pressures_pa[desired_indices]

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    axes_object.scatter(
        x=pressures_pa * PASCALS_TO_MB, y=heating_rate_errors_k_day01,
        s=MARKER_SIZE, c=MARKER_FACE_COLOUR, marker=MARKER_TYPE,
        edgecolors=MARKER_EDGE_COLOUR
    )
    axes_object.plot(
        axes_object.get_xlim(), numpy.full(2, 0.),
        color=REFERENCE_LINE_COLOUR, linestyle='dashed', linewidth=3
    )
    axes_object.set_title(
        'Heating-rate error at {0:d} m AGL'.format(height_m_agl)
    )
    axes_object.set_xlabel('Pressure (mb) at {0:d} m AGL'.format(height_m_agl))
    axes_object.set_ylabel(r'Heating-rate error (K day$^{-1}$)')

    output_file_name = '{0:s}/scatter_plot_by_layer_pressure.jpg'.format(
        output_dir_name
    )
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    axes_object.scatter(
        x=surface_pressures_pa * PASCALS_TO_MB, y=heating_rate_errors_k_day01,
        s=MARKER_SIZE, c=MARKER_FACE_COLOUR, marker=MARKER_TYPE,
        edgecolors=MARKER_EDGE_COLOUR
    )
    axes_object.plot(
        axes_object.get_xlim(), numpy.full(2, 0.),
        color=REFERENCE_LINE_COLOUR, linestyle='dashed', linewidth=3
    )
    axes_object.set_title(
        'Heating-rate error at {0:d} m AGL'.format(height_m_agl)
    )
    axes_object.set_xlabel('Surface pressure (mb)')
    axes_object.set_ylabel(r'Heating-rate error (K day$^{-1}$)')

    output_file_name = '{0:s}/scatter_plot_by_surface_pressure.jpg'.format(
        output_dir_name
    )
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_FILE_ARG_NAME
        ),
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        height_m_agl=getattr(INPUT_ARG_OBJECT, HEIGHT_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
