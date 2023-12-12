"""Plots results of discard test (error vs. discard fraction)."""

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
import example_utils
import discard_test_utils as dt_utils
import evaluation_plotting
import uq_evaluation_plotting as uq_eval_plotting

ERROR_PROFILE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255

METRES_TO_MICRONS = 1e6

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

INPUT_FILE_ARG_NAME = 'input_file_name'
WAVELENGTHS_ARG_NAME = 'wavelengths_metres'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by `dt_utils.read_results`.'
)
WAVELENGTHS_HELP_STRING = (
    'List of wavelengths.  Will create one set of plots for each.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + WAVELENGTHS_ARG_NAME, type=float, nargs='+', required=False,
    default=[example_utils.DUMMY_BROADBAND_WAVELENGTH_METRES],
    help=WAVELENGTHS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_file_name, wavelengths_metres, output_dir_name):
    """Plots results of discard test (error vs. discard fraction).

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param wavelengths_metres: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    result_table_xarray = dt_utils.read_results(input_file_name)
    rtx = result_table_xarray

    for this_var_name in rtx.coords[dt_utils.SCALAR_FIELD_DIM].values:
        for this_wavelength_metres in wavelengths_metres:
            figure_object, _ = uq_eval_plotting.plot_discard_test(
                result_table_xarray=result_table_xarray,
                target_var_name=this_var_name,
                target_wavelength_metres=this_wavelength_metres
            )

            this_figure_file_name = (
                '{0:s}/discard_test_{1:s}_{2:.2f}microns.jpg'
            ).format(
                output_dir_name,
                this_var_name.replace('_', '-'),
                METRES_TO_MICRONS * this_wavelength_metres
            )

            print('Saving figure to file: "{0:s}"...'.format(
                this_figure_file_name
            ))
            figure_object.savefig(
                this_figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
                pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(figure_object)

    for this_var_name in rtx.coords[dt_utils.AUX_TARGET_FIELD_DIM].values:
        for this_wavelength_metres in wavelengths_metres:
            figure_object, _ = uq_eval_plotting.plot_discard_test(
                result_table_xarray=result_table_xarray,
                target_var_name=this_var_name,
                target_wavelength_metres=this_wavelength_metres
            )

            this_figure_file_name = (
                '{0:s}/discard_test_{1:s}_{2:.2f}microns.jpg'
            ).format(
                output_dir_name,
                this_var_name.replace('_', '-'),
                METRES_TO_MICRONS * this_wavelength_metres
            )

            print('Saving figure to file: "{0:s}"...'.format(
                this_figure_file_name
            ))
            figure_object.savefig(
                this_figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
                pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(figure_object)

    for this_var_name in rtx.coords[dt_utils.VECTOR_FIELD_DIM].values:
        for this_wavelength_metres in wavelengths_metres:
            figure_object, _ = uq_eval_plotting.plot_discard_test(
                result_table_xarray=result_table_xarray,
                target_var_name=this_var_name,
                target_wavelength_metres=this_wavelength_metres
            )

            this_figure_file_name = (
                '{0:s}/discard_test_{1:s}_{2:.2f}microns.jpg'
            ).format(
                output_dir_name,
                this_var_name.replace('_', '-'),
                METRES_TO_MICRONS * this_wavelength_metres
            )

            print('Saving figure to file: "{0:s}"...'.format(
                this_figure_file_name
            ))
            figure_object.savefig(
                this_figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
                pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(figure_object)

            for this_height_m_agl in rtx.coords[dt_utils.HEIGHT_DIM].values:
                figure_object, _ = uq_eval_plotting.plot_discard_test(
                    result_table_xarray=result_table_xarray,
                    target_var_name=this_var_name,
                    target_wavelength_metres=this_wavelength_metres,
                    target_height_m_agl=this_height_m_agl
                )

                this_figure_file_name = (
                    '{0:s}/discard_test_{1:s}_{2:.2f}microns_{3:05d}-m-agl.jpg'
                ).format(
                    output_dir_name,
                    this_var_name.replace('_', '-'),
                    METRES_TO_MICRONS * this_wavelength_metres,
                    int(numpy.round(this_height_m_agl))
                )

                print('Saving figure to file: "{0:s}"...'.format(
                    this_figure_file_name
                ))
                figure_object.savefig(
                    this_figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
                    pad_inches=0, bbox_inches='tight'
                )
                pyplot.close(figure_object)

    for t in range(len(rtx.coords[dt_utils.VECTOR_FIELD_DIM].values)):
        for this_wavelength_metres in wavelengths_metres:
            w = example_utils.match_wavelengths(
                wavelengths_metres=rtx.coords[dt_utils.WAVELENGTH_DIM].values,
                desired_wavelength_metres=this_wavelength_metres
            )

            figure_object, axes_object = pyplot.subplots(
                1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
            )
            evaluation_plotting.plot_score_profile(
                heights_m_agl=rtx.coords[dt_utils.HEIGHT_DIM].values,
                score_values=
                rtx[dt_utils.VECTOR_MONO_FRACTION_KEY].values[t, :, w],
                score_name=evaluation_plotting.MONO_FRACTION_NAME,
                line_colour=ERROR_PROFILE_COLOUR, line_width=4,
                line_style='solid', use_log_scale=True,
                axes_object=axes_object, are_axes_new=True
            )

            axes_object.set_xlabel('Monotonicity fraction in discard test (MF)')

            this_var_name = rtx.coords[dt_utils.VECTOR_FIELD_DIM].values[t]
            title_string = 'MF for {0:s} at {1:.2f}'.format(
                uq_eval_plotting.TARGET_NAME_ABBREV_TO_FANCY[this_var_name],
                METRES_TO_MICRONS * this_wavelength_metres
            )
            title_string += r' $\mu$m'
            axes_object.set_title(title_string)

            figure_file_name = (
                '{0:s}/mono_fraction_{1:s}_{2:.2f}microns.jpg'
            ).format(
                output_dir_name,
                this_var_name.replace('_', '-'),
                METRES_TO_MICRONS * this_wavelength_metres
            )

            print('Saving figure to file: "{0:s}"...'.format(figure_file_name))
            figure_object.savefig(
                figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
                pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        wavelengths_metres=numpy.array(
            getattr(INPUT_ARG_OBJECT, WAVELENGTHS_ARG_NAME), dtype=float
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
