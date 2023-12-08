"""Plots CRPS (continuous ranked probability score) for each target variable."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from ml4rt.utils import example_utils
from ml4rt.utils import uq_evaluation
from ml4rt.utils import crps_utils
from ml4rt.plotting import evaluation_plotting
from ml4rt.plotting import uq_evaluation_plotting as uq_eval_plotting

METRES_TO_MICRONS = 1e6

TARGET_NAME_TO_CUBED_UNITS = {
    example_utils.SHORTWAVE_HEATING_RATE_NAME: r'K$^{3}$ day$^{-3}$',
    example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME: r'W$^{3}$ m$^{-6}$',
    example_utils.SHORTWAVE_TOA_UP_FLUX_NAME: r'W$^{3}$ m$^{-6}$',
    uq_evaluation.SHORTWAVE_NET_FLUX_NAME: r'W$^{3}$ m$^{-6}$',
    example_utils.LONGWAVE_HEATING_RATE_NAME: r'K$^{3}$ day$^{-3}$',
    example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME: r'W$^{3}$ m$^{-6}$',
    example_utils.LONGWAVE_TOA_UP_FLUX_NAME: r'W$^{3}$ m$^{-6}$',
    uq_evaluation.LONGWAVE_NET_FLUX_NAME: r'W$^{3}$ m$^{-6}$'
}

LINE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255

LEGEND_BOUNDING_BOX_DICT = {
    'facecolor': 'white',
    'alpha': 0.7,
    'edgecolor': 'black',
    'linewidth': 1,
    'boxstyle': 'round'
}

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

INPUT_FILE_ARG_NAME = 'input_file_name'
WAVELENGTHS_ARG_NAME = 'wavelengths_metres'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by `crps_utils.read_results`.'
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


def _plot_crps_one_wavelength(
        result_table_xarray, wavelength_metres, output_dir_name):
    """Plots CRPS for one wavelength.

    :param result_table_xarray: xarray table returned by
        `crps_utils.read_results`.
    :param wavelength_metres: Wavelength.
    :param output_dir_name: Name of output directory.
    """

    rtx = result_table_xarray
    w = example_utils.match_wavelengths(
        wavelengths_metres=rtx.coords[crps_utils.WAVELENGTH_DIM].values,
        desired_wavelength_metres=wavelength_metres
    )

    scalar_target_names = (
        rtx.coords[crps_utils.SCALAR_FIELD_DIM].values.tolist()
    )
    scalar_crps_values = rtx[crps_utils.SCALAR_CRPS_KEY].values[:, w]
    scalar_crpss_values = rtx[crps_utils.SCALAR_CRPSS_KEY].values[:, w]
    scalar_dwcrps_values = rtx[crps_utils.SCALAR_DWCRPS_KEY].values[:, w]

    aux_target_names = (
        rtx.coords[crps_utils.AUX_TARGET_FIELD_DIM].values.tolist()
    )

    if len(aux_target_names) > 0:
        scalar_target_names += aux_target_names
        scalar_crps_values = numpy.concatenate(
            (scalar_crps_values, rtx[crps_utils.AUX_CRPS_KEY].values[:, w]),
            axis=0
        )
        scalar_crpss_values = numpy.concatenate(
            (scalar_crpss_values, rtx[crps_utils.AUX_CRPSS_KEY].values[:, w]),
            axis=0
        )
        scalar_dwcrps_values = numpy.concatenate(
            (scalar_dwcrps_values, rtx[crps_utils.AUX_DWCRPS_KEY].values[:, w]),
            axis=0
        )

    vector_target_names = rtx.coords[crps_utils.VECTOR_FIELD_DIM].values

    for t in range(len(vector_target_names)):
        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )
        evaluation_plotting.plot_score_profile(
            heights_m_agl=rtx.coords[crps_utils.HEIGHT_DIM].values,
            score_values=rtx[crps_utils.VECTOR_CRPS_KEY].values[t, :, w],
            score_name=evaluation_plotting.CRPS_NAME,
            line_colour=LINE_COLOUR, line_width=4, line_style='solid',
            use_log_scale=True, axes_object=axes_object,
            are_axes_new=True
        )

        axes_object.set_xlabel(
            'Continuous ranked probability score (CRPS; {0:s})'.format(
                uq_eval_plotting.TARGET_NAME_TO_UNITS[vector_target_names[t]]
            )
        )

        title_string = 'CRPS for {0:s} at {1:.2f}'.format(
            uq_eval_plotting.TARGET_NAME_ABBREV_TO_FANCY[vector_target_names[t]],
            METRES_TO_MICRONS * wavelength_metres
        )
        title_string += r' $\mu$m'
        axes_object.set_title(title_string)

        annotation_string = 'CRPS for {0:s} = {1:.3g} {2:s}'.format(
            uq_eval_plotting.TARGET_NAME_ABBREV_TO_FANCY[
                scalar_target_names[0]
            ],
            scalar_crps_values[0],
            uq_eval_plotting.TARGET_NAME_TO_UNITS[scalar_target_names[0]]
        )
        for j in range(1, len(scalar_target_names)):
            annotation_string += '\nCRPS for {0:s} = {1:.3g} {2:s}'.format(
                uq_eval_plotting.TARGET_NAME_ABBREV_TO_FANCY[
                    scalar_target_names[j]
                ],
                scalar_crps_values[j],
                uq_eval_plotting.TARGET_NAME_TO_UNITS[scalar_target_names[j]]
            )

        axes_object.text(
            0.99, 0.3, annotation_string,
            fontsize=20, color='k',
            bbox=LEGEND_BOUNDING_BOX_DICT,
            horizontalalignment='right', verticalalignment='center',
            transform=axes_object.transAxes, zorder=1e10
        )

        figure_file_name = '{0:s}/crps_{1:s}_{2:.2f}microns.jpg'.format(
            output_dir_name,
            vector_target_names[t].replace('_', '-'),
            METRES_TO_MICRONS * wavelength_metres
        )
        print('Saving figure to file: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )
        evaluation_plotting.plot_score_profile(
            heights_m_agl=rtx.coords[crps_utils.HEIGHT_DIM].values,
            score_values=rtx[crps_utils.VECTOR_CRPSS_KEY].values[t, :, w],
            score_name=evaluation_plotting.CRPSS_NAME,
            line_colour=LINE_COLOUR, line_width=4, line_style='solid',
            use_log_scale=True, axes_object=axes_object,
            are_axes_new=True
        )

        axes_object.set_xlabel(
            'Continuous ranked probability skill score (CRPSS)'
        )
        title_string = title_string.replace('CRPS', 'CRPSS')
        axes_object.set_title(title_string)

        annotation_string = 'CRPSS for {0:s} = {1:.3f}'.format(
            uq_eval_plotting.TARGET_NAME_ABBREV_TO_FANCY[
                scalar_target_names[0]
            ],
            scalar_crpss_values[0]
        )
        for j in range(1, len(scalar_target_names)):
            annotation_string += '\nCRPSS for {0:s} = {1:.3f}'.format(
                uq_eval_plotting.TARGET_NAME_ABBREV_TO_FANCY[
                    scalar_target_names[j]
                ],
                scalar_crpss_values[j]
            )

        axes_object.text(
            0.99, 0.3, annotation_string,
            fontsize=20, color='k',
            bbox=LEGEND_BOUNDING_BOX_DICT,
            horizontalalignment='right', verticalalignment='center',
            transform=axes_object.transAxes, zorder=1e10
        )

        figure_file_name = '{0:s}/crpss_{1:s}_{2:.2f}microns.jpg'.format(
            output_dir_name,
            vector_target_names[t].replace('_', '-'),
            METRES_TO_MICRONS * wavelength_metres
        )
        print('Saving figure to file: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )
        evaluation_plotting.plot_score_profile(
            heights_m_agl=rtx.coords[crps_utils.HEIGHT_DIM].values,
            score_values=rtx[crps_utils.VECTOR_DWCRPS_KEY].values[t, :, w],
            score_name=evaluation_plotting.CRPS_NAME,
            line_colour=LINE_COLOUR, line_width=4, line_style='solid',
            use_log_scale=True, axes_object=axes_object,
            are_axes_new=True
        )

        axes_object.set_xlabel('Dual-weighted CRPS ({0:s})'.format(
            TARGET_NAME_TO_CUBED_UNITS[vector_target_names[t]]
        ))
        title_string = title_string.replace('CRPSS', 'Dual-weighted CRPS')
        axes_object.set_title(title_string)

        annotation_string = 'DWCRPS for {0:s} = {1:.3g} {2:s}'.format(
            uq_eval_plotting.TARGET_NAME_ABBREV_TO_FANCY[
                scalar_target_names[0]
            ],
            scalar_dwcrps_values[0],
            TARGET_NAME_TO_CUBED_UNITS[scalar_target_names[0]]
        )
        for j in range(1, len(scalar_target_names)):
            annotation_string += '\nDWCRPS for {0:s} = {1:.3g} {2:s}'.format(
                uq_eval_plotting.TARGET_NAME_ABBREV_TO_FANCY[
                    scalar_target_names[j]
                ],
                scalar_dwcrps_values[j],
                TARGET_NAME_TO_CUBED_UNITS[scalar_target_names[j]]
            )

        axes_object.text(
            0.99, 0.3, annotation_string,
            fontsize=20, color='k',
            bbox=LEGEND_BOUNDING_BOX_DICT,
            horizontalalignment='right', verticalalignment='center',
            transform=axes_object.transAxes, zorder=1e10
        )

        figure_file_name = '{0:s}/dwcrps_{1:s}_{2:.2f}microns.jpg'.format(
            output_dir_name,
            vector_target_names[t].replace('_', '-'),
            METRES_TO_MICRONS * wavelength_metres
        )
        print('Saving figure to file: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)


def _run(input_file_name, wavelengths_metres, output_dir_name):
    """Plots CRPS (continuous ranked probability score) for each target var.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param wavelengths_metres: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    result_table_xarray = crps_utils.read_results(input_file_name)

    for this_wavelength_metres in wavelengths_metres:
        _plot_crps_one_wavelength(
            result_table_xarray=result_table_xarray,
            wavelength_metres=this_wavelength_metres,
            output_dir_name=output_dir_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        wavelengths_metres=numpy.array(
            getattr(INPUT_ARG_OBJECT, WAVELENGTHS_ARG_NAME), dtype=float
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
