"""Plots distribution of each target variable."""

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

import histograms
import time_conversion
import file_system_utils
import gg_plotting_utils
import imagemagick_utils
import example_io
import example_utils

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
METRES_TO_MICRONS = 1e6

SHORTWAVE_TARGET_NAMES_IN_FILE = [
    example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME,
    example_utils.SHORTWAVE_TOA_UP_FLUX_NAME,
    example_utils.SHORTWAVE_HEATING_RATE_NAME
]
SHORTWAVE_NET_FLUX_NAME = 'shortwave_net_flux_w_m02'
SHORTWAVE_TARGET_NAMES = (
    SHORTWAVE_TARGET_NAMES_IN_FILE + [SHORTWAVE_NET_FLUX_NAME]
)

LONGWAVE_TARGET_NAMES_IN_FILE = [
    example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME,
    example_utils.LONGWAVE_TOA_UP_FLUX_NAME,
    example_utils.LONGWAVE_HEATING_RATE_NAME
]
LONGWAVE_NET_FLUX_NAME = 'longwave_net_flux_w_m02'
LONGWAVE_TARGET_NAMES = (
    LONGWAVE_TARGET_NAMES_IN_FILE + [LONGWAVE_NET_FLUX_NAME]
)

TARGET_NAME_TO_VERBOSE = {
    example_utils.SHORTWAVE_HEATING_RATE_NAME:
        r'Shortwave heating rate (K day$^{-1}$)',
    example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME:
        r'Shortwave $F_{down}^{sfc}$ (W m$^{-2}$)',
    example_utils.SHORTWAVE_TOA_UP_FLUX_NAME:
        r'Shortwave $F_{up}^{TOA}$ (W m$^{-2}$)',
    SHORTWAVE_NET_FLUX_NAME: r'Shortwave net flux (W m$^{-2}$)',
    example_utils.LONGWAVE_HEATING_RATE_NAME:
        r'Longwave heating rate (K day$^{-1}$)',
    example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME:
        r'Longwave $F_{down}^{sfc}$ (W m$^{-2}$)',
    example_utils.LONGWAVE_TOA_UP_FLUX_NAME:
        r'Longwave $F_{up}^{TOA}$ (W m$^{-2}$)',
    LONGWAVE_NET_FLUX_NAME: r'Longwave net flux (W m$^{-2}$)'
}

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

FACE_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
EDGE_COLOUR = numpy.full(3, 0.)
EDGE_WIDTH = 1.5

FONT_SIZE = 44
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

INPUT_DIR_ARG_NAME = 'input_example_dir_name'
NUM_BINS_ARG_NAME = 'num_histogram_bins'
PLOT_SHORTWAVE_ARG_NAME = 'plot_shortwave'
WAVELENGTHS_ARG_NAME = 'wavelengths_metres'
START_TIME_ARG_NAME = 'start_time_string'
END_TIME_ARG_NAME = 'end_time_string'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of directory with data examples.  Files therein will be found by '
    '`example_io.find_file` and read by `example_io.read_file`.'
)
NUM_BINS_HELP_STRING = 'Number of bins in each histogram.'
PLOT_SHORTWAVE_HELP_STRING = (
    'Boolean flag.  If 1 (0), will plot shortwave (longwave) values.'
)
WAVELENGTHS_HELP_STRING = (
    'List of wavelengths.  Will create one figure for each.'
)
START_TIME_HELP_STRING = (
    'Beginning of time period (format "yyyy-mm-dd-HHMMSS").  Will plot '
    'distributions only for the time period `{0:s}`...`{1:s}`.'
).format(START_TIME_ARG_NAME, END_TIME_ARG_NAME)

END_TIME_HELP_STRING = (
    'End of time period (format "yyyy-mm-dd-HHMMSS").  Will plot '
    'distributions only for the time period `{0:s}`...`{1:s}`.'
).format(START_TIME_ARG_NAME, END_TIME_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_BINS_ARG_NAME, type=int, required=False, default=50,
    help=NUM_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_SHORTWAVE_ARG_NAME, type=int, required=True,
    help=PLOT_SHORTWAVE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + WAVELENGTHS_ARG_NAME, type=float, nargs='+', required=False,
    default=[example_utils.DUMMY_BROADBAND_WAVELENGTH_METRES],
    help=WAVELENGTHS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + START_TIME_ARG_NAME, type=str, required=True,
    help=START_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + END_TIME_ARG_NAME, type=str, required=True,
    help=END_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_histogram_one_target(
        target_values, target_name, wavelength_metres,
        num_bins, letter_label, output_dir_name):
    """Plots histogram for one target variable.

    :param target_values: 1-D numpy array of values.
    :param target_name: Name of target variable.
    :param wavelength_metres: Wavelength for target variable.
    :param num_bins: Number of bins in histogram.
    :param letter_label: Letter label (will be used to label panel).
    :param output_dir_name: Name of output directory.  Figure will be saved
        here.
    :return: output_file_name: Path to output file.
    """

    if target_name in [
            SHORTWAVE_NET_FLUX_NAME, LONGWAVE_NET_FLUX_NAME,
            example_utils.LONGWAVE_HEATING_RATE_NAME
    ]:
        min_value = numpy.min(target_values)
    else:
        min_value = 0.

    max_value = numpy.max(target_values)

    num_examples_by_bin = histograms.create_histogram(
        input_values=target_values, num_bins=num_bins,
        min_value=min_value, max_value=max_value
    )[1]
    frequency_by_bin = (
        num_examples_by_bin.astype(float) / numpy.sum(num_examples_by_bin)
    )

    bin_edges = numpy.linspace(min_value, max_value, num=num_bins + 1)
    bin_centers = numpy.array([
        numpy.mean(bin_edges[[k, k + 1]])
        for k in range(num_bins)
    ])

    x_tick_coords = 0.5 + numpy.linspace(
        0, num_bins - 1, num=num_bins, dtype=float
    )

    if target_name in [
            example_utils.SHORTWAVE_HEATING_RATE_NAME,
            example_utils.LONGWAVE_HEATING_RATE_NAME
    ]:
        x_tick_labels = ['{0:.2f}'.format(c) for c in bin_centers]
    else:
        x_tick_labels = ['{0:.1f}'.format(c) for c in bin_centers]

    x_tick_labels = [
        x_tick_labels[k] if numpy.mod(k, 3) == 0 else ' '
        for k in range(num_bins)
    ]

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    axes_object.bar(
        x=x_tick_coords, height=frequency_by_bin, width=1.,
        color=FACE_COLOUR, edgecolor=EDGE_COLOUR, linewidth=EDGE_WIDTH
    )

    axes_object.set_xlim([
        x_tick_coords[0] - 0.5, x_tick_coords[-1] + 0.5
    ])
    axes_object.set_xticks(x_tick_coords)
    axes_object.set_xticklabels(x_tick_labels, rotation=90.)

    axes_object.set_ylabel('Frequency')
    axes_object.set_xlabel('{0:s} at {1:.2f} microns'.format(
        TARGET_NAME_TO_VERBOSE[target_name],
        METRES_TO_MICRONS * wavelength_metres
    ))
    gg_plotting_utils.label_axes(
        axes_object=axes_object, label_string='({0:s})'.format(letter_label)
    )

    output_file_name = '{0:s}/histogram_{1:s}_{2:.2f}microns.jpg'.format(
        output_dir_name,
        target_name.replace('_', '-'),
        METRES_TO_MICRONS * wavelength_metres
    )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name, output_file_name=output_file_name
    )
    imagemagick_utils.resize_image(
        input_file_name=output_file_name, output_file_name=output_file_name,
        output_size_pixels=int(2.5e6)
    )

    return output_file_name


def _plot_distributions_one_wavelength(
        example_dict, plot_shortwave, num_histogram_bins, wavelength_metres,
        output_dir_name):
    """Plots target distributions for one wavelength.

    :param example_dict: Dictionary returned by `example_io.read_file`.
    :param plot_shortwave: See documentation at top of this script.
    :param num_histogram_bins: Same.
    :param wavelength_metres: Same.
    :param output_dir_name: Same.
    """

    letter_label = None
    panel_file_names = []

    if plot_shortwave:
        target_names = SHORTWAVE_TARGET_NAMES
        target_names_in_file = SHORTWAVE_TARGET_NAMES_IN_FILE
    else:
        target_names = LONGWAVE_TARGET_NAMES
        target_names_in_file = LONGWAVE_TARGET_NAMES_IN_FILE

    for this_target_name in target_names:
        if this_target_name in target_names_in_file:
            these_target_values = example_utils.get_field_from_dict(
                example_dict=example_dict,
                field_name=this_target_name,
                target_wavelength_metres=wavelength_metres
            )
        else:
            down_flux_name = (
                example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME
                if plot_shortwave
                else example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME
            )
            up_flux_name = (
                example_utils.SHORTWAVE_TOA_UP_FLUX_NAME
                if plot_shortwave
                else example_utils.LONGWAVE_TOA_UP_FLUX_NAME
            )

            down_fluxes_w_m02 = example_utils.get_field_from_dict(
                example_dict=example_dict,
                field_name=down_flux_name,
                target_wavelength_metres=wavelength_metres
            )
            up_fluxes_w_m02 = example_utils.get_field_from_dict(
                example_dict=example_dict,
                field_name=up_flux_name,
                target_wavelength_metres=wavelength_metres
            )
            these_target_values = down_fluxes_w_m02 - up_fluxes_w_m02

        these_target_values = numpy.ravel(these_target_values)

        if letter_label is None:
            letter_label = 'a'
        else:
            letter_label = chr(ord(letter_label) + 1)

        this_file_name = _plot_histogram_one_target(
            target_values=these_target_values,
            target_name=this_target_name,
            wavelength_metres=wavelength_metres,
            num_bins=num_histogram_bins,
            letter_label=letter_label,
            output_dir_name=output_dir_name
        )
        panel_file_names.append(this_file_name)

    concat_figure_file_name = (
        '{0:s}/target_distributions_{1:.2f}microns.jpg'
    ).format(
        output_dir_name,
        METRES_TO_MICRONS * wavelength_metres
    )

    print('Concatenating panels to: "{0:s}"...'.format(
        concat_figure_file_name
    ))
    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=concat_figure_file_name,
        num_panel_rows=2, num_panel_columns=3, border_width_pixels=25
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=concat_figure_file_name,
        output_file_name=concat_figure_file_name
    )


def _run(example_dir_name, num_histogram_bins, plot_shortwave,
         wavelengths_metres, start_time_string, end_time_string,
         output_dir_name):
    """Plots distribution of each target variable.

    This is effectively the main method.

    :param example_dir_name: See documentation at top of file.
    :param num_histogram_bins: Same.
    :param plot_shortwave: Same.
    :param wavelengths_metres: Same.
    :param start_time_string: Same.
    :param end_time_string: Same.
    :param output_dir_name: Same.
    """

    # Handle input args.
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )
    start_time_unix_sec = time_conversion.string_to_unix_sec(
        start_time_string, TIME_FORMAT
    )
    end_time_unix_sec = time_conversion.string_to_unix_sec(
        end_time_string, TIME_FORMAT
    )

    # Do actual stuff.
    example_file_names = example_io.find_many_files(
        directory_name=example_dir_name,
        first_time_unix_sec=start_time_unix_sec,
        last_time_unix_sec=end_time_unix_sec,
        raise_error_if_all_missing=True,
        raise_error_if_any_missing=True
    )

    example_file_names = list(set(example_file_names))
    example_dicts = []

    for this_file_name in example_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_example_dict = example_io.read_file(
            netcdf_file_name=this_file_name,
            max_shortwave_heating_k_day01=numpy.inf,
            min_longwave_heating_k_day01=-1 * numpy.inf,
            max_longwave_heating_k_day01=numpy.inf
        )

        this_example_dict = example_utils.subset_by_wavelength(
            example_dict=this_example_dict,
            target_wavelengths_metres=wavelengths_metres
        )
        this_example_dict = example_utils.subset_by_time(
            example_dict=this_example_dict,
            first_time_unix_sec=start_time_unix_sec,
            last_time_unix_sec=end_time_unix_sec
        )

        if plot_shortwave:
            field_names = SHORTWAVE_TARGET_NAMES_IN_FILE
        else:
            field_names = LONGWAVE_TARGET_NAMES_IN_FILE

        if (
                example_utils.HEIGHT_NAME in
                this_example_dict[example_utils.VECTOR_PREDICTOR_NAMES_KEY]
        ):
            field_names.append(example_utils.HEIGHT_NAME)

        this_example_dict = example_utils.subset_by_field(
            example_dict=this_example_dict, field_names=field_names
        )
        example_dicts.append(this_example_dict)

    example_dict = example_utils.concat_examples(example_dicts)
    del example_dicts

    for this_wavelength_metres in wavelengths_metres:
        _plot_distributions_one_wavelength(
            example_dict=example_dict,
            plot_shortwave=plot_shortwave,
            num_histogram_bins=num_histogram_bins,
            wavelength_metres=this_wavelength_metres,
            output_dir_name=output_dir_name
        )

    if (
            example_utils.HEIGHT_NAME in
            example_dict[example_utils.VECTOR_PREDICTOR_NAMES_KEY]
    ):
        height_matrix_m_agl = example_utils.get_field_from_dict(
            example_dict=example_dict,
            field_name=example_utils.HEIGHT_NAME
        )
        height_diff_matrix_metres = numpy.diff(height_matrix_m_agl, axis=1)

        num_sigma_levels = height_matrix_m_agl.shape[1]

        for j in range(num_sigma_levels - 1):
            print((
                'Difference between {0:d}th and {1:d}th sigma-levels ... '
                'mean = {2:.2f} m ... stdev = {3:.2f} m'
            ).format(
                j + 1,
                j + 2,
                numpy.mean(height_diff_matrix_metres[:, j]),
                numpy.std(height_diff_matrix_metres[:, j], ddof=1)
            ))


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        example_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        num_histogram_bins=getattr(INPUT_ARG_OBJECT, NUM_BINS_ARG_NAME),
        plot_shortwave=bool(getattr(INPUT_ARG_OBJECT, PLOT_SHORTWAVE_ARG_NAME)),
        wavelengths_metres=numpy.array(
            getattr(INPUT_ARG_OBJECT, WAVELENGTHS_ARG_NAME), dtype=float
        ),
        start_time_string=getattr(INPUT_ARG_OBJECT, START_TIME_ARG_NAME),
        end_time_string=getattr(INPUT_ARG_OBJECT, END_TIME_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
