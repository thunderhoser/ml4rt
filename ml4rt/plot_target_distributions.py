"""Plots distribution of each target variable."""

import os
import sys
import copy
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

FIRST_YEAR = 2020
LAST_YEAR = 2020

SHORTWAVE_TARGET_NAMES_IN_FILE = [
    example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME,
    example_utils.SHORTWAVE_TOA_UP_FLUX_NAME,
    example_utils.SHORTWAVE_HEATING_RATE_NAME
]
SHORTWAVE_HEAT_FLUX_NAME = 'shortwave_heat_flux_w_m02'
SHORTWAVE_NET_FLUX_NAME = 'shortwave_net_flux_w_m02'
SHORTWAVE_TARGET_NAMES = copy.deepcopy(SHORTWAVE_TARGET_NAMES_IN_FILE)
SHORTWAVE_TARGET_NAMES.insert(-1, SHORTWAVE_NET_FLUX_NAME)

LONGWAVE_TARGET_NAMES_IN_FILE = [
    example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME,
    example_utils.LONGWAVE_TOA_UP_FLUX_NAME,
    example_utils.LONGWAVE_HEATING_RATE_NAME
]
LONGWAVE_HEAT_FLUX_NAME = 'longwave_heat_flux_w_m02'
LONGWAVE_NET_FLUX_NAME = 'longwave_net_flux_w_m02'
LONGWAVE_TARGET_NAMES = copy.deepcopy(LONGWAVE_TARGET_NAMES_IN_FILE)
LONGWAVE_TARGET_NAMES.insert(-1, LONGWAVE_NET_FLUX_NAME)

TARGET_NAME_TO_VERBOSE = {
    example_utils.SHORTWAVE_HEATING_RATE_NAME:
        r'Shortwave heating rate (K day$^{-1}$)',
    example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME:
        r'Shortwave $F_{down}^{sfc}$ (W m$^{-2}$)',
    example_utils.SHORTWAVE_TOA_UP_FLUX_NAME:
        r'Shortwave $F_{up}^{TOA}$ (W m$^{-2}$)',
    SHORTWAVE_NET_FLUX_NAME: r'Shortwave net flux (W m$^{-2}$)',
    SHORTWAVE_HEAT_FLUX_NAME: r'Shortwave heat flux (W m$^{-2}$)',
    example_utils.LONGWAVE_HEATING_RATE_NAME:
        r'Longwave heating rate (K day$^{-1}$)',
    example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME:
        r'Longwave $F_{down}^{sfc}$ (W m$^{-2}$)',
    example_utils.LONGWAVE_TOA_UP_FLUX_NAME:
        r'Longwave $F_{up}^{TOA}$ (W m$^{-2}$)',
    LONGWAVE_NET_FLUX_NAME: r'Longwave net flux (W m$^{-2}$)',
    LONGWAVE_HEAT_FLUX_NAME: r'Longwave heat flux (W m$^{-2}$)'
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

TROPICAL_DIR_ARG_NAME = 'input_tropical_dir_name'
NON_TROPICAL_DIR_ARG_NAME = 'input_non_tropical_dir_name'
NUM_BINS_ARG_NAME = 'num_histogram_bins'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

TROPICAL_DIR_HELP_STRING = (
    'Name of directory with examples for tropical sites.  Files therein will be'
    ' found by `example_io.find_file` and read by `example_io.read_file`.'
)
NON_TROPICAL_DIR_HELP_STRING = (
    'Same as `{0:s}` but for non-tropical sites.'.format(TROPICAL_DIR_ARG_NAME)
)
NUM_BINS_HELP_STRING = 'Number of bins in each histogram.'
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + TROPICAL_DIR_ARG_NAME, type=str, required=True,
    help=TROPICAL_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NON_TROPICAL_DIR_ARG_NAME, type=str, required=True,
    help=NON_TROPICAL_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_BINS_ARG_NAME, type=int, required=False, default=50,
    help=NUM_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_histogram_one_target(
        target_values, target_name, num_bins, letter_label, output_dir_name):
    """Plots histogram for one target variable.

    :param target_values: 1-D numpy array of values.
    :param target_name: Name of target variable.
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
        x_tick_labels = ['{0:.1f}'.format(c) for c in bin_centers]
    else:
        x_tick_labels = [
            '{0:d}'.format(int(numpy.round(c))) for c in bin_centers
        ]

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
    axes_object.set_xlabel(TARGET_NAME_TO_VERBOSE[target_name])
    gg_plotting_utils.label_axes(
        axes_object=axes_object, label_string='({0:s})'.format(letter_label)
    )

    output_file_name = '{0:s}/histogram_{1:s}.jpg'.format(
        output_dir_name, target_name.replace('_', '-')
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


def _run(tropical_example_dir_name, non_tropical_example_dir_name,
         num_histogram_bins, output_dir_name):
    """Plots distribution of each target variable.

    This is effectively the main method.

    :param tropical_example_dir_name: See documentation at top of file.
    :param non_tropical_example_dir_name: Same.
    :param num_histogram_bins: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    first_time_unix_sec = (
        time_conversion.first_and_last_times_in_year(FIRST_YEAR)[0]
    )
    last_time_unix_sec = (
        time_conversion.first_and_last_times_in_year(LAST_YEAR)[-1]
    )

    example_file_names = example_io.find_many_files(
        directory_name=tropical_example_dir_name,
        first_time_unix_sec=first_time_unix_sec,
        last_time_unix_sec=last_time_unix_sec,
        raise_error_if_all_missing=True, raise_error_if_any_missing=True
    )

    example_file_names += example_io.find_many_files(
        directory_name=non_tropical_example_dir_name,
        first_time_unix_sec=first_time_unix_sec,
        last_time_unix_sec=last_time_unix_sec,
        raise_error_if_all_missing=True, raise_error_if_any_missing=True
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

        these_field_names = (
            SHORTWAVE_TARGET_NAMES_IN_FILE + LONGWAVE_TARGET_NAMES_IN_FILE +
            [example_utils.PRESSURE_NAME, example_utils.HEIGHT_NAME]
        )
        this_example_dict = example_utils.subset_by_field(
            example_dict=this_example_dict, field_names=these_field_names
        )

        example_dicts.append(this_example_dict)

    example_dict = example_utils.concat_examples(example_dicts)
    del example_dicts

    letter_label = None
    shortwave_panel_file_names = []

    for this_target_name in SHORTWAVE_TARGET_NAMES:
        if this_target_name in SHORTWAVE_TARGET_NAMES_IN_FILE:
            these_target_values = example_utils.get_field_from_dict(
                example_dict=example_dict, field_name=this_target_name
            )
        else:
            down_fluxes_w_m02 = example_utils.get_field_from_dict(
                example_dict=example_dict,
                field_name=example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME
            )
            up_fluxes_w_m02 = example_utils.get_field_from_dict(
                example_dict=example_dict,
                field_name=example_utils.SHORTWAVE_TOA_UP_FLUX_NAME
            )
            these_target_values = down_fluxes_w_m02 - up_fluxes_w_m02

        these_target_values = numpy.ravel(these_target_values)

        if letter_label is None:
            letter_label = 'a'
        else:
            letter_label = chr(ord(letter_label) + 1)

        this_file_name = _plot_histogram_one_target(
            target_values=these_target_values, target_name=this_target_name,
            num_bins=num_histogram_bins, letter_label=letter_label,
            output_dir_name=output_dir_name
        )
        shortwave_panel_file_names.append(this_file_name)

    example_dict = example_utils.multiply_hr_by_layer_thickness(example_dict)
    heat_flux_matrix_w_m02 = example_utils.get_field_from_dict(
        example_dict=example_dict,
        field_name=example_utils.SHORTWAVE_HEATING_RATE_NAME
    )

    if letter_label is None:
        letter_label = 'a'
    else:
        letter_label = chr(ord(letter_label) + 1)

    this_file_name = _plot_histogram_one_target(
        target_values=numpy.ravel(heat_flux_matrix_w_m02),
        target_name=SHORTWAVE_HEAT_FLUX_NAME,
        num_bins=num_histogram_bins, letter_label=letter_label,
        output_dir_name=output_dir_name
    )
    shortwave_panel_file_names.append(this_file_name)

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
                j + 1, j + 2, numpy.mean(height_diff_matrix_metres[:, j]),
                numpy.std(height_diff_matrix_metres[:, j], ddof=1)
            ))

    shortwave_figure_file_name = (
        '{0:s}/shortwave_target_distributions.jpg'
    ).format(output_dir_name)

    print('Concatenating panels to: "{0:s}"...'.format(
        shortwave_figure_file_name
    ))
    imagemagick_utils.concatenate_images(
        input_file_names=shortwave_panel_file_names,
        output_file_name=shortwave_figure_file_name,
        num_panel_rows=2, num_panel_columns=3, border_width_pixels=25
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=shortwave_figure_file_name,
        output_file_name=shortwave_figure_file_name
    )

    letter_label = None
    longwave_panel_file_names = []

    for this_target_name in LONGWAVE_TARGET_NAMES:
        if this_target_name in LONGWAVE_TARGET_NAMES_IN_FILE:
            these_target_values = example_utils.get_field_from_dict(
                example_dict=example_dict, field_name=this_target_name
            )
        else:
            down_fluxes_w_m02 = example_utils.get_field_from_dict(
                example_dict=example_dict,
                field_name=example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME
            )
            up_fluxes_w_m02 = example_utils.get_field_from_dict(
                example_dict=example_dict,
                field_name=example_utils.LONGWAVE_TOA_UP_FLUX_NAME
            )
            these_target_values = down_fluxes_w_m02 - up_fluxes_w_m02

        these_target_values = numpy.ravel(these_target_values)

        if letter_label is None:
            letter_label = 'a'
        else:
            letter_label = chr(ord(letter_label) + 1)

        this_file_name = _plot_histogram_one_target(
            target_values=these_target_values, target_name=this_target_name,
            num_bins=num_histogram_bins, letter_label=letter_label,
            output_dir_name=output_dir_name
        )
        longwave_panel_file_names.append(this_file_name)

    example_dict = example_utils.multiply_hr_by_layer_thickness(example_dict)
    heat_flux_matrix_w_m02 = example_utils.get_field_from_dict(
        example_dict=example_dict,
        field_name=example_utils.LONGWAVE_HEATING_RATE_NAME
    )

    if letter_label is None:
        letter_label = 'a'
    else:
        letter_label = chr(ord(letter_label) + 1)

    this_file_name = _plot_histogram_one_target(
        target_values=numpy.ravel(heat_flux_matrix_w_m02),
        target_name=LONGWAVE_HEAT_FLUX_NAME,
        num_bins=num_histogram_bins, letter_label=letter_label,
        output_dir_name=output_dir_name
    )
    longwave_panel_file_names.append(this_file_name)

    longwave_figure_file_name = (
        '{0:s}/longwave_target_distributions.jpg'
    ).format(output_dir_name)

    print('Concatenating panels to: "{0:s}"...'.format(
        longwave_figure_file_name
    ))
    imagemagick_utils.concatenate_images(
        input_file_names=longwave_panel_file_names,
        output_file_name=longwave_figure_file_name,
        num_panel_rows=2, num_panel_columns=3, border_width_pixels=25
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=longwave_figure_file_name,
        output_file_name=longwave_figure_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        tropical_example_dir_name=getattr(
            INPUT_ARG_OBJECT, TROPICAL_DIR_ARG_NAME
        ),
        non_tropical_example_dir_name=getattr(
            INPUT_ARG_OBJECT, NON_TROPICAL_DIR_ARG_NAME
        ),
        num_histogram_bins=getattr(INPUT_ARG_OBJECT, NUM_BINS_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
