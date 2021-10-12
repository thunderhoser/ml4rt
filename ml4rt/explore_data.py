"""Explores data by creating plots for different subsets of profiles."""

import os
import sys
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import plotting_utils
import example_io
import example_utils
import profile_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NUM_LATITUDE_BINS = 36
MIN_PATH_FOR_CLOUD_KG_M02 = 0.05
PERCENTILE_LEVELS = numpy.array([
    50, 75, 90, 95, 96, 97, 98, 99, 99.5, 99.75, 99.9, 100
])

LATITUDE_COLOUR_MAP_OBJECT = pyplot.get_cmap('twilight_shifted')

NO_CLOUD_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
SINGLE_LAYER_CLOUD_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
MULTI_LAYER_CLOUD_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255

FIGURE_RESOLUTION_DPI = 300

INPUT_FILES_ARG_NAME = 'input_example_file_names'
OUTPUT_DIR_ARG_NAME = 'output_figure_dir_name'

INPUT_FILES_HELP_STRING = (
    'List of paths to input files.  Each file will be read by '
    '`example_io.read_file`, and all examples in these files will be explored '
    'together.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=INPUT_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_by_latitude(example_dict, output_dir_name):
    """Plots mean heating-rate profile by latitude.

    :param example_dict: Dictionary in format returned by
        `example_io.read_file`.
    :param output_dir_name: Name of output directory.  Figure will be saved
        here.
    """

    example_latitudes_deg_n = example_utils.parse_example_ids(
        example_dict[example_utils.EXAMPLE_IDS_KEY]
    )[example_utils.LATITUDES_KEY]

    bin_edges_deg_n = numpy.linspace(
        -90, 90, num=NUM_LATITUDE_BINS + 1, dtype=float
    )
    bin_centers_deg_n = bin_edges_deg_n[:-1] + numpy.diff(bin_edges_deg_n) / 2
    colour_norm_object = matplotlib.colors.Normalize(
        vmin=bin_centers_deg_n[0], vmax=bin_centers_deg_n[-1]
    )

    example_to_bin_indices = numpy.digitize(
        x=example_latitudes_deg_n, bins=bin_edges_deg_n, right=False
    )
    example_to_bin_indices = numpy.maximum(example_to_bin_indices, 0)
    example_to_bin_indices = numpy.minimum(
        example_to_bin_indices, NUM_LATITUDE_BINS - 1
    )

    heating_rate_matrix_k_day01 = example_utils.get_field_from_dict(
        example_dict=example_dict,
        field_name=example_utils.SHORTWAVE_HEATING_RATE_NAME
    )
    figure_object = None
    axes_object = None

    for k in range(NUM_LATITUDE_BINS):
        these_example_indices = numpy.where(example_to_bin_indices == k)[0]
        if len(these_example_indices) == 0:
            continue

        this_colour = LATITUDE_COLOUR_MAP_OBJECT(
            colour_norm_object(bin_centers_deg_n[k])
        )

        figure_object, axes_object = profile_plotting.plot_one_variable(
            values=numpy.mean(
                heating_rate_matrix_k_day01[these_example_indices, :], axis=0
            ),
            heights_m_agl=example_dict[example_utils.HEIGHTS_KEY],
            use_log_scale=True, line_colour=this_colour,
            figure_object=figure_object
        )

    colour_bar_object = plotting_utils.plot_linear_colour_bar(
        axes_object_or_matrix=axes_object, data_matrix=bin_centers_deg_n,
        colour_map_object=LATITUDE_COLOUR_MAP_OBJECT,
        min_value=bin_centers_deg_n[0], max_value=bin_centers_deg_n[-1],
        orientation_string='vertical', extend_min=False, extend_max=False
    )

    tick_values = colour_bar_object.get_ticks()
    tick_strings = ['{0:.1f}'.format(v) for v in tick_values]
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    output_file_name = '{0:s}/latitude_heating_rates.jpg'.format(
        output_dir_name
    )
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _plot_by_cloud_regime(example_dict, output_dir_name):
    """Plots mean heating-rate profile by cloud regime.

    :param example_dict: Dictionary in format returned by
        `example_io.read_file`.
    :param output_dir_name: Name of output directory.  Figure will be saved
        here.
    """

    cloud_layer_counts = example_utils.find_cloud_layers(
        example_dict=example_dict, min_path_kg_m02=MIN_PATH_FOR_CLOUD_KG_M02,
        for_ice=False
    )[-1]

    no_cloud_indices = numpy.where(cloud_layer_counts == 0)[0]
    single_layer_cloud_indices = numpy.where(cloud_layer_counts == 1)[0]
    multi_layer_cloud_indices = numpy.where(cloud_layer_counts > 1)[0]

    print((
        'Number of profiles with no liquid cloud = {0:d}; '
        'single-layer cloud = {1:d}; multi-layer cloud = {2:d}'
    ).format(
        len(no_cloud_indices), len(single_layer_cloud_indices),
        len(multi_layer_cloud_indices))
    )

    heating_rate_matrix_k_day01 = example_utils.get_field_from_dict(
        example_dict=example_dict,
        field_name=example_utils.SHORTWAVE_HEATING_RATE_NAME
    )

    figure_object, axes_object = profile_plotting.plot_one_variable(
        values=numpy.mean(
            heating_rate_matrix_k_day01[no_cloud_indices, :], axis=0
        ),
        heights_m_agl=example_dict[example_utils.HEIGHTS_KEY],
        use_log_scale=True, line_colour=NO_CLOUD_COLOUR, figure_object=None
    )

    axes_object.set_title('Heating rate by cloud regime')
    axes_object.set_xlabel(r'Heating rate (K day$^{-1}$)')

    profile_plotting.plot_one_variable(
        values=numpy.mean(
            heating_rate_matrix_k_day01[single_layer_cloud_indices, :], axis=0
        ),
        heights_m_agl=example_dict[example_utils.HEIGHTS_KEY],
        use_log_scale=True, line_colour=SINGLE_LAYER_CLOUD_COLOUR,
        figure_object=figure_object
    )

    profile_plotting.plot_one_variable(
        values=numpy.mean(
            heating_rate_matrix_k_day01[multi_layer_cloud_indices, :], axis=0
        ),
        heights_m_agl=example_dict[example_utils.HEIGHTS_KEY],
        use_log_scale=True, line_colour=MULTI_LAYER_CLOUD_COLOUR,
        figure_object=figure_object
    )

    output_file_name = '{0:s}/cloud_regime_heating_rates.jpg'.format(
        output_dir_name
    )
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run(example_file_names, output_dir_name):
    """Explores data by creating plots for different subsets of profiles.

    This is effectively the main method.

    :param example_file_names: See documentation at top of file.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    example_dicts = []

    for this_file_name in example_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_example_dict = example_io.read_file(
            netcdf_file_name=this_file_name, max_heating_rate_k_day=numpy.inf
        )
        example_dicts.append(this_example_dict)

    example_dict = example_utils.concat_examples(example_dicts)

    heating_rate_matrix_k_day01 = example_utils.get_field_from_dict(
        example_dict=example_dict,
        field_name=example_utils.SHORTWAVE_HEATING_RATE_NAME
    )
    heating_rate_percentiles_k_day01 = numpy.percentile(
        heating_rate_matrix_k_day01, PERCENTILE_LEVELS
    )

    for a, b in zip(PERCENTILE_LEVELS, heating_rate_percentiles_k_day01):
        print('{0:.1f}th-percentile heating rate = {1:.4f} K day^-1'.format(
            a, b
        ))
    print(SEPARATOR_STRING)

    # _plot_by_cloud_regime(
    #     example_dict=example_dict, output_dir_name=output_dir_name
    # )
    _plot_by_latitude(
        example_dict=example_dict, output_dir_name=output_dir_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        example_file_names=getattr(INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
