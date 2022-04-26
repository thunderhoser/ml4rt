"""Explores data by creating plots for different subsets of profiles."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from ml4rt.io import example_io
from ml4rt.utils import example_utils
from ml4rt.plotting import profile_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

RADIANS_TO_DEGREES = 180. / numpy.pi

NUM_LATITUDE_BINS = 36
NUM_ZENITH_ANGLE_BINS = 17
NUM_EXTREME_PROFILES = 100
MIN_PATH_FOR_CLOUD_KG_M02 = 0.05
PERCENTILE_LEVELS = numpy.array([
    50, 75, 90, 95, 96, 97, 98, 99, 99.5, 99.75, 99.9, 100
])

LATITUDE_COLOUR_MAP_OBJECT = pyplot.get_cmap('twilight_shifted')
ZENITH_ANGLE_COLOUR_MAP_OBJECT = pyplot.get_cmap('inferno')

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


def _plot_profiles(example_dict, example_indices):
    """Plots a subset of heating-rate profiles.

    :param example_dict: Dictionary in format returned by
        `example_io.read_file`.
    :param example_indices: 1-D numpy array with indices of profiles to plot.
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    heating_rate_matrix_k_day01 = example_utils.get_field_from_dict(
        example_dict=example_dict,
        field_name=example_utils.SHORTWAVE_HEATING_RATE_NAME
    )

    figure_object = None
    axes_object = None

    for i in example_indices:
        this_colour = numpy.random.uniform(low=0., high=1., size=3)

        figure_object, axes_object = profile_plotting.plot_one_variable(
            values=heating_rate_matrix_k_day01[i, :],
            heights_m_agl=example_dict[example_utils.HEIGHTS_KEY],
            use_log_scale=True, line_colour=this_colour,
            figure_object=figure_object
        )

    axes_object.set_xlabel(r'Heating rate (K day$^{-1}$)')

    return figure_object, axes_object


def _plot_by_zenith_angle(example_dict, output_dir_name):
    """Plots mean heating-rate profile by solar zenith angle.

    :param example_dict: Dictionary in format returned by
        `example_io.read_file`.
    :param output_dir_name: Name of output directory.  Figure will be saved
        here.
    """

    example_zenith_angles_rad = example_utils.get_field_from_dict(
        example_dict=example_dict, field_name=example_utils.ZENITH_ANGLE_NAME
    )
    example_zenith_angles_deg = RADIANS_TO_DEGREES * example_zenith_angles_rad

    bin_edges_deg = numpy.linspace(
        0, 85, num=NUM_ZENITH_ANGLE_BINS + 1, dtype=float
    )
    bin_centers_deg = bin_edges_deg[:-1] + numpy.diff(bin_edges_deg) / 2
    colour_norm_object = matplotlib.colors.Normalize(
        vmin=bin_centers_deg[0], vmax=bin_centers_deg[-1]
    )

    example_to_bin_indices = numpy.digitize(
        x=example_zenith_angles_deg, bins=bin_edges_deg, right=False
    )
    example_to_bin_indices = numpy.maximum(example_to_bin_indices, 0)
    example_to_bin_indices = numpy.minimum(
        example_to_bin_indices, NUM_ZENITH_ANGLE_BINS - 1
    )

    heating_rate_matrix_k_day01 = example_utils.get_field_from_dict(
        example_dict=example_dict,
        field_name=example_utils.SHORTWAVE_HEATING_RATE_NAME
    )
    figure_object = None
    axes_object = None

    for k in range(NUM_ZENITH_ANGLE_BINS):
        these_example_indices = numpy.where(example_to_bin_indices == k)[0]
        if len(these_example_indices) == 0:
            continue

        this_colour = ZENITH_ANGLE_COLOUR_MAP_OBJECT(
            colour_norm_object(bin_centers_deg[k])
        )

        figure_object, axes_object = profile_plotting.plot_one_variable(
            values=numpy.mean(
                heating_rate_matrix_k_day01[these_example_indices, :], axis=0
            ),
            heights_m_agl=example_dict[example_utils.HEIGHTS_KEY],
            use_log_scale=True, line_colour=this_colour,
            figure_object=figure_object
        )

    axes_object.set_title('Heating rate by solar zenith angle')
    axes_object.set_xlabel(r'Heating rate (K day$^{-1}$)')

    colour_bar_object = plotting_utils.plot_linear_colour_bar(
        axes_object_or_matrix=axes_object, data_matrix=bin_centers_deg,
        colour_map_object=ZENITH_ANGLE_COLOUR_MAP_OBJECT,
        min_value=bin_centers_deg[0], max_value=bin_centers_deg[-1],
        orientation_string='vertical', extend_min=False, extend_max=False
    )

    tick_values = colour_bar_object.get_ticks()
    tick_strings = ['{0:.1f}'.format(v) for v in tick_values]
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)
    colour_bar_object.set_label(r'Solar zenith angle ($^{\circ}$)')

    output_file_name = '{0:s}/zenith_angle_heating_rates.jpg'.format(
        output_dir_name
    )
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)


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

    axes_object.set_title('Heating rate by latitude')
    axes_object.set_xlabel(r'Heating rate (K day$^{-1}$)')

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
    colour_bar_object.set_label(r'Latitude ($^{\circ}$N)')

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
        len(multi_layer_cloud_indices)
    ))

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

    axes_object.set_title(
        'Heating rate by cloud regime (green = clear sky; purple = multi-layer)'
    )
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
            netcdf_file_name=this_file_name,
            max_shortwave_heating_k_day01=numpy.inf
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

    _plot_by_cloud_regime(
        example_dict=example_dict, output_dir_name=output_dir_name
    )
    _plot_by_latitude(
        example_dict=example_dict, output_dir_name=output_dir_name
    )
    _plot_by_zenith_angle(
        example_dict=example_dict, output_dir_name=output_dir_name
    )

    mean_heating_rates_k_day01 = numpy.mean(heating_rate_matrix_k_day01, axis=1)
    sort_indices = numpy.argsort(-mean_heating_rates_k_day01)
    figure_object, axes_object = _plot_profiles(
        example_dict=example_dict,
        example_indices=sort_indices[:NUM_EXTREME_PROFILES]
    )
    axes_object.set_title(
        '{0:d} profiles with max average heating rates'.format(
            NUM_EXTREME_PROFILES
        )
    )
    output_file_name = '{0:s}/max_average_heating_rates.jpg'.format(
        output_dir_name
    )
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    sort_indices = numpy.argsort(mean_heating_rates_k_day01)
    figure_object, axes_object = _plot_profiles(
        example_dict=example_dict,
        example_indices=sort_indices[:NUM_EXTREME_PROFILES]
    )
    axes_object.set_title(
        '{0:d} profiles with minimum average heating rates'.format(
            NUM_EXTREME_PROFILES
        )
    )
    output_file_name = '{0:s}/min_average_heating_rates.jpg'.format(
        output_dir_name
    )
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    max_heating_rates_k_day01 = numpy.max(heating_rate_matrix_k_day01, axis=1)
    sort_indices = numpy.argsort(-max_heating_rates_k_day01)
    figure_object, axes_object = _plot_profiles(
        example_dict=example_dict,
        example_indices=sort_indices[:NUM_EXTREME_PROFILES]
    )
    axes_object.set_title(
        '{0:d} profiles with max max heating rates'.format(
            NUM_EXTREME_PROFILES
        )
    )
    output_file_name = '{0:s}/max_max_heating_rates.jpg'.format(
        output_dir_name
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
        example_file_names=getattr(INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
