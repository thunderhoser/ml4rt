"""Explores data by creating plots for different subsets of profiles."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from ml4rt.io import example_io
from ml4rt.utils import example_utils
from ml4rt.plotting import profile_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MIN_PATH_FOR_CLOUD_KG_M02 = 0.05
PERCENTILE_LEVELS = numpy.array([
    50, 75, 90, 95, 96, 97, 98, 99, 99.5, 99.75, 99.9, 100
])

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


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        example_file_names=getattr(INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
