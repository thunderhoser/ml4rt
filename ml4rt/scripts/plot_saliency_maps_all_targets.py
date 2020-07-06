"""Plots saliency maps for all target variables."""

import os.path
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4rt.io import example_io
from ml4rt.machine_learning import saliency
from ml4rt.machine_learning import neural_net
from ml4rt.plotting import profile_plotting

# TODO(thunderhoser): Find some way to incorporate prediction quality in the
# plots.

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
METRES_TO_KM = 0.001

TARGET_NAME_TO_VERBOSE = {
    example_io.SHORTWAVE_DOWN_FLUX_NAME: 'down flux',
    example_io.SHORTWAVE_UP_FLUX_NAME: 'up flux',
    example_io.SHORTWAVE_HEATING_RATE_NAME: 'heating rate',
    example_io.SHORTWAVE_SURFACE_DOWN_FLUX_NAME: 'sfc down flux',
    example_io.SHORTWAVE_TOA_UP_FLUX_NAME: 'TOA up flux'
}

PREDICTOR_NAME_TO_VERBOSE = {
    example_io.TEMPERATURE_NAME: 'temperature',
    example_io.SPECIFIC_HUMIDITY_NAME: 'humidity',
    example_io.PRESSURE_NAME: 'pressure',
    example_io.LIQUID_WATER_CONTENT_NAME: 'LWC',
    example_io.ICE_WATER_CONTENT_NAME: 'IWC',
    example_io.ZENITH_ANGLE_NAME: 'zenith angle',
    example_io.LATITUDE_NAME: 'latitude',
    example_io.LONGITUDE_NAME: 'longitude',
    example_io.ALBEDO_NAME: 'albedo',
    example_io.LIQUID_WATER_PATH_NAME: 'LWP',
    example_io.ICE_WATER_PATH_NAME: 'IWP'
}

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

DEFAULT_FONT_SIZE = 20
TICK_LABEL_FONT_SIZE = 20

pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

SALIENCY_FILE_ARG_NAME = 'input_saliency_file_name'
COLOUR_MAP_ARG_NAME = 'colour_map_name'
MAX_PERCENTILE_ARG_NAME = 'max_colour_percentile'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

SALIENCY_FILE_HELP_STRING = (
    'Path to saliency file (will be read by `saliency.read_all_targets_file`).'
)
COLOUR_MAP_HELP_STRING = (
    'Colour scheme (must be accepted by `matplotlib.pyplot.get_cmap`).'
)
MAX_PERCENTILE_HELP_STRING = (
    'Used to determine limits of colour bar.  For each plot, max absolute value'
    ' in colour bar will be [q]th percentile of all values in plot, where '
    'q = `{0:s}`.'
).format(MAX_PERCENTILE_ARG_NAME)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (figures will be saved here).'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + SALIENCY_FILE_ARG_NAME, type=str, required=True,
    help=SALIENCY_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + COLOUR_MAP_ARG_NAME, type=str, required=False, default='seismic',
    help=COLOUR_MAP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=False, default=99.,
    help=MAX_PERCENTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_saliency_one_example(
        saliency_dict, example_index, model_metadata_dict, colour_map_object,
        max_colour_percentile, output_dir_name):
    """Plots saliency maps for one example.

    :param saliency_dict: Dictionary read by `saliency.read_all_targets_file`.
    :param example_index: Will plot saliency maps for example with this array
        index.
    :param model_metadata_dict: Dictionary read by `neural_net.read_metafile`.
    :param colour_map_object: See documentation at top of file.
    :param max_colour_percentile: Same.
    :param output_dir_name: Same.
    """

    # Housekeeping.
    generator_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]

    scalar_predictor_names = (
        generator_option_dict[neural_net.SCALAR_PREDICTOR_NAMES_KEY]
    )
    scalar_target_names = (
        generator_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY]
    )
    vector_predictor_names = (
        generator_option_dict[neural_net.VECTOR_PREDICTOR_NAMES_KEY]
    )
    vector_target_names = (
        generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY]
    )

    scalar_predictor_names_verbose = [
        PREDICTOR_NAME_TO_VERBOSE[n] for n in scalar_predictor_names
    ]
    scalar_target_names_verbose = [
        TARGET_NAME_TO_VERBOSE[n] for n in scalar_target_names
    ]
    vector_predictor_names_verbose = [
        PREDICTOR_NAME_TO_VERBOSE[n] for n in vector_predictor_names
    ]
    vector_target_names_verbose = [
        TARGET_NAME_TO_VERBOSE[n] for n in vector_target_names
    ]

    heights_km_agl = (
        METRES_TO_KM * generator_option_dict[neural_net.HEIGHTS_KEY]
    )
    height_labels = profile_plotting.create_log_height_labels(heights_km_agl)

    num_scalar_predictors = len(scalar_predictor_names)
    num_scalar_targets = len(scalar_target_names)
    num_vector_predictors = len(vector_predictor_names)
    num_vector_targets = len(vector_target_names)
    num_heights = len(heights_km_agl)

    i = example_index
    example_id_string = saliency_dict[saliency.EXAMPLE_IDS_KEY][i]
    saliency_matrix_scalar_p_scalar_t = (
        saliency_dict[saliency.SALIENCY_SCALAR_P_SCALAR_T_KEY][i, ...]
    )
    saliency_matrix_vector_p_scalar_t = (
        saliency_dict[saliency.SALIENCY_VECTOR_P_SCALAR_T_KEY][i, ...]
    )
    saliency_matrix_scalar_p_vector_t = (
        saliency_dict[saliency.SALIENCY_SCALAR_P_VECTOR_T_KEY][i, ...]
    )
    saliency_matrix_vector_p_vector_t = (
        saliency_dict[saliency.SALIENCY_VECTOR_P_VECTOR_T_KEY][i, ...]
    )

    if saliency_matrix_scalar_p_scalar_t.size > 0:
        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

        max_colour_value = numpy.percentile(
            saliency_matrix_scalar_p_scalar_t, max_colour_percentile
        )
        min_colour_value = -1 * max_colour_value

        axes_object.imshow(
            numpy.transpose(saliency_matrix_scalar_p_scalar_t),
            cmap=colour_map_object, vmin=min_colour_value,
            vmax=max_colour_value, origin='lower'
        )

        x_tick_values = numpy.linspace(
            0, num_scalar_predictors - 1, num=num_scalar_predictors, dtype=float
        )
        y_tick_values = numpy.linspace(
            0, num_scalar_targets - 1, num=num_scalar_targets, dtype=float
        )
        axes_object.set_xticks(x_tick_values)
        axes_object.set_yticks(y_tick_values)

        x_tick_labels = [
            '{0:s}{1:s}'.format(n[0].upper(), n[1:])
            for n in scalar_predictor_names_verbose
        ]
        y_tick_labels = [
            '{0:s}{1:s}'.format(n[0].upper(), n[1:])
            for n in scalar_target_names_verbose
        ]
        axes_object.set_xticklabels(
            x_tick_labels, fontsize=TICK_LABEL_FONT_SIZE, rotation=90.
        )
        axes_object.set_yticklabels(
            y_tick_labels, fontsize=TICK_LABEL_FONT_SIZE
        )

        axes_object.set_xlabel('Predictor')
        axes_object.set_ylabel('Target')

        output_file_name = '{0:s}/{1:s}_scalars.jpg'.format(
            output_dir_name, example_id_string.replace('_', '-')
        )
        print('Saving figure to: "{0:s}"...'.format(output_file_name))

        figure_object.savefig(
            output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
            bbox_inches='tight'
        )
        pyplot.close(figure_object)

    for k in range(num_scalar_targets):
        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

        max_colour_value = numpy.percentile(
            saliency_matrix_vector_p_scalar_t[..., k], max_colour_percentile
        )
        min_colour_value = -1 * max_colour_value

        axes_object.imshow(
            saliency_matrix_vector_p_scalar_t[..., k], cmap=colour_map_object,
            vmin=min_colour_value, vmax=max_colour_value, origin='lower'
        )

        x_tick_values = numpy.linspace(
            0, num_vector_predictors - 1, num=num_vector_predictors, dtype=float
        )
        y_tick_values = numpy.linspace(
            0, num_heights - 1, num=num_heights, dtype=float
        )
        axes_object.set_xticks(x_tick_values)
        axes_object.set_yticks(y_tick_values)

        x_tick_labels = [
            '{0:s}{1:s}'.format(n[0].upper(), n[1:])
            for n in vector_predictor_names_verbose
        ]
        axes_object.set_xticklabels(
            x_tick_labels, fontsize=TICK_LABEL_FONT_SIZE, rotation=90.
        )
        axes_object.set_yticklabels(
            height_labels, fontsize=TICK_LABEL_FONT_SIZE
        )

        axes_object.set_xlabel('Predictor variable')
        axes_object.set_ylabel('Predictor height (km AGL)')

        output_file_name = '{0:s}/{1:s}_{2:s}.jpg'.format(
            output_dir_name, example_id_string.replace('_', '-'),
            scalar_target_names[k].replace('_', '-')
        )
        print('Saving figure to: "{0:s}"...'.format(output_file_name))

        figure_object.savefig(
            output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
            bbox_inches='tight'
        )
        pyplot.close(figure_object)

    for j in range(num_scalar_predictors):
        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

        max_colour_value = numpy.percentile(
            saliency_matrix_scalar_p_vector_t[j, ...], max_colour_percentile
        )
        min_colour_value = -1 * max_colour_value

        axes_object.imshow(
            saliency_matrix_scalar_p_vector_t[j, ...], cmap=colour_map_object,
            vmin=min_colour_value, vmax=max_colour_value, origin='lower'
        )

        x_tick_values = numpy.linspace(
            0, num_vector_targets - 1, num=num_vector_targets, dtype=float
        )
        y_tick_values = numpy.linspace(
            0, num_heights - 1, num=num_heights, dtype=float
        )
        axes_object.set_xticks(x_tick_values)
        axes_object.set_yticks(y_tick_values)

        x_tick_labels = [
            '{0:s}{1:s}'.format(n[0].upper(), n[1:])
            for n in vector_target_names_verbose
        ]
        axes_object.set_xticklabels(
            x_tick_labels, fontsize=TICK_LABEL_FONT_SIZE, rotation=90.
        )
        axes_object.set_yticklabels(
            height_labels, fontsize=TICK_LABEL_FONT_SIZE
        )

        axes_object.set_xlabel('Target variable')
        axes_object.set_ylabel('Target height (km AGL)')

        output_file_name = '{0:s}/{1:s}_{2:s}.jpg'.format(
            output_dir_name, example_id_string.replace('_', '-'),
            scalar_predictor_names[j].replace('_', '-')
        )
        print('Saving figure to: "{0:s}"...'.format(output_file_name))

        figure_object.savefig(
            output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
            bbox_inches='tight'
        )
        pyplot.close(figure_object)

    for j in range(num_vector_predictors):
        for k in range(num_vector_targets):
            figure_object, axes_object = pyplot.subplots(
                1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
            )

            max_colour_value = numpy.percentile(
                saliency_matrix_vector_p_vector_t[:, j, :, k],
                max_colour_percentile
            )
            min_colour_value = -1 * max_colour_value

            axes_object.imshow(
                numpy.transpose(saliency_matrix_vector_p_vector_t[:, j, :, k]),
                cmap=colour_map_object, vmin=min_colour_value,
                vmax=max_colour_value, origin='lower'
            )

            tick_values = numpy.linspace(
                0, num_heights - 1, num=num_heights, dtype=float
            )
            axes_object.set_xticks(tick_values)
            axes_object.set_yticks(tick_values)

            axes_object.set_xticklabels(
                height_labels, fontsize=TICK_LABEL_FONT_SIZE
            )
            axes_object.set_yticklabels(
                height_labels, fontsize=TICK_LABEL_FONT_SIZE
            )

            axes_object.set_xlabel('Predictor height (km AGL)')
            axes_object.set_ylabel('Target height (km AGL)')

            output_file_name = '{0:s}/{1:s}_{2:s}_{3:s}.jpg'.format(
                output_dir_name, example_id_string.replace('_', '-'),
                vector_predictor_names[j].replace('_', '-'),
                vector_target_names[k].replace('_', '-')
            )
            print('Saving figure to: "{0:s}"...'.format(output_file_name))

            figure_object.savefig(
                output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
                bbox_inches='tight'
            )
            pyplot.close(figure_object)


def _run(saliency_file_name, colour_map_name, max_colour_percentile,
         output_dir_name):
    """Plots saliency maps for all target variables.

    This is effectively the main method.

    :param saliency_file_name: See documentation at top of file.
    :param colour_map_name: Same.
    :param max_colour_percentile: Same.
    :param output_dir_name: Same.
    """

    colour_map_object = pyplot.get_cmap(colour_map_name)
    error_checking.assert_is_geq(max_colour_percentile, 90.)
    error_checking.assert_is_leq(max_colour_percentile, 100.)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    print('Reading saliency values from: "{0:s}"...'.format(saliency_file_name))
    saliency_dict = saliency.read_standard_file(saliency_file_name)

    model_file_name = saliency_dict[saliency.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0]
    )

    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)

    num_examples = len(saliency_dict[saliency.EXAMPLE_IDS_KEY])
    print(SEPARATOR_STRING)

    for i in range(num_examples):
        _plot_saliency_one_example(
            saliency_dict=saliency_dict, example_index=i,
            model_metadata_dict=model_metadata_dict,
            colour_map_object=colour_map_object,
            max_colour_percentile=max_colour_percentile,
            output_dir_name=output_dir_name
        )
        print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        saliency_file_name=getattr(INPUT_ARG_OBJECT, SALIENCY_FILE_ARG_NAME),
        colour_map_name=getattr(INPUT_ARG_OBJECT, COLOUR_MAP_ARG_NAME),
        max_colour_percentile=getattr(
            INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
