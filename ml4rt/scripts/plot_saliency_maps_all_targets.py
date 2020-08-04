"""Plots saliency maps for all target variables."""

import os.path
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils
from ml4rt.utils import example_utils
from ml4rt.machine_learning import saliency
from ml4rt.machine_learning import neural_net
from ml4rt.plotting import profile_plotting
from ml4rt.scripts import plot_saliency_maps

# TODO(thunderhoser): Find some way to incorporate prediction quality in the
# plots.

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
METRES_TO_KM = 0.001

TARGET_NAME_TO_VERBOSE = plot_saliency_maps.TARGET_NAME_TO_VERBOSE

PREDICTOR_NAME_TO_VERBOSE = {
    example_utils.TEMPERATURE_NAME: 'temperature',
    example_utils.SPECIFIC_HUMIDITY_NAME: 'spec humidity',
    example_utils.RELATIVE_HUMIDITY_NAME: 'rel humidity',
    example_utils.WATER_VAPOUR_PATH_NAME: 'downward WVP',
    example_utils.UPWARD_WATER_VAPOUR_PATH_NAME: 'upward WVP',
    example_utils.PRESSURE_NAME: 'pressure',
    example_utils.LIQUID_WATER_CONTENT_NAME: 'LWC',
    example_utils.ICE_WATER_CONTENT_NAME: 'IWC',
    example_utils.LIQUID_WATER_PATH_NAME: 'downward LWP',
    example_utils.ICE_WATER_PATH_NAME: 'downward IWP',
    example_utils.UPWARD_LIQUID_WATER_PATH_NAME: 'upward LWP',
    example_utils.UPWARD_ICE_WATER_PATH_NAME: 'upward IWP',
    example_utils.ZENITH_ANGLE_NAME: 'zenith angle',
    example_utils.LATITUDE_NAME: 'latitude',
    example_utils.LONGITUDE_NAME: 'longitude',
    example_utils.ALBEDO_NAME: 'albedo',
    example_utils.COLUMN_LIQUID_WATER_PATH_NAME: 'column LWP',
    example_utils.COLUMN_ICE_WATER_PATH_NAME: 'column IWP'
}

REFERENCE_LINE_COLOUR = numpy.full(3, 152. / 255)
REFERENCE_LINE_WIDTH = 2

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


def _plot_saliency_scalar_p_scalar_t(
        saliency_matrix, predictor_names, target_names,
        example_id_string, colour_map_object, max_colour_percentile,
        output_dir_name):
    """Plots saliency for one example: scalar predictors, scalar targets.

    P = number of predictor variables
    T = number of target variables

    :param saliency_matrix: P-by-T numpy array of saliency values.
    :param predictor_names: length-P list of predictor names.
    :param target_names: length-T list of target names.
    :param example_id_string: Example ID.
    :param colour_map_object: See documentation at top of file.
    :param max_colour_percentile: Same.
    :param output_dir_name: Same.
    """

    predictor_names_verbose = [
        PREDICTOR_NAME_TO_VERBOSE[n] for n in predictor_names
    ]
    target_names_verbose = [
        TARGET_NAME_TO_VERBOSE[n] for n in target_names
    ]

    max_colour_value = numpy.percentile(
        numpy.absolute(saliency_matrix), max_colour_percentile
    )
    max_colour_value = numpy.maximum(max_colour_value, 0.001)
    min_colour_value = -1 * max_colour_value

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    axes_object.imshow(
        numpy.transpose(saliency_matrix), cmap=colour_map_object,
        vmin=min_colour_value, vmax=max_colour_value, origin='lower'
    )

    num_predictors = len(predictor_names)
    num_targets = len(target_names)
    x_tick_values = numpy.linspace(
        0, num_predictors - 1, num=num_predictors, dtype=float
    )
    y_tick_values = numpy.linspace(
        0, num_targets - 1, num=num_targets, dtype=float
    )
    axes_object.set_xticks(x_tick_values)
    axes_object.set_yticks(y_tick_values)

    x_tick_labels = [
        '{0:s}{1:s}'.format(n[0].upper(), n[1:])
        for n in predictor_names_verbose
    ]
    y_tick_labels = [
        '{0:s}{1:s}'.format(n[0].upper(), n[1:]) for n in target_names_verbose
    ]
    axes_object.set_xticklabels(
        x_tick_labels, fontsize=TICK_LABEL_FONT_SIZE, rotation=90.
    )
    axes_object.set_yticklabels(
        y_tick_labels, fontsize=TICK_LABEL_FONT_SIZE
    )

    axes_object.set_xlabel('Predictor')
    axes_object.set_ylabel('Target')

    orientation_string = (
        'horizontal' if len(x_tick_values) >= len(y_tick_values)
        else 'vertical'
    )

    colour_bar_object = plotting_utils.plot_linear_colour_bar(
        axes_object_or_matrix=axes_object, data_matrix=saliency_matrix,
        colour_map_object=colour_map_object,
        min_value=min_colour_value, max_value=max_colour_value,
        orientation_string=orientation_string,
        padding=0.1 if orientation_string == 'horizontal' else 0.01,
        extend_min=True, extend_max=True,
        fraction_of_axis_length=0.8, font_size=DEFAULT_FONT_SIZE
    )

    tick_values = colour_bar_object.get_ticks()
    tick_strings = ['{0:.1f}'.format(v) for v in tick_values]
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    axes_object.set_title(
        'Saliency for scalar targets with respect to scalar predictors',
        fontsize=DEFAULT_FONT_SIZE
    )

    output_file_name = '{0:s}/{1:s}_scalars.jpg'.format(
        output_dir_name, example_id_string.replace('_', '-')
    )
    print('Saving figure to: "{0:s}"...'.format(output_file_name))

    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _plot_saliency_vector_p_scalar_t(
        saliency_matrix, predictor_names, target_names, height_labels,
        example_id_string, colour_map_object, max_colour_percentile,
        output_dir_name):
    """Plots saliency for one example: vector predictors, scalar targets.

    P = number of predictor variables
    T = number of target variables
    H = number of heights

    :param saliency_matrix: H-by-P-by-T numpy array of saliency values.
    :param predictor_names: length-P list of predictor names.
    :param target_names: length-T list of target names.
    :param height_labels: length-H list of height labels (strings).
    :param example_id_string: Example ID.
    :param colour_map_object: See documentation at top of file.
    :param max_colour_percentile: Same.
    :param output_dir_name: Same.
    """

    predictor_names_verbose = [
        PREDICTOR_NAME_TO_VERBOSE[n] for n in predictor_names
    ]
    target_names_verbose = [
        TARGET_NAME_TO_VERBOSE[n] for n in target_names
    ]

    num_targets = len(target_names)
    num_predictors = len(predictor_names)
    num_heights = len(height_labels)

    for k in range(num_targets):
        max_colour_value = numpy.percentile(
            numpy.absolute(saliency_matrix[..., k]), max_colour_percentile
        )
        max_colour_value = numpy.maximum(max_colour_value, 0.001)
        min_colour_value = -1 * max_colour_value

        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )
        axes_object.imshow(
            saliency_matrix[..., k], cmap=colour_map_object,
            vmin=min_colour_value, vmax=max_colour_value, origin='lower'
        )

        x_tick_values = numpy.linspace(
            0, num_predictors - 1, num=num_predictors, dtype=float
        )
        y_tick_values = numpy.linspace(
            0, num_heights - 1, num=num_heights, dtype=float
        )
        axes_object.set_xticks(x_tick_values)
        axes_object.set_yticks(y_tick_values)

        x_tick_labels = [
            '{0:s}{1:s}'.format(n[0].upper(), n[1:])
            for n in predictor_names_verbose
        ]
        axes_object.set_xticklabels(
            x_tick_labels, fontsize=TICK_LABEL_FONT_SIZE, rotation=90.
        )
        axes_object.set_yticklabels(
            height_labels, fontsize=TICK_LABEL_FONT_SIZE
        )

        axes_object.set_xlabel('Predictor variable')
        axes_object.set_ylabel('Predictor height (km AGL)')

        orientation_string = (
            'horizontal' if len(x_tick_values) >= len(y_tick_values)
            else 'vertical'
        )

        colour_bar_object = plotting_utils.plot_linear_colour_bar(
            axes_object_or_matrix=axes_object,
            data_matrix=saliency_matrix[..., k],
            colour_map_object=colour_map_object,
            min_value=min_colour_value, max_value=max_colour_value,
            orientation_string=orientation_string,
            padding=0.1 if orientation_string == 'horizontal' else 0.01,
            extend_min=True, extend_max=True,
            fraction_of_axis_length=0.8, font_size=DEFAULT_FONT_SIZE
        )

        tick_values = colour_bar_object.get_ticks()
        tick_strings = ['{0:.1f}'.format(v) for v in tick_values]
        colour_bar_object.set_ticks(tick_values)
        colour_bar_object.set_ticklabels(tick_strings)

        title_string = 'Saliency for {0:s}'.format(target_names_verbose[k])
        axes_object.set_title(title_string, fontsize=DEFAULT_FONT_SIZE)

        output_file_name = '{0:s}/{1:s}_{2:s}.jpg'.format(
            output_dir_name, example_id_string.replace('_', '-'),
            target_names[k].replace('_', '-')
        )
        print('Saving figure to: "{0:s}"...'.format(output_file_name))

        figure_object.savefig(
            output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
            bbox_inches='tight'
        )
        pyplot.close(figure_object)


def _plot_saliency_scalar_p_vector_t(
        saliency_matrix, predictor_names, target_names, height_labels,
        example_id_string, colour_map_object, max_colour_percentile,
        output_dir_name):
    """Plots saliency for one example: scalar predictors, vector targets.

    P = number of predictor variables
    T = number of target variables
    H = number of heights

    :param saliency_matrix: P-by-H-by-T numpy array of saliency values.
    :param predictor_names: length-P list of predictor names.
    :param target_names: length-T list of target names.
    :param height_labels: length-H list of height labels (strings).
    :param example_id_string: Example ID.
    :param colour_map_object: See documentation at top of file.
    :param max_colour_percentile: Same.
    :param output_dir_name: Same.
    """

    predictor_names_verbose = [
        PREDICTOR_NAME_TO_VERBOSE[n] for n in predictor_names
    ]
    target_names_verbose = [
        TARGET_NAME_TO_VERBOSE[n] for n in target_names
    ]

    num_targets = len(target_names)
    num_predictors = len(predictor_names)
    num_heights = len(height_labels)

    for k in range(num_targets):
        max_colour_value = numpy.percentile(
            numpy.absolute(saliency_matrix[..., k]), max_colour_percentile
        )
        max_colour_value = numpy.maximum(max_colour_value, 0.001)
        min_colour_value = -1 * max_colour_value

        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

        axes_object.imshow(
            numpy.transpose(saliency_matrix[..., k]), cmap=colour_map_object,
            vmin=min_colour_value, vmax=max_colour_value, origin='lower'
        )

        x_tick_values = numpy.linspace(
            0, num_predictors - 1, num=num_predictors, dtype=float
        )
        y_tick_values = numpy.linspace(
            0, num_heights - 1, num=num_heights, dtype=float
        )
        axes_object.set_xticks(x_tick_values)
        axes_object.set_yticks(y_tick_values)

        x_tick_labels = [
            '{0:s}{1:s}'.format(n[0].upper(), n[1:])
            for n in predictor_names_verbose
        ]
        axes_object.set_xticklabels(
            x_tick_labels, fontsize=TICK_LABEL_FONT_SIZE, rotation=90.
        )
        axes_object.set_yticklabels(
            height_labels, fontsize=TICK_LABEL_FONT_SIZE
        )

        axes_object.set_xlabel('Predictor variable')
        axes_object.set_ylabel('Target height (km AGL)')

        orientation_string = (
            'horizontal' if len(x_tick_values) >= len(y_tick_values)
            else 'vertical'
        )

        colour_bar_object = plotting_utils.plot_linear_colour_bar(
            axes_object_or_matrix=axes_object,
            data_matrix=saliency_matrix[..., k],
            colour_map_object=colour_map_object,
            min_value=min_colour_value, max_value=max_colour_value,
            orientation_string=orientation_string,
            padding=0.1 if orientation_string == 'horizontal' else 0.01,
            extend_min=True, extend_max=True,
            fraction_of_axis_length=0.8, font_size=DEFAULT_FONT_SIZE
        )

        tick_values = colour_bar_object.get_ticks()
        tick_strings = ['{0:.1f}'.format(v) for v in tick_values]
        colour_bar_object.set_ticks(tick_values)
        colour_bar_object.set_ticklabels(tick_strings)

        title_string = 'Saliency for {0:s}'.format(target_names_verbose[k])
        axes_object.set_title(title_string, fontsize=DEFAULT_FONT_SIZE)

        output_file_name = '{0:s}/{1:s}_{2:s}.jpg'.format(
            output_dir_name, example_id_string.replace('_', '-'),
            target_names[k].replace('_', '-')
        )
        print('Saving figure to: "{0:s}"...'.format(output_file_name))

        figure_object.savefig(
            output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
            bbox_inches='tight'
        )
        pyplot.close(figure_object)


def _plot_saliency_vector_p_vector_t(
        saliency_matrix, predictor_names, target_names, height_labels,
        example_id_string, colour_map_object, max_colour_percentile,
        output_dir_name):
    """Plots saliency for one example: vector predictors, vector targets.

    P = number of predictor variables
    T = number of target variables
    H = number of heights

    :param saliency_matrix: H-by-P-by-H-by-T numpy array of saliency values.
    :param predictor_names: length-P list of predictor names.
    :param target_names: length-T list of target names.
    :param height_labels: length-H list of height labels (strings).
    :param example_id_string: Example ID.
    :param colour_map_object: See documentation at top of file.
    :param max_colour_percentile: Same.
    :param output_dir_name: Same.
    """

    predictor_names_verbose = [
        PREDICTOR_NAME_TO_VERBOSE[n] for n in predictor_names
    ]
    target_names_verbose = [
        TARGET_NAME_TO_VERBOSE[n] for n in target_names
    ]

    num_targets = len(target_names)
    num_predictors = len(predictor_names)
    num_heights = len(height_labels)

    for j in range(num_predictors):
        for k in range(num_targets):
            max_colour_value = numpy.percentile(
                numpy.abs(saliency_matrix[:, j, :, k]), max_colour_percentile
            )
            max_colour_value = numpy.maximum(max_colour_value, 0.001)
            min_colour_value = -1 * max_colour_value

            figure_object, axes_object = pyplot.subplots(
                1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
            )

            axes_object.imshow(
                numpy.transpose(saliency_matrix[:, j, :, k]),
                cmap=colour_map_object, vmin=min_colour_value,
                vmax=max_colour_value, origin='lower'
            )

            tick_values = numpy.linspace(
                0, num_heights - 1, num=num_heights, dtype=float
            )
            axes_object.set_xticks(tick_values)
            axes_object.set_yticks(tick_values)

            axes_object.set_xticklabels(
                height_labels, fontsize=TICK_LABEL_FONT_SIZE, rotation=90.
            )
            axes_object.set_yticklabels(
                height_labels, fontsize=TICK_LABEL_FONT_SIZE
            )

            axes_object.set_xlabel('Predictor height (km AGL)')
            axes_object.set_ylabel('Target height (km AGL)')

            axes_object.plot(
                axes_object.get_xlim(), axes_object.get_ylim(),
                color=REFERENCE_LINE_COLOUR, linestyle='dashed',
                linewidth=REFERENCE_LINE_WIDTH
            )

            colour_bar_object = plotting_utils.plot_linear_colour_bar(
                axes_object_or_matrix=axes_object,
                data_matrix=saliency_matrix[:, j, :, k],
                colour_map_object=colour_map_object,
                min_value=min_colour_value, max_value=max_colour_value,
                orientation_string='horizontal', padding=0.1,
                extend_min=True, extend_max=True,
                fraction_of_axis_length=0.8, font_size=DEFAULT_FONT_SIZE
            )

            tick_values = colour_bar_object.get_ticks()
            tick_strings = ['{0:.1f}'.format(v) for v in tick_values]
            colour_bar_object.set_ticks(tick_values)
            colour_bar_object.set_ticklabels(tick_strings)

            title_string = 'Saliency for {0:s} with respect to {1:s}'.format(
                target_names_verbose[k], predictor_names_verbose[j]
            )
            axes_object.set_title(title_string, fontsize=DEFAULT_FONT_SIZE)

            output_file_name = '{0:s}/{1:s}_{2:s}_{3:s}.jpg'.format(
                output_dir_name, example_id_string.replace('_', '-'),
                predictor_names[j].replace('_', '-'),
                target_names[k].replace('_', '-')
            )
            print('Saving figure to: "{0:s}"...'.format(output_file_name))

            figure_object.savefig(
                output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
                bbox_inches='tight'
            )
            pyplot.close(figure_object)


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

    heights_km_agl = (
        METRES_TO_KM * generator_option_dict[neural_net.HEIGHTS_KEY]
    )
    height_labels = profile_plotting.create_height_labels(
        tick_values_km_agl=heights_km_agl, use_log_scale=False
    )
    height_labels = [
        height_labels[k] if numpy.mod(k, 4) == 0 else ' '
        for k in range(len(height_labels))
    ]

    example_id_string = saliency_dict[saliency.EXAMPLE_IDS_KEY][example_index]
    this_matrix = (
        saliency_dict[saliency.SALIENCY_SCALAR_P_SCALAR_T_KEY][
            example_index, ...
        ]
    )

    is_dense_net = len(this_matrix.shape) == 2

    if this_matrix.size > 0:
        if is_dense_net:
            _plot_saliency_scalar_p_scalar_t(
                saliency_matrix=this_matrix,
                predictor_names=scalar_predictor_names,
                target_names=scalar_target_names,
                example_id_string=example_id_string,
                colour_map_object=colour_map_object,
                max_colour_percentile=max_colour_percentile,
                output_dir_name=output_dir_name
            )
        else:
            _plot_saliency_vector_p_scalar_t(
                saliency_matrix=this_matrix,
                predictor_names=scalar_predictor_names,
                target_names=scalar_target_names,
                height_labels=height_labels,
                example_id_string=example_id_string,
                colour_map_object=colour_map_object,
                max_colour_percentile=max_colour_percentile,
                output_dir_name=output_dir_name
            )

    this_matrix = (
        saliency_dict[saliency.SALIENCY_VECTOR_P_SCALAR_T_KEY][
            example_index, ...
        ]
    )

    if this_matrix.size > 0:
        _plot_saliency_vector_p_scalar_t(
            saliency_matrix=this_matrix,
            predictor_names=vector_predictor_names,
            target_names=scalar_target_names,
            height_labels=height_labels,
            example_id_string=example_id_string,
            colour_map_object=colour_map_object,
            max_colour_percentile=max_colour_percentile,
            output_dir_name=output_dir_name
        )

    this_matrix = (
        saliency_dict[saliency.SALIENCY_SCALAR_P_VECTOR_T_KEY][
            example_index, ...
        ]
    )

    if this_matrix.size > 0:
        if is_dense_net:
            _plot_saliency_scalar_p_vector_t(
                saliency_matrix=this_matrix,
                predictor_names=scalar_predictor_names,
                target_names=vector_target_names,
                height_labels=height_labels,
                example_id_string=example_id_string,
                colour_map_object=colour_map_object,
                max_colour_percentile=max_colour_percentile,
                output_dir_name=output_dir_name
            )
        else:
            _plot_saliency_vector_p_vector_t(
                saliency_matrix=this_matrix,
                predictor_names=scalar_predictor_names,
                target_names=vector_target_names,
                height_labels=height_labels,
                example_id_string=example_id_string,
                colour_map_object=colour_map_object,
                max_colour_percentile=max_colour_percentile,
                output_dir_name=output_dir_name
            )

    this_matrix = (
        saliency_dict[saliency.SALIENCY_VECTOR_P_VECTOR_T_KEY][
            example_index, ...
        ]
    )

    if this_matrix.size > 0:
        _plot_saliency_vector_p_vector_t(
            saliency_matrix=this_matrix,
            predictor_names=vector_predictor_names,
            target_names=vector_target_names,
            height_labels=height_labels,
            example_id_string=example_id_string,
            colour_map_object=colour_map_object,
            max_colour_percentile=max_colour_percentile,
            output_dir_name=output_dir_name
        )


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
    saliency_dict = saliency.read_all_targets_file(saliency_file_name)

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
