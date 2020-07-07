"""Plots Grad-CAM output (class-activation maps) for all target variables."""

import os.path
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils
from ml4rt.io import example_io
from ml4rt.machine_learning import gradcam
from ml4rt.machine_learning import neural_net
from ml4rt.plotting import profile_plotting

# TODO(thunderhoser): Find some way to incorporate prediction quality in the
# plots.

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
METRES_TO_KM = 0.001

TARGET_NAME_TO_VERBOSE = {
    example_io.SHORTWAVE_DOWN_FLUX_NAME: 'down flux',
    example_io.SHORTWAVE_UP_FLUX_NAME: 'up flux',
    example_io.SHORTWAVE_HEATING_RATE_NAME: 'heating rate'
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

GRADCAM_FILE_ARG_NAME = 'input_gradcam_file_name'
COLOUR_MAP_ARG_NAME = 'colour_map_name'
MAX_PERCENTILE_ARG_NAME = 'max_colour_percentile'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

GRADCAM_FILE_HELP_STRING = (
    'Path to Grad-CAM file (will be read by `gradcam.read_all_targets_file`).'
)
COLOUR_MAP_HELP_STRING = (
    'Colour scheme (must be accepted by `matplotlib.pyplot.get_cmap`).'
)
MAX_PERCENTILE_HELP_STRING = (
    'Used to determine limits of colour bar.  For each plot, max value in '
    'colour bar will be [q]th percentile of all values in plot, where '
    'q = `{0:s}`.'
).format(MAX_PERCENTILE_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (figures will be saved here).'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + GRADCAM_FILE_ARG_NAME, type=str, required=True,
    help=GRADCAM_FILE_HELP_STRING
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


def _plot_gradcam_one_example(
        gradcam_dict, example_index, model_metadata_dict, colour_map_object,
        max_colour_percentile, output_dir_name):
    """Plots class-activation map for one example, all target variables.

    :param gradcam_dict: Dictionary read by `gradcam.read_all_targets_file`.
    :param example_index: Will plot class-activation maps for example with this
        array index.
    :param model_metadata_dict: Dictionary read by `neural_net.read_metafile`.
    :param colour_map_object: See documentation at top of file.
    :param max_colour_percentile: Same.
    :param output_dir_name: Same.
    """

    generator_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    target_names = generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY]
    target_names_verbose = [
        TARGET_NAME_TO_VERBOSE[n] for n in target_names
    ]

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

    example_id_string = gradcam_dict[gradcam.EXAMPLE_IDS_KEY][example_index]
    class_activation_matrix_3d = (
        gradcam_dict[gradcam.CLASS_ACTIVATIONS_KEY][example_index, ...]
    )

    num_targets = len(target_names)
    num_heights = len(height_labels)

    for k in range(num_targets):
        class_activation_matrix_2d = class_activation_matrix_3d[..., k]

        max_colour_value = numpy.percentile(
            class_activation_matrix_2d, max_colour_percentile
        )
        max_colour_value = numpy.maximum(max_colour_value, 0.001)

        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

        axes_object.imshow(
            numpy.transpose(class_activation_matrix_2d),
            cmap=colour_map_object, vmin=0., vmax=max_colour_value,
            origin='lower'
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
            data_matrix=class_activation_matrix_2d,
            colour_map_object=colour_map_object,
            min_value=0., max_value=max_colour_value,
            orientation_string='horizontal', padding=0.1,
            extend_min=True, extend_max=True,
            fraction_of_axis_length=0.8, font_size=DEFAULT_FONT_SIZE
        )

        tick_values = colour_bar_object.get_ticks()
        tick_strings = ['{0:.1f}'.format(v) for v in tick_values]
        colour_bar_object.set_ticks(tick_values)
        colour_bar_object.set_ticklabels(tick_strings)

        title_string = 'Class-activation map for {0:s}'.format(
            target_names_verbose[k]
        )
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


def _run(gradcam_file_name, colour_map_name, max_colour_percentile,
         output_dir_name):
    """Plots Grad-CAM output (class-activation maps) for all target variables.

    This is effectively the main method.

    :param gradcam_file_name: See documentation at top of file.
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

    print('Reading class-activation maps from: "{0:s}"...'.format(
        gradcam_file_name
    ))
    gradcam_dict = gradcam.read_all_targets_file(gradcam_file_name)

    model_file_name = gradcam_dict[gradcam.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0]
    )

    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)

    num_examples = len(gradcam_dict[gradcam.EXAMPLE_IDS_KEY])
    print(SEPARATOR_STRING)

    for i in range(num_examples):
        _plot_gradcam_one_example(
            gradcam_dict=gradcam_dict, example_index=i,
            model_metadata_dict=model_metadata_dict,
            colour_map_object=colour_map_object,
            max_colour_percentile=max_colour_percentile,
            output_dir_name=output_dir_name
        )
        print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        gradcam_file_name=getattr(INPUT_ARG_OBJECT, GRADCAM_FILE_ARG_NAME),
        colour_map_name=getattr(INPUT_ARG_OBJECT, COLOUR_MAP_ARG_NAME),
        max_colour_percentile=getattr(
            INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
