"""Plots Grad-CAM output (class-activation maps)."""

import os.path
import argparse
import numpy
import matplotlib

matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from ml4rt.io import example_io
from ml4rt.machine_learning import gradcam
from ml4rt.machine_learning import neural_net
from ml4rt.plotting import profile_plotting
from ml4rt.scripts import plot_saliency_maps

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

METRES_TO_KM = 0.001

LINE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
LINE_WIDTH = 3

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

DEFAULT_FONT_SIZE = 24
TITLE_FONT_SIZE = 16

pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

GRADCAM_FILE_ARG_NAME = 'input_gradcam_file_name'
PREDICTION_FILE_ARG_NAME = 'input_prediction_file_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

GRADCAM_FILE_HELP_STRING = (
    'Path to Grad-CAM file (will be read by `gradcam.read_standard_file`).'
)
PREDICTION_FILE_HELP_STRING = (
    'Path to prediction file (will be read by `prediction_io.read_file`).  For '
    'each example in the saliency file, this script will find the predicted/'
    'actual target values in the prediction file and include these in the title'
    ' of the plot.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (figures will be saved here).'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + GRADCAM_FILE_ARG_NAME, type=str, required=True,
    help=GRADCAM_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_FILE_ARG_NAME, type=str, required=False, default='',
    help=PREDICTION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_one_example(
        gradcam_dict, example_index, model_metadata_dict, title_string,
        output_dir_name):
    """Plots class-activation map for one example.

    :param gradcam_dict: Dictionary read by `gradcam.read_standard_file`.
    :param example_index: Will plot class-activation map for example with this
        array index.
    :param model_metadata_dict: Dictionary read by `neural_net.read_metafile`.
    :param title_string: Figure title.
    :param output_dir_name: Name of output directory.  Figure will be saved
        here.
    """

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    axes_object.set_yscale('log')

    class_activations = (
        gradcam_dict[gradcam.CLASS_ACTIVATIONS_KEY][example_index, ...]
    )
    example_id_string = gradcam_dict[gradcam.EXAMPLE_IDS_KEY][example_index]

    generator_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    heights_km_agl = METRES_TO_KM * (
        generator_option_dict[neural_net.HEIGHTS_KEY]
    )

    axes_object.plot(
        class_activations, heights_km_agl, color=LINE_COLOUR,
        linewidth=LINE_WIDTH
    )

    y_tick_strings = profile_plotting.create_height_labels(
        tick_values_km_agl=axes_object.get_yticks(), use_log_scale=True
    )
    axes_object.set_yticklabels(y_tick_strings)

    axes_object.set_xlabel('Class activation')
    axes_object.set_ylabel('Height (km AGL)')

    axes_object.set_title(title_string, fontsize=TITLE_FONT_SIZE)

    output_file_name = '{0:s}/{1:s}.jpg'.format(
        output_dir_name, example_id_string.replace('_', '-')
    )
    print('Saving figure to: "{0:s}"...'.format(output_file_name))

    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run(gradcam_file_name, prediction_file_name, output_dir_name):
    """Plots Grad-CAM output (class-activation maps).

    This is effectively the main method.

    :param gradcam_file_name: See documentation at top of file.
    :param prediction_file_name: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    print('Reading class-activation maps from: "{0:s}"...'.format(
        gradcam_file_name
    ))
    gradcam_dict = gradcam.read_standard_file(gradcam_file_name)

    example_id_strings = gradcam_dict[gradcam.EXAMPLE_IDS_KEY]
    output_neuron_indices = gradcam_dict[gradcam.OUTPUT_NEURONS_KEY]
    model_file_name = gradcam_dict[gradcam.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0]
    )

    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    generator_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]

    dummy_example_dict = {
        example_io.VECTOR_TARGET_NAMES_KEY:
            generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY],
        example_io.HEIGHTS_KEY: generator_option_dict[neural_net.HEIGHTS_KEY]
    }

    target_field_name, target_height_m_agl = (
        neural_net.neuron_indices_to_target_var(
            neuron_indices=output_neuron_indices,
            example_dict=dummy_example_dict,
            net_type_string=model_metadata_dict[neural_net.NET_TYPE_KEY]
        )
    )

    predicted_target_values, actual_target_values = (
        plot_saliency_maps._get_target_values(
            prediction_file_name=prediction_file_name,
            model_metadata_dict=model_metadata_dict,
            example_id_strings=example_id_strings,
            target_field_name=target_field_name,
            target_height_m_agl=target_height_m_agl
        )
    )

    print(SEPARATOR_STRING)
    num_examples = len(example_id_strings)

    for i in range(num_examples):
        this_title_string = 'Actual and predicted {0:s}'.format(
            target_field_name
        )

        if target_height_m_agl is not None:
            this_title_string += ' at {0:.2f} km AGL'.format(
                METRES_TO_KM * target_height_m_agl
            )

        this_title_string += ' = {0:.2f}, {1:.2f}'.format(
            actual_target_values[i], predicted_target_values[i]
        )

        _plot_one_example(
            gradcam_dict=gradcam_dict, example_index=i,
            model_metadata_dict=model_metadata_dict,
            title_string=this_title_string, output_dir_name=output_dir_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        gradcam_file_name=getattr(INPUT_ARG_OBJECT, GRADCAM_FILE_ARG_NAME),
        prediction_file_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_FILE_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
