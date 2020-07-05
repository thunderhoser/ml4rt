"""Plots saliency maps (one for each example)."""

import os.path
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from ml4rt.io import prediction_io
from ml4rt.io import example_io
from ml4rt.machine_learning import saliency
from ml4rt.machine_learning import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

METRES_TO_KM = 0.001

ORANGE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
PURPLE_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255
GREEN_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255

PREDICTOR_NAME_TO_COLOUR = {
    example_io.TEMPERATURE_NAME: ORANGE_COLOUR,
    example_io.SPECIFIC_HUMIDITY_NAME: ORANGE_COLOUR,
    example_io.PRESSURE_NAME: PURPLE_COLOUR,
    example_io.LIQUID_WATER_CONTENT_NAME: GREEN_COLOUR,
    example_io.ICE_WATER_CONTENT_NAME: GREEN_COLOUR
}

PREDICTOR_NAME_TO_LINE_STYLE = {
    example_io.TEMPERATURE_NAME: 'solid',
    example_io.SPECIFIC_HUMIDITY_NAME: 'dashed',
    example_io.PRESSURE_NAME: 'solid',
    example_io.LIQUID_WATER_CONTENT_NAME: 'solid',
    example_io.ICE_WATER_CONTENT_NAME: 'dashed'
}

PREDICTOR_NAME_TO_LINE_WIDTH = {
    example_io.TEMPERATURE_NAME: 2,
    example_io.SPECIFIC_HUMIDITY_NAME: 4,
    example_io.PRESSURE_NAME: 2,
    example_io.LIQUID_WATER_CONTENT_NAME: 2,
    example_io.ICE_WATER_CONTENT_NAME: 4
}

PREDICTOR_NAME_TO_VERBOSE = {
    example_io.TEMPERATURE_NAME: 'Temperature',
    example_io.SPECIFIC_HUMIDITY_NAME: 'Specific\nhumidity',
    example_io.PRESSURE_NAME: 'Pressure',
    example_io.LIQUID_WATER_CONTENT_NAME: 'LWC',
    example_io.ICE_WATER_CONTENT_NAME: 'IWC',
    example_io.ZENITH_ANGLE_NAME: 'Zenith angle',
    example_io.LATITUDE_NAME: 'Latitude',
    example_io.LONGITUDE_NAME: 'Longitude',
    example_io.ALBEDO_NAME: 'Albedo',
    example_io.LIQUID_WATER_PATH_NAME: 'LWP',
    example_io.ICE_WATER_PATH_NAME: 'IWP'
}

REFERENCE_LINE_COLOUR = numpy.full(3, 152. / 255)
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

SALIENCY_FILE_ARG_NAME = 'input_saliency_file_name'
PREDICTION_FILE_ARG_NAME = 'input_prediction_file_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

SALIENCY_FILE_HELP_STRING = (
    'Path to saliency file (will be read by `saliency.read_standard_file`).'
)
PREDICTION_FILE_HELP_STRING = (
    'Path to prediction file (will be read by `prediction_io.read_file`).  For '
    'each example in the saliency file, this script will find the predicted/'
    'actual target values in the prediction file and include these in the '
    'legend of the saliency plot.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (figures will be saved here).'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + SALIENCY_FILE_ARG_NAME, type=str, required=True,
    help=SALIENCY_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_FILE_ARG_NAME, type=str, required=True,
    help=PREDICTION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_saliency_one_example(
        saliency_dict, example_index, model_metadata_dict,
        predicted_target_value, actual_target_value, output_dir_name):
    """Plots saliency map for one example.

    :param saliency_dict: Dictionary read by `saliency.read_standard_file`.
    :param example_index: Will plot saliency map for example with this array
        index.
    :param model_metadata_dict: Dictionary read by `neural_net.read_metafile`.
    :param predicted_target_value: Predicted target value.
    :param actual_target_value: Actual target value.
    :param output_dir_name: Name of output directory.  Figure will be saved
        here.
    """

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    scalar_saliency_values = (
        saliency_dict[saliency.SCALAR_SALIENCY_KEY][example_index, :]
    )
    vector_saliency_matrix = (
        saliency_dict[saliency.VECTOR_SALIENCY_KEY][example_index, ...]
    )
    example_id_string = saliency_dict[saliency.EXAMPLE_IDS_KEY][example_index]

    generator_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    scalar_predictor_names = (
        generator_option_dict[neural_net.SCALAR_PREDICTOR_NAMES_KEY]
    )
    vector_predictor_names = (
        generator_option_dict[neural_net.VECTOR_PREDICTOR_NAMES_KEY]
    )
    heights_km_agl = METRES_TO_KM * (
        generator_option_dict[neural_net.HEIGHTS_KEY]
    )

    y_min = numpy.min(heights_km_agl)
    y_max = numpy.max(heights_km_agl)

    # these_x = numpy.array([0, 0])
    # these_y = numpy.array([y_min, y_max])
    # axes_object.plot(
    #     these_x, these_y, color=REFERENCE_LINE_COLOUR,
    #     linewidth=2, linestyle='dashed'
    # )

    num_vector_predictors = len(vector_predictor_names)
    legend_handles = [None] * num_vector_predictors
    legend_strings = [None] * num_vector_predictors

    for k in range(num_vector_predictors):
        print(vector_predictor_names[k])
        print(vector_saliency_matrix[:, k])
        print('\n\n')

        legend_handles[k] = axes_object.plot(
            vector_saliency_matrix[:, k], heights_km_agl,
            color=PREDICTOR_NAME_TO_COLOUR[vector_predictor_names[k]],
            linewidth=PREDICTOR_NAME_TO_LINE_WIDTH[vector_predictor_names[k]],
            linestyle=PREDICTOR_NAME_TO_LINE_STYLE[vector_predictor_names[k]]
        )[0]

        legend_strings[k] = PREDICTOR_NAME_TO_VERBOSE[vector_predictor_names[k]]

    x_min = numpy.percentile(vector_saliency_matrix, 1)
    print(x_min)
    x_max = numpy.percentile(vector_saliency_matrix, 90)
    print(x_max)
    axes_object.set_xlim(x_min, x_max)
    axes_object.set_xlim(y_min, y_max)

    axes_object.set_xlabel('Saliency')
    axes_object.set_ylabel('Height (km AGL)')

    axes_object.legend(
        legend_handles, legend_strings, loc='upper left',
        bbox_to_anchor=(0, 1), fancybox=True, shadow=False,
        facecolor='white', edgecolor='k', framealpha=0.5, ncol=1
    )

    num_scalar_predictors = len(scalar_predictor_names)
    title_string = ''

    for k in range(num_scalar_predictors):
        if k == 3:
            title_string += '\n'
        elif k == 0:
            pass
        else:
            title_string += ' ... '

        title_string += '{0:s}: {1:.2f}'.format(
            scalar_predictor_names[k], scalar_saliency_values[k]
        )

    title_string += (
        '\nactual and predicted target values = {0:.2f}, {1:.2f}'
    ).format(actual_target_value, predicted_target_value)

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


def _run(saliency_file_name, prediction_file_name, output_dir_name):
    """Plots saliency maps (one for each example).

    This is effectively the main method.

    :param saliency_file_name: See documentation at top of file.
    :param prediction_file_name: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    print('Reading saliency values from: "{0:s}"...'.format(saliency_file_name))
    saliency_dict = saliency.read_standard_file(saliency_file_name)

    example_id_strings = saliency_dict[saliency.EXAMPLE_IDS_KEY]
    model_file_name = saliency_dict[saliency.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0]
    )

    print('Reading predicted and actual target values from: "{0:s}"...'.format(
        prediction_file_name
    ))
    prediction_dict = prediction_io.read_file(prediction_file_name)

    # TODO(thunderhoser): Need to find the right target variable as well.
    these_indices = numpy.array([
        prediction_dict[prediction_io.EXAMPLE_IDS_KEY].index(id)
        for id in example_id_strings
    ], dtype=int)

    actual_target_values = (
        prediction_dict[prediction_io.VECTOR_TARGETS_KEY][these_indices, 0, 0]
    )
    predicted_target_values = (
        prediction_dict[
            prediction_io.VECTOR_PREDICTIONS_KEY
        ][these_indices, 0, 0]
    )

    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)

    print(SEPARATOR_STRING)

    for k in range(len(example_id_strings)):
        _plot_saliency_one_example(
            saliency_dict=saliency_dict, example_index=k,
            model_metadata_dict=model_metadata_dict,
            predicted_target_value=predicted_target_values[k],
            actual_target_value=actual_target_values[k],
            output_dir_name=output_dir_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        saliency_file_name=getattr(INPUT_ARG_OBJECT, SALIENCY_FILE_ARG_NAME),
        prediction_file_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_FILE_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
