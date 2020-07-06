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
from ml4rt.plotting import profile_plotting

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
    example_io.ICE_WATER_CONTENT_NAME: GREEN_COLOUR,
    example_io.ZENITH_ANGLE_NAME: ORANGE_COLOUR,
    example_io.ALBEDO_NAME: ORANGE_COLOUR,
    example_io.LATITUDE_NAME: PURPLE_COLOUR,
    example_io.LONGITUDE_NAME: PURPLE_COLOUR,
    example_io.LIQUID_WATER_PATH_NAME: GREEN_COLOUR,
    example_io.ICE_WATER_PATH_NAME: GREEN_COLOUR
}

PREDICTOR_NAME_TO_LINE_STYLE = {
    example_io.TEMPERATURE_NAME: 'solid',
    example_io.SPECIFIC_HUMIDITY_NAME: 'dashed',
    example_io.PRESSURE_NAME: 'solid',
    example_io.LIQUID_WATER_CONTENT_NAME: 'solid',
    example_io.ICE_WATER_CONTENT_NAME: 'dashed',
    example_io.ZENITH_ANGLE_NAME: 'solid',
    example_io.ALBEDO_NAME: 'dashed',
    example_io.LATITUDE_NAME: 'solid',
    example_io.LONGITUDE_NAME: 'dashed',
    example_io.LIQUID_WATER_PATH_NAME: 'solid',
    example_io.ICE_WATER_PATH_NAME: 'dashed'
}

PREDICTOR_NAME_TO_LINE_WIDTH = {
    example_io.TEMPERATURE_NAME: 2,
    example_io.SPECIFIC_HUMIDITY_NAME: 4,
    example_io.PRESSURE_NAME: 2,
    example_io.LIQUID_WATER_CONTENT_NAME: 2,
    example_io.ICE_WATER_CONTENT_NAME: 4,
    example_io.ZENITH_ANGLE_NAME: 4,
    example_io.ALBEDO_NAME: 2,
    example_io.LATITUDE_NAME: 4,
    example_io.LONGITUDE_NAME: 2,
    example_io.LIQUID_WATER_PATH_NAME: 4,
    example_io.ICE_WATER_PATH_NAME: 2
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
    'legend of the saliency plot.  If saliency file contains values for '
    'non-output neuron, this file is not needed.'
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
    '--' + PREDICTION_FILE_ARG_NAME, type=str, required=False, default='',
    help=PREDICTION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _get_target_values(
        prediction_file_name, model_metadata_dict, example_id_strings,
        target_field_name, target_height_m_agl):
    """Returns predicted and actual target values.

    E = number of examples

    :param prediction_file_name: See documentation at top of file.
    :param model_metadata_dict: Dictionary returned by
        `neural_net.read_metafile`.
    :param example_id_strings: length-E list of example IDs.  Will return target
        values only for these examples.
    :param target_field_name: Name of target variable.
    :param target_height_m_agl: Height of target variable (metres above ground
        level).
    :return: predicted_values: length-E numpy array of predicted target values.
    :return: actual_values: length-E numpy array of actual target values.
    """

    print((
        'Reading predicted and actual target values from: "{0:s}"...'
    ).format(
        prediction_file_name
    ))
    prediction_dict = prediction_io.read_file(prediction_file_name)

    example_indices = numpy.array([
        prediction_dict[prediction_io.EXAMPLE_IDS_KEY].index(id)
        for id in example_id_strings
    ], dtype=int)

    generator_option_dict = (
        model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    )

    if target_height_m_agl is None:
        scalar_target_names = (
            generator_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY]
        )
        channel_index = scalar_target_names.index(target_field_name)

        actual_values = (
            prediction_dict[prediction_io.SCALAR_TARGETS_KEY][
                example_indices, channel_index
            ]
        )
        predicted_values = (
            prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY][
                example_indices, channel_index
            ]
        )

        return predicted_values, actual_values

    vector_target_names = (
        generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY]
    )
    channel_index = vector_target_names.index(target_field_name)

    height_index = example_io.match_heights(
        heights_m_agl=generator_option_dict[neural_net.HEIGHTS_KEY],
        desired_height_m_agl=target_height_m_agl
    )

    actual_values = (
        prediction_dict[prediction_io.VECTOR_TARGETS_KEY][
            example_indices, height_index, channel_index
        ]
    )
    predicted_values = (
        prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY][
            example_indices, height_index, channel_index
        ]
    )

    return predicted_values, actual_values


def _plot_saliency_one_example(
        saliency_dict, example_index, model_metadata_dict, title_suffix,
        output_dir_name):
    """Plots saliency map for one example.

    :param saliency_dict: Dictionary read by `saliency.read_standard_file`.
    :param example_index: Will plot saliency map for example with this array
        index.
    :param model_metadata_dict: Dictionary read by `neural_net.read_metafile`.
    :param title_suffix: End of figure title.
    :param output_dir_name: Name of output directory.  Figure will be saved
        here.
    """

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    axes_object.set_yscale('log')

    scalar_saliency_matrix = (
        saliency_dict[saliency.SCALAR_SALIENCY_KEY][example_index, ...]
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

    num_vector_predictors = len(vector_predictor_names)
    legend_handles = [None] * num_vector_predictors
    legend_strings = [None] * num_vector_predictors

    for k in range(num_vector_predictors):
        legend_handles[k] = axes_object.plot(
            vector_saliency_matrix[:, k], heights_km_agl,
            color=PREDICTOR_NAME_TO_COLOUR[vector_predictor_names[k]],
            linewidth=PREDICTOR_NAME_TO_LINE_WIDTH[vector_predictor_names[k]],
            linestyle=PREDICTOR_NAME_TO_LINE_STYLE[vector_predictor_names[k]]
        )[0]

        legend_strings[k] = PREDICTOR_NAME_TO_VERBOSE[vector_predictor_names[k]]

    # x_min = numpy.percentile(vector_saliency_matrix, 1)
    # x_max = numpy.percentile(vector_saliency_matrix, 90)
    # axes_object.set_xlim(x_min, x_max)

    y_tick_strings = profile_plotting.create_height_labels(
        tick_values_km_agl=axes_object.get_yticks(), use_log_scale=True
    )
    axes_object.set_yticklabels(y_tick_strings)

    axes_object.set_xlabel('Saliency')
    axes_object.set_ylabel('Height (km AGL)')

    axes_object.legend(
        legend_handles, legend_strings, loc='upper left',
        bbox_to_anchor=(0, 1), fancybox=True, shadow=False,
        facecolor='white', edgecolor='k', framealpha=0.5, ncol=1
    )

    num_scalar_dim = len(scalar_saliency_matrix.shape)
    num_scalar_predictors = len(scalar_predictor_names)
    title_string = ''

    if num_scalar_dim == 1:
        for k in range(num_scalar_predictors):
            if k == 3:
                title_string += '\n'
            elif k == 0:
                pass
            else:
                title_string += ' ... '

            title_string += '{0:s}: {1:.2f}'.format(
                scalar_predictor_names[k], scalar_saliency_matrix[k]
            )

        title_string += '\n' + title_suffix
    else:
        title_string = title_suffix[0].upper() + title_suffix[1:]

    axes_object.set_title(title_string, fontsize=TITLE_FONT_SIZE)

    output_file_name = '{0:s}/{1:s}_vector-predictors.jpg'.format(
        output_dir_name, example_id_string.replace('_', '-')
    )
    print('Saving figure to: "{0:s}"...'.format(output_file_name))

    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    if num_scalar_dim == 1:
        return

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    axes_object.set_yscale('log')

    num_scalar_predictors = len(scalar_predictor_names)
    legend_handles = [None] * num_vector_predictors
    legend_strings = [None] * num_vector_predictors

    for k in range(num_scalar_predictors):
        print(scalar_saliency_matrix.shape)
        print(scalar_saliency_matrix[:, k].shape)

        legend_handles[k] = axes_object.plot(
            scalar_saliency_matrix[:, k], heights_km_agl,
            color=PREDICTOR_NAME_TO_COLOUR[scalar_predictor_names[k]],
            linewidth=PREDICTOR_NAME_TO_LINE_WIDTH[scalar_predictor_names[k]],
            linestyle=PREDICTOR_NAME_TO_LINE_STYLE[scalar_predictor_names[k]]
        )[0]

        legend_strings[k] = PREDICTOR_NAME_TO_VERBOSE[scalar_predictor_names[k]]

    y_tick_strings = profile_plotting.create_height_labels(
        tick_values_km_agl=axes_object.get_yticks(), use_log_scale=True
    )
    axes_object.set_yticklabels(y_tick_strings)

    axes_object.set_xlabel('Saliency')
    axes_object.set_ylabel('Height (km AGL)')

    axes_object.legend(
        legend_handles, legend_strings, loc='upper left',
        bbox_to_anchor=(0, 1), fancybox=True, shadow=False,
        facecolor='white', edgecolor='k', framealpha=0.5, ncol=1
    )
    axes_object.set_title(title_string, fontsize=TITLE_FONT_SIZE)

    output_file_name = '{0:s}/{1:s}_scalar-predictors.jpg'.format(
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

    target_field_name = saliency_dict[saliency.TARGET_FIELD_KEY]
    target_height_m_agl = saliency_dict[saliency.TARGET_HEIGHT_KEY]
    example_id_strings = saliency_dict[saliency.EXAMPLE_IDS_KEY]
    model_file_name = saliency_dict[saliency.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0]
    )

    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    num_examples = len(example_id_strings)

    if target_field_name is None:
        predicted_target_values = [None] * num_examples
        actual_target_values = [None] * num_examples
    else:
        predicted_target_values, actual_target_values = _get_target_values(
            prediction_file_name=prediction_file_name,
            model_metadata_dict=model_metadata_dict,
            example_id_strings=example_id_strings,
            target_field_name=target_field_name,
            target_height_m_agl=target_height_m_agl
        )

    print(SEPARATOR_STRING)

    for k in range(len(example_id_strings)):
        if target_field_name is None:
            this_title_suffix = ''
        else:
            this_title_suffix = 'actual and predicted {0:s}'.format(
                target_field_name
            )

            if target_height_m_agl is not None:
                this_title_suffix += ' at {0:.2f} km AGL'.format(
                    METRES_TO_KM * target_height_m_agl
                )

            this_title_suffix += ' = {0:.2f}, {1:.2f}'.format(
                actual_target_values[k], predicted_target_values[k]
            )

        _plot_saliency_one_example(
            saliency_dict=saliency_dict, example_index=k,
            model_metadata_dict=model_metadata_dict,
            title_suffix=this_title_suffix, output_dir_name=output_dir_name
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
