"""Plots results of backwards optimization."""

import copy
import os.path
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import temperature_conversions as temperature_conv
from gewittergefahr.gg_utils import file_system_utils
from ml4rt.utils import example_utils
from ml4rt.machine_learning import neural_net
from ml4rt.machine_learning import backwards_optimization as bwo
from ml4rt.plotting import profile_plotting
from ml4rt.scripts import plot_examples

# TODO(thunderhoser): Report init/final scalar-predictor values and activations
# in the figure titles.

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

SOLID_LINE_WIDTH = 2
DASHED_LINE_WIDTH = 4
FIGURE_RESOLUTION_DPI = 300

PREDICTOR_NAMES_BY_SET = [
    plot_examples.FIRST_PREDICTOR_NAMES, plot_examples.SECOND_PREDICTOR_NAMES,
    plot_examples.THIRD_PREDICTOR_NAMES
]
PREDICTOR_COLOURS_BY_SET = [
    plot_examples.FIRST_PREDICTOR_COLOURS,
    plot_examples.SECOND_PREDICTOR_COLOURS,
    plot_examples.THIRD_PREDICTOR_COLOURS
]

INPUT_FILE_ARG_NAME = 'input_file_name'
USE_LOG_SCALE_ARG_NAME = 'use_log_scale'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by '
    '`backwards_optimization.read_file`.'
)
USE_LOG_SCALE_HELP_STRING = (
    'Boolean flag.  If 1 (0), will use logarithmic (linear) scale for height '
    'axis.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (figures will be saved here).'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + USE_LOG_SCALE_ARG_NAME, type=int, required=False, default=1,
    help=USE_LOG_SCALE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_results_one_example(
        bwo_dict, example_index, model_metadata_dict, use_log_scale,
        output_dir_name):
    """Plots results for one example.

    :param bwo_dict: Dictionary read by
        `backwards_optimization.read_file`.
    :param example_index: Will plot results for example with this array index.
    :param model_metadata_dict: Dictionary read by `neural_net.read_metafile`.
    :param use_log_scale: See documentation at top of file.
    :param output_dir_name: Name of output directory.  Figure will be saved
        here.
    """

    # Housekeeping.
    example_id_string = bwo_dict[bwo.EXAMPLE_IDS_KEY][example_index]
    generator_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]

    base_example_dict = {
        example_utils.SCALAR_PREDICTOR_NAMES_KEY:
            generator_option_dict[neural_net.SCALAR_PREDICTOR_NAMES_KEY],
        example_utils.VECTOR_PREDICTOR_NAMES_KEY:
            generator_option_dict[neural_net.VECTOR_PREDICTOR_NAMES_KEY],
        example_utils.HEIGHTS_KEY: generator_option_dict[neural_net.HEIGHTS_KEY]
    }

    init_example_dict = copy.deepcopy(base_example_dict)
    init_example_dict[example_utils.SCALAR_PREDICTOR_VALS_KEY] = (
        bwo_dict[bwo.INIT_SCALAR_PREDICTORS_KEY][[example_index], ...]
    )
    init_example_dict[example_utils.VECTOR_PREDICTOR_VALS_KEY] = (
        bwo_dict[bwo.INIT_VECTOR_PREDICTORS_KEY][[example_index], ...]
    )

    final_example_dict = copy.deepcopy(base_example_dict)
    final_example_dict[example_utils.SCALAR_PREDICTOR_VALS_KEY] = (
        bwo_dict[bwo.FINAL_SCALAR_PREDICTORS_KEY][[example_index], ...]
    )
    final_example_dict[example_utils.VECTOR_PREDICTOR_VALS_KEY] = (
        bwo_dict[bwo.FINAL_VECTOR_PREDICTORS_KEY][[example_index], ...]
    )

    diff_example_dict = copy.deepcopy(base_example_dict)
    diff_example_dict[example_utils.SCALAR_PREDICTOR_VALS_KEY] = (
        final_example_dict[example_utils.SCALAR_PREDICTOR_VALS_KEY] -
        init_example_dict[example_utils.SCALAR_PREDICTOR_VALS_KEY]
    )
    diff_example_dict[example_utils.VECTOR_PREDICTOR_VALS_KEY] = (
        final_example_dict[example_utils.VECTOR_PREDICTOR_VALS_KEY] -
        init_example_dict[example_utils.VECTOR_PREDICTOR_VALS_KEY]
    )

    predictor_names = (
        diff_example_dict[example_utils.VECTOR_PREDICTOR_NAMES_KEY]
    )

    if example_utils.TEMPERATURE_NAME in predictor_names:
        temperature_index = predictor_names.index(
            example_utils.TEMPERATURE_NAME
        )
        diff_predictor_matrix = (
            diff_example_dict[example_utils.VECTOR_PREDICTOR_VALS_KEY]
        )
        diff_predictor_matrix[..., temperature_index] = (
            temperature_conv.celsius_to_kelvins(
                diff_predictor_matrix[..., temperature_index]
            )
        )
        diff_example_dict[example_utils.VECTOR_PREDICTOR_VALS_KEY] = (
            diff_predictor_matrix
        )

    num_predictor_sets = len(PREDICTOR_NAMES_BY_SET)

    for k in range(num_predictor_sets):
        these_flags = numpy.array([
            n in base_example_dict[example_utils.VECTOR_PREDICTOR_NAMES_KEY]
            for n in PREDICTOR_NAMES_BY_SET[k]
        ], dtype=bool)

        these_indices = numpy.where(these_flags)[0]
        if len(these_indices) == 0:
            continue

        predictor_names = [PREDICTOR_NAMES_BY_SET[k][i] for i in these_indices]
        predictor_colours = [
            PREDICTOR_COLOURS_BY_SET[k][i] for i in these_indices
        ]

        # Plot initial and final values on the same set of axes.
        handle_dict = profile_plotting.plot_predictors(
            example_dict=init_example_dict, example_index=0,
            predictor_names=predictor_names,
            predictor_colours=predictor_colours,
            predictor_line_widths=
            numpy.full(len(these_indices), SOLID_LINE_WIDTH),
            predictor_line_styles=['solid'] * len(these_indices),
            use_log_scale=use_log_scale, handle_dict=None
        )

        profile_plotting.plot_predictors(
            example_dict=final_example_dict, example_index=0,
            predictor_names=predictor_names,
            predictor_colours=predictor_colours,
            predictor_line_widths=
            numpy.full(len(these_indices), DASHED_LINE_WIDTH),
            predictor_line_styles=['dashed'] * len(these_indices),
            use_log_scale=use_log_scale, handle_dict=handle_dict
        )

        output_file_name = '{0:s}/{1:s}_predictor-set-{2:d}.jpg'.format(
            output_dir_name, example_id_string.replace('_', '-'), k
        )
        figure_object = handle_dict[profile_plotting.FIGURE_HANDLE_KEY]

        print('Saving figure to: "{0:s}"...'.format(output_file_name))
        figure_object.savefig(
            output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
            bbox_inches='tight'
        )
        pyplot.close(figure_object)

        # Plot differences (final minus initial).
        handle_dict = profile_plotting.plot_predictors(
            example_dict=diff_example_dict, example_index=0,
            predictor_names=predictor_names,
            predictor_colours=predictor_colours,
            predictor_line_widths=
            numpy.full(len(these_indices), SOLID_LINE_WIDTH),
            predictor_line_styles=['solid'] * len(these_indices),
            use_log_scale=use_log_scale, handle_dict=None
        )

        output_file_name = '{0:s}/{1:s}_predictor-set-{2:d}_diffs.jpg'.format(
            output_dir_name, example_id_string.replace('_', '-'), k
        )
        figure_object = handle_dict[profile_plotting.FIGURE_HANDLE_KEY]

        print('Saving figure to: "{0:s}"...'.format(output_file_name))
        figure_object.savefig(
            output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
            bbox_inches='tight'
        )
        pyplot.close(figure_object)


def _run(input_file_name, use_log_scale, output_dir_name):
    """Plots results of backwards optimization.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param use_log_scale: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    print('Reading backwards-optimization results from: "{0:s}"...'.format(
        input_file_name
    ))
    bwo_dict = bwo.read_file(input_file_name)

    model_file_name = bwo_dict[bwo.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0]
    )

    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    print(SEPARATOR_STRING)

    num_examples = len(bwo_dict[bwo.EXAMPLE_IDS_KEY])

    for i in range(num_examples):
        _plot_results_one_example(
            bwo_dict=bwo_dict, example_index=i,
            model_metadata_dict=model_metadata_dict,
            use_log_scale=use_log_scale, output_dir_name=output_dir_name
        )

        if i != num_examples - 1:
            print(MINOR_SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        use_log_scale=bool(getattr(INPUT_ARG_OBJECT, USE_LOG_SCALE_ARG_NAME)),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
