"""Plots comparisons between predicted and actual (target) profiles."""

import os
import sys
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import prediction_io
import example_utils
import neural_net
import profile_plotting

FIGURE_RESOLUTION_DPI = 300

SHORTWAVE_VECTOR_TARGET_NAMES = [
    example_utils.SHORTWAVE_HEATING_RATE_NAME,
    example_utils.SHORTWAVE_DOWN_FLUX_NAME, example_utils.SHORTWAVE_UP_FLUX_NAME
]
LONGWAVE_VECTOR_TARGET_NAMES = [
    example_utils.LONGWAVE_HEATING_RATE_NAME,
    example_utils.LONGWAVE_DOWN_FLUX_NAME, example_utils.LONGWAVE_UP_FLUX_NAME
]

TARGET_NAME_TO_VERBOSE = {
    example_utils.SHORTWAVE_DOWN_FLUX_NAME: 'Downwelling shortwave flux',
    example_utils.SHORTWAVE_UP_FLUX_NAME: 'Upwelling shortwave flux',
    example_utils.SHORTWAVE_HEATING_RATE_NAME: 'Shortwave heating rate',
    example_utils.LONGWAVE_DOWN_FLUX_NAME: 'Downwelling longwave flux',
    example_utils.LONGWAVE_UP_FLUX_NAME: 'Upwelling longwave flux',
    example_utils.LONGWAVE_HEATING_RATE_NAME: 'Longwave heating rate'
}

TARGET_NAME_TO_UNITS = {
    example_utils.SHORTWAVE_DOWN_FLUX_NAME: r'W m$^{-2}$',
    example_utils.SHORTWAVE_UP_FLUX_NAME: r'W m$^{-2}$',
    example_utils.SHORTWAVE_HEATING_RATE_NAME: r'K day$^{-1}$',
    example_utils.LONGWAVE_DOWN_FLUX_NAME: r'W m$^{-2}$',
    example_utils.LONGWAVE_UP_FLUX_NAME: r'W m$^{-2}$',
    example_utils.LONGWAVE_HEATING_RATE_NAME: r'K day$^{-1}$'
}

TARGET_NAME_TO_COLOUR = {
    example_utils.SHORTWAVE_DOWN_FLUX_NAME:
        profile_plotting.DOWNWELLING_FLUX_COLOUR,
    example_utils.SHORTWAVE_UP_FLUX_NAME:
        profile_plotting.UPWELLING_FLUX_COLOUR,
    example_utils.SHORTWAVE_HEATING_RATE_NAME:
        profile_plotting.HEATING_RATE_COLOUR,
    example_utils.LONGWAVE_DOWN_FLUX_NAME:
        profile_plotting.DOWNWELLING_FLUX_COLOUR,
    example_utils.LONGWAVE_UP_FLUX_NAME:
        profile_plotting.UPWELLING_FLUX_COLOUR,
    example_utils.LONGWAVE_HEATING_RATE_NAME:
        profile_plotting.HEATING_RATE_COLOUR
}

PREDICTION_FILE_ARG_NAME = 'input_prediction_file_name'
PLOT_SHORTWAVE_ARG_NAME = 'plot_shortwave'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
USE_LOG_SCALE_ARG_NAME = 'use_log_scale'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

PREDICTION_FILE_HELP_STRING = (
    'Path to prediction file, containing both predicted and actual (target) '
    'profiles.  Will be read by `prediction_io.read_file`.'
)
PLOT_SHORTWAVE_HELP_STRING = (
    'Boolean flag.  If 1 (0), will plot shortwave (longwave) values.'
)
NUM_EXAMPLES_HELP_STRING = (
    'Will plot the first N examples, where N = `{0:s}`.  If you want to plot '
    'all examples, leave this alone.'
).format(NUM_EXAMPLES_ARG_NAME)

USE_LOG_SCALE_HELP_STRING = (
    'Boolean flag.  If 1 (0), will use logarithmic (linear) scale for height '
    'axis.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_FILE_ARG_NAME, type=str, required=True,
    help=PREDICTION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_SHORTWAVE_ARG_NAME, type=int, required=True,
    help=PLOT_SHORTWAVE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_EXAMPLES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + USE_LOG_SCALE_ARG_NAME, type=int, required=False, default=1,
    help=USE_LOG_SCALE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_comparisons_fancy(
        vector_target_matrix, vector_prediction_matrix, example_id_strings,
        model_metadata_dict, use_log_scale, plot_shortwave, output_dir_name):
    """Plots fancy comparisons (with all target variables in the same plot).

    E = number of examples
    H = number of heights
    T = number of target variables

    :param vector_target_matrix: E-by-H-by-T numpy array of target (actual)
        values.
    :param vector_prediction_matrix: E-by-H-by-T numpy array of predicted
        values.
    :param example_id_strings: length-E list of example IDs.
    :param model_metadata_dict: Dictionary returned by
        `neural_net.read_metadata`.
    :param use_log_scale: See documentation at top of file.
    :param plot_shortwave: Same.
    :param output_dir_name: Path to output directory (figures will be saved
        here).
    """

    generator_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]

    target_example_dict = {
        example_utils.HEIGHTS_KEY:
            generator_option_dict[neural_net.HEIGHTS_KEY],
        example_utils.VECTOR_TARGET_NAMES_KEY:
            generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY],
        example_utils.VECTOR_TARGET_VALS_KEY: vector_target_matrix
    }

    prediction_example_dict = {
        example_utils.HEIGHTS_KEY:
            generator_option_dict[neural_net.HEIGHTS_KEY],
        example_utils.VECTOR_TARGET_NAMES_KEY:
            generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY],
        example_utils.VECTOR_TARGET_VALS_KEY: vector_prediction_matrix
    }

    num_examples = vector_target_matrix.shape[0]

    for i in range(num_examples):
        this_handle_dict = profile_plotting.plot_targets(
            example_dict=target_example_dict, example_index=i,
            use_log_scale=use_log_scale, for_shortwave=plot_shortwave,
            line_style='solid', handle_dict=None
        )
        profile_plotting.plot_targets(
            example_dict=prediction_example_dict, example_index=i,
            use_log_scale=use_log_scale, for_shortwave=plot_shortwave,
            line_style='dashed', handle_dict=this_handle_dict
        )

        this_file_name = '{0:s}/{1:s}_comparison.jpg'.format(
            output_dir_name, example_id_strings[i].replace('_', '-')
        )
        print('Saving figure to: "{0:s}"...'.format(this_file_name))

        this_figure_object = (
            this_handle_dict[profile_plotting.FIGURE_HANDLE_KEY]
        )
        this_figure_object.savefig(
            this_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(this_figure_object)


def _plot_comparisons_simple(
        vector_target_matrix, vector_prediction_matrix, example_id_strings,
        model_metadata_dict, use_log_scale, plot_shortwave, title_strings,
        output_dir_name):
    """Plots simple comparisons (with each target var in a different plot).

    :param vector_target_matrix: See doc for `_plot_comparisons_fancy`.
    :param vector_prediction_matrix: Same.
    :param example_id_strings: Same.
    :param model_metadata_dict: Same.
    :param use_log_scale: Same.
    :param plot_shortwave: Same.
    :param title_strings: 1-D list of titles, one per example.
    :param output_dir_name: See doc for `_plot_comparisons_fancy`.
    """

    generator_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    heights_m_agl = generator_option_dict[neural_net.HEIGHTS_KEY]

    if plot_shortwave:
        target_names = [
            n for n in generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY]
            if n in SHORTWAVE_VECTOR_TARGET_NAMES
        ]
    else:
        target_names = [
            n for n in generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY]
            if n in LONGWAVE_VECTOR_TARGET_NAMES
        ]

    num_examples = vector_target_matrix.shape[0]
    num_target_vars = len(target_names)

    for i in range(num_examples):
        for j in range(num_target_vars):
            k = generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY].index(
                target_names[j]
            )

            this_figure_object, this_axes_object = (
                profile_plotting.plot_one_variable(
                    values=vector_target_matrix[i, :, k],
                    heights_m_agl=heights_m_agl, use_log_scale=use_log_scale,
                    line_colour=TARGET_NAME_TO_COLOUR[target_names[j]],
                    line_style='solid', figure_object=None
                )
            )

            profile_plotting.plot_one_variable(
                values=vector_prediction_matrix[i, :, k],
                heights_m_agl=heights_m_agl, use_log_scale=use_log_scale,
                line_colour=TARGET_NAME_TO_COLOUR[target_names[j]],
                line_style='dashed', figure_object=this_figure_object
            )

            this_axes_object.set_xlabel('{0:s} ({1:s})'.format(
                TARGET_NAME_TO_VERBOSE[target_names[j]],
                TARGET_NAME_TO_UNITS[target_names[j]]
            ))

            this_axes_object.set_title(title_strings[i])

            this_file_name = '{0:s}/{1:s}_{2:s}.jpg'.format(
                output_dir_name, example_id_strings[i].replace('_', '-'),
                target_names[j].replace('_', '-')
            )
            print('Saving figure to: "{0:s}"...'.format(this_file_name))

            this_figure_object.savefig(
                this_file_name, dpi=FIGURE_RESOLUTION_DPI,
                pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(this_figure_object)


def _get_flux_strings(
        scalar_target_matrix, scalar_prediction_matrix, model_metadata_dict,
        plot_shortwave):
    """For each example, returns string with actual and predicted fluxes.

    E = number of examples
    S = number of scalar target variables

    :param scalar_target_matrix: E-by-S numpy array of actual values.
    :param scalar_prediction_matrix: E-by-S numpy array of predicted values.
    :param model_metadata_dict: Dictionary returned by
        `neural_net.read_metadata`.
    :param plot_shortwave: See documentation at top of file.
    :return: flux_strings: length-E list of strings.
    """

    scalar_target_names = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY][
        neural_net.SCALAR_TARGET_NAMES_KEY
    ]
    num_examples = scalar_target_matrix.shape[0]

    try:
        down_flux_index = scalar_target_names.index(
            example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME if plot_shortwave
            else example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME
        )

        down_flux_strings = [
            '{0:.1f}, {1:.1f}'.format(a, p) for a, p in zip(
                scalar_target_matrix[:, down_flux_index],
                scalar_prediction_matrix[:, down_flux_index]
            )
        ]
        down_flux_strings = [
            r'True and pred $F_{down}^{sfc}$ = ' + s + r' W m$^{-2}$'
            for s in down_flux_strings
        ]
    except ValueError:
        down_flux_strings = None

    try:
        up_flux_index = scalar_target_names.index(
            example_utils.SHORTWAVE_TOA_UP_FLUX_NAME if plot_shortwave
            else example_utils.LONGWAVE_TOA_UP_FLUX_NAME
        )

        up_flux_strings = [
            '{0:.1f}, {1:.1f}'.format(a, p) for a, p in zip(
                scalar_target_matrix[:, up_flux_index],
                scalar_prediction_matrix[:, up_flux_index]
            )
        ]
        up_flux_strings = [
            r'True and pred $F_{up}^{TOA}$ = ' + s + r' W m$^{-2}$'
            for s in up_flux_strings
        ]
    except ValueError:
        up_flux_strings = None

    if down_flux_strings is None and up_flux_strings is None:
        return [' '] * num_examples

    if down_flux_strings is not None and up_flux_strings is not None:
        return [
            d + '\n' + u for d, u in zip(down_flux_strings, up_flux_strings)
        ]

    if down_flux_strings is not None:
        return down_flux_strings

    return up_flux_strings


def _run(prediction_file_name, plot_shortwave, num_examples, use_log_scale,
         output_dir_name):
    """Plots comparisons between predicted and actual (target) profiles.

    This is effectively the main method.

    :param prediction_file_name: See documentation at top of file.
    :param plot_shortwave: Same.
    :param num_examples: Same.
    :param use_log_scale: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )
    if num_examples < 1:
        num_examples = None

    print((
        'Reading predicted and actual (target) profiles from: "{0:s}"...'
    ).format(
        prediction_file_name
    ))

    prediction_dict = prediction_io.read_file(prediction_file_name)
    num_examples_orig = len(prediction_dict[prediction_io.EXAMPLE_IDS_KEY])

    if num_examples is not None and num_examples < num_examples_orig:
        desired_indices = numpy.linspace(
            0, num_examples - 1, num=num_examples, dtype=int
        )
        prediction_dict = prediction_io.subset_by_index(
            prediction_dict=prediction_dict, desired_indices=desired_indices
        )

    vector_target_matrix = prediction_dict[prediction_io.VECTOR_TARGETS_KEY]
    vector_prediction_matrix = (
        prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY]
    )
    scalar_target_matrix = prediction_dict[prediction_io.SCALAR_TARGETS_KEY]
    scalar_prediction_matrix = (
        prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY]
    )

    model_file_name = prediction_dict[prediction_io.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY][
        neural_net.HEIGHTS_KEY
    ] = prediction_dict[prediction_io.HEIGHTS_KEY]

    generator_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    vector_target_names = (
        generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY]
    )

    if plot_shortwave:
        plot_fancy = all([
            t in vector_target_names for t in SHORTWAVE_VECTOR_TARGET_NAMES
        ])
    else:
        plot_fancy = all([
            t in vector_target_names for t in LONGWAVE_VECTOR_TARGET_NAMES
        ])

    if plot_fancy:
        _plot_comparisons_fancy(
            vector_target_matrix=vector_target_matrix,
            vector_prediction_matrix=vector_prediction_matrix,
            example_id_strings=prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
            model_metadata_dict=model_metadata_dict,
            use_log_scale=use_log_scale, plot_shortwave=plot_shortwave,
            output_dir_name=output_dir_name
        )
    else:
        title_strings = _get_flux_strings(
            scalar_target_matrix=scalar_target_matrix,
            scalar_prediction_matrix=scalar_prediction_matrix,
            model_metadata_dict=model_metadata_dict,
            plot_shortwave=plot_shortwave
        )

        _plot_comparisons_simple(
            vector_target_matrix=vector_target_matrix,
            vector_prediction_matrix=vector_prediction_matrix,
            example_id_strings=prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
            model_metadata_dict=model_metadata_dict,
            use_log_scale=use_log_scale, plot_shortwave=plot_shortwave,
            title_strings=title_strings, output_dir_name=output_dir_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_FILE_ARG_NAME
        ),
        plot_shortwave=bool(getattr(INPUT_ARG_OBJECT, PLOT_SHORTWAVE_ARG_NAME)),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        use_log_scale=bool(getattr(INPUT_ARG_OBJECT, USE_LOG_SCALE_ARG_NAME)),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
