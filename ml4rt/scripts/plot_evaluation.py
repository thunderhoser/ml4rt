"""Plots model evaluation."""

import os.path
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from ml4rt.io import example_io
from ml4rt.utils import evaluation
from ml4rt.utils import normalization
from ml4rt.machine_learning import neural_net
from ml4rt.plotting import evaluation_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

SCORE_NAME_TO_VERBOSE = {
    evaluation_plotting.MSE_NAME: 'Mean squared error',
    evaluation_plotting.MSE_SKILL_SCORE_NAME:
        'MSE (mean squared error) skill score',
    evaluation_plotting.MAE_NAME: 'Mean absolute error',
    evaluation_plotting.MAE_SKILL_SCORE_NAME:
        'MAE (mean absolute error) skill score',
    evaluation_plotting.BIAS_NAME: 'Bias',
    evaluation_plotting.CORRELATION_NAME: 'Correlation'
}

SCORE_NAME_TO_PROFILE_KEY = {
    evaluation_plotting.MSE_NAME: evaluation.VECTOR_MSE_KEY,
    evaluation_plotting.MSE_SKILL_SCORE_NAME: evaluation.VECTOR_MSE_SKILL_KEY,
    evaluation_plotting.MAE_NAME: evaluation.VECTOR_MAE_KEY,
    evaluation_plotting.MAE_SKILL_SCORE_NAME: evaluation.VECTOR_MAE_SKILL_KEY,
    evaluation_plotting.BIAS_NAME: evaluation.VECTOR_BIAS_KEY,
    evaluation_plotting.CORRELATION_NAME: evaluation.VECTOR_CORRELATION_KEY
}

TARGET_NAME_TO_CONV_RATIO = {
    example_io.SHORTWAVE_DOWN_FLUX_NAME: 1.,
    example_io.SHORTWAVE_UP_FLUX_NAME: 1.,
    example_io.SHORTWAVE_HEATING_RATE_NAME: 86400.,
    example_io.SHORTWAVE_SURFACE_DOWN_FLUX_NAME: 1.,
    example_io.SHORTWAVE_TOA_UP_FLUX_NAME: 1.
}

TARGET_NAME_TO_VERBOSE = {
    example_io.SHORTWAVE_DOWN_FLUX_NAME: r'downwelling flux (W m$^{-2}$)',
    example_io.SHORTWAVE_UP_FLUX_NAME: r'upwelling flux (W m$^{-2}$)',
    example_io.SHORTWAVE_HEATING_RATE_NAME: r'heating rate (K day$^{-1}$)',
    example_io.SHORTWAVE_SURFACE_DOWN_FLUX_NAME:
        r'surface downwelling flux (W m$^{-2}$)',
    example_io.SHORTWAVE_TOA_UP_FLUX_NAME: r'TOA upwelling flux (W m$^{-2}$)',
}

# TODO(thunderhoser): This is a HACK.
HEIGHTS_M_AGL = numpy.array([
    10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 225, 250, 275, 300, 350,
    400, 450, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600,
    1700, 1800, 1900, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600,
    3800, 4000, 4200, 4400, 4600, 4800, 5000, 5500, 6000, 6500, 7000, 8000,
    9000, 10000, 11000, 12000, 13000, 14000, 15000, 18000, 20000, 22000, 24000,
    27000, 30000, 33000, 36000, 39000, 42000, 46000, 50000
], dtype=float)

PROFILE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
TAYLOR_MARKER_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

INPUT_FILE_ARG_NAME = 'input_eval_file_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file (will be read by `evaluation.write_file`).'
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
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_file_name, output_dir_name):
    """Plots model evaluation.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    evaluation_dict = evaluation.read_file(input_file_name)

    model_file_name = evaluation_dict[evaluation.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metadata(model_metafile_name)
    generator_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]

    # TODO(thunderhoser): Make sure that variables end up in correct order.
    all_target_names = generator_option_dict[neural_net.TARGET_NAMES_KEY]
    scalar_target_names = [
        t for t in all_target_names if t in example_io.SCALAR_TARGET_NAMES
    ]
    vector_target_names = [
        t for t in all_target_names if t in example_io.VECTOR_TARGET_NAMES
    ]

    all_predictor_names = generator_option_dict[neural_net.PREDICTOR_NAMES_KEY]
    scalar_predictor_names = [
        p for p in all_predictor_names if p in example_io.SCALAR_PREDICTOR_NAMES
    ]
    vector_predictor_names = [
        p for p in all_predictor_names if p in example_io.VECTOR_PREDICTOR_NAMES
    ]

    example_dict = {
        example_io.SCALAR_TARGET_NAMES_KEY: scalar_target_names,
        example_io.VECTOR_TARGET_NAMES_KEY: vector_target_names,
        example_io.SCALAR_PREDICTOR_NAMES_KEY: scalar_predictor_names,
        example_io.VECTOR_PREDICTOR_NAMES_KEY: vector_predictor_names,
        example_io.HEIGHTS_KEY: HEIGHTS_M_AGL
    }

    mean_training_example_dict = normalization.create_mean_example(
        example_dict=example_dict,
        normalization_file_name=
        generator_option_dict[neural_net.NORMALIZATION_FILE_KEY]
    )

    for k in range(len(vector_target_names)):
        this_target_name_verbose = (
            TARGET_NAME_TO_VERBOSE[vector_target_names[k]]
        )
        this_conv_ratio = TARGET_NAME_TO_CONV_RATIO[vector_target_names[k]]

        # Plot error profiles.
        for this_score_name in list(SCORE_NAME_TO_PROFILE_KEY.keys()):
            this_figure_object, this_axes_object = pyplot.subplots(
                1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
            )

            this_key = SCORE_NAME_TO_PROFILE_KEY[this_score_name]
            this_prmse = (
                this_conv_ratio *
                evaluation_dict[evaluation.VECTOR_PRMSE_KEY][k]
            )

            these_profile_values = evaluation_dict[this_key][:, k]

            if this_score_name in [
                    evaluation_plotting.MAE_NAME, evaluation_plotting.BIAS_NAME
            ]:
                these_profile_values = these_profile_values * this_conv_ratio
            elif this_score_name == evaluation_plotting.MSE_NAME:
                these_profile_values *= this_conv_ratio ** 2

            evaluation_plotting.plot_score_profile(
                heights_m_agl=HEIGHTS_M_AGL, score_values=these_profile_values,
                score_name=this_score_name, line_colour=PROFILE_COLOUR,
                line_width=2, use_log_scale=True, axes_object=this_axes_object
            )

            this_score_name_verbose = SCORE_NAME_TO_VERBOSE[this_score_name]
            this_title_string = '{0:s} for {1:s} ... PRMSE = {2:.2f}'.format(
                this_score_name_verbose, this_target_name_verbose, this_prmse
            )

            this_axes_object.set_xlabel(this_score_name_verbose)
            this_axes_object.set_title(this_title_string)

            this_file_name = '{0:s}/{1:s}_{2:s}_profile.jpg'.format(
                output_dir_name, vector_target_names[k].replace('_', '-'),
                this_score_name.replace('_', '-')
            )
            print('Saving figure to: "{0:s}"...'.format(this_file_name))

            this_figure_object.savefig(
                this_file_name, dpi=FIGURE_RESOLUTION_DPI,
                pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(this_figure_object)

        # Plot reliability curves for all heights in the same figure.
        this_mean_prediction_matrix = this_conv_ratio * (
            evaluation_dict[evaluation.VECTOR_RELIABILITY_X_KEY][:, k, :]
        )
        this_mean_target_matrix = this_conv_ratio * (
            evaluation_dict[evaluation.VECTOR_RELIABILITY_Y_KEY][:, k, :]
        )
        this_combined_matrix = numpy.concatenate(
            (this_mean_prediction_matrix, this_mean_target_matrix), axis=0
        )
        this_max_value = numpy.percentile(this_combined_matrix, 99)

        this_figure_object, this_axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

        evaluation_plotting.plot_rel_curve_many_heights(
            mean_target_matrix=this_mean_target_matrix,
            mean_prediction_matrix=this_mean_prediction_matrix,
            heights_m_agl=HEIGHTS_M_AGL, max_value_to_plot=this_max_value,
            axes_object=this_axes_object
        )

        this_axes_object.set_title(
            'Reliability curves for {0:s}'.format(this_target_name_verbose)
        )

        this_file_name = '{0:s}/{1:s}_reliability_profile.jpg'.format(
            output_dir_name, vector_target_names[k].replace('_', '-')
        )
        print('Saving figure to: "{0:s}"...'.format(this_file_name))

        this_figure_object.savefig(
            this_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(this_figure_object)

        # Plot Taylor diagram for all heights in the same figure.
        these_target_stdevs = this_conv_ratio * (
            evaluation_dict[evaluation.VECTOR_TARGET_STDEV_KEY][:, k]
        )
        these_prediction_stdevs = this_conv_ratio * (
            evaluation_dict[evaluation.VECTOR_PREDICTION_STDEV_KEY][:, k]
        )
        these_correlations = (
            evaluation_dict[evaluation.VECTOR_CORRELATION_KEY][:, k]
        )

        evaluation_plotting.plot_taylor_diagram_many_heights(
            target_stdevs=these_target_stdevs,
            prediction_stdevs=these_prediction_stdevs,
            correlations=these_correlations, heights_m_agl=HEIGHTS_M_AGL,
            figure_object=this_figure_object
        )

        this_axes_object.set_title(
            'Taylor diagram for {0:s}'.format(this_target_name_verbose)
        )

        this_file_name = '{0:s}/{1:s}_taylor_profile.jpg'.format(
            output_dir_name, vector_target_names[k].replace('_', '-')
        )
        print('Saving figure to: "{0:s}"...'.format(this_file_name))

        this_figure_object.savefig(
            this_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(this_figure_object)

        print(SEPARATOR_STRING)

    for k in range(len(scalar_target_names)):
        this_target_name_verbose = (
            TARGET_NAME_TO_VERBOSE[scalar_target_names[k]]
        )
        this_conv_ratio = TARGET_NAME_TO_CONV_RATIO[scalar_target_names[k]]

        # Plot attributes diagram.
        these_mean_predictions = this_conv_ratio * (
            evaluation_dict[evaluation.SCALAR_RELIABILITY_X_KEY][k, :]
        )
        these_mean_targets = this_conv_ratio * (
            evaluation_dict[evaluation.SCALAR_RELIABILITY_Y_KEY][k, :]
        )
        these_example_counts = (
            evaluation_dict[evaluation.SCALAR_RELIABILITY_COUNT_KEY][k, :]
        )
        this_climo_value = (
            mean_training_example_dict[example_io.SCALAR_TARGET_VALS_KEY][0, k]
        )

        this_figure_object, this_axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

        evaluation_plotting.plot_attributes_diagram(
            figure_object=this_figure_object, axes_object=this_axes_object,
            mean_predictions=these_mean_predictions,
            mean_observations=these_mean_targets,
            example_counts=these_example_counts,
            mean_value_in_training=this_climo_value
        )

        this_axes_object.set_title(
            'Attributes diagram for {0:s}'.format(this_target_name_verbose)
        )

        this_file_name = '{0:s}/{1:s}_reliability.jpg'.format(
            output_dir_name, scalar_target_names[k].replace('_', '-')
        )
        print('Saving figure to: "{0:s}"...'.format(this_file_name))

        this_figure_object.savefig(
            this_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(this_figure_object)

        # Plot Taylor diagram.
        this_target_stdev = this_conv_ratio * (
            evaluation_dict[evaluation.SCALAR_TARGET_STDEV_KEY][k]
        )
        this_prediction_stdev = this_conv_ratio * (
            evaluation_dict[evaluation.SCALAR_TARGET_STDEV_KEY][k]
        )
        this_correlation = evaluation_dict[evaluation.SCALAR_CORRELATION_KEY][k]

        this_figure_object, this_axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

        evaluation_plotting.plot_taylor_diagram(
            target_stdev=this_target_stdev,
            prediction_stdev=this_prediction_stdev,
            correlation=this_correlation, marker_colour=TAYLOR_MARKER_COLOUR,
            figure_object=this_figure_object
        )

        this_axes_object.set_title(
            'Taylor diagram for {0:s}'.format(this_target_name_verbose)
        )

        this_file_name = '{0:s}/{1:s}_taylor.jpg'.format(
            output_dir_name, vector_target_names[k].replace('_', '-')
        )
        print('Saving figure to: "{0:s}"...'.format(this_file_name))

        this_figure_object.savefig(
            this_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(this_figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
