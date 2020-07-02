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

# TODO(thunderhoser): Probably want to modularize the code in this script.

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

SCORE_NAME_TO_VERBOSE = {
    evaluation_plotting.MSE_NAME: 'Mean squared error',
    evaluation_plotting.MSE_SKILL_SCORE_NAME: 'MSE skill score',
    evaluation_plotting.MAE_NAME: 'Mean absolute error',
    evaluation_plotting.MAE_SKILL_SCORE_NAME: 'MAE skill score',
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

ORIG_UNIT_SCORE_NAMES = [
    evaluation_plotting.MAE_NAME, evaluation_plotting.BIAS_NAME
]
SQUARED_UNIT_SCORE_NAMES = [evaluation_plotting.MSE_NAME]

TARGET_NAME_TO_VERBOSE = {
    example_io.SHORTWAVE_DOWN_FLUX_NAME: 'downwelling flux',
    example_io.SHORTWAVE_UP_FLUX_NAME: 'upwelling flux',
    example_io.SHORTWAVE_HEATING_RATE_NAME: 'heating rate',
    example_io.SHORTWAVE_SURFACE_DOWN_FLUX_NAME: 'surface downwelling flux',
    example_io.SHORTWAVE_TOA_UP_FLUX_NAME: 'TOA upwelling flux',
    evaluation.NET_FLUX_NAME: 'net flux',
    evaluation.HIGHEST_UP_FLUX_NAME: 'top-of-profile upwelling flux',
    evaluation.LOWEST_DOWN_FLUX_NAME: 'bottom-of-profile downwelling flux'
}

TARGET_NAME_TO_UNITS = {
    example_io.SHORTWAVE_DOWN_FLUX_NAME: r'W m$^{-2}$',
    example_io.SHORTWAVE_UP_FLUX_NAME: r'W m$^{-2}$',
    example_io.SHORTWAVE_HEATING_RATE_NAME: r'K day$^{-1}$',
    example_io.SHORTWAVE_SURFACE_DOWN_FLUX_NAME: r'W m$^{-2}$',
    example_io.SHORTWAVE_TOA_UP_FLUX_NAME: r'W m$^{-2}$',
    evaluation.NET_FLUX_NAME: r'W m$^{-2}$',
    evaluation.HIGHEST_UP_FLUX_NAME: r'W m$^{-2}$',
    evaluation.LOWEST_DOWN_FLUX_NAME: r'W m$^{-2}$'
}

TARGET_NAME_TO_SQUARED_UNITS = {
    example_io.SHORTWAVE_DOWN_FLUX_NAME: r'W$^{2}$ m$^{-4}$',
    example_io.SHORTWAVE_UP_FLUX_NAME: r'W$^{2}$ m$^{-4}$',
    example_io.SHORTWAVE_HEATING_RATE_NAME: r'K$^{2}$ day$^{-2}$',
    example_io.SHORTWAVE_SURFACE_DOWN_FLUX_NAME: r'W$^{2}$ m$^{-4}$',
    example_io.SHORTWAVE_TOA_UP_FLUX_NAME: r'W$^{2}$ m$^{-4}$',
    evaluation.NET_FLUX_NAME: r'W$^{2}$ m$^{-4}$',
    evaluation.HIGHEST_UP_FLUX_NAME: r'W$^{2}$ m$^{-4}$',
    evaluation.LOWEST_DOWN_FLUX_NAME: r'W$^{2}$ m$^{-4}$'
}

PROFILE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
TAYLOR_MARKER_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

INPUT_FILE_ARG_NAME = 'input_eval_file_name'
PLOT_BY_HEIGHT_ARG_NAME = 'plot_by_height'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file (will be read by `evaluation.write_file`).'
)
PLOT_BY_HEIGHT_HELP_STRING = (
    'Boolean flag.  If 1, will plot Taylor diagram and attributes diagram for '
    'each vector field at each height.  If 0, will not plot these things.'
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
    '--' + PLOT_BY_HEIGHT_ARG_NAME, type=int, required=False, default=1,
    help=PLOT_BY_HEIGHT_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_file_name, plot_by_height, output_dir_name):
    """Plots model evaluation.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param plot_by_height: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    result_table_xarray = evaluation.read_file(input_file_name)

    model_file_name = result_table_xarray.attrs[evaluation.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    generator_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]

    example_dict = {
        example_io.SCALAR_TARGET_NAMES_KEY:
            generator_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY],
        example_io.VECTOR_TARGET_NAMES_KEY:
            generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY],
        example_io.SCALAR_PREDICTOR_NAMES_KEY:
            generator_option_dict[neural_net.SCALAR_PREDICTOR_NAMES_KEY],
        example_io.VECTOR_PREDICTOR_NAMES_KEY:
            generator_option_dict[neural_net.VECTOR_PREDICTOR_NAMES_KEY],
        example_io.HEIGHTS_KEY: generator_option_dict[neural_net.HEIGHTS_KEY]
    }

    normalization_file_name = (
        generator_option_dict[neural_net.NORMALIZATION_FILE_KEY]
    )
    print((
        'Reading training examples (for climatology) from: "{0:s}"...'
    ).format(
        normalization_file_name
    ))
    training_example_dict = example_io.read_file(normalization_file_name)

    mean_training_example_dict = normalization.create_mean_example(
        new_example_dict=example_dict,
        training_example_dict=training_example_dict
    )

    vector_target_names = (
        generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY]
    )
    heights_m_agl = generator_option_dict[neural_net.HEIGHTS_KEY]

    for k in range(len(vector_target_names)):
        this_target_name_verbose = (
            TARGET_NAME_TO_VERBOSE[vector_target_names[k]]
        )
        this_unit_string = TARGET_NAME_TO_UNITS[vector_target_names[k]]
        this_squared_unit_string = (
            TARGET_NAME_TO_SQUARED_UNITS[vector_target_names[k]]
        )

        # Plot error profiles.
        for this_score_name in list(SCORE_NAME_TO_PROFILE_KEY.keys()):
            this_figure_object, this_axes_object = pyplot.subplots(
                1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
            )

            this_key = SCORE_NAME_TO_PROFILE_KEY[this_score_name]

            evaluation_plotting.plot_score_profile(
                heights_m_agl=heights_m_agl,
                score_values=result_table_xarray[this_key].values[:, k],
                score_name=this_score_name, line_colour=PROFILE_COLOUR,
                line_width=2, use_log_scale=False, axes_object=this_axes_object
            )

            this_score_name_verbose = SCORE_NAME_TO_VERBOSE[this_score_name]
            this_prmse = (
                result_table_xarray[evaluation.VECTOR_PRMSE_KEY].values[k]
            )
            this_title_string = (
                '{0:s} for {1:s} (PRMSE = {2:.2f} {3:s})'
            ).format(
                this_score_name_verbose, this_target_name_verbose, this_prmse,
                this_unit_string
            )

            this_x_label_string = '{0:s}'.format(this_score_name_verbose)

            if this_score_name in SQUARED_UNIT_SCORE_NAMES:
                this_x_label_string += ' ({0:s})'.format(
                    this_squared_unit_string
                )
            elif this_score_name in ORIG_UNIT_SCORE_NAMES:
                this_x_label_string += ' ({0:s})'.format(this_unit_string)

            this_axes_object.set_xlabel(this_x_label_string)
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
        this_mean_prediction_matrix = (
            result_table_xarray[
                evaluation.VECTOR_RELIABILITY_X_KEY
            ].values[:, k, :]
        )
        this_mean_target_matrix = (
            result_table_xarray[
                evaluation.VECTOR_RELIABILITY_Y_KEY
            ].values[:, k, :]
        )
        this_combined_matrix = numpy.concatenate(
            (this_mean_prediction_matrix, this_mean_target_matrix), axis=0
        )
        this_max_value = numpy.nanpercentile(this_combined_matrix, 99)

        this_figure_object, this_axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

        evaluation_plotting.plot_rel_curve_many_heights(
            mean_target_matrix=this_mean_target_matrix,
            mean_prediction_matrix=this_mean_prediction_matrix,
            heights_m_agl=heights_m_agl, min_value_to_plot=0.,
            max_value_to_plot=this_max_value, axes_object=this_axes_object
        )

        this_axes_object.set_title(
            'Reliability curves for {0:s}'.format(this_target_name_verbose)
        )
        this_axes_object.set_xlabel(
            'Prediction ({0:s})'.format(this_unit_string)
        )
        this_axes_object.set_ylabel(
            'Conditional mean observation ({0:s})'.format(this_unit_string)
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
        these_target_stdevs = (
            result_table_xarray[evaluation.VECTOR_TARGET_STDEV_KEY].values[:, k]
        )
        these_prediction_stdevs = (
            result_table_xarray[
                evaluation.VECTOR_PREDICTION_STDEV_KEY].values[:, k]
        )
        these_correlations = (
            result_table_xarray[evaluation.VECTOR_CORRELATION_KEY].values[:, k]
        )

        this_figure_object = pyplot.figure(
            figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

        this_taylor_diag_object = (
            evaluation_plotting.plot_taylor_diagram_many_heights(
                target_stdevs=these_target_stdevs,
                prediction_stdevs=these_prediction_stdevs,
                correlations=these_correlations, heights_m_agl=heights_m_agl,
                figure_object=this_figure_object
            )
        )

        this_figure_object.suptitle(
            'Taylor diagram for {0:s}'.format(this_target_name_verbose),
            y=0.85
        )
        this_taylor_diag_object._ax.axis['left'].label.set_text(
            'Standard deviation ({0:s})'.format(this_unit_string)
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

    scalar_target_names = (
        generator_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY]
    )

    for k in range(len(scalar_target_names)):
        this_target_name_verbose = (
            TARGET_NAME_TO_VERBOSE[scalar_target_names[k]]
        )
        this_unit_string = TARGET_NAME_TO_UNITS[scalar_target_names[k]]

        # Plot attributes diagram.
        these_mean_predictions = (
            result_table_xarray[
                evaluation.SCALAR_RELIABILITY_X_KEY
            ].values[k, :]
        )
        these_mean_targets = (
            result_table_xarray[
                evaluation.SCALAR_RELIABILITY_Y_KEY
            ].values[k, :]
        )
        these_example_counts = (
            result_table_xarray[
                evaluation.SCALAR_RELIABILITY_COUNT_KEY
            ].values[k, :]
        )
        these_inv_mean_targets = (
            result_table_xarray[
                evaluation.SCALAR_INV_RELIABILITY_X_KEY
            ].values[k, :]
        )
        these_inv_example_counts = (
            result_table_xarray[
                evaluation.SCALAR_INV_RELIABILITY_COUNT_KEY
            ].values[k, :]
        )

        this_climo_value = (
            mean_training_example_dict[example_io.SCALAR_TARGET_VALS_KEY][0, k]
        )
        this_max_value = numpy.nanpercentile(
            numpy.concatenate((these_mean_predictions, these_mean_targets)),
            99.
        )

        this_figure_object, this_axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

        evaluation_plotting.plot_attributes_diagram(
            figure_object=this_figure_object, axes_object=this_axes_object,
            mean_predictions=these_mean_predictions,
            mean_observations=these_mean_targets,
            example_counts=these_example_counts,
            mean_value_in_training=this_climo_value,
            min_value_to_plot=0., max_value_to_plot=this_max_value,
            inv_mean_observations=these_inv_mean_targets,
            inv_example_counts=these_inv_example_counts
        )

        this_axes_object.set_title(
            'Attributes diagram for {0:s}'.format(this_target_name_verbose)
        )
        this_axes_object.set_xlabel(
            'Prediction ({0:s})'.format(this_unit_string)
        )
        this_axes_object.set_ylabel(
            'Conditional mean observation ({0:s})'.format(this_unit_string)
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
        this_target_stdev = (
            result_table_xarray[evaluation.SCALAR_TARGET_STDEV_KEY].values[k]
        )
        this_prediction_stdev = (
            result_table_xarray[
                evaluation.SCALAR_PREDICTION_STDEV_KEY].values[k]
        )
        this_correlation = (
            result_table_xarray[evaluation.SCALAR_CORRELATION_KEY].values[k]
        )

        this_figure_object = pyplot.figure(
            figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

        this_taylor_diag_object = evaluation_plotting.plot_taylor_diagram(
            target_stdev=this_target_stdev,
            prediction_stdev=this_prediction_stdev,
            correlation=this_correlation, marker_colour=TAYLOR_MARKER_COLOUR,
            figure_object=this_figure_object
        )

        this_figure_object.suptitle(
            'Taylor diagram for {0:s}'.format(this_target_name_verbose)
        )
        this_taylor_diag_object._ax.axis['left'].label.set_text(
            'Standard deviation ({0:s})'.format(this_unit_string)
        )

        this_file_name = '{0:s}/{1:s}_taylor.jpg'.format(
            output_dir_name, scalar_target_names[k].replace('_', '-')
        )
        print('Saving figure to: "{0:s}"...'.format(this_file_name))

        this_figure_object.savefig(
            this_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(this_figure_object)

    print(SEPARATOR_STRING)

    try:
        aux_target_field_names = (
            result_table_xarray.coords[evaluation.AUX_TARGET_FIELD_DIM].values
        )
        aux_predicted_field_names = (
            result_table_xarray.coords[
                evaluation.AUX_PREDICTED_FIELD_DIM].values
        )
    except:
        aux_target_field_names = []
        aux_predicted_field_names = []

    for k in range(len(aux_target_field_names)):
        this_target_field_name_verbose = (
            TARGET_NAME_TO_VERBOSE[aux_target_field_names[k]]
        )
        this_predicted_field_name_verbose = (
            TARGET_NAME_TO_VERBOSE[aux_predicted_field_names[k]]
        )
        this_unit_string = TARGET_NAME_TO_UNITS[aux_target_field_names[k]]

        # Plot attributes diagram.
        these_mean_predictions = (
            result_table_xarray[evaluation.AUX_RELIABILITY_X_KEY].values[k, :]
        )
        these_mean_targets = (
            result_table_xarray[evaluation.AUX_RELIABILITY_Y_KEY].values[k, :]
        )
        these_example_counts = (
            result_table_xarray[
                evaluation.AUX_RELIABILITY_COUNT_KEY].values[k, :]
        )
        these_inv_mean_targets = (
            result_table_xarray[
                evaluation.AUX_INV_RELIABILITY_X_KEY
            ].values[k, :]
        )
        these_inv_example_counts = (
            result_table_xarray[
                evaluation.AUX_INV_RELIABILITY_COUNT_KEY
            ].values[k, :]
        )

        if aux_target_field_names[k] == evaluation.NET_FLUX_NAME:
            surface_down_flux_index = scalar_target_names.index(
                example_io.SHORTWAVE_SURFACE_DOWN_FLUX_NAME
            )
            toa_up_flux_index = scalar_target_names.index(
                example_io.SHORTWAVE_TOA_UP_FLUX_NAME
            )
            this_climo_value = (
                mean_training_example_dict[example_io.SCALAR_TARGET_VALS_KEY][
                    0, surface_down_flux_index] -
                mean_training_example_dict[example_io.SCALAR_TARGET_VALS_KEY][
                    0, toa_up_flux_index]
            )
        else:
            this_index = scalar_target_names.index(aux_target_field_names[k])
            this_climo_value = (
                mean_training_example_dict[example_io.SCALAR_TARGET_VALS_KEY][
                    0, this_index]
            )

        this_min_value = numpy.nanpercentile(
            numpy.concatenate((these_mean_predictions, these_mean_targets)),
            1.
        )
        this_min_value = numpy.minimum(this_min_value, 0.)
        this_max_value = numpy.nanpercentile(
            numpy.concatenate((these_mean_predictions, these_mean_targets)),
            99.
        )

        this_figure_object, this_axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

        evaluation_plotting.plot_attributes_diagram(
            figure_object=this_figure_object, axes_object=this_axes_object,
            mean_predictions=these_mean_predictions,
            mean_observations=these_mean_targets,
            example_counts=these_example_counts,
            mean_value_in_training=this_climo_value,
            min_value_to_plot=this_min_value, max_value_to_plot=this_max_value,
            inv_mean_observations=these_inv_mean_targets,
            inv_example_counts=these_inv_example_counts
        )

        this_title_string = (
            'Attributes diagram (prediction = {0:s};\nobservation = {1:s})'
        ).format(
            this_predicted_field_name_verbose, this_target_field_name_verbose
        )

        this_axes_object.set_title(this_title_string)
        this_axes_object.set_xlabel('Prediction ({0:s})'.format(
            this_unit_string
        ))
        this_axes_object.set_ylabel(
            'Conditional mean observation ({0:s})'.format(this_unit_string)
        )

        this_file_name = '{0:s}/aux_{1:s}_reliability.jpg'.format(
            output_dir_name, aux_target_field_names[k].replace('_', '-')
        )
        print('Saving figure to: "{0:s}"...'.format(this_file_name))

        this_figure_object.savefig(
            this_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(this_figure_object)

        # Plot Taylor diagram.
        this_target_stdev = (
            result_table_xarray[evaluation.AUX_TARGET_STDEV_KEY].values[k]
        )
        this_prediction_stdev = (
            result_table_xarray[evaluation.AUX_PREDICTION_STDEV_KEY].values[k]
        )
        this_correlation = (
            result_table_xarray[evaluation.AUX_CORRELATION_KEY].values[k]
        )

        this_figure_object = pyplot.figure(
            figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

        this_taylor_diag_object = evaluation_plotting.plot_taylor_diagram(
            target_stdev=this_target_stdev,
            prediction_stdev=this_prediction_stdev,
            correlation=this_correlation, marker_colour=TAYLOR_MARKER_COLOUR,
            figure_object=this_figure_object
        )

        this_title_string = (
            'Taylor diagram (prediction = {0:s};\nobservation = {1:s})'
        ).format(
            this_predicted_field_name_verbose, this_target_field_name_verbose
        )

        this_figure_object.suptitle(this_title_string)
        this_taylor_diag_object._ax.axis['left'].label.set_text(
            'Standard deviation for observation ({0:s})'.format(
                this_unit_string
            )
        )

        this_file_name = '{0:s}/aux_{1:s}_taylor.jpg'.format(
            output_dir_name, aux_target_field_names[k].replace('_', '-')
        )
        print('Saving figure to: "{0:s}"...'.format(this_file_name))

        this_figure_object.savefig(
            this_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(this_figure_object)

    if not plot_by_height:
        return

    if len(aux_target_field_names) > 0:
        print(SEPARATOR_STRING)

    for k in range(len(vector_target_names)):
        this_target_name_verbose = (
            TARGET_NAME_TO_VERBOSE[vector_target_names[k]]
        )
        this_unit_string = TARGET_NAME_TO_UNITS[vector_target_names[k]]

        for j in range(len(heights_m_agl)):

            # Plot attributes diagram.
            these_mean_predictions = (
                result_table_xarray[
                    evaluation.VECTOR_RELIABILITY_X_KEY].values[j, k, :]
            )
            these_mean_targets = (
                result_table_xarray[
                    evaluation.VECTOR_RELIABILITY_Y_KEY].values[j, k, :]
            )
            these_example_counts = (
                result_table_xarray[
                    evaluation.VECTOR_RELIABILITY_COUNT_KEY].values[j, k, :]
            )
            these_inv_mean_targets = (
                result_table_xarray[
                    evaluation.VECTOR_INV_RELIABILITY_X_KEY
                ].values[j, k, :]
            )
            these_inv_example_counts = (
                result_table_xarray[
                    evaluation.VECTOR_INV_RELIABILITY_COUNT_KEY
                ].values[j, k, :]
            )

            this_climo_value = (
                mean_training_example_dict[
                    example_io.VECTOR_TARGET_VALS_KEY
                ][0, j, k]
            )
            this_max_value = numpy.nanpercentile(
                numpy.concatenate((these_mean_predictions, these_mean_targets)),
                99.
            )

            this_figure_object, this_axes_object = pyplot.subplots(
                1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
            )

            evaluation_plotting.plot_attributes_diagram(
                figure_object=this_figure_object, axes_object=this_axes_object,
                mean_predictions=these_mean_predictions,
                mean_observations=these_mean_targets,
                example_counts=these_example_counts,
                mean_value_in_training=this_climo_value,
                min_value_to_plot=0., max_value_to_plot=this_max_value,
                inv_mean_observations=these_inv_mean_targets,
                inv_example_counts=these_inv_example_counts
            )

            this_height_string_unpadded = '{0:d}'.format(
                int(numpy.round(heights_m_agl[j]))
            )
            this_height_string_padded = '{0:05d}'.format(
                int(numpy.round(heights_m_agl[j]))
            )

            this_axes_object.set_title(
                'Attributes diagram for {0:s} at {1:s} m AGL'.format(
                    this_target_name_verbose, this_height_string_unpadded
                )
            )
            this_axes_object.set_xlabel(
                'Prediction ({0:s})'.format(this_unit_string)
            )
            this_axes_object.set_ylabel(
                'Conditional mean observation ({0:s})'.format(this_unit_string)
            )

            this_file_name = '{0:s}/{1:s}_{2:s}metres_reliability.jpg'.format(
                output_dir_name, vector_target_names[k].replace('_', '-'),
                this_height_string_padded
            )
            print('Saving figure to: "{0:s}"...'.format(this_file_name))

            this_figure_object.savefig(
                this_file_name, dpi=FIGURE_RESOLUTION_DPI,
                pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(this_figure_object)

            # Plot Taylor diagram.
            this_target_stdev = (
                result_table_xarray[
                    evaluation.VECTOR_TARGET_STDEV_KEY].values[j, k]
            )
            this_prediction_stdev = (
                result_table_xarray[
                    evaluation.VECTOR_PREDICTION_STDEV_KEY].values[j, k]
            )
            this_correlation = (
                result_table_xarray[
                    evaluation.VECTOR_CORRELATION_KEY].values[j, k]
            )

            this_figure_object = pyplot.figure(
                figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
            )

            this_taylor_diag_object = evaluation_plotting.plot_taylor_diagram(
                target_stdev=this_target_stdev,
                prediction_stdev=this_prediction_stdev,
                correlation=this_correlation,
                marker_colour=TAYLOR_MARKER_COLOUR,
                figure_object=this_figure_object
            )

            this_figure_object.suptitle(
                'Taylor diagram for {0:s} at {1:s} m AGL'.format(
                    this_target_name_verbose, this_height_string_unpadded
                )
            )
            this_taylor_diag_object._ax.axis['left'].label.set_text(
                'Standard deviation ({0:s})'.format(this_unit_string)
            )

            this_file_name = '{0:s}/{1:s}_{2:s}metres_taylor.jpg'.format(
                output_dir_name, vector_target_names[k].replace('_', '-'),
                this_height_string_padded
            )
            print('Saving figure to: "{0:s}"...'.format(this_file_name))

            this_figure_object.savefig(
                this_file_name, dpi=FIGURE_RESOLUTION_DPI,
                pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(this_figure_object)

            if j != len(heights_m_agl) - 1:
                print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        plot_by_height=bool(getattr(INPUT_ARG_OBJECT, PLOT_BY_HEIGHT_ARG_NAME)),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
