"""Plots model evaluation."""

import os
import sys
import argparse
import numpy
import xarray
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import error_checking
import prediction_io
import example_io
import example_utils
import evaluation
import normalization
import neural_net
import evaluation_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

SCORE_NAME_TO_VERBOSE = {
    evaluation_plotting.MSE_NAME: 'Mean squared error',
    evaluation_plotting.MSE_SKILL_SCORE_NAME: 'MSE skill score',
    evaluation_plotting.MAE_NAME: 'Mean absolute error',
    evaluation_plotting.MAE_SKILL_SCORE_NAME: 'MAE skill score',
    evaluation_plotting.BIAS_NAME: 'Bias',
    evaluation_plotting.CORRELATION_NAME: 'Correlation',
    evaluation_plotting.KGE_NAME: 'Kling-Gupta efficiency'
}

SCORE_NAME_TO_PROFILE_KEY = {
    evaluation_plotting.MSE_NAME: evaluation.VECTOR_MSE_KEY,
    evaluation_plotting.MSE_SKILL_SCORE_NAME: evaluation.VECTOR_MSE_SKILL_KEY,
    evaluation_plotting.MAE_NAME: evaluation.VECTOR_MAE_KEY,
    evaluation_plotting.MAE_SKILL_SCORE_NAME: evaluation.VECTOR_MAE_SKILL_KEY,
    evaluation_plotting.BIAS_NAME: evaluation.VECTOR_BIAS_KEY,
    evaluation_plotting.CORRELATION_NAME: evaluation.VECTOR_CORRELATION_KEY,
    evaluation_plotting.KGE_NAME: evaluation.VECTOR_KGE_KEY
}

ORIG_UNIT_SCORE_NAMES = [
    evaluation_plotting.MAE_NAME, evaluation_plotting.BIAS_NAME
]
SQUARED_UNIT_SCORE_NAMES = [evaluation_plotting.MSE_NAME]

TARGET_NAME_TO_VERBOSE = {
    example_utils.SHORTWAVE_DOWN_FLUX_NAME: 'downwelling flux',
    example_utils.SHORTWAVE_DOWN_FLUX_INC_NAME:
        r'$\frac{\Delta F_{down}}{\Delta z}$',
    example_utils.SHORTWAVE_UP_FLUX_NAME: 'upwelling flux',
    example_utils.SHORTWAVE_UP_FLUX_INC_NAME:
        r'$\frac{\Delta F_{up}}{\Delta z}$',
    example_utils.SHORTWAVE_HEATING_RATE_NAME: 'heating rate',
    example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME: 'surface downwelling flux',
    example_utils.SHORTWAVE_TOA_UP_FLUX_NAME: 'TOA upwelling flux',
    evaluation.NET_FLUX_NAME: 'net flux',
    evaluation.HIGHEST_UP_FLUX_NAME: 'top-of-profile upwelling flux',
    evaluation.LOWEST_DOWN_FLUX_NAME: 'bottom-of-profile downwelling flux'
}

TARGET_NAME_TO_UNITS = {
    example_utils.SHORTWAVE_DOWN_FLUX_NAME: r'W m$^{-2}$',
    example_utils.SHORTWAVE_DOWN_FLUX_INC_NAME: r'W m$^{-3}$',
    example_utils.SHORTWAVE_UP_FLUX_NAME: r'W m$^{-2}$',
    example_utils.SHORTWAVE_UP_FLUX_INC_NAME: r'W m$^{-3}$',
    example_utils.SHORTWAVE_HEATING_RATE_NAME: r'K day$^{-1}$',
    example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME: r'W m$^{-2}$',
    example_utils.SHORTWAVE_TOA_UP_FLUX_NAME: r'W m$^{-2}$',
    evaluation.NET_FLUX_NAME: r'W m$^{-2}$',
    evaluation.HIGHEST_UP_FLUX_NAME: r'W m$^{-2}$',
    evaluation.LOWEST_DOWN_FLUX_NAME: r'W m$^{-2}$'
}

TARGET_NAME_TO_SQUARED_UNITS = {
    example_utils.SHORTWAVE_DOWN_FLUX_NAME: r'W$^{2}$ m$^{-4}$',
    example_utils.SHORTWAVE_DOWN_FLUX_INC_NAME: r'W$^{2}$ m$^{-6}$',
    example_utils.SHORTWAVE_UP_FLUX_NAME: r'W$^{2}$ m$^{-4}$',
    example_utils.SHORTWAVE_UP_FLUX_INC_NAME: r'W$^{2}$ m$^{-6}$',
    example_utils.SHORTWAVE_HEATING_RATE_NAME: r'K$^{2}$ day$^{-2}$',
    example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME: r'W$^{2}$ m$^{-4}$',
    example_utils.SHORTWAVE_TOA_UP_FLUX_NAME: r'W$^{2}$ m$^{-4}$',
    evaluation.NET_FLUX_NAME: r'W$^{2}$ m$^{-4}$',
    evaluation.HIGHEST_UP_FLUX_NAME: r'W$^{2}$ m$^{-4}$',
    evaluation.LOWEST_DOWN_FLUX_NAME: r'W$^{2}$ m$^{-4}$'
}

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

INPUT_FILES_ARG_NAME = 'input_eval_file_names'
LINE_STYLES_ARG_NAME = 'line_styles'
LINE_COLOURS_ARG_NAME = 'line_colours'
SET_DESCRIPTIONS_ARG_NAME = 'set_descriptions'
USE_LOG_SCALE_ARG_NAME = 'use_log_scale'
PLOT_BY_HEIGHT_ARG_NAME = 'plot_by_height'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILES_HELP_STRING = (
    'Space-separated list of paths to input files (each will be read by '
    '`evaluation.write_file`).'
)
LINE_STYLES_HELP_STRING = (
    'Space-separated list of line styles (in any format accepted by '
    'matplotlib).  Must have same length as `{0:s}`.'
).format(INPUT_FILES_ARG_NAME)

LINE_COLOURS_HELP_STRING = (
    'Space-separated list of line colours.  Each colour must be a length-3 '
    'array of (R, G, B) intensities ranging from 0...255.  Colours in each '
    'array should be underscore-separated, so the list should look like '
    '"0_0_0 217_95_2", for examples.  List must have same length as `{0:s}`.'
).format(INPUT_FILES_ARG_NAME)

SET_DESCRIPTIONS_HELP_STRING = (
    'Space-separated list of set descriptions, to be used in legends.  Must '
    'have same length as `{0:s}`.'
).format(INPUT_FILES_ARG_NAME)

USE_LOG_SCALE_HELP_STRING = (
    'Boolean flag.  If 1 (0), will use logarithmic (linear) scale for height '
    'axis.'
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
    '--' + INPUT_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=INPUT_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LINE_STYLES_ARG_NAME, type=str, nargs='+', required=True,
    help=LINE_STYLES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LINE_COLOURS_ARG_NAME, type=str, nargs='+', required=True,
    help=LINE_COLOURS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SET_DESCRIPTIONS_ARG_NAME, type=str, nargs='+', required=True,
    help=SET_DESCRIPTIONS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + USE_LOG_SCALE_ARG_NAME, type=int, required=False, default=1,
    help=USE_LOG_SCALE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_BY_HEIGHT_ARG_NAME, type=int, required=False, default=1,
    help=PLOT_BY_HEIGHT_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_attributes_diagram(
        evaluation_tables_xarray, line_styles, line_colours,
        set_descriptions_abbrev, set_descriptions_verbose,
        mean_training_example_dict, target_name, output_dir_name,
        height_m_agl=None):
    """Plots attributes diagram for each set and each target variable.

    S = number of evaluation sets
    T_v = number of vector target variables
    T_s = number of scalar target variables
    H = number of heights

    :param evaluation_tables_xarray: length-S list of xarray tables in format
        returned by `evaluation.read_file`.
    :param line_styles: length-S list of line styles.
    :param line_colours: length-S list of line colours.
    :param set_descriptions_abbrev: length-S list of abbreviated descriptions
        for evaluation sets.
    :param set_descriptions_verbose: length-S list of verbose descriptions for
        evaluation sets.
    :param mean_training_example_dict: Dictionary created by
        `normalization.create_mean_example`.
    :param target_name: Name of target variable.
    :param output_dir_name: Name of output directory.  Figures will be saved
        here.
    :param height_m_agl: Height (metres above ground level).
    """

    t = evaluation_tables_xarray[0]
    is_scalar = target_name in t.coords[evaluation.SCALAR_FIELD_DIM].values
    is_aux = target_name in t.coords[evaluation.AUX_TARGET_FIELD_DIM].values
    is_vector = target_name in t.coords[evaluation.VECTOR_FIELD_DIM].values

    if is_scalar:
        target_indices = numpy.array([
            numpy.where(
                t.coords[evaluation.SCALAR_FIELD_DIM].values == target_name
            )[0][0]
            for t in evaluation_tables_xarray
        ], dtype=int)

        mean_predictions_by_set = [
            t[evaluation.SCALAR_RELIABILITY_X_KEY].values[k, ...]
            for t, k in zip(evaluation_tables_xarray, target_indices)
        ]
        mean_observations_by_set = [
            t[evaluation.SCALAR_RELIABILITY_Y_KEY].values[k, ...]
            for t, k in zip(evaluation_tables_xarray, target_indices)
        ]
        example_counts_by_set = [
            t[evaluation.SCALAR_RELIABILITY_COUNT_KEY].values[k, ...]
            for t, k in zip(evaluation_tables_xarray, target_indices)
        ]
        inverted_mean_obs_by_set = [
            t[evaluation.SCALAR_INV_RELIABILITY_X_KEY].values[k, ...]
            for t, k in zip(evaluation_tables_xarray, target_indices)
        ]
        inverted_example_counts_by_set = [
            t[evaluation.SCALAR_INV_RELIABILITY_COUNT_KEY].values[k, ...]
            for t, k in zip(evaluation_tables_xarray, target_indices)
        ]

        k = mean_training_example_dict[
            example_utils.SCALAR_TARGET_NAMES_KEY
        ].index(target_name)

        climo_value = mean_training_example_dict[
            example_utils.SCALAR_TARGET_VALS_KEY
        ][0, k]

    elif is_aux:
        target_indices = numpy.array([
            numpy.where(
                t.coords[evaluation.AUX_TARGET_FIELD_DIM].values == target_name
            )[0][0]
            for t in evaluation_tables_xarray
        ], dtype=int)

        mean_predictions_by_set = [
            t[evaluation.AUX_RELIABILITY_X_KEY].values[k, ...]
            for t, k in zip(evaluation_tables_xarray, target_indices)
        ]
        mean_observations_by_set = [
            t[evaluation.AUX_RELIABILITY_Y_KEY].values[k, ...]
            for t, k in zip(evaluation_tables_xarray, target_indices)
        ]
        example_counts_by_set = [
            t[evaluation.AUX_RELIABILITY_COUNT_KEY].values[k, ...]
            for t, k in zip(evaluation_tables_xarray, target_indices)
        ]
        inverted_mean_obs_by_set = [
            t[evaluation.AUX_INV_RELIABILITY_X_KEY].values[k, ...]
            for t, k in zip(evaluation_tables_xarray, target_indices)
        ]
        inverted_example_counts_by_set = [
            t[evaluation.AUX_INV_RELIABILITY_COUNT_KEY].values[k, ...]
            for t, k in zip(evaluation_tables_xarray, target_indices)
        ]

        training_target_names = mean_training_example_dict[
            example_utils.SCALAR_TARGET_NAMES_KEY
        ]
        training_target_matrix = mean_training_example_dict[
            example_utils.SCALAR_TARGET_VALS_KEY
        ]

        if target_name == evaluation.NET_FLUX_NAME:
            down_flux_index = training_target_names.index(
                example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME
            )
            up_flux_index = training_target_names.index(
                example_utils.SHORTWAVE_TOA_UP_FLUX_NAME
            )
            climo_value = (
                training_target_matrix[0, down_flux_index] -
                training_target_matrix[0, up_flux_index]
            )
        else:
            k = training_target_names.index(target_name)
            climo_value = training_target_matrix[0, k]
    else:
        target_indices = numpy.array([
            numpy.where(
                t.coords[evaluation.VECTOR_FIELD_DIM].values == target_name
            )[0][0]
            for t in evaluation_tables_xarray
        ], dtype=int)

        height_indices = numpy.array([
            numpy.argmin(numpy.absolute(
                t.coords[evaluation.HEIGHT_DIM].values - height_m_agl
            )) for t in evaluation_tables_xarray
        ], dtype=int)

        mean_predictions_by_set = [
            t[evaluation.VECTOR_RELIABILITY_X_KEY].values[j, k, ...]
            for t, j, k in
            zip(evaluation_tables_xarray, height_indices, target_indices)
        ]
        mean_observations_by_set = [
            t[evaluation.VECTOR_RELIABILITY_Y_KEY].values[j, k, ...]
            for t, j, k in
            zip(evaluation_tables_xarray, height_indices, target_indices)
        ]
        example_counts_by_set = [
            t[evaluation.VECTOR_RELIABILITY_COUNT_KEY].values[j, k, ...]
            for t, j, k in
            zip(evaluation_tables_xarray, height_indices, target_indices)
        ]
        inverted_mean_obs_by_set = [
            t[evaluation.VECTOR_INV_RELIABILITY_X_KEY].values[j, k, ...]
            for t, j, k in
            zip(evaluation_tables_xarray, height_indices, target_indices)
        ]
        inverted_example_counts_by_set = [
            t[evaluation.VECTOR_INV_RELIABILITY_COUNT_KEY].values[j, k, ...]
            for t, j, k in
            zip(evaluation_tables_xarray, height_indices, target_indices)
        ]

        j = numpy.argmin(numpy.absolute(
            mean_training_example_dict[example_utils.HEIGHTS_KEY] - height_m_agl
        ))

        k = mean_training_example_dict[
            example_utils.VECTOR_TARGET_NAMES_KEY
        ].index(target_name)

        climo_value = mean_training_example_dict[
            example_utils.VECTOR_TARGET_VALS_KEY
        ][0, j, k]

    concat_values = numpy.concatenate([
        a for a in mean_predictions_by_set + mean_observations_by_set
        if a is not None
    ])
    max_value_to_plot = numpy.nanpercentile(concat_values, 99.9)
    min_value_to_plot = numpy.nanpercentile(concat_values, 0.1)
    min_value_to_plot = numpy.minimum(min_value_to_plot, 0.)

    num_evaluation_sets = len(evaluation_tables_xarray)

    for main_index in range(num_evaluation_sets):
        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

        legend_handles = []
        legend_strings = []

        this_handle = evaluation_plotting.plot_attributes_diagram(
            figure_object=figure_object, axes_object=axes_object,
            mean_predictions=mean_predictions_by_set[main_index],
            mean_observations=mean_observations_by_set[main_index],
            example_counts=example_counts_by_set[main_index],
            mean_value_in_training=climo_value,
            min_value_to_plot=min_value_to_plot,
            max_value_to_plot=max_value_to_plot,
            inv_mean_observations=inverted_mean_obs_by_set[main_index],
            inv_example_counts=inverted_example_counts_by_set[main_index],
            line_colour=line_colours[main_index],
            line_style=line_styles[main_index], line_width=4
        )

        if this_handle is not None:
            legend_handles.append(this_handle)
            legend_strings.append(set_descriptions_verbose[main_index])

        axes_object.set_xlabel('Prediction ({0:s})'.format(
            TARGET_NAME_TO_UNITS[target_name]
        ))
        axes_object.set_ylabel('Conditional mean observation ({0:s})'.format(
            TARGET_NAME_TO_UNITS[target_name]
        ))

        title_string = 'Attributes diagram for {0:s}'.format(
            TARGET_NAME_TO_VERBOSE[target_name]
        )
        if is_vector:
            title_string += ' at {0:d} m AGL'.format(
                int(numpy.round(height_m_agl))
            )

        axes_object.set_title(title_string)

        for i in range(num_evaluation_sets):
            if i == main_index:
                continue

            this_handle = evaluation_plotting._plot_reliability_curve(
                axes_object=axes_object,
                mean_predictions=mean_predictions_by_set[i],
                mean_observations=mean_observations_by_set[i],
                min_value_to_plot=min_value_to_plot,
                max_value_to_plot=max_value_to_plot,
                line_colour=line_colours[i], line_style=line_styles[i],
                line_width=4
            )

            if this_handle is not None:
                legend_handles.append(this_handle)
                legend_strings.append(set_descriptions_verbose[i])

        if len(legend_handles) > 1:
            axes_object.legend(
                legend_handles, legend_strings, loc='center left',
                bbox_to_anchor=(0, 0.5), fancybox=True, shadow=False,
                facecolor='white', edgecolor='k', framealpha=0.5, ncol=1
            )

        figure_file_name = '{0:s}/{1:s}'.format(
            output_dir_name, target_name.replace('_', '-')
        )

        if is_vector:
            figure_file_name += '_{0:05d}m-agl'.format(
                int(numpy.round(height_m_agl))
            )

        figure_file_name += '_attributes_{0:s}.jpg'.format(
            set_descriptions_abbrev[main_index]
        )

        print('Saving figure to: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)


def _plot_score_profile(
        evaluation_tables_xarray, line_styles, line_colours,
        set_descriptions_verbose, target_name, score_name, use_log_scale,
        output_dir_name):
    """Plots vertical profile of one score.

    :param evaluation_tables_xarray: See doc for `_plot_attributes_diagram`.
    :param line_styles: Same.
    :param line_colours: Same.
    :param set_descriptions_verbose: Same.
    :param target_name: Name of target variable for which score is being plotted.
    :param score_name: Name of score being plotted.
    :param use_log_scale: Boolean flag.  If True, will plot heights (y-axis) in
        log scale.
    :param output_dir_name: Name of output directory.  Figure will be saved
        here.
    """

    target_indices = numpy.array([
        numpy.where(
            t.coords[evaluation.VECTOR_FIELD_DIM].values == target_name
        )[0][0]
        for t in evaluation_tables_xarray
    ], dtype=int)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    score_key = SCORE_NAME_TO_PROFILE_KEY[score_name]
    legend_handles = []
    legend_strings = []
    num_evaluation_sets = len(evaluation_tables_xarray)

    for i in range(num_evaluation_sets):
        k = target_indices[i]

        this_handle = evaluation_plotting.plot_score_profile(
            heights_m_agl=
            evaluation_tables_xarray[i].coords[evaluation.HEIGHT_DIM].values,
            score_values=evaluation_tables_xarray[i][score_key].values[:, k],
            score_name=score_name, line_colour=line_colours[i],
            line_width=4, line_style=line_styles[i],
            use_log_scale=use_log_scale, axes_object=axes_object,
            are_axes_new=i == 0
        )

        legend_handles.append(this_handle)
        legend_strings.append(set_descriptions_verbose[i])

    if len(legend_handles) > 1:
        axes_object.legend(
            legend_handles, legend_strings, loc='center left',
            bbox_to_anchor=(0, 0.5), fancybox=True, shadow=False,
            facecolor='white', edgecolor='k', framealpha=0.5, ncol=1
        )

    title_string = '{0:s} for {1:s}'.format(
        SCORE_NAME_TO_VERBOSE[score_name],
        TARGET_NAME_TO_VERBOSE[target_name]
    )

    if num_evaluation_sets == 1:
        k = target_indices[0]
        prmse = (
            evaluation_tables_xarray[0][evaluation.VECTOR_PRMSE_KEY].values[k]
        )
        title_string += ' (PRMSE = {0:.2f} {1:s})'.format(
            prmse, TARGET_NAME_TO_UNITS[target_name]
        )

    x_label_string = '{0:s}'.format(SCORE_NAME_TO_VERBOSE[score_name])

    if score_name in SQUARED_UNIT_SCORE_NAMES:
        x_label_string += ' ({0:s})'.format(
            TARGET_NAME_TO_SQUARED_UNITS[target_name]
        )
    elif score_name in ORIG_UNIT_SCORE_NAMES:
        x_label_string += ' ({0:s})'.format(TARGET_NAME_TO_UNITS[target_name])

    axes_object.set_xlabel(x_label_string)
    axes_object.set_title(title_string)

    figure_file_name = '{0:s}/{1:s}_{2:s}_profile.jpg'.format(
        output_dir_name, target_name.replace('_', '-'),
        score_name.replace('_', '-')
    )

    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _plot_error_distributions(
        prediction_dicts, model_metadata_dict, aux_target_names,
        set_descriptions_abbrev, set_descriptions_verbose, output_dir_name):
    """Plots error distribution for each set and each target variable.

    S = number of evaluation sets
    T_v = number of vector target variables
    T_s = number of scalar target variables
    H = number of heights

    :param prediction_dicts: length-S list of dictionaries in format returned by
        `prediction_io.read_file`.
    :param model_metadata_dict: Dictionary with metadata for model that
        generated predictions (in format returned by
        `neural_net.read_metafile`).
    :param aux_target_names: 1-D list with names of auxiliary target variables.
    :param set_descriptions_abbrev: length-S list of abbreviated descriptions
        for evaluation sets.
    :param set_descriptions_verbose: length-S list of verbose descriptions for
        evaluation sets.
    :param output_dir_name: Name of output directory.  Figures will be saved
        here.
    """

    generator_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    scalar_target_names = (
        generator_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY]
    )
    vector_target_names = (
        generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY]
    )
    heights_m_agl = generator_option_dict[neural_net.HEIGHTS_KEY]

    num_vector_targets = len(vector_target_names)
    num_scalar_targets = len(scalar_target_names)
    num_aux_targets = len(aux_target_names)
    num_evaluation_sets = len(set_descriptions_verbose)

    for k in range(num_vector_targets):
        for i in range(num_evaluation_sets):
            figure_object, axes_object = pyplot.subplots(
                1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
            )

            error_matrix = (
                prediction_dicts[i][prediction_io.VECTOR_PREDICTIONS_KEY][
                    ..., k
                ] -
                prediction_dicts[i][prediction_io.VECTOR_TARGETS_KEY][..., k]
            )

            evaluation_plotting.plot_error_dist_many_heights(
                error_matrix=error_matrix, heights_m_agl=heights_m_agl,
                min_error_to_plot=numpy.percentile(error_matrix, 1.),
                max_error_to_plot=numpy.percentile(error_matrix, 99.),
                axes_object=axes_object
            )

            axes_object.set_title(
                'Error distribution for {0:s} ({1:s})\n{2:s}'.format(
                    TARGET_NAME_TO_VERBOSE[vector_target_names[k]],
                    TARGET_NAME_TO_UNITS[vector_target_names[k]],
                    set_descriptions_verbose[i]
                )
            )

            figure_file_name = '{0:s}/{1:s}_error-dist_{2:s}.jpg'.format(
                output_dir_name, vector_target_names[k].replace('_', '-'),
                set_descriptions_abbrev[i]
            )

            print('Saving figure to: "{0:s}"...'.format(figure_file_name))
            figure_object.savefig(
                figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
                pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(figure_object)

    for k in range(num_scalar_targets):
        for i in range(num_evaluation_sets):
            figure_object, axes_object = pyplot.subplots(
                1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
            )

            error_values = (
                prediction_dicts[i][prediction_io.SCALAR_PREDICTIONS_KEY][
                    ..., k
                ] -
                prediction_dicts[i][prediction_io.SCALAR_TARGETS_KEY][..., k]
            )

            evaluation_plotting.plot_error_distribution(
                error_values=error_values,
                min_error_to_plot=numpy.percentile(error_values, 1.),
                max_error_to_plot=numpy.percentile(error_values, 99.),
                axes_object=axes_object
            )

            axes_object.set_title(
                'Error distribution for {0:s} ({1:s})\n{2:s}'.format(
                    TARGET_NAME_TO_VERBOSE[scalar_target_names[k]],
                    TARGET_NAME_TO_UNITS[scalar_target_names[k]],
                    set_descriptions_verbose[i]
                )
            )

            figure_file_name = '{0:s}/{1:s}_error-dist_{2:s}.jpg'.format(
                output_dir_name, scalar_target_names[k].replace('_', '-'),
                set_descriptions_abbrev[i]
            )

            print('Saving figure to: "{0:s}"...'.format(figure_file_name))
            figure_object.savefig(
                figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
                pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(figure_object)

    for k in range(num_aux_targets):
        for i in range(num_evaluation_sets):
            figure_object, axes_object = pyplot.subplots(
                1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
            )

            if aux_target_names[k] == evaluation.NET_FLUX_NAME:
                down_flux_index = scalar_target_names.index(
                    example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME
                )
                up_flux_index = scalar_target_names.index(
                    example_utils.SHORTWAVE_TOA_UP_FLUX_NAME
                )

                predicted_values = (
                    prediction_dicts[i][prediction_io.SCALAR_PREDICTIONS_KEY][
                        ..., down_flux_index
                    ] -
                    prediction_dicts[i][prediction_io.SCALAR_PREDICTIONS_KEY][
                        ..., up_flux_index
                    ]
                )

                target_values = (
                    prediction_dicts[i][prediction_io.SCALAR_TARGETS_KEY][
                        ..., down_flux_index
                    ] -
                    prediction_dicts[i][prediction_io.SCALAR_TARGETS_KEY][
                        ..., up_flux_index
                    ]
                )

                error_values = predicted_values - target_values
            else:
                this_index = scalar_target_names.index(aux_target_names[k])

                error_values = (
                    prediction_dicts[i][prediction_io.SCALAR_PREDICTIONS_KEY][
                        ..., this_index
                    ] -
                    prediction_dicts[i][prediction_io.SCALAR_TARGETS_KEY][
                        ..., this_index
                    ]
                )

            evaluation_plotting.plot_error_distribution(
                error_values=error_values,
                min_error_to_plot=numpy.percentile(error_values, 1.),
                max_error_to_plot=numpy.percentile(error_values, 99.),
                axes_object=axes_object
            )

            axes_object.set_title(
                'Error distribution for {0:s} ({1:s})\n{2:s}'.format(
                    TARGET_NAME_TO_VERBOSE[aux_target_names[k]],
                    TARGET_NAME_TO_UNITS[aux_target_names[k]],
                    set_descriptions_verbose[i]
                )
            )

            figure_file_name = '{0:s}/{1:s}_error-dist_{2:s}.jpg'.format(
                output_dir_name, aux_target_names[k].replace('_', '-'),
                set_descriptions_abbrev[i]
            )

            print('Saving figure to: "{0:s}"...'.format(figure_file_name))
            figure_object.savefig(
                figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
                pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(figure_object)


def _plot_reliability_by_height(
        evaluation_tables_xarray, vector_target_names, heights_m_agl,
        set_descriptions_abbrev, set_descriptions_verbose, output_dir_name):
    """Plots height-dependent reliability curve for each set and vector target.

    :param evaluation_tables_xarray: See doc for `_plot_attributes_diagram`.
    :param vector_target_names: See doc for `_plot_error_distributions`.
    :param heights_m_agl: Same.
    :param set_descriptions_abbrev: Same.
    :param set_descriptions_verbose: Same.
    :param output_dir_name: Same.
    """

    num_vector_targets = len(vector_target_names)
    num_evaluation_sets = len(set_descriptions_verbose)

    for k in range(num_vector_targets):
        for i in range(num_evaluation_sets):
            t = evaluation_tables_xarray[i]

            mean_prediction_matrix = numpy.take(
                t[evaluation.VECTOR_RELIABILITY_X_KEY].values, axis=1, indices=k
            )
            mean_target_matrix = numpy.take(
                t[evaluation.VECTOR_RELIABILITY_Y_KEY].values, axis=1, indices=k
            )
            concat_matrix = numpy.concatenate(
                (mean_prediction_matrix, mean_target_matrix), axis=0
            )
            max_value_to_plot = numpy.nanpercentile(concat_matrix, 99)

            figure_object, axes_object = pyplot.subplots(
                1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
            )

            evaluation_plotting.plot_rel_curve_many_heights(
                mean_target_matrix=mean_target_matrix,
                mean_prediction_matrix=mean_prediction_matrix,
                heights_m_agl=heights_m_agl, min_value_to_plot=0.,
                max_value_to_plot=max_value_to_plot, axes_object=axes_object
            )

            axes_object.set_title(
                'Reliability curves for {0:s}\n{1:s}'.format(
                    TARGET_NAME_TO_VERBOSE[vector_target_names[k]],
                    set_descriptions_verbose[i]
                )
            )
            axes_object.set_xlabel('Prediction ({0:s})'.format(
                TARGET_NAME_TO_UNITS[vector_target_names[k]]
            ))
            axes_object.set_ylabel(
                'Conditional mean observation ({0:s})'.format(
                    TARGET_NAME_TO_UNITS[vector_target_names[k]]
                )
            )

            figure_file_name = '{0:s}/{1:s}_reliability_{2:s}.jpg'.format(
                output_dir_name, vector_target_names[k].replace('_', '-'),
                set_descriptions_abbrev[i]
            )

            print('Saving figure to: "{0:s}"...'.format(figure_file_name))
            figure_object.savefig(
                figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
                pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(figure_object)


def _run(evaluation_file_names, line_styles, line_colour_strings,
         set_descriptions_verbose, use_log_scale, plot_by_height,
         output_dir_name):
    """Plots model evaluation.

    This is effectively the main method.

    :param evaluation_file_names: See documentation at top of file.
    :param line_styles: Same.
    :param line_colour_strings: Same.
    :param set_descriptions_verbose: Same.
    :param use_log_scale: Same.
    :param plot_by_height: Same.
    :param output_dir_name: Same.
    """

    # Check input args.
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    num_evaluation_sets = len(evaluation_file_names)
    expected_dim = numpy.array([num_evaluation_sets], dtype=int)

    error_checking.assert_is_string_list(line_styles)
    error_checking.assert_is_numpy_array(
        numpy.array(line_styles), exact_dimensions=expected_dim
    )

    error_checking.assert_is_string_list(set_descriptions_verbose)
    error_checking.assert_is_numpy_array(
        numpy.array(set_descriptions_verbose), exact_dimensions=expected_dim
    )

    set_descriptions_verbose = [
        s.replace('_', ' ') for s in set_descriptions_verbose
    ]
    set_descriptions_abbrev = [
        s.lower().replace(' ', '-') for s in set_descriptions_verbose
    ]

    error_checking.assert_is_string_list(line_colour_strings)
    error_checking.assert_is_numpy_array(
        numpy.array(line_colour_strings), exact_dimensions=expected_dim
    )
    line_colours = [
        numpy.fromstring(s, dtype=float, sep='_') / 255
        for s in line_colour_strings
    ]

    for i in range(num_evaluation_sets):
        error_checking.assert_is_numpy_array(
            line_colours[i], exact_dimensions=numpy.array([3], dtype=int)
        )
        error_checking.assert_is_geq_numpy_array(line_colours[i], 0.)
        error_checking.assert_is_leq_numpy_array(line_colours[i], 1.)

    # Read files.
    evaluation_tables_xarray = [xarray.Dataset()] * num_evaluation_sets
    prediction_dicts = [dict()] * num_evaluation_sets

    for i in range(num_evaluation_sets):
        print('Reading data from: "{0:s}"...'.format(evaluation_file_names[i]))
        evaluation_tables_xarray[i] = evaluation.read_file(
            evaluation_file_names[i]
        )

        this_prediction_file_name = (
            evaluation_tables_xarray[i].attrs[evaluation.PREDICTION_FILE_KEY]
        )

        print('Reading data from: "{0:s}"...'.format(this_prediction_file_name))
        prediction_dicts[i] = prediction_io.read_file(this_prediction_file_name)

    model_file_name = (
        evaluation_tables_xarray[0].attrs[evaluation.MODEL_FILE_KEY]
    )
    model_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    generator_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]

    scalar_target_names = (
        generator_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY]
    )
    vector_target_names = (
        generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY]
    )
    heights_m_agl = generator_option_dict[neural_net.HEIGHTS_KEY]

    try:
        t = evaluation_tables_xarray[0]
        aux_target_names = t.coords[evaluation.AUX_TARGET_FIELD_DIM].values
    except:
        aux_target_names = []

    num_scalar_targets = len(scalar_target_names)
    num_vector_targets = len(vector_target_names)
    num_heights = len(heights_m_agl)
    num_aux_targets = len(aux_target_names)

    example_dict = {
        example_utils.SCALAR_TARGET_NAMES_KEY: scalar_target_names,
        example_utils.VECTOR_TARGET_NAMES_KEY: vector_target_names,
        example_utils.HEIGHTS_KEY: heights_m_agl,
        example_utils.SCALAR_PREDICTOR_NAMES_KEY:
            generator_option_dict[neural_net.SCALAR_PREDICTOR_NAMES_KEY],
        example_utils.VECTOR_PREDICTOR_NAMES_KEY:
            generator_option_dict[neural_net.VECTOR_PREDICTOR_NAMES_KEY]
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
    training_example_dict = example_utils.subset_by_height(
        example_dict=training_example_dict, heights_m_agl=heights_m_agl
    )
    mean_training_example_dict = normalization.create_mean_example(
        new_example_dict=example_dict,
        training_example_dict=training_example_dict
    )

    print(SEPARATOR_STRING)

    # Do actual stuff.
    _plot_error_distributions(
        prediction_dicts=prediction_dicts,
        model_metadata_dict=model_metadata_dict,
        aux_target_names=aux_target_names,
        set_descriptions_abbrev=set_descriptions_abbrev,
        set_descriptions_verbose=set_descriptions_verbose,
        output_dir_name=output_dir_name
    )
    print(SEPARATOR_STRING)

    _plot_reliability_by_height(
        evaluation_tables_xarray=evaluation_tables_xarray,
        vector_target_names=vector_target_names,
        heights_m_agl=heights_m_agl,
        set_descriptions_abbrev=set_descriptions_abbrev,
        set_descriptions_verbose=set_descriptions_verbose,
        output_dir_name=output_dir_name
    )
    print(SEPARATOR_STRING)

    for k in range(num_vector_targets):
        for this_score_name in list(SCORE_NAME_TO_PROFILE_KEY.keys()):
            _plot_score_profile(
                evaluation_tables_xarray=evaluation_tables_xarray,
                line_styles=line_styles, line_colours=line_colours,
                set_descriptions_verbose=set_descriptions_verbose,
                target_name=vector_target_names[k], score_name=this_score_name,
                use_log_scale=use_log_scale, output_dir_name=output_dir_name
            )

    print(SEPARATOR_STRING)

    for k in range(num_scalar_targets):
        _plot_attributes_diagram(
            evaluation_tables_xarray=evaluation_tables_xarray,
            line_styles=line_styles,
            line_colours=line_colours,
            set_descriptions_abbrev=set_descriptions_abbrev,
            set_descriptions_verbose=set_descriptions_verbose,
            mean_training_example_dict=mean_training_example_dict,
            target_name=scalar_target_names[k],
            output_dir_name=output_dir_name
        )

    for k in range(num_aux_targets):
        _plot_attributes_diagram(
            evaluation_tables_xarray=evaluation_tables_xarray,
            line_styles=line_styles,
            line_colours=line_colours,
            set_descriptions_abbrev=set_descriptions_abbrev,
            set_descriptions_verbose=set_descriptions_verbose,
            mean_training_example_dict=mean_training_example_dict,
            target_name=aux_target_names[k],
            output_dir_name=output_dir_name
        )

    if not plot_by_height:
        return

    print(SEPARATOR_STRING)

    for k in range(num_vector_targets):
        for j in range(num_heights):
            _plot_attributes_diagram(
                evaluation_tables_xarray=evaluation_tables_xarray,
                line_styles=line_styles,
                line_colours=line_colours,
                set_descriptions_abbrev=set_descriptions_abbrev,
                set_descriptions_verbose=set_descriptions_verbose,
                mean_training_example_dict=mean_training_example_dict,
                height_m_agl=heights_m_agl[j],
                target_name=vector_target_names[k],
                output_dir_name=output_dir_name
            )

        if k != num_vector_targets - 1:
            print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        evaluation_file_names=getattr(INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME),
        line_styles=getattr(INPUT_ARG_OBJECT, LINE_STYLES_ARG_NAME),
        line_colour_strings=getattr(INPUT_ARG_OBJECT, LINE_COLOURS_ARG_NAME),
        set_descriptions_verbose=getattr(
            INPUT_ARG_OBJECT, SET_DESCRIPTIONS_ARG_NAME
        ),
        use_log_scale=bool(getattr(INPUT_ARG_OBJECT, USE_LOG_SCALE_ARG_NAME)),
        plot_by_height=bool(getattr(INPUT_ARG_OBJECT, PLOT_BY_HEIGHT_ARG_NAME)),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
