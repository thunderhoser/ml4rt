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

PROFILE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
TAYLOR_MARKER_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

INPUT_FILES_ARG_NAME = 'input_eval_file_names'
LINE_STYLES_ARG_NAME = 'line_styles'
LINE_COLOURS_ARG_NAME = 'line_colours'
LINE_LEGENDS_ARG_NAME = 'line_legend_strings'
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

LINE_LEGENDS_HELP_STRING = (
    'Space-separated list of line legends (in any format accepted by '
    'matplotlib).  Must have same length as `{0:s}`.'
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
    '--' + LINE_LEGENDS_ARG_NAME, type=str, nargs='+', required=True,
    help=LINE_LEGENDS_HELP_STRING
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


def _plot_taylor_diagram(
        evaluation_table_array, output_dir_name, scalar_target_index=None,
        aux_target_index=None, vector_target_index=None, height_index=None):
    """Plots Taylor diagram for one field.

    In this case, "field" means one scalar variable or one vector variable at
    one height.

    :param evaluation_table_array: Evaluation results (in format returned by
        `evaluation.read_file`).
    :param output_dir_name: Name of output directory (figure will be saved
        here).
    :param scalar_target_index: Index for scalar target variable.
    :param aux_target_index: Index for auxiliary target variable.
    :param vector_target_index: Index for vector target variable.
    :param height_index: Height index for target variable.
    """

    if scalar_target_index is not None:
        target_name = evaluation_table_array.coords[
            evaluation.SCALAR_FIELD_DIM
        ].values[scalar_target_index]

        target_stdev = evaluation_table_array[
            evaluation.SCALAR_TARGET_STDEV_KEY
        ].values[scalar_target_index]

        prediction_stdev = evaluation_table_array[
            evaluation.SCALAR_PREDICTION_STDEV_KEY
        ].values[scalar_target_index]

        correlation = evaluation_table_array[
            evaluation.SCALAR_CORRELATION_KEY
        ].values[scalar_target_index]

        output_file_name = '{0:s}/{1:s}_taylor.jpg'.format(
            output_dir_name, target_name.replace('_', '-')
        )

    elif aux_target_index is not None:
        target_name = evaluation_table_array.coords[
            evaluation.AUX_TARGET_FIELD_DIM
        ].values[aux_target_index]

        predicted_target_name = evaluation_table_array.coords[
            evaluation.AUX_PREDICTED_FIELD_DIM
        ].values[aux_target_index]

        target_stdev = evaluation_table_array[
            evaluation.AUX_TARGET_STDEV_KEY
        ].values[aux_target_index]

        prediction_stdev = evaluation_table_array[
            evaluation.AUX_PREDICTION_STDEV_KEY
        ].values[aux_target_index]

        correlation = evaluation_table_array[
            evaluation.AUX_CORRELATION_KEY
        ].values[aux_target_index]

        output_file_name = '{0:s}/aux_{1:s}_taylor.jpg'.format(
            output_dir_name, target_name.replace('_', '-')
        )

    else:
        target_name = evaluation_table_array.coords[
            evaluation.VECTOR_FIELD_DIM
        ].values[vector_target_index]

        height_m_agl = evaluation_table_array.coords[
            evaluation.HEIGHT_DIM
        ].values[height_index]

        target_stdev = evaluation_table_array[
            evaluation.VECTOR_TARGET_STDEV_KEY
        ].values[height_index, vector_target_index]

        prediction_stdev = evaluation_table_array[
            evaluation.VECTOR_PREDICTION_STDEV_KEY
        ].values[height_index, vector_target_index]

        correlation = evaluation_table_array[
            evaluation.VECTOR_CORRELATION_KEY
        ].values[height_index, vector_target_index]

        output_file_name = '{0:s}/{1:s}_{2:05d}metres_taylor.jpg'.format(
            output_dir_name, target_name.replace('_', '-'),
            int(numpy.round(height_m_agl))
        )

    figure_object = pyplot.figure(
        figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    taylor_diagram_object = evaluation_plotting.plot_taylor_diagram(
        target_stdev=target_stdev, prediction_stdev=prediction_stdev,
        correlation=correlation, marker_colour=TAYLOR_MARKER_COLOUR,
        figure_object=figure_object
    )

    if aux_target_index is None:
        taylor_diagram_object._ax.axis['left'].label.set_text(
            'Standard deviation ({0:s})'.format(
                TARGET_NAME_TO_UNITS[target_name]
            )
        )

        title_string = 'Taylor diagram for {0:s}'.format(
            TARGET_NAME_TO_VERBOSE[target_name]
        )
    else:
        taylor_diagram_object._ax.axis['left'].label.set_text(
            'Standard deviation for observation ({0:s})'.format(
                TARGET_NAME_TO_UNITS[target_name]
            )
        )

        title_string = (
            'Taylor diagram (prediction = {0:s};\nobservation = {1:s})'
        ).format(
            TARGET_NAME_TO_VERBOSE[target_name],
            TARGET_NAME_TO_VERBOSE[predicted_target_name]
        )

    if vector_target_index is not None:
        title_string += ' at {0:d} m AGL'.format(
            int(numpy.round(height_m_agl))
        )

    figure_object.suptitle(title_string)

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _plot_attributes_diagram(
        evaluation_tables_xarray, line_styles, line_colours,
        line_legend_strings, mean_training_example_dict, output_dir_name,
        scalar_target_index=None, aux_target_index=None,
        vector_target_index=None, height_index=None):
    """Plots attributes diagram for one field.

    In this case, "field" means one scalar variable or one vector variable at
    one height.

    :param evaluation_tables_xarray: See doc for `_plot_score_profile`.
    :param line_styles: Same.
    :param line_colours: Same.
    :param line_legend_strings: Same.
    :param mean_training_example_dict: Dictionary created by
        `normalization.create_mean_example`.
    :param output_dir_name: Name of output directory.  Figure will be saved
        here.
    :param scalar_target_index: See doc for `_plot_taylor_diagram`.
    :param aux_target_index: Same.
    :param vector_target_index: Same.
    :param height_index: Same.
    """

    scalar_target_names_by_file = [
        t.coords[evaluation.SCALAR_FIELD_DIM].values.tolist()
        for t in evaluation_tables_xarray
    ]
    vector_target_names_by_file = [
        t.coords[evaluation.VECTOR_FIELD_DIM].values.tolist()
        for t in evaluation_tables_xarray
    ]
    heights_by_file_m_agl = [
        t.coords[evaluation.HEIGHT_DIM].values for t in evaluation_tables_xarray
    ]
    aux_target_names_by_file = [
        t.coords[evaluation.AUX_TARGET_FIELD_DIM].values.tolist()
        if evaluation.AUX_TARGET_FIELD_DIM in t.coords
        else []
        for t in evaluation_tables_xarray
    ]

    num_files = len(evaluation_tables_xarray)
    scalar_target_index_by_file = [None] * num_files
    vector_target_index_by_file = [None] * num_files
    height_index_by_file = [None] * num_files
    aux_target_index_by_file = [None] * num_files

    t = evaluation_tables_xarray[0]

    if scalar_target_index is not None:
        target_name = (
            t.coords[evaluation.SCALAR_FIELD_DIM].values[scalar_target_index]
        )
        mean_predictions = (
            t[evaluation.SCALAR_RELIABILITY_X_KEY].values[
                scalar_target_index, ...
            ]
        )
        mean_observations = (
            t[evaluation.SCALAR_RELIABILITY_Y_KEY].values[
                scalar_target_index, ...
            ]
        )
        example_counts = (
            t[evaluation.SCALAR_RELIABILITY_COUNT_KEY].values[
                scalar_target_index, ...
            ]
        )
        inverted_mean_observations = (
            t[evaluation.SCALAR_INV_RELIABILITY_X_KEY].values[
                scalar_target_index, ...
            ]
        )
        inverted_example_counts = (
            t[evaluation.SCALAR_INV_RELIABILITY_COUNT_KEY].values[
                scalar_target_index, ...
            ]
        )

        climo_value = mean_training_example_dict[
            example_utils.SCALAR_TARGET_VALS_KEY
        ][0, scalar_target_index]

        output_file_name = '{0:s}/{1:s}_reliability.jpg'.format(
            output_dir_name, target_name.replace('_', '-')
        )

        scalar_target_index_by_file = [
            n.index(target_name) if target_name in n else -1
            for n in scalar_target_names_by_file
        ]

    elif aux_target_index is not None:
        target_name = (
            t.coords[evaluation.AUX_TARGET_FIELD_DIM].values[aux_target_index]
        )
        predicted_target_name = (
            t.coords[evaluation.AUX_PREDICTED_FIELD_DIM].values[
                aux_target_index
            ]
        )
        mean_predictions = (
            t[evaluation.AUX_RELIABILITY_X_KEY].values[aux_target_index, ...]
        )
        mean_observations = (
            t[evaluation.AUX_RELIABILITY_Y_KEY].values[aux_target_index, ...]
        )
        example_counts = (
            t[evaluation.AUX_RELIABILITY_COUNT_KEY].values[
                aux_target_index, ...
            ]
        )
        inverted_mean_observations = (
            t[evaluation.AUX_INV_RELIABILITY_X_KEY].values[
                aux_target_index, ...
            ]
        )
        inverted_example_counts = (
            t[evaluation.AUX_INV_RELIABILITY_COUNT_KEY].values[
                aux_target_index, ...
            ]
        )

        scalar_target_names = (
            t.coords[evaluation.SCALAR_FIELD_DIM].values
        ).tolist()

        mean_scalar_target_matrix = (
            mean_training_example_dict[example_utils.SCALAR_TARGET_VALS_KEY]
        )

        if target_name == evaluation.NET_FLUX_NAME:
            surface_down_flux_index = scalar_target_names.index(
                example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME
            )
            toa_up_flux_index = scalar_target_names.index(
                example_utils.SHORTWAVE_TOA_UP_FLUX_NAME
            )
            climo_value = (
                mean_scalar_target_matrix[0, surface_down_flux_index] -
                mean_scalar_target_matrix[0, toa_up_flux_index]
            )
        else:
            this_index = scalar_target_names.index(target_name)
            climo_value = mean_scalar_target_matrix[0, this_index]

        output_file_name = '{0:s}/aux_{1:s}_reliability.jpg'.format(
            output_dir_name, target_name.replace('_', '-')
        )

        aux_target_index_by_file = [
            n.index(target_name) if target_name in n else -1
            for n in aux_target_names_by_file
        ]
    else:
        target_name = (
            t.coords[evaluation.VECTOR_FIELD_DIM].values[vector_target_index]
        )
        height_m_agl = t.coords[evaluation.HEIGHT_DIM].values[height_index]

        mean_predictions = (
            t[evaluation.VECTOR_RELIABILITY_X_KEY].values[
                height_index, vector_target_index, ...
            ]
        )
        mean_observations = (
            t[evaluation.VECTOR_RELIABILITY_Y_KEY].values[
                height_index, vector_target_index, ...
            ]
        )
        example_counts = (
            t[evaluation.VECTOR_RELIABILITY_COUNT_KEY].values[
                height_index, vector_target_index, ...
            ]
        )
        inverted_mean_observations = (
            t[evaluation.VECTOR_INV_RELIABILITY_X_KEY].values[
                height_index, vector_target_index, ...
            ]
        )
        inverted_example_counts = (
            t[evaluation.VECTOR_INV_RELIABILITY_COUNT_KEY].values[
                height_index, vector_target_index, ...
            ]
        )

        climo_value = mean_training_example_dict[
            example_utils.VECTOR_TARGET_VALS_KEY
        ][0, height_index, vector_target_index]

        output_file_name = '{0:s}/{1:s}_{2:05d}metres_reliability.jpg'.format(
            output_dir_name, target_name.replace('_', '-'),
            int(numpy.round(height_m_agl))
        )

        vector_target_index_by_file = [
            n.index(target_name) if target_name in n else -1
            for n in vector_target_names_by_file
        ]

        for i in range(num_files):
            try:
                height_index_by_file[i] = example_utils.match_heights(
                    heights_m_agl=heights_by_file_m_agl[i],
                    desired_height_m_agl=height_m_agl
                )
            except ValueError:
                vector_target_index_by_file[i] = None

    mean_predictions_by_file = [None] * num_files
    mean_observations_by_file = [None] * num_files
    mean_predictions_by_file[0] = mean_predictions
    mean_observations_by_file[0] = mean_observations

    for i in range(1, num_files):
        t = evaluation_tables_xarray[i]

        if scalar_target_index_by_file[i] is not None:
            k = scalar_target_index_by_file[i]
            mean_predictions_by_file[i] = (
                t[evaluation.SCALAR_RELIABILITY_X_KEY].values[k, ...]
            )
            mean_observations_by_file[i] = (
                t[evaluation.SCALAR_RELIABILITY_Y_KEY].values[k, ...]
            )
        elif aux_target_index_by_file[i] is not None:
            k = aux_target_index_by_file[i]
            mean_predictions_by_file[i] = (
                t[evaluation.AUX_RELIABILITY_X_KEY].values[k, ...]
            )
            mean_observations_by_file[i] = (
                t[evaluation.AUX_RELIABILITY_Y_KEY].values[k, ...]
            )
        elif vector_target_index_by_file[i] is not None:
            j = height_index_by_file[i]
            k = vector_target_index_by_file[i]

            mean_predictions_by_file[i] = (
                t[evaluation.VECTOR_RELIABILITY_X_KEY].values[j, k, ...]
            )
            mean_observations_by_file[i] = (
                t[evaluation.VECTOR_RELIABILITY_Y_KEY].values[j, k, ...]
            )

    these_arrays = [
        a for a in mean_predictions_by_file + mean_observations_by_file
        if a is not None
    ]

    concat_values = numpy.concatenate(these_arrays)
    max_value_to_plot = numpy.nanpercentile(concat_values, 99.9)
    min_value_to_plot = numpy.nanpercentile(concat_values, 0.1)
    min_value_to_plot = numpy.minimum(min_value_to_plot, 0.)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    legend_handles = []
    legend_strings = []

    this_handle = evaluation_plotting.plot_attributes_diagram(
        figure_object=figure_object, axes_object=axes_object,
        mean_predictions=mean_predictions, mean_observations=mean_observations,
        example_counts=example_counts, mean_value_in_training=climo_value,
        min_value_to_plot=min_value_to_plot,
        max_value_to_plot=max_value_to_plot,
        inv_mean_observations=inverted_mean_observations,
        inv_example_counts=inverted_example_counts,
        line_colour=line_colours[0], line_style=line_styles[0], line_width=4
    )

    if this_handle is not None:
        legend_handles.append(this_handle)
        legend_strings.append(line_legend_strings[0])

    axes_object.set_xlabel('Prediction ({0:s})'.format(
        TARGET_NAME_TO_UNITS[target_name]
    ))
    axes_object.set_ylabel('Conditional mean observation ({0:s})'.format(
        TARGET_NAME_TO_UNITS[target_name]
    ))

    if aux_target_index is None:
        title_string = 'Attributes diagram for {0:s}'.format(
            TARGET_NAME_TO_VERBOSE[target_name]
        )
    else:
        title_string = (
            'Attributes diagram (prediction = {0:s};\nobservation = {1:s})'
        ).format(
            TARGET_NAME_TO_VERBOSE[target_name],
            TARGET_NAME_TO_VERBOSE[predicted_target_name]
        )

    if vector_target_index is not None:
        title_string += ' at {0:d} m AGL'.format(
            int(numpy.round(height_m_agl))
        )

    axes_object.set_title(title_string)

    for i in range(1, num_files):
        if mean_predictions_by_file[i] is None:
            continue

        this_handle = evaluation_plotting._plot_reliability_curve(
            axes_object=axes_object,
            mean_predictions=mean_predictions_by_file[i],
            mean_observations=mean_observations_by_file[i],
            min_value_to_plot=min_value_to_plot,
            max_value_to_plot=max_value_to_plot,
            line_colour=line_colours[i], line_style=line_styles[i], line_width=2
        )

        if this_handle is None:
            continue

        legend_handles.append(this_handle)
        legend_strings.append(line_legend_strings[i])

    if len(legend_handles) > 0:
        axes_object.legend(
            legend_handles, legend_strings, loc='center left',
            bbox_to_anchor=(0, 0.5), fancybox=True, shadow=False,
            facecolor='white', edgecolor='k', framealpha=0.5, ncol=1
        )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _plot_score_profile(
        evaluation_tables_xarray, line_styles, line_colours,
        line_legend_strings, target_name, score_name, use_log_scale,
        output_dir_name):
    """Plots vertical profile of one score.

    N = number of models being evaluated

    :param evaluation_tables_xarray: length-N list of evaluation tables (in
        format returned by `evaluation.read_file`).
    :param line_styles: length-N list of line styles.
    :param line_colours: length-N list of line colours.
    :param line_legend_strings: length-N list of line legends.
    :param target_name: Name of target variable for which score is being plotted.
    :param score_name: Name of score being plotted.
    :param use_log_scale: Boolean flag.  If True, will plot heights (y-axis) in
        log scale.
    :param output_dir_name: Name of output directory.  Figure will be saved
        here.
    """

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    for this_table in evaluation_tables_xarray:
        print(this_table)
        print('\n\n\n**********************\n\n\n')

    vector_target_names_by_file = [
        t.coords[evaluation.VECTOR_FIELD_DIM].values.tolist()
        for t in evaluation_tables_xarray
    ]
    heights_by_file_m_agl = [
        t.coords[evaluation.HEIGHT_DIM].values for t in evaluation_tables_xarray
    ]

    score_key = SCORE_NAME_TO_PROFILE_KEY[score_name]
    legend_handles = []
    legend_strings = []
    num_files = len(evaluation_tables_xarray)

    for i in range(num_files):
        try:
            k = vector_target_names_by_file[i].index(target_name)
        except ValueError:
            if i == 0:
                raise
            continue

        this_handle = evaluation_plotting.plot_score_profile(
            heights_m_agl=heights_by_file_m_agl[i],
            score_values=evaluation_tables_xarray[i][score_key].values[:, k],
            score_name=score_name, line_colour=line_colours[i],
            line_width=4, line_style=line_styles[i],
            use_log_scale=use_log_scale, axes_object=axes_object,
            are_axes_new=i == 0
        )

        legend_handles.append(this_handle)
        legend_strings.append(line_legend_strings[i])

    axes_object.legend(
        legend_handles, legend_strings, loc='center left',
        bbox_to_anchor=(0, 0.5), fancybox=True, shadow=False,
        facecolor='white', edgecolor='k', framealpha=0.5, ncol=1
    )

    score_name_verbose = SCORE_NAME_TO_VERBOSE[score_name]
    k = vector_target_names_by_file[0].index(target_name)
    prmse = evaluation_tables_xarray[0][evaluation.VECTOR_PRMSE_KEY].values[k]
    title_string = '{0:s} for {1:s} (PRMSE = {2:.2f} {3:s})'.format(
        score_name_verbose, TARGET_NAME_TO_VERBOSE[target_name],
        prmse, TARGET_NAME_TO_UNITS[target_name]
    )

    x_label_string = '{0:s}'.format(score_name_verbose)

    if score_name in SQUARED_UNIT_SCORE_NAMES:
        x_label_string += ' ({0:s})'.format(
            TARGET_NAME_TO_SQUARED_UNITS[target_name]
        )
    elif score_name in ORIG_UNIT_SCORE_NAMES:
        x_label_string += ' ({0:s})'.format(TARGET_NAME_TO_UNITS[target_name])

    axes_object.set_xlabel(x_label_string)
    axes_object.set_title(title_string)

    this_file_name = '{0:s}/{1:s}_{2:s}_profile.jpg'.format(
        output_dir_name, target_name.replace('_', '-'),
        score_name.replace('_', '-')
    )
    print('Saving figure to: "{0:s}"...'.format(this_file_name))
    figure_object.savefig(
        this_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run(evaluation_file_names, line_styles, line_colour_strings,
         line_legend_strings, use_log_scale, plot_by_height, output_dir_name):
    """Plots model evaluation.

    This is effectively the main method.

    :param evaluation_file_names: See documentation at top of file.
    :param line_styles: Same.
    :param line_colour_strings: Same.
    :param line_legend_strings: Same.
    :param use_log_scale: Same.
    :param plot_by_height: Same.
    :param output_dir_name: Same.
    """

    # Check input args.
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    num_files = len(evaluation_file_names)
    expected_dim = numpy.array([num_files], dtype=int)

    error_checking.assert_is_string_list(line_styles)
    error_checking.assert_is_numpy_array(
        numpy.array(line_styles), exact_dimensions=expected_dim
    )

    error_checking.assert_is_string_list(line_legend_strings)
    error_checking.assert_is_numpy_array(
        numpy.array(line_legend_strings), exact_dimensions=expected_dim
    )
    line_legend_strings = [s.replace('_', ' ') for s in line_legend_strings]

    error_checking.assert_is_string_list(line_colour_strings)
    error_checking.assert_is_numpy_array(
        numpy.array(line_colour_strings), exact_dimensions=expected_dim
    )
    line_colours = [
        numpy.fromstring(s, dtype=float, sep='_') / 255
        for s in line_colour_strings
    ]

    for i in range(num_files):
        error_checking.assert_is_numpy_array(
            line_colours[i], exact_dimensions=numpy.array([3], dtype=int)
        )
        error_checking.assert_is_geq_numpy_array(line_colours[i], 0.)
        error_checking.assert_is_leq_numpy_array(line_colours[i], 1.)

    # Housekeeping.
    evaluation_tables_xarray = [xarray.Dataset()] * num_files

    for i in range(num_files):
        print('Reading data from: "{0:s}"...'.format(evaluation_file_names[i]))
        evaluation_tables_xarray[i] = evaluation.read_file(
            evaluation_file_names[i]
        )

    prediction_file_name = (
        evaluation_tables_xarray[0].attrs[evaluation.PREDICTION_FILE_KEY]
    )
    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(prediction_file_name)

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

    scalar_target_names = evaluation_tables_xarray[0].coords[
        evaluation.SCALAR_FIELD_DIM
    ].values.tolist()

    vector_target_names = evaluation_tables_xarray[0].coords[
        evaluation.VECTOR_FIELD_DIM
    ].values.tolist()

    heights_m_agl = (
        evaluation_tables_xarray[0].coords[evaluation.HEIGHT_DIM].values
    )

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

    # Do actual stuff.
    for k in range(len(vector_target_names)):
        this_target_name_verbose = (
            TARGET_NAME_TO_VERBOSE[vector_target_names[k]]
        )
        this_unit_string = TARGET_NAME_TO_UNITS[vector_target_names[k]]

        # Plot error distribution.
        this_figure_object, this_axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

        this_error_matrix = (
            prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY][..., k] -
            prediction_dict[prediction_io.VECTOR_TARGETS_KEY][..., k]
        )

        evaluation_plotting.plot_error_dist_many_heights(
            error_matrix=this_error_matrix, heights_m_agl=heights_m_agl,
            min_error_to_plot=numpy.percentile(this_error_matrix, 1.),
            max_error_to_plot=numpy.percentile(this_error_matrix, 99.),
            axes_object=this_axes_object
        )

        this_axes_object.set_title(
            'Error distribution for {0:s} ({1:s})'.format(
                this_target_name_verbose, this_unit_string
            )
        )

        this_file_name = '{0:s}/{1:s}_error-dist.jpg'.format(
            output_dir_name, vector_target_names[k].replace('_', '-'),
        )
        print('Saving figure to: "{0:s}"...'.format(this_file_name))

        this_figure_object.savefig(
            this_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(this_figure_object)

        # Plot error profiles.
        for this_score_name in list(SCORE_NAME_TO_PROFILE_KEY.keys()):
            _plot_score_profile(
                evaluation_tables_xarray=evaluation_tables_xarray,
                line_styles=line_styles, line_colours=line_colours,
                line_legend_strings=line_legend_strings,
                target_name=vector_target_names[k],
                score_name=this_score_name, use_log_scale=use_log_scale,
                output_dir_name=output_dir_name
            )

        # Plot reliability curves for all heights in the same figure.
        this_mean_prediction_matrix = numpy.take(
            evaluation_tables_xarray[0][
                evaluation.VECTOR_RELIABILITY_X_KEY
            ].values,
            axis=1, indices=k
        )
        this_mean_target_matrix = numpy.take(
            evaluation_tables_xarray[0][
                evaluation.VECTOR_RELIABILITY_Y_KEY
            ].values,
            axis=1, indices=k
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
        these_target_stdevs = evaluation_tables_xarray[0][
            evaluation.VECTOR_TARGET_STDEV_KEY
        ].values[..., k]

        these_prediction_stdevs = evaluation_tables_xarray[0][
            evaluation.VECTOR_PREDICTION_STDEV_KEY
        ].values[..., k]

        these_correlations = evaluation_tables_xarray[0][
            evaluation.VECTOR_CORRELATION_KEY
        ].values[..., k]

        this_figure_object = pyplot.figure(
            figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

        taylor_diagram_object = (
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
        taylor_diagram_object._ax.axis['left'].label.set_text(
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

    for k in range(len(scalar_target_names)):
        this_figure_object, this_axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

        these_error_values = (
            prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY][..., k] -
            prediction_dict[prediction_io.SCALAR_TARGETS_KEY][..., k]
        )

        evaluation_plotting.plot_error_distribution(
            error_values=these_error_values,
            min_error_to_plot=numpy.percentile(these_error_values, 1.),
            max_error_to_plot=numpy.percentile(these_error_values, 99.),
            axes_object=this_axes_object
        )

        this_axes_object.set_title(
            'Error distribution for {0:s} ({1:s})'.format(
                TARGET_NAME_TO_VERBOSE[scalar_target_names[k]],
                TARGET_NAME_TO_UNITS[scalar_target_names[k]]
            )
        )

        this_file_name = '{0:s}/{1:s}_error-dist.jpg'.format(
            output_dir_name, scalar_target_names[k].replace('_', '-'),
        )
        print('Saving figure to: "{0:s}"...'.format(this_file_name))

        this_figure_object.savefig(
            this_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(this_figure_object)

        _plot_attributes_diagram(
            evaluation_tables_xarray=evaluation_tables_xarray,
            line_styles=line_styles, line_colours=line_colours,
            line_legend_strings=line_legend_strings,
            mean_training_example_dict=mean_training_example_dict,
            output_dir_name=output_dir_name, scalar_target_index=k
        )

        _plot_taylor_diagram(
            evaluation_table_array=evaluation_tables_xarray[0],
            output_dir_name=output_dir_name, scalar_target_index=k
        )

    print(SEPARATOR_STRING)

    try:
        aux_target_field_names = (
            evaluation_tables_xarray[0].coords[
                evaluation.AUX_TARGET_FIELD_DIM
            ].values
        )
    except:
        aux_target_field_names = []

    for k in range(len(aux_target_field_names)):
        _plot_attributes_diagram(
            evaluation_tables_xarray=evaluation_tables_xarray,
            line_styles=line_styles, line_colours=line_colours,
            line_legend_strings=line_legend_strings,
            mean_training_example_dict=mean_training_example_dict,
            output_dir_name=output_dir_name, aux_target_index=k
        )

        _plot_taylor_diagram(
            evaluation_table_array=evaluation_tables_xarray[0],
            output_dir_name=output_dir_name, aux_target_index=k
        )

    if not plot_by_height:
        return

    if len(aux_target_field_names) > 0:
        print(SEPARATOR_STRING)

    for k in range(len(vector_target_names)):
        for j in range(len(heights_m_agl)):
            _plot_attributes_diagram(
                evaluation_tables_xarray=evaluation_tables_xarray,
                line_styles=line_styles, line_colours=line_colours,
                line_legend_strings=line_legend_strings,
                mean_training_example_dict=mean_training_example_dict,
                output_dir_name=output_dir_name,
                vector_target_index=k, height_index=j
            )

            _plot_taylor_diagram(
                evaluation_table_array=evaluation_tables_xarray[0],
                output_dir_name=output_dir_name,
                vector_target_index=k, height_index=j
            )

            if j != len(heights_m_agl) - 1:
                print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        evaluation_file_names=getattr(INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME),
        line_styles=getattr(INPUT_ARG_OBJECT, LINE_STYLES_ARG_NAME),
        line_colour_strings=getattr(INPUT_ARG_OBJECT, LINE_COLOURS_ARG_NAME),
        line_legend_strings=getattr(INPUT_ARG_OBJECT, LINE_LEGENDS_ARG_NAME),
        use_log_scale=bool(getattr(INPUT_ARG_OBJECT, USE_LOG_SCALE_ARG_NAME)),
        plot_by_height=bool(getattr(INPUT_ARG_OBJECT, PLOT_BY_HEIGHT_ARG_NAME)),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
