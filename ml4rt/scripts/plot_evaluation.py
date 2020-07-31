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
    example_io.SHORTWAVE_DOWN_FLUX_INC_NAME:
        r'$\frac{\Delta F_{down}}{\Delta z}$',
    example_io.SHORTWAVE_UP_FLUX_NAME: 'upwelling flux',
    example_io.SHORTWAVE_UP_FLUX_INC_NAME: r'$\frac{\Delta F_{up}}{\Delta z}$',
    example_io.SHORTWAVE_HEATING_RATE_NAME: 'heating rate',
    example_io.SHORTWAVE_SURFACE_DOWN_FLUX_NAME: 'surface downwelling flux',
    example_io.SHORTWAVE_TOA_UP_FLUX_NAME: 'TOA upwelling flux',
    evaluation.NET_FLUX_NAME: 'net flux',
    evaluation.HIGHEST_UP_FLUX_NAME: 'top-of-profile upwelling flux',
    evaluation.LOWEST_DOWN_FLUX_NAME: 'bottom-of-profile downwelling flux'
}

TARGET_NAME_TO_UNITS = {
    example_io.SHORTWAVE_DOWN_FLUX_NAME: r'W m$^{-2}$',
    example_io.SHORTWAVE_DOWN_FLUX_INC_NAME: r'W m$^{-3}$',
    example_io.SHORTWAVE_UP_FLUX_NAME: r'W m$^{-2}$',
    example_io.SHORTWAVE_UP_FLUX_INC_NAME: r'W m$^{-3}$',
    example_io.SHORTWAVE_HEATING_RATE_NAME: r'K day$^{-1}$',
    example_io.SHORTWAVE_SURFACE_DOWN_FLUX_NAME: r'W m$^{-2}$',
    example_io.SHORTWAVE_TOA_UP_FLUX_NAME: r'W m$^{-2}$',
    evaluation.NET_FLUX_NAME: r'W m$^{-2}$',
    evaluation.HIGHEST_UP_FLUX_NAME: r'W m$^{-2}$',
    evaluation.LOWEST_DOWN_FLUX_NAME: r'W m$^{-2}$'
}

TARGET_NAME_TO_SQUARED_UNITS = {
    example_io.SHORTWAVE_DOWN_FLUX_NAME: r'W$^{2}$ m$^{-4}$',
    example_io.SHORTWAVE_DOWN_FLUX_INC_NAME: r'W$^{2}$ m$^{-6}$',
    example_io.SHORTWAVE_UP_FLUX_NAME: r'W$^{2}$ m$^{-4}$',
    example_io.SHORTWAVE_UP_FLUX_INC_NAME: r'W$^{2}$ m$^{-6}$',
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

INPUT_FILE_ARG_NAME = 'main_eval_file_name'
BASELINE_FILE_ARG_NAME = 'baseline_eval_file_name'
USE_LOG_SCALE_ARG_NAME = 'use_log_scale'
PLOT_BY_HEIGHT_ARG_NAME = 'plot_by_height'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file (will be read by `evaluation.write_file`).'
)
BASELINE_FILE_HELP_STRING = (
    'Same as `{0:s}` but containing evaluation results for a baseline model.  '
    'If you do not want to plot a baseline, leave this argument alone.'
)
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
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + BASELINE_FILE_ARG_NAME, type=str, required=False, default='',
    help=BASELINE_FILE_HELP_STRING
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
        main_results_xarray, output_dir_name, scalar_target_index=None,
        aux_target_index=None, vector_target_index=None, height_index=None):
    """Plots Taylor diagram for one field.

    In this case, "field" means one scalar variable or one vector variable at
    one height.

    :param main_results_xarray: xarray table with main evaluation results
        (format described in `evaluation.read_file`).
    :param output_dir_name: Name of output directory (figure will be saved
        here).
    :param scalar_target_index: Index for scalar target variable.
    :param aux_target_index: Index for auxiliary target variable.
    :param vector_target_index: Index for vector target variable.
    :param height_index: Height index for target variable.
    """

    if scalar_target_index is not None:
        target_name = main_results_xarray.coords[
            evaluation.SCALAR_FIELD_DIM
        ].values[scalar_target_index]

        target_stdev = main_results_xarray[
            evaluation.SCALAR_TARGET_STDEV_KEY
        ].values[scalar_target_index]

        prediction_stdev = main_results_xarray[
            evaluation.SCALAR_PREDICTION_STDEV_KEY
        ].values[scalar_target_index]

        correlation = main_results_xarray[
            evaluation.SCALAR_CORRELATION_KEY
        ].values[scalar_target_index]

        output_file_name = '{0:s}/{1:s}_taylor.jpg'.format(
            output_dir_name, target_name.replace('_', '-')
        )

    elif aux_target_index is not None:
        target_name = main_results_xarray.coords[
            evaluation.AUX_TARGET_FIELD_DIM
        ].values[aux_target_index]

        predicted_target_name = main_results_xarray.coords[
            evaluation.AUX_PREDICTED_FIELD_DIM
        ].values[aux_target_index]

        target_stdev = main_results_xarray[
            evaluation.AUX_TARGET_STDEV_KEY
        ].values[aux_target_index]

        prediction_stdev = main_results_xarray[
            evaluation.AUX_PREDICTION_STDEV_KEY
        ].values[aux_target_index]

        correlation = main_results_xarray[
            evaluation.AUX_CORRELATION_KEY
        ].values[aux_target_index]

        output_file_name = '{0:s}/aux_{1:s}_taylor.jpg'.format(
            output_dir_name, target_name.replace('_', '-')
        )

    else:
        target_name = main_results_xarray.coords[
            evaluation.VECTOR_FIELD_DIM
        ].values[vector_target_index]

        height_m_agl = main_results_xarray.coords[
            evaluation.HEIGHT_DIM
        ].values[height_index]

        target_stdev = main_results_xarray[
            evaluation.VECTOR_TARGET_STDEV_KEY
        ].values[height_index, vector_target_index]

        prediction_stdev = main_results_xarray[
            evaluation.VECTOR_PREDICTION_STDEV_KEY
        ].values[height_index, vector_target_index]

        correlation = main_results_xarray[
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
        title_string += ' at {0:d} m AGL'.format(height_m_agl)

    figure_object.suptitle(title_string)

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _plot_attributes_diagram(
        main_results_xarray, baseline_results_xarray, mean_training_example_dict,
        output_dir_name, scalar_target_index=None, aux_target_index=None,
        vector_target_index=None, height_index=None):
    """Plots attributes diagram for one field.

    In this case, "field" means one scalar variable or one vector variable at
    one height.

    :param main_results_xarray: xarray table with main evaluation results
        (format described in `evaluation.read_file`).
    :param baseline_results_xarray: Same but with evaluation results for baseline
        model.
    :param mean_training_example_dict: Dictionary created by
        `normalization.create_mean_example`.
    :param output_dir_name: See doc for `_plot_taylor_diagram`.
    :param scalar_target_index: Same.
    :param aux_target_index: Same.
    :param vector_target_index: Same.
    :param height_index: Same.
    """

    if baseline_results_xarray is None:
        baseline_scalar_target_names = []
        baseline_aux_target_names = []
        baseline_vector_target_names = []
        baseline_heights_m_agl = []
    else:
        baseline_scalar_target_names = (
            baseline_results_xarray.coords[evaluation.SCALAR_FIELD_DIM].values
        ).tolist()

        try:
            baseline_aux_target_names = baseline_results_xarray.coords[
                evaluation.AUX_TARGET_FIELD_DIM
            ].values.tolist()
        except:
            baseline_aux_target_names = []

        baseline_vector_target_names = (
            baseline_results_xarray.coords[evaluation.VECTOR_FIELD_DIM].values
        ).tolist()

        baseline_heights_m_agl = (
            baseline_results_xarray.coords[evaluation.HEIGHT_DIM].values
        )

    baseline_scalar_target_index = None
    baseline_aux_target_index = None
    baseline_vector_target_index = None
    baseline_height_index = None

    if scalar_target_index is not None:
        target_name = main_results_xarray.coords[
            evaluation.SCALAR_FIELD_DIM
        ].values[scalar_target_index]

        mean_predictions = main_results_xarray[
            evaluation.SCALAR_RELIABILITY_X_KEY
        ].values[scalar_target_index, ...]

        mean_observations = main_results_xarray[
            evaluation.SCALAR_RELIABILITY_Y_KEY
        ].values[scalar_target_index, ...]

        example_counts = main_results_xarray[
            evaluation.SCALAR_RELIABILITY_COUNT_KEY
        ].values[scalar_target_index, ...]

        inverted_mean_observations = main_results_xarray[
            evaluation.SCALAR_INV_RELIABILITY_X_KEY
        ].values[scalar_target_index, ...]

        inverted_example_counts = main_results_xarray[
            evaluation.SCALAR_INV_RELIABILITY_COUNT_KEY
        ].values[scalar_target_index, ...]

        climo_value = mean_training_example_dict[
            example_io.SCALAR_TARGET_VALS_KEY
        ][0, scalar_target_index]

        output_file_name = '{0:s}/{1:s}_reliability.jpg'.format(
            output_dir_name, target_name.replace('_', '-')
        )

        if target_name in baseline_scalar_target_names:
            baseline_scalar_target_index = baseline_scalar_target_names.index(
                target_name
            )

    elif aux_target_index is not None:
        target_name = main_results_xarray.coords[
            evaluation.AUX_TARGET_FIELD_DIM
        ].values[aux_target_index]

        predicted_target_name = main_results_xarray.coords[
            evaluation.AUX_PREDICTED_FIELD_DIM
        ].values[aux_target_index]

        mean_predictions = main_results_xarray[
            evaluation.AUX_RELIABILITY_X_KEY
        ].values[aux_target_index, ...]

        mean_observations = main_results_xarray[
            evaluation.AUX_RELIABILITY_Y_KEY
        ].values[aux_target_index, ...]

        example_counts = main_results_xarray[
            evaluation.AUX_RELIABILITY_COUNT_KEY
        ].values[aux_target_index, ...]

        inverted_mean_observations = main_results_xarray[
            evaluation.AUX_INV_RELIABILITY_X_KEY
        ].values[aux_target_index, ...]

        inverted_example_counts = main_results_xarray[
            evaluation.AUX_INV_RELIABILITY_COUNT_KEY
        ].values[aux_target_index, ...]

        scalar_target_names = (
            main_results_xarray.coords[evaluation.SCALAR_FIELD_DIM].values
        ).tolist()

        mean_scalar_target_matrix = (
            mean_training_example_dict[example_io.SCALAR_TARGET_VALS_KEY]
        )

        if target_name == evaluation.NET_FLUX_NAME:
            surface_down_flux_index = scalar_target_names.index(
                example_io.SHORTWAVE_SURFACE_DOWN_FLUX_NAME
            )
            toa_up_flux_index = scalar_target_names.index(
                example_io.SHORTWAVE_TOA_UP_FLUX_NAME
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

        if target_name in baseline_aux_target_names:
            baseline_aux_target_index = baseline_aux_target_names.index(
                target_name
            )

    else:
        target_name = main_results_xarray.coords[
            evaluation.VECTOR_FIELD_DIM
        ].values[vector_target_index]

        height_m_agl = main_results_xarray.coords[
            evaluation.HEIGHT_DIM
        ].values[height_index]

        mean_predictions = main_results_xarray[
            evaluation.VECTOR_RELIABILITY_X_KEY
        ].values[height_index, vector_target_index, ...]

        mean_observations = main_results_xarray[
            evaluation.VECTOR_RELIABILITY_Y_KEY
        ].values[height_index, vector_target_index, ...]

        example_counts = main_results_xarray[
            evaluation.VECTOR_RELIABILITY_COUNT_KEY
        ].values[height_index, vector_target_index, ...]

        inverted_mean_observations = main_results_xarray[
            evaluation.VECTOR_INV_RELIABILITY_X_KEY
        ].values[height_index, vector_target_index, ...]

        inverted_example_counts = main_results_xarray[
            evaluation.VECTOR_INV_RELIABILITY_COUNT_KEY
        ].values[height_index, vector_target_index, ...]

        climo_value = mean_training_example_dict[
            example_io.VECTOR_TARGET_VALS_KEY
        ][0, height_index, vector_target_index]

        output_file_name = '{0:s}/{1:s}_{2:05d}metres_reliability.jpg'.format(
            output_dir_name, target_name.replace('_', '-'),
            int(numpy.round(height_m_agl))
        )

        if target_name in baseline_vector_target_names:
            baseline_vector_target_index = baseline_vector_target_names.index(
                target_name
            )
            baseline_height_index = example_io.match_heights(
                heights_m_agl=baseline_heights_m_agl,
                desired_height_m_agl=height_m_agl
            )

    concat_values = numpy.concatenate((mean_predictions, mean_observations))
    max_value_to_plot = numpy.nanpercentile(concat_values, 99.9)
    min_value_to_plot = numpy.nanpercentile(concat_values, 0.1)
    min_value_to_plot = numpy.minimum(min_value_to_plot, 0.)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    evaluation_plotting.plot_attributes_diagram(
        figure_object=figure_object, axes_object=axes_object,
        mean_predictions=mean_predictions, mean_observations=mean_observations,
        example_counts=example_counts, mean_value_in_training=climo_value,
        min_value_to_plot=min_value_to_plot,
        max_value_to_plot=max_value_to_plot,
        inv_mean_observations=inverted_mean_observations,
        inv_example_counts=inverted_example_counts
    )

    axes_object.set_title('Attributes diagram for {0:s}'.format(
        TARGET_NAME_TO_VERBOSE[target_name]
    ))
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
        title_string += ' at {0:d} m AGL'.format(height_m_agl)

    figure_object.suptitle(title_string)

    mean_predictions = None
    mean_observations = None

    if baseline_scalar_target_index is not None:
        mean_predictions = baseline_results_xarray[
            evaluation.SCALAR_RELIABILITY_X_KEY
        ].values[baseline_scalar_target_index, ...]

        mean_observations = baseline_results_xarray[
            evaluation.SCALAR_RELIABILITY_Y_KEY
        ].values[baseline_scalar_target_index, ...]

    elif baseline_aux_target_index is not None:
        mean_predictions = baseline_results_xarray[
            evaluation.AUX_RELIABILITY_X_KEY
        ].values[baseline_aux_target_index, ...]

        mean_observations = baseline_results_xarray[
            evaluation.AUX_RELIABILITY_Y_KEY
        ].values[baseline_aux_target_index, ...]

    elif baseline_vector_target_index is not None:
        mean_predictions = baseline_results_xarray[
            evaluation.VECTOR_RELIABILITY_X_KEY
        ].values[baseline_height_index, baseline_vector_target_index, ...]

        mean_observations = baseline_results_xarray[
            evaluation.VECTOR_RELIABILITY_Y_KEY
        ].values[baseline_height_index, baseline_vector_target_index, ...]

    if mean_predictions is not None:
        evaluation_plotting._plot_reliability_curve(
            axes_object=axes_object, mean_predictions=mean_predictions,
            mean_observations=mean_observations,
            min_value_to_plot=min_value_to_plot,
            max_value_to_plot=max_value_to_plot,
            line_style='dashed'
        )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run(main_eval_file_name, baseline_eval_file_name, use_log_scale,
         plot_by_height, output_dir_name):
    """Plots model evaluation.

    This is effectively the main method.

    :param main_eval_file_name: See documentation at top of file.
    :param baseline_eval_file_name: Same.
    :param use_log_scale: Same.
    :param plot_by_height: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    print('Reading data from: "{0:s}"...'.format(main_eval_file_name))
    main_results_xarray = evaluation.read_file(main_eval_file_name)

    if baseline_eval_file_name == '':
        baseline_results_xarray = None
    else:
        print('Reading data from: "{0:s}"...'.format(baseline_eval_file_name))
        baseline_results_xarray = evaluation.read_file(baseline_eval_file_name)

    model_file_name = main_results_xarray.attrs[evaluation.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    generator_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]

    scalar_target_names = main_results_xarray.coords[
        evaluation.SCALAR_FIELD_DIM
    ].values.tolist()

    vector_target_names = main_results_xarray.coords[
        evaluation.VECTOR_FIELD_DIM
    ].values.tolist()

    heights_m_agl = main_results_xarray.coords[evaluation.HEIGHT_DIM].values

    example_dict = {
        example_io.SCALAR_TARGET_NAMES_KEY: scalar_target_names,
        example_io.VECTOR_TARGET_NAMES_KEY: vector_target_names,
        example_io.HEIGHTS_KEY: heights_m_agl,
        example_io.SCALAR_PREDICTOR_NAMES_KEY:
            generator_option_dict[neural_net.SCALAR_PREDICTOR_NAMES_KEY],
        example_io.VECTOR_PREDICTOR_NAMES_KEY:
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
    training_example_dict = example_io.subset_by_height(
        example_dict=training_example_dict, heights_m_agl=heights_m_agl
    )
    mean_training_example_dict = normalization.create_mean_example(
        new_example_dict=example_dict,
        training_example_dict=training_example_dict
    )

    if baseline_results_xarray is None:
        baseline_vector_target_names = []
        baseline_heights_m_agl = []
    else:
        baseline_vector_target_names = baseline_results_xarray.coords[
            evaluation.VECTOR_FIELD_DIM
        ].values.tolist()

        baseline_heights_m_agl = (
            baseline_results_xarray.coords[evaluation.HEIGHT_DIM].values
        )

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
                score_values=main_results_xarray[this_key].values[:, k],
                score_name=this_score_name, line_colour=PROFILE_COLOUR,
                line_width=2, line_style='solid', use_log_scale=use_log_scale,
                axes_object=this_axes_object, are_axes_new=True
            )

            if vector_target_names[k] in baseline_vector_target_names:
                this_index = baseline_vector_target_names.index(
                    vector_target_names[k]
                )

                evaluation_plotting.plot_score_profile(
                    heights_m_agl=baseline_heights_m_agl,
                    score_values=
                    baseline_results_xarray[this_key].values[:, this_index],
                    score_name=this_score_name, line_colour=PROFILE_COLOUR,
                    line_width=2, line_style='dashed',
                    use_log_scale=use_log_scale,
                    axes_object=this_axes_object, are_axes_new=False
                )

            this_score_name_verbose = SCORE_NAME_TO_VERBOSE[this_score_name]
            this_prmse = (
                main_results_xarray[evaluation.VECTOR_PRMSE_KEY].values[k]
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
        this_mean_prediction_matrix = numpy.take(
            main_results_xarray[evaluation.VECTOR_RELIABILITY_X_KEY].values,
            axis=1, indices=k
        )
        this_mean_target_matrix = numpy.take(
            main_results_xarray[evaluation.VECTOR_RELIABILITY_Y_KEY].values,
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
        these_target_stdevs = main_results_xarray[
            evaluation.VECTOR_TARGET_STDEV_KEY
        ].values[..., k]

        these_prediction_stdevs = main_results_xarray[
            evaluation.VECTOR_PREDICTION_STDEV_KEY
        ].values[..., k]

        these_correlations = main_results_xarray[
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
        _plot_attributes_diagram(
            main_results_xarray=main_results_xarray,
            baseline_results_xarray=baseline_results_xarray,
            mean_training_example_dict=mean_training_example_dict,
            output_dir_name=output_dir_name, scalar_target_index=k
        )

        _plot_taylor_diagram(
            main_results_xarray=main_results_xarray,
            output_dir_name=output_dir_name, scalar_target_index=k
        )

    print(SEPARATOR_STRING)

    try:
        aux_target_field_names = (
            main_results_xarray.coords[evaluation.AUX_TARGET_FIELD_DIM].values
        )
    except:
        aux_target_field_names = []

    for k in range(len(aux_target_field_names)):
        _plot_attributes_diagram(
            main_results_xarray=main_results_xarray,
            baseline_results_xarray=baseline_results_xarray,
            mean_training_example_dict=mean_training_example_dict,
            output_dir_name=output_dir_name, aux_target_index=k
        )

        _plot_taylor_diagram(
            main_results_xarray=main_results_xarray,
            output_dir_name=output_dir_name, aux_target_index=k
        )

    if not plot_by_height:
        return

    if len(aux_target_field_names) > 0:
        print(SEPARATOR_STRING)

    for k in range(len(vector_target_names)):
        for j in range(len(heights_m_agl)):
            _plot_attributes_diagram(
                main_results_xarray=main_results_xarray,
                baseline_results_xarray=baseline_results_xarray,
                mean_training_example_dict=mean_training_example_dict,
                output_dir_name=output_dir_name,
                vector_target_index=k, height_index=j
            )

            _plot_taylor_diagram(
                main_results_xarray=main_results_xarray,
                output_dir_name=output_dir_name,
                vector_target_index=k, height_index=j
            )

            if j != len(heights_m_agl) - 1:
                print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        main_eval_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        baseline_eval_file_name=getattr(
            INPUT_ARG_OBJECT, BASELINE_FILE_ARG_NAME
        ),
        use_log_scale=bool(getattr(INPUT_ARG_OBJECT, USE_LOG_SCALE_ARG_NAME)),
        plot_by_height=bool(getattr(INPUT_ARG_OBJECT, PLOT_BY_HEIGHT_ARG_NAME)),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
