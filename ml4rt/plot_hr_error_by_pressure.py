"""Plots heating-rate errors as a function of pressure."""

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
import example_io
import example_utils
import evaluation
import evaluation_plotting as eval_plotting

TOLERANCE = 1e-6
PASCALS_TO_MB = 0.01

MARKER_SIZE = 2
MARKER_TYPE = 'o'
MARKER_FACE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
MARKER_EDGE_COLOUR = MARKER_FACE_COLOUR
PROFILE_COLOUR = MARKER_FACE_COLOUR
REFERENCE_LINE_COLOUR = numpy.full(3, 152. / 255)

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

FONT_SIZE = 30
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

PREDICTION_FILE_ARG_NAME = 'input_prediction_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
SCATTERPLOT_HEIGHT_ARG_NAME = 'scatterplot_height_m_agl'
NUM_BINS_FOR_PROFILES_ARG_NAME = 'num_bins_for_profiles'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

PREDICTION_FILE_HELP_STRING = (
    'Path to file with predicted and actual target values.  Will be read by '
    '`prediction_io.read_file`.'
)
EXAMPLE_DIR_HELP_STRING = (
    'Name of directory with learning examples.  Files therein will be found by '
    '`example_io.find_file` and read by `example_io.read_file`.  Only pressure '
    'will be read from these files.'
)
SCATTERPLOT_HEIGHT_HELP_STRING = (
    'Will create scatterplot of heating-rate errors at this height (metres '
    'above ground level).'
)
NUM_BINS_FOR_PROFILES_HELP_STRING = (
    'Number of pressure bins for error profiles.'
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
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SCATTERPLOT_HEIGHT_ARG_NAME, type=int, required=True,
    help=SCATTERPLOT_HEIGHT_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_BINS_FOR_PROFILES_ARG_NAME, type=int, required=True,
    help=NUM_BINS_FOR_PROFILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(prediction_file_name, example_dir_name, scatterplot_height_m_agl,
         num_bins_for_profiles, output_dir_name):
    """Plots heating-rate error as a fcn of pressure, using scatter plot.

    This is effectively the main method.

    :param prediction_file_name: See documentation at top of file.
    :param example_dir_name: Same.
    :param scatterplot_height_m_agl: Same.
    :param num_bins_for_profiles: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(prediction_file_name)
    example_id_strings = prediction_dict[prediction_io.EXAMPLE_IDS_KEY]
    height_diffs_metres = numpy.absolute(
        prediction_dict[prediction_io.HEIGHTS_KEY] - scatterplot_height_m_agl
    )
    height_index = numpy.argmin(height_diffs_metres)

    assert numpy.min(height_diffs_metres) <= TOLERANCE
    assert prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY].shape[-1] == 1

    heating_rate_errors_k_day01 = (
        prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY][
            :, height_index, 0
        ]
        - prediction_dict[prediction_io.VECTOR_TARGETS_KEY][:, height_index, 0]
    )

    valid_times_unix_sec = example_utils.parse_example_ids(example_id_strings)[
        example_utils.VALID_TIMES_KEY
    ]
    example_file_names = example_io.find_many_files(
        directory_name=example_dir_name,
        first_time_unix_sec=numpy.min(valid_times_unix_sec),
        last_time_unix_sec=numpy.max(valid_times_unix_sec),
        raise_error_if_any_missing=False
    )

    all_example_id_strings = []
    all_pressure_matrix_pa = None
    all_pressures_pa = numpy.array([])
    all_surface_pressures_pa = numpy.array([])

    for this_file_name in example_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_example_dict = example_io.read_file(
            netcdf_file_name=this_file_name, exclude_summit_greenland=False,
            max_heating_rate_k_day=numpy.inf
        )
        this_example_dict = example_utils.subset_by_field(
            example_dict=this_example_dict,
            field_names=[example_utils.PRESSURE_NAME]
        )

        all_example_id_strings += (
            this_example_dict[example_utils.EXAMPLE_IDS_KEY]
        )
        this_pressure_matrix_pa = example_utils.get_field_from_dict(
            example_dict=this_example_dict,
            field_name=example_utils.PRESSURE_NAME
        )

        if all_pressure_matrix_pa is None:
            all_pressure_matrix_pa = this_pressure_matrix_pa + 0.
        else:
            all_pressure_matrix_pa = numpy.concatenate(
                (all_pressure_matrix_pa, this_pressure_matrix_pa), axis=0
            )

        these_pressures_pa = example_utils.get_field_from_dict(
            example_dict=this_example_dict,
            field_name=example_utils.PRESSURE_NAME,
            height_m_agl=scatterplot_height_m_agl
        )
        these_surface_pressures_pa = example_utils.get_field_from_dict(
            example_dict=this_example_dict,
            field_name=example_utils.PRESSURE_NAME
        )[:, 0]

        all_pressures_pa = numpy.concatenate(
            (all_pressures_pa, these_pressures_pa), axis=0
        )
        all_surface_pressures_pa = numpy.concatenate(
            (all_surface_pressures_pa, these_surface_pressures_pa), axis=0
        )

    desired_indices = example_utils.find_examples(
        all_id_strings=all_example_id_strings,
        desired_id_strings=example_id_strings, allow_missing=False
    )
    pressure_matrix_pa = all_pressure_matrix_pa[desired_indices, :]
    pressures_pa = all_pressures_pa[desired_indices]
    surface_pressures_pa = all_surface_pressures_pa[desired_indices]

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    axes_object.scatter(
        x=pressures_pa * PASCALS_TO_MB, y=heating_rate_errors_k_day01,
        s=MARKER_SIZE, c=MARKER_FACE_COLOUR, marker=MARKER_TYPE,
        edgecolors=MARKER_EDGE_COLOUR
    )
    axes_object.plot(
        axes_object.get_xlim(), numpy.full(2, 0.),
        color=REFERENCE_LINE_COLOUR, linestyle='dashed', linewidth=3
    )
    axes_object.set_title(
        'Heating-rate error at {0:d} m AGL'.format(scatterplot_height_m_agl)
    )
    axes_object.set_xlabel('Pressure (mb) at {0:d} m AGL'.format(
        scatterplot_height_m_agl
    ))
    axes_object.set_ylabel(r'Heating-rate error (K day$^{-1}$)')

    output_file_name = '{0:s}/scatter_plot_by_layer_pressure.jpg'.format(
        output_dir_name
    )
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    axes_object.scatter(
        x=surface_pressures_pa * PASCALS_TO_MB, y=heating_rate_errors_k_day01,
        s=MARKER_SIZE, c=MARKER_FACE_COLOUR, marker=MARKER_TYPE,
        edgecolors=MARKER_EDGE_COLOUR
    )
    axes_object.plot(
        axes_object.get_xlim(), numpy.full(2, 0.),
        color=REFERENCE_LINE_COLOUR, linestyle='dashed', linewidth=3
    )
    axes_object.set_title(
        'Heating-rate error at {0:d} m AGL'.format(scatterplot_height_m_agl)
    )
    axes_object.set_xlabel('Surface pressure (mb)')
    axes_object.set_ylabel(r'Heating-rate error (K day$^{-1}$)')

    output_file_name = '{0:s}/scatter_plot_by_surface_pressure.jpg'.format(
        output_dir_name
    )
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    percentile_levels = numpy.linspace(
        0, 100, num=num_bins_for_profiles + 1, dtype=float
    )
    bin_edges_pa = numpy.percentile(pressure_matrix_pa, percentile_levels)
    bin_centers_pa = (bin_edges_pa[:-1] + bin_edges_pa[1:]) / 2

    bias_profile_k_day01 = numpy.full(num_bins_for_profiles, numpy.nan)
    correlation_profile = numpy.full(num_bins_for_profiles, numpy.nan)
    mae_profile_k_day01 = numpy.full(num_bins_for_profiles, numpy.nan)
    mae_skill_score_profile = numpy.full(num_bins_for_profiles, numpy.nan)
    mse_profile_k2_day02 = numpy.full(num_bins_for_profiles, numpy.nan)
    mse_skill_score_profile = numpy.full(num_bins_for_profiles, numpy.nan)
    kge_profile = numpy.full(num_bins_for_profiles, numpy.nan)

    target_matrix_k_day01 = (
        prediction_dict[prediction_io.VECTOR_TARGETS_KEY][..., 0]
    )
    prediction_matrix_k_day01 = (
        prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY][..., 0]
    )

    num_examples_per_bin = int(1e10)

    for k in range(num_bins_for_profiles):
        if numpy.mod(k, 10) == 0:
            print((
                'Have computed heating-rate errors (first pass) for {0:d} of '
                '{1:d} pressure bins...'
            ).format(
                k, num_bins_for_profiles
            ))

        if k == num_bins_for_profiles - 1:
            these_i, these_j = numpy.where(
                pressure_matrix_pa >= bin_edges_pa[k]
            )
        else:
            these_i, these_j = numpy.where(numpy.logical_and(
                pressure_matrix_pa >= bin_edges_pa[k],
                pressure_matrix_pa < bin_edges_pa[k + 1]
            ))

        num_examples_per_bin = min([
            num_examples_per_bin, len(these_i)
        ])

        bias_profile_k_day01[k] = evaluation._get_bias_one_scalar(
            target_values=target_matrix_k_day01[these_i, these_j],
            predicted_values=prediction_matrix_k_day01[these_i, these_j]
        )
        correlation_profile[k] = evaluation._get_correlation_one_scalar(
            target_values=target_matrix_k_day01[these_i, these_j],
            predicted_values=prediction_matrix_k_day01[these_i, these_j]
        )
        mae_profile_k_day01[k] = evaluation._get_mae_one_scalar(
            target_values=target_matrix_k_day01[these_i, these_j],
            predicted_values=prediction_matrix_k_day01[these_i, these_j]
        )
        mse_profile_k2_day02[k] = evaluation._get_mse_one_scalar(
            target_values=target_matrix_k_day01[these_i, these_j],
            predicted_values=prediction_matrix_k_day01[these_i, these_j]
        )[0]
        kge_profile[k] = evaluation._get_kge_one_scalar(
            target_values=target_matrix_k_day01[these_i, these_j],
            predicted_values=prediction_matrix_k_day01[these_i, these_j]
        )

        # TODO(thunderhoser): These are not true skill scores, because the climo
        # value comes from the evaluation data, not the training data.  I am
        # just being lazy for now.
        mae_skill_score_profile[k] = evaluation._get_mae_ss_one_scalar(
            target_values=target_matrix_k_day01[these_i, these_j],
            predicted_values=prediction_matrix_k_day01[these_i, these_j],
            mean_training_target_value=
            numpy.mean(target_matrix_k_day01[these_i, these_j])
        )
        mse_skill_score_profile[k] = evaluation._get_mse_ss_one_scalar(
            target_values=target_matrix_k_day01[these_i, these_j],
            predicted_values=prediction_matrix_k_day01[these_i, these_j],
            mean_training_target_value=
            numpy.mean(target_matrix_k_day01[these_i, these_j])
        )

    error_matrix_k_day01 = numpy.full(
        (num_examples_per_bin, num_bins_for_profiles), numpy.nan
    )
    print('\n')

    for k in range(num_bins_for_profiles):
        if numpy.mod(k, 10) == 0:
            print((
                'Have computed heating-rate errors (second pass) for {0:d} of '
                '{1:d} pressure bins...'
            ).format(
                k, num_bins_for_profiles
            ))

        if k == num_bins_for_profiles - 1:
            these_i, these_j = numpy.where(
                pressure_matrix_pa >= bin_edges_pa[k]
            )
        else:
            these_i, these_j = numpy.where(numpy.logical_and(
                pressure_matrix_pa >= bin_edges_pa[k],
                pressure_matrix_pa < bin_edges_pa[k + 1]
            ))

        these_errors_k_day01 = (
            prediction_matrix_k_day01[these_i, these_j] -
            target_matrix_k_day01[these_i, these_j]
        )
        error_matrix_k_day01[:, k] = these_errors_k_day01[:num_examples_per_bin]

    print((
        'Have computed heating-rate errors for all {0:d} pressure bins!\n'
    ).format(num_bins_for_profiles))

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    eval_plotting.plot_score_profile_by_pressure(
        pressures_pa=bin_centers_pa, score_values=bias_profile_k_day01,
        score_name=eval_plotting.BIAS_NAME, line_colour=PROFILE_COLOUR,
        line_width=2, axes_object=axes_object, are_axes_new=True
    )
    axes_object.set_title('Heating-rate bias vs. pressure')
    axes_object.set_xlabel(r'Bias (K day$^{-1}$)')

    figure_file_name = '{0:s}/bias_profile.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    eval_plotting.plot_score_profile_by_pressure(
        pressures_pa=bin_centers_pa, score_values=correlation_profile,
        score_name=eval_plotting.CORRELATION_NAME, line_colour=PROFILE_COLOUR,
        line_width=2, axes_object=axes_object, are_axes_new=True
    )
    axes_object.set_title('Heating-rate correlation vs. pressure')
    axes_object.set_xlabel('Correlation')

    figure_file_name = '{0:s}/correlation_profile.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    eval_plotting.plot_score_profile_by_pressure(
        pressures_pa=bin_centers_pa, score_values=mae_profile_k_day01,
        score_name=eval_plotting.MAE_NAME, line_colour=PROFILE_COLOUR,
        line_width=2, axes_object=axes_object, are_axes_new=True
    )
    axes_object.set_title('Heating-rate MAE vs. pressure')
    axes_object.set_xlabel(r'Mean absolute error (K day$^{-1}$)')

    figure_file_name = '{0:s}/mae_profile.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    eval_plotting.plot_score_profile_by_pressure(
        pressures_pa=bin_centers_pa, score_values=mse_profile_k2_day02,
        score_name=eval_plotting.MSE_NAME, line_colour=PROFILE_COLOUR,
        line_width=2, axes_object=axes_object, are_axes_new=True
    )
    axes_object.set_title('Heating-rate MSE vs. pressure')
    axes_object.set_xlabel(r'Mean squared error (K$^{2}$ day$^{-2}$)')

    figure_file_name = '{0:s}/mse_profile.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    eval_plotting.plot_score_profile_by_pressure(
        pressures_pa=bin_centers_pa, score_values=kge_profile,
        score_name=eval_plotting.KGE_NAME, line_colour=PROFILE_COLOUR,
        line_width=2, axes_object=axes_object, are_axes_new=True
    )
    axes_object.set_title('Heating-rate KGE vs. pressure')
    axes_object.set_xlabel('Kling-Gupta efficiency')

    figure_file_name = '{0:s}/kge_profile.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    eval_plotting.plot_score_profile_by_pressure(
        pressures_pa=bin_centers_pa, score_values=mae_skill_score_profile,
        score_name=eval_plotting.MAE_SKILL_SCORE_NAME,
        line_colour=PROFILE_COLOUR, line_width=2,
        axes_object=axes_object, are_axes_new=True
    )
    axes_object.set_title('Heating-rate MAESS vs. pressure')
    axes_object.set_xlabel('Mean absolute error (MAE) skill score')

    figure_file_name = '{0:s}/mae_skill_score_profile.jpg'.format(
        output_dir_name
    )
    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    eval_plotting.plot_score_profile_by_pressure(
        pressures_pa=bin_centers_pa, score_values=mse_skill_score_profile,
        score_name=eval_plotting.MSE_SKILL_SCORE_NAME,
        line_colour=PROFILE_COLOUR, line_width=2,
        axes_object=axes_object, are_axes_new=True
    )
    axes_object.set_title('Heating-rate MSESS vs. pressure')
    axes_object.set_xlabel('Mean squared error (MSE) skill score')

    figure_file_name = '{0:s}/mse_skill_score_profile.jpg'.format(
        output_dir_name
    )
    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    eval_plotting.plot_error_dist_many_pressures(
        error_matrix=error_matrix_k_day01, pressures_pa=bin_centers_pa,
        min_error_to_plot=numpy.percentile(error_matrix_k_day01, 1.),
        max_error_to_plot=numpy.percentile(error_matrix_k_day01, 99.),
        axes_object=axes_object
    )
    axes_object.set_title(r'Error distribution for heating rate (K day$^{-1}$')

    figure_file_name = '{0:s}/boxplot_profile.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_FILE_ARG_NAME
        ),
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        scatterplot_height_m_agl=getattr(
            INPUT_ARG_OBJECT, SCATTERPLOT_HEIGHT_ARG_NAME
        ),
        num_bins_for_profiles=getattr(
            INPUT_ARG_OBJECT, NUM_BINS_FOR_PROFILES_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
