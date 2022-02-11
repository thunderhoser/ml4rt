"""Plots errors as a function of two metadata variables.

For the desired target variable, this script creates three sets of plots:

- Error metrics as a function of aerosol optical depth (AOD) and solar zenith
  angle (SZA)
- Error metrics as a function of AOD and surface downwelling flux
- Error metrics as a function of SZA and surface downwelling flux
"""

import os
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from scipy.integrate import simps
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils as gg_plotting_utils
from ml4rt.io import example_io
from ml4rt.io import prediction_io
from ml4rt.utils import evaluation
from ml4rt.utils import example_utils
from ml4rt.machine_learning import neural_net

TOLERANCE = 1e-6
RADIANS_TO_DEGREES = 180. / numpy.pi

MIN_ZENITH_ANGLE_RAD = 0.
MAX_ZENITH_ANGLE_RAD = numpy.pi / 2
MAX_SURFACE_DOWN_FLUX_W_M02 = 1200.
MAX_AEROSOL_OPTICAL_DEPTH = 1.8

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

MAIN_COLOUR_MAP_OBJECT = pyplot.get_cmap(name='viridis')
BIAS_COLOUR_MAP_OBJECT = pyplot.get_cmap(name='seismic')
NUM_EXAMPLES_COLOUR_MAP_OBJECT = pyplot.get_cmap(name='plasma')

MAIN_COLOUR_MAP_OBJECT.set_bad(numpy.full(3, 152. / 255))
BIAS_COLOUR_MAP_OBJECT.set_bad(numpy.full(3, 152. / 255))

INPUT_FILE_ARG_NAME = 'input_prediction_file_name'
HEATING_RATE_HEIGHT_ARG_NAME = 'heating_rate_height_m_agl'
NUM_ANGLE_BINS_ARG_NAME = 'num_zenith_angle_bins'
NUM_DOWN_FLUX_BINS_ARG_NAME = 'num_surface_down_flux_bins'
NUM_AOD_BINS_ARG_NAME = 'num_aod_bins'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing predictions and observations.  Will be read'
    ' by `prediction_io.read_file`.'
)
HEATING_RATE_HEIGHT_HELP_STRING = (
    'Will plot error metrics for heating rate at this height (metres above '
    'ground level).  If you want to plot error metrics for net flux instead, '
    'leave this argument alone.'
)
NUM_ANGLE_BINS_HELP_STRING = 'Number of bins for zenith angle.'
NUM_DOWN_FLUX_BINS_HELP_STRING = 'Number of bins for surface downwelling flux.'
NUM_AOD_BINS_HELP_STRING = 'Number of bins for aerosol optical depth (AOD).'
EXAMPLE_DIR_HELP_STRING = (
    'Name of directory with example files.  Aerosol optical depths will be '
    'computed from aerosol-extinction profiles in these files.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + HEATING_RATE_HEIGHT_ARG_NAME, type=int, required=False, default=-1,
    help=HEATING_RATE_HEIGHT_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_ANGLE_BINS_ARG_NAME, type=int, required=False, default=9,
    help=NUM_ANGLE_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_DOWN_FLUX_BINS_ARG_NAME, type=int, required=False, default=12,
    help=NUM_DOWN_FLUX_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_AOD_BINS_ARG_NAME, type=int, required=False, default=9,
    help=NUM_AOD_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _get_aerosol_optical_depths(prediction_dict, example_dir_name):
    """Computes aerosol optical depth (AOD) for each profile.

    :param prediction_dict: Dictionary returned by `prediction_io.read_file`.
    :param example_dir_name: See documentation at top of file.
    :return: aerosol_optical_depths: 1-D numpy array of AOD values, one per
        example.
    """

    valid_times_unix_sec = example_utils.parse_example_ids(
        prediction_dict[prediction_io.EXAMPLE_IDS_KEY]
    )[example_utils.VALID_TIMES_KEY]

    example_file_names = example_io.find_many_files(
        directory_name=example_dir_name,
        first_time_unix_sec=numpy.min(valid_times_unix_sec),
        last_time_unix_sec=numpy.max(valid_times_unix_sec),
        raise_error_if_any_missing=False
    )

    example_id_strings = []
    aerosol_extinction_matrix_metres01 = numpy.array([])
    height_matrix_m_agl = numpy.array([])

    for this_file_name in example_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_example_dict = example_io.read_file(
            netcdf_file_name=this_file_name, exclude_summit_greenland=False,
            max_heating_rate_k_day=numpy.inf
        )

        example_id_strings += this_example_dict[example_utils.EXAMPLE_IDS_KEY]
        this_extinction_matrix_metres01 = example_utils.get_field_from_dict(
            example_dict=this_example_dict,
            field_name=example_utils.AEROSOL_EXTINCTION_NAME
        )

        if aerosol_extinction_matrix_metres01.size == 0:
            aerosol_extinction_matrix_metres01 = (
                this_extinction_matrix_metres01 + 0.
            )
        else:
            aerosol_extinction_matrix_metres01 = numpy.concatenate((
                aerosol_extinction_matrix_metres01,
                this_extinction_matrix_metres01
            ), axis=0)

        if (
                example_utils.HEIGHT_NAME in
                this_example_dict[example_utils.VECTOR_PREDICTOR_NAMES_KEY]
        ):
            this_height_matrix_m_agl = example_utils.get_field_from_dict(
                example_dict=this_example_dict,
                field_name=example_utils.HEIGHT_NAME
            )

            if height_matrix_m_agl.size == 0:
                height_matrix_m_agl = this_height_matrix_m_agl + 0.
            else:
                height_matrix_m_agl = numpy.concatenate(
                    (height_matrix_m_agl, this_height_matrix_m_agl), axis=0
                )
        else:
            if height_matrix_m_agl.size == 0:
                height_matrix_m_agl = (
                    this_example_dict[example_utils.HEIGHTS_KEY] + 0.
                )

    desired_indices = example_utils.find_examples(
        all_id_strings=example_id_strings,
        desired_id_strings=prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
        allow_missing=False
    )
    del example_id_strings

    aerosol_extinction_matrix_metres01 = (
        aerosol_extinction_matrix_metres01[desired_indices, :]
    )

    if len(height_matrix_m_agl.shape) == 2:
        height_matrix_m_agl = height_matrix_m_agl[desired_indices, :]
        num_examples = aerosol_extinction_matrix_metres01.shape[0]
        aerosol_optical_depths = numpy.full(num_examples, numpy.nan)
        print('\n')

        for i in range(num_examples):
            if numpy.mod(i, 1000) == 0:
                print((
                    'Have computed aerosol optical depth for {0:d} of {1:d} '
                    'profiles...'
                ).format(
                    i, num_examples
                ))

            aerosol_optical_depths[i] = simps(
                y=aerosol_extinction_matrix_metres01[i, :],
                x=height_matrix_m_agl[i, :],
                even='avg'
            )

        print((
            'Have computed aerosol optical depth for all {0:d} profiles!\n'
        ).format(
            num_examples
        ))
    else:
        aerosol_optical_depths = simps(
            y=aerosol_extinction_matrix_metres01, x=height_matrix_m_agl,
            axis=-1, even='avg'
        )

    return aerosol_optical_depths


def _plot_scores_2d(
        score_matrix, colour_map_object, colour_norm_object, x_tick_labels,
        y_tick_labels):
    """Plots scores on 2-D grid.

    M = number of rows in grid
    N = number of columns in grid

    :param score_matrix: M-by-N numpy array of scores.
    :param colour_map_object: Colour scheme (instance of
        `matplotlib.pyplot.cm`).
    :param colour_norm_object: Normalizer for colour scheme (instance of
        `matplotlib.pyplot.Normalize` or similar).
    :param x_tick_labels: length-N list of tick labels.
    :param y_tick_labels: length-M list of tick labels.
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    axes_object.imshow(
        score_matrix, cmap=colour_map_object, norm=colour_norm_object,
        origin='lower'
    )

    x_tick_values = numpy.linspace(
        0, score_matrix.shape[1] - 1, num=score_matrix.shape[1], dtype=float
    )
    y_tick_values = numpy.linspace(
        0, score_matrix.shape[0] - 1, num=score_matrix.shape[0], dtype=float
    )
    pyplot.xticks(x_tick_values, x_tick_labels)
    pyplot.yticks(y_tick_values, y_tick_labels)

    colour_bar_object = gg_plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object,
        data_matrix=score_matrix[numpy.invert(numpy.isnan(score_matrix))],
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', extend_min=True, extend_max=True,
        fraction_of_axis_length=0.8
    )

    tick_values = colour_bar_object.get_ticks()
    tick_strings = ['{0:.2g}'.format(v) for v in tick_values]
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    return figure_object, axes_object


def _run(prediction_file_name, heating_rate_height_m_agl, num_zenith_angle_bins,
         num_surface_down_flux_bins, num_aod_bins, example_dir_name,
         output_dir_name):
    """Plots errors as a function of two metadata variables.

    This is effectively the main method.

    :param prediction_file_name: See documentation at top of file.
    :param heating_rate_height_m_agl: Same.
    :param num_zenith_angle_bins: Same.
    :param num_surface_down_flux_bins: Same.
    :param num_aod_bins: Same.
    :param example_dir_name: Same.
    :param output_dir_name: Same.
    """

    if heating_rate_height_m_agl < 0:
        heating_rate_height_m_agl = None

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(prediction_file_name)

    model_file_name = prediction_dict[prediction_io.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    training_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    normalization_file_name = (
        training_option_dict[neural_net.NORMALIZATION_FILE_KEY]
    )

    print((
        'Reading training examples (for climatology) from: "{0:s}"...'
    ).format(
        normalization_file_name
    ))
    training_example_dict = example_io.read_file(normalization_file_name)

    # Bin examples by surface downwelling flux.
    edge_down_fluxes_w_m02 = numpy.linspace(
        0, MAX_SURFACE_DOWN_FLUX_W_M02,
        num=num_surface_down_flux_bins + 1, dtype=float
    )
    edge_down_fluxes_w_m02[-1] = numpy.inf

    k = training_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY].index(
        example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME
    )
    down_flux_bin_indices = numpy.digitize(
        x=prediction_dict[prediction_io.SCALAR_TARGETS_KEY][:, k],
        bins=edge_down_fluxes_w_m02, right=False
    ) - 1

    # Bin examples by solar zenith angle.
    edge_zenith_angles_rad = numpy.linspace(
        MIN_ZENITH_ANGLE_RAD, MAX_ZENITH_ANGLE_RAD,
        num=num_zenith_angle_bins + 1, dtype=float
    )
    edge_zenith_angles_rad[-1] = numpy.inf

    actual_zenith_angles_rad = example_utils.parse_example_ids(
        prediction_dict[prediction_io.EXAMPLE_IDS_KEY]
    )[example_utils.ZENITH_ANGLES_KEY]

    zenith_angle_bin_indices = numpy.digitize(
        x=actual_zenith_angles_rad, bins=edge_zenith_angles_rad, right=False
    ) - 1

    # Bin examples by AOD.
    edge_aerosol_optical_depths = numpy.linspace(
        0, MAX_AEROSOL_OPTICAL_DEPTH,
        num=num_aod_bins + 1, dtype=float
    )
    edge_aerosol_optical_depths[-1] = numpy.inf

    actual_aerosol_optical_depths = _get_aerosol_optical_depths(
        prediction_dict=prediction_dict, example_dir_name=example_dir_name
    )
    aod_bin_indices = numpy.digitize(
        x=actual_aerosol_optical_depths, bins=edge_aerosol_optical_depths,
        right=False
    ) - 1

    # Extracted predicted and observed values of target variable.
    if heating_rate_height_m_agl is None:
        j = training_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY].index(
            example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME
        )
        k = training_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY].index(
            example_utils.SHORTWAVE_TOA_UP_FLUX_NAME
        )

        target_values = (
            prediction_dict[prediction_io.SCALAR_TARGETS_KEY][:, j] -
            prediction_dict[prediction_io.SCALAR_TARGETS_KEY][:, k]
        )
        predicted_values = (
            prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY][:, j] -
            prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY][:, k]
        )

        training_down_fluxes_w_m02 = example_utils.get_field_from_dict(
            example_dict=training_example_dict,
            field_name=example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME
        )
        training_up_fluxes_w_m02 = example_utils.get_field_from_dict(
            example_dict=training_example_dict,
            field_name=example_utils.SHORTWAVE_TOA_UP_FLUX_NAME
        )
        climo_value = numpy.mean(
            training_down_fluxes_w_m02 - training_up_fluxes_w_m02
        )
    else:
        k = training_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY].index(
            example_utils.SHORTWAVE_HEATING_RATE_NAME
        )

        height_diffs_metres = numpy.absolute(
            heating_rate_height_m_agl -
            prediction_dict[prediction_io.HEIGHTS_KEY]
        )
        j = numpy.where(height_diffs_metres <= TOLERANCE)[0][0]

        target_values = (
            prediction_dict[prediction_io.VECTOR_TARGETS_KEY][:, j, k]
        )
        predicted_values = (
            prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY][:, j, k]
        )

        training_values = example_utils.get_field_from_dict(
            example_dict=training_example_dict,
            field_name=example_utils.SHORTWAVE_HEATING_RATE_NAME,
            height_m_agl=heating_rate_height_m_agl
        )
        climo_value = numpy.mean(training_values)

    # TODO(thunderhoser): Modularize this shit!
    dimensions = (num_aod_bins, num_zenith_angle_bins)
    bias_matrix = numpy.full(dimensions, numpy.nan)
    correlation_matrix = numpy.full(dimensions, numpy.nan)
    mae_matrix = numpy.full(dimensions, numpy.nan)
    mae_skill_score_matrix = numpy.full(dimensions, numpy.nan)
    mse_matrix = numpy.full(dimensions, numpy.nan)
    mse_skill_score_matrix = numpy.full(dimensions, numpy.nan)
    kge_matrix = numpy.full(dimensions, numpy.nan)
    num_examples_matrix = numpy.full(dimensions, 0, dtype=int)

    for i in range(num_aod_bins):
        for j in range(num_zenith_angle_bins):
            these_indices = numpy.where(numpy.logical_and(
                aod_bin_indices == i, zenith_angle_bin_indices == j
            ))[0]

            if len(these_indices) == 0:
                continue

            bias_matrix[i, j] = evaluation._get_bias_one_scalar(
                target_values=target_values[these_indices],
                predicted_values=predicted_values[these_indices]
            )
            correlation_matrix[i, j] = evaluation._get_correlation_one_scalar(
                target_values=target_values[these_indices],
                predicted_values=predicted_values[these_indices]
            )
            mae_matrix[i, j] = evaluation._get_mae_one_scalar(
                target_values=target_values[these_indices],
                predicted_values=predicted_values[these_indices]
            )
            mse_matrix[i, j] = evaluation._get_mse_one_scalar(
                target_values=target_values[these_indices],
                predicted_values=predicted_values[these_indices]
            )[0]
            kge_matrix[i, j] = evaluation._get_kge_one_scalar(
                target_values=target_values[these_indices],
                predicted_values=predicted_values[these_indices]
            )
            mae_skill_score_matrix[i, j] = evaluation._get_mae_ss_one_scalar(
                target_values=target_values[these_indices],
                predicted_values=predicted_values[these_indices],
                mean_training_target_value=climo_value
            )
            mse_skill_score_matrix[i, j] = evaluation._get_mse_ss_one_scalar(
                target_values=target_values[these_indices],
                predicted_values=predicted_values[these_indices],
                mean_training_target_value=climo_value
            )
            num_examples_matrix[i, j] = len(these_indices)

    edge_zenith_angles_rad[-1] = MAX_ZENITH_ANGLE_RAD
    edge_zenith_angles_deg = edge_zenith_angles_rad * RADIANS_TO_DEGREES
    x_tick_labels = [
        '[{0:d}, {1:d}]'.format(
            int(numpy.round(a)), int(numpy.round(a))
        ) for a, b in
        zip(edge_zenith_angles_deg[:-1], edge_zenith_angles_deg[1:])
    ]

    edge_aerosol_optical_depths[-1] = MAX_AEROSOL_OPTICAL_DEPTH
    y_tick_labels = [
        '[{0:.1f}, {1:.1f}]'.format(a, b) for a, b in
        zip(edge_aerosol_optical_depths[:-1], edge_aerosol_optical_depths[1:])
    ]

    colour_norm_object = pyplot.Normalize(
        vmin=-1 * numpy.nanpercentile(numpy.absolute(bias_matrix), 99),
        vmax=numpy.nanpercentile(numpy.absolute(bias_matrix), 99)
    )
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=bias_matrix, colour_map_object=BIAS_COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object,
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
    )
    axes_object.set_xlabel('Solar zenith angle (deg)')
    axes_object.set_ylabel('Aerosol optical depth (unitless)')
    axes_object.set_title('Bias')

    output_file_name = '{0:s}/sza_aod_bias.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    colour_norm_object = pyplot.Normalize(
        vmin=numpy.nanpercentile(correlation_matrix, 1),
        vmax=numpy.nanpercentile(correlation_matrix, 99)
    )
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=correlation_matrix,
        colour_map_object=MAIN_COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object,
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
    )
    axes_object.set_xlabel('Solar zenith angle (deg)')
    axes_object.set_ylabel('Aerosol optical depth (unitless)')
    axes_object.set_title('Correlation')

    output_file_name = '{0:s}/sza_aod_correlation.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    colour_norm_object = pyplot.Normalize(
        vmin=numpy.nanpercentile(mae_matrix, 1),
        vmax=numpy.nanpercentile(mae_matrix, 99)
    )
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=mae_matrix, colour_map_object=MAIN_COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object,
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
    )
    axes_object.set_xlabel('Solar zenith angle (deg)')
    axes_object.set_ylabel('Aerosol optical depth (unitless)')
    axes_object.set_title('Mean absolute error')

    output_file_name = '{0:s}/sza_aod_mae.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    colour_norm_object = pyplot.Normalize(
        vmin=numpy.nanpercentile(mse_matrix, 1),
        vmax=numpy.nanpercentile(mse_matrix, 99)
    )
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=mse_matrix, colour_map_object=MAIN_COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object,
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
    )
    axes_object.set_xlabel('Solar zenith angle (deg)')
    axes_object.set_ylabel('Aerosol optical depth (unitless)')
    axes_object.set_title('Mean squared error')

    output_file_name = '{0:s}/sza_aod_mse.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    colour_norm_object = pyplot.Normalize(
        vmin=numpy.nanpercentile(kge_matrix, 1),
        vmax=numpy.nanpercentile(kge_matrix, 99)
    )
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=kge_matrix, colour_map_object=MAIN_COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object,
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
    )
    axes_object.set_xlabel('Solar zenith angle (deg)')
    axes_object.set_ylabel('Aerosol optical depth (unitless)')
    axes_object.set_title('Kling-Gupta efficiency')

    output_file_name = '{0:s}/sza_aod_kge.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    colour_norm_object = pyplot.Normalize(
        vmin=numpy.nanpercentile(mae_skill_score_matrix, 1),
        vmax=numpy.nanpercentile(mae_skill_score_matrix, 99)
    )
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=mae_skill_score_matrix,
        colour_map_object=MAIN_COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object,
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
    )
    axes_object.set_xlabel('Solar zenith angle (deg)')
    axes_object.set_ylabel('Aerosol optical depth (unitless)')
    axes_object.set_title('MAE skill score')

    output_file_name = '{0:s}/sza_aod_maess.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    colour_norm_object = pyplot.Normalize(
        vmin=numpy.nanpercentile(mse_skill_score_matrix, 1),
        vmax=numpy.nanpercentile(mse_skill_score_matrix, 99)
    )
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=mse_skill_score_matrix,
        colour_map_object=MAIN_COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object,
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
    )
    axes_object.set_xlabel('Solar zenith angle (deg)')
    axes_object.set_ylabel('Aerosol optical depth (unitless)')
    axes_object.set_title('MSE skill score')

    output_file_name = '{0:s}/sza_aod_msess.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    colour_norm_object = pyplot.Normalize(
        vmin=numpy.nanpercentile(num_examples_matrix, 1),
        vmax=numpy.nanpercentile(num_examples_matrix, 99)
    )
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=num_examples_matrix,
        colour_map_object=NUM_EXAMPLES_COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object,
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
    )
    axes_object.set_xlabel('Solar zenith angle (deg)')
    axes_object.set_ylabel('Aerosol optical depth (unitless)')
    axes_object.set_title('Number of examples')

    output_file_name = '{0:s}/sza_aod_num-examples.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        heating_rate_height_m_agl=getattr(
            INPUT_ARG_OBJECT, HEATING_RATE_HEIGHT_ARG_NAME
        ),
        num_zenith_angle_bins=getattr(
            INPUT_ARG_OBJECT, NUM_ANGLE_BINS_ARG_NAME
        ),
        num_surface_down_flux_bins=getattr(
            INPUT_ARG_OBJECT, NUM_DOWN_FLUX_BINS_ARG_NAME
        ),
        num_aod_bins=getattr(INPUT_ARG_OBJECT, NUM_AOD_BINS_ARG_NAME),
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
