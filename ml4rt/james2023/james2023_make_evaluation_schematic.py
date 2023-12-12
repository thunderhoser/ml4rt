"""Makes schematic to explain evaluation methods."""

import os
import argparse
import numpy
import xarray
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import imagemagick_utils
from ml4rt.utils import evaluation
from ml4rt.utils import uq_evaluation
from ml4rt.utils import example_utils
from ml4rt.utils import pit_utils
from ml4rt.utils import spread_skill_utils as ss_utils
from ml4rt.utils import discard_test_utils as dt_utils
from ml4rt.plotting import evaluation_plotting as eval_plotting
from ml4rt.plotting import uq_evaluation_plotting as uq_eval_plotting

SAMPLE_SIZE = int(1e6)
ENSEMBLE_SIZE = 50

MIN_HEATING_RATE_K_DAY01 = 0.
MAX_HEATING_RATE_K_DAY01 = 41.
NUM_BINS_FOR_ATTR_DIAG = 41
NUM_SPREAD_BINS = 100
NUM_PIT_HISTOGRAM_BINS = 20

CONVERT_EXE_NAME = '/usr/bin/convert'
TITLE_FONT_SIZE = 250
TITLE_FONT_NAME = 'DejaVu-Sans-Bold'

SCATTERPLOT_MEAN_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
SCATTERPLOT_MEMBER_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255
REFERENCE_LINE_COLOUR = numpy.full(3, 0.)

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

PANEL_SIZE_PX = int(2.5e6)
CONCAT_FIGURE_SIZE_PX = int(1e7)

OUTPUT_DIR_ARG_NAME = 'output_dir_name'
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _create_data_good_model():
    """Creates data (predictions and targets) for good model.

    E = number of examples
    S = ensemble size

    :return: actual_heating_rates_k_day01: length-E numpy array of actual
        heating rates.
    :return: predicted_hr_matrix_k_day01: E-by-S numpy array of predicted
        heating rates.
    """

    mean_predicted_heating_rates_k_day01 = numpy.random.gamma(
        shape=4, scale=4, size=SAMPLE_SIZE
    )
    mean_predicted_heating_rates_k_day01 = numpy.maximum(
        mean_predicted_heating_rates_k_day01, MIN_HEATING_RATE_K_DAY01
    )
    mean_predicted_heating_rates_k_day01 = numpy.minimum(
        mean_predicted_heating_rates_k_day01, MAX_HEATING_RATE_K_DAY01
    )

    actual_heating_rates_k_day01 = (
        mean_predicted_heating_rates_k_day01 +
        numpy.random.normal(loc=0., scale=2., size=SAMPLE_SIZE)
    )
    actual_heating_rates_k_day01 = numpy.maximum(
        actual_heating_rates_k_day01, MIN_HEATING_RATE_K_DAY01
    )
    actual_heating_rates_k_day01 = numpy.minimum(
        actual_heating_rates_k_day01, MAX_HEATING_RATE_K_DAY01
    )

    predicted_hr_matrix_k_day01 = numpy.full(
        (SAMPLE_SIZE, ENSEMBLE_SIZE), numpy.nan
    )

    for i in range(SAMPLE_SIZE):
        if numpy.mod(i, 10000) == 0:
            print('Have created ensemble for {0:d} of {1:d} examples...'.format(
                i, SAMPLE_SIZE
            ))

        this_error_k_day01 = (
            mean_predicted_heating_rates_k_day01[i] -
            actual_heating_rates_k_day01[i]
        )

        predicted_hr_matrix_k_day01[i, :] = (
            mean_predicted_heating_rates_k_day01[i] +
            numpy.random.uniform(
                low=-2 * this_error_k_day01, high=0, size=ENSEMBLE_SIZE
            )
        )

    print('Have created ensemble for all {0:d} examples!'.format(SAMPLE_SIZE))
    predicted_hr_matrix_k_day01 = numpy.maximum(
        predicted_hr_matrix_k_day01, MIN_HEATING_RATE_K_DAY01
    )
    predicted_hr_matrix_k_day01 = numpy.minimum(
        predicted_hr_matrix_k_day01, MAX_HEATING_RATE_K_DAY01
    )

    stdev_predicted_heating_rates_k_day01 = numpy.std(
        predicted_hr_matrix_k_day01, axis=1, ddof=1
    )
    good_indices = numpy.where(stdev_predicted_heating_rates_k_day01 <= 5)[0]

    predicted_hr_matrix_k_day01 = predicted_hr_matrix_k_day01[good_indices, :]
    actual_heating_rates_k_day01 = actual_heating_rates_k_day01[good_indices]

    return actual_heating_rates_k_day01, predicted_hr_matrix_k_day01


def _create_data_poor_model():
    """Creates data (predictions and targets) for poor model.

    :return: actual_heating_rates_k_day01: Same.
    :return: predicted_hr_matrix_k_day01: Same.
    """

    mean_predicted_heating_rates_k_day01 = numpy.random.gamma(
        shape=4, scale=4, size=SAMPLE_SIZE
    )
    mean_predicted_heating_rates_k_day01 = numpy.maximum(
        mean_predicted_heating_rates_k_day01, MIN_HEATING_RATE_K_DAY01
    )
    mean_predicted_heating_rates_k_day01 = numpy.minimum(
        mean_predicted_heating_rates_k_day01, MAX_HEATING_RATE_K_DAY01
    )

    sigmoid_denoms = 1. + numpy.exp(
        0.25 *
        (MAX_HEATING_RATE_K_DAY01 / 2 - mean_predicted_heating_rates_k_day01)
    )
    actual_heating_rates_k_day01 = MAX_HEATING_RATE_K_DAY01 / sigmoid_denoms
    actual_heating_rates_k_day01 += numpy.random.normal(
        loc=0., scale=2., size=SAMPLE_SIZE
    )
    actual_heating_rates_k_day01 = numpy.maximum(
        actual_heating_rates_k_day01, MIN_HEATING_RATE_K_DAY01
    )
    actual_heating_rates_k_day01 = numpy.minimum(
        actual_heating_rates_k_day01, MAX_HEATING_RATE_K_DAY01
    )

    predicted_hr_matrix_k_day01 = numpy.vstack([
        mp + numpy.random.normal(
            loc=0., scale=max([0.1 * mp, 0.01]), size=ENSEMBLE_SIZE
        )
        for mp in mean_predicted_heating_rates_k_day01
    ])
    predicted_hr_matrix_k_day01 = numpy.maximum(
        predicted_hr_matrix_k_day01, MIN_HEATING_RATE_K_DAY01
    )
    predicted_hr_matrix_k_day01 = numpy.minimum(
        predicted_hr_matrix_k_day01, MAX_HEATING_RATE_K_DAY01
    )

    return actual_heating_rates_k_day01, predicted_hr_matrix_k_day01


def _overlay_text(
        image_file_name, x_offset_from_left_px, y_offset_from_top_px,
        text_string):
    """Creates two figures showing overall evaluation of uncertainty quant (UQ).

    :param image_file_name: Path to image file.
    :param x_offset_from_left_px: Left-relative x-coordinate (pixels).
    :param y_offset_from_top_px: Top-relative y-coordinate (pixels).
    :param text_string: String to overlay.
    :raises: ValueError: if ImageMagick command (which is ultimately a Unix
        command) fails.
    """

    command_string = (
        '"{0:s}" "{1:s}" -pointsize {2:d} -font "{3:s}" '
        '-fill "rgb(0, 0, 0)" -annotate {4:+d}{5:+d} "{6:s}" "{1:s}"'
    ).format(
        CONVERT_EXE_NAME, image_file_name, TITLE_FONT_SIZE, TITLE_FONT_NAME,
        x_offset_from_left_px, y_offset_from_top_px, text_string
    )

    exit_code = os.system(command_string)
    if exit_code == 0:
        return

    raise ValueError(imagemagick_utils.ERROR_STRING)


def _make_scatterplot_1model(output_dir_name, for_good_model, panel_letter):
    """Creates scatterplot for one model.

    :param output_dir_name: Name of output directory.  Figure will be saved
        here.
    :param for_good_model: Boolean flag.
    :param panel_letter: Letter used to label panel.
    :return: output_file_name: Full path to image file where figure was saved.
    """

    if for_good_model:
        actual_heating_rates_k_day01, predicted_hr_matrix_k_day01 = (
            _create_data_good_model()
        )
    else:
        actual_heating_rates_k_day01, predicted_hr_matrix_k_day01 = (
            _create_data_poor_model()
        )

    mean_predicted_heating_rates_k_day01 = numpy.mean(
        predicted_hr_matrix_k_day01, axis=1
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    legend_handles = [None] * 2

    for j in range(0, ENSEMBLE_SIZE, 5):
        legend_handles[1] = axes_object.plot(
            predicted_hr_matrix_k_day01[::10, j],
            actual_heating_rates_k_day01[::10],
            linestyle='None', marker='o', markersize=2.5, markeredgewidth=0,
            markerfacecolor=SCATTERPLOT_MEMBER_COLOUR,
            markeredgecolor=SCATTERPLOT_MEMBER_COLOUR
        )[0]

    legend_handles[0] = axes_object.plot(
        mean_predicted_heating_rates_k_day01[::10],
        actual_heating_rates_k_day01[::10],
        linestyle='None', marker='o', markersize=5, markeredgewidth=0,
        markerfacecolor=SCATTERPLOT_MEAN_COLOUR,
        markeredgecolor=SCATTERPLOT_MEAN_COLOUR
    )[0]

    axes_object.plot(
        [MIN_HEATING_RATE_K_DAY01, MAX_HEATING_RATE_K_DAY01],
        [MIN_HEATING_RATE_K_DAY01, MAX_HEATING_RATE_K_DAY01],
        linestyle='dashed', color=REFERENCE_LINE_COLOUR, linewidth=4
    )

    legend_strings = ['Ensemble mean', 'Ensemble member']
    the_one_legend_handle = axes_object.legend(
        legend_handles, legend_strings, loc='upper left',
        bbox_to_anchor=(0, 0.99), fancybox=True, shadow=False,
        facecolor='white', edgecolor='k', framealpha=1., ncol=1,
        fontsize=36
    )
    the_one_legend_handle.legendHandles[0]._legmarker.set_markersize(12)
    the_one_legend_handle.legendHandles[1]._legmarker.set_markersize(12)

    axes_object.set_xlim(MIN_HEATING_RATE_K_DAY01, MAX_HEATING_RATE_K_DAY01)
    axes_object.set_ylim(MIN_HEATING_RATE_K_DAY01, MAX_HEATING_RATE_K_DAY01)

    axes_object.set_xlabel(r'Prediction (K day$^{-1}$)')
    axes_object.set_ylabel(r'Observation (K day$^{-1}$)')
    axes_object.set_title('Scatterplot for Model {0:s}'.format(
        'A' if for_good_model else 'B'
    ))

    output_file_name = '{0:s}/scatterplot_{1:s}_model.jpg'.format(
        output_dir_name,
        'good' if for_good_model else 'poor'
    )
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name,
        output_file_name=output_file_name,
        border_width_pixels=TITLE_FONT_SIZE + 75
    )
    _overlay_text(
        image_file_name=output_file_name,
        x_offset_from_left_px=TITLE_FONT_SIZE + 50,
        y_offset_from_top_px=TITLE_FONT_SIZE + 200,
        text_string='({0:s})'.format(panel_letter)
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name,
        output_file_name=output_file_name
    )

    return output_file_name


def _plot_attr_diagram_1model(output_dir_name, for_good_model, panel_letter):
    """Plots attributes diagram for one model.

    :param output_dir_name: Name of output directory.  Figure will be saved
        here.
    :param for_good_model: Boolean flag.
    :param panel_letter: Letter used to label panel.
    :return: output_file_name: Full path to image file where figure was saved.
    """

    if for_good_model:
        actual_heating_rates_k_day01, predicted_heating_rates_k_day01 = (
            _create_data_good_model()
        )
    else:
        actual_heating_rates_k_day01, predicted_heating_rates_k_day01 = (
            _create_data_poor_model()
        )

    predicted_heating_rates_k_day01 = numpy.mean(
        predicted_heating_rates_k_day01, axis=1
    )

    mean_predictions, mean_observations, example_counts = (
        evaluation._get_rel_curve_one_scalar(
            target_values=actual_heating_rates_k_day01,
            predicted_values=predicted_heating_rates_k_day01,
            num_bins=NUM_BINS_FOR_ATTR_DIAG,
            min_bin_edge=MIN_HEATING_RATE_K_DAY01,
            max_bin_edge=MAX_HEATING_RATE_K_DAY01,
            invert=False
        )
    )

    squared_diffs = (mean_predictions - mean_observations) ** 2
    reliability_k2_day02 = (
        numpy.nansum(example_counts * squared_diffs) /
        numpy.sum(example_counts)
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    eval_plotting.plot_attributes_diagram(
        figure_object=figure_object, axes_object=axes_object,
        mean_predictions=mean_predictions, mean_observations=mean_observations,
        mean_value_in_training=numpy.mean(actual_heating_rates_k_day01),
        min_value_to_plot=MIN_HEATING_RATE_K_DAY01,
        max_value_to_plot=MAX_HEATING_RATE_K_DAY01
    )
    eval_plotting.plot_inset_histogram(
        figure_object=figure_object,
        bin_centers=mean_predictions,
        bin_counts=example_counts,
        has_predictions=True
    )

    _, _, example_counts = evaluation._get_rel_curve_one_scalar(
        target_values=actual_heating_rates_k_day01,
        predicted_values=predicted_heating_rates_k_day01,
        num_bins=NUM_BINS_FOR_ATTR_DIAG,
        min_bin_edge=MIN_HEATING_RATE_K_DAY01,
        max_bin_edge=MAX_HEATING_RATE_K_DAY01,
        invert=True
    )
    eval_plotting.plot_inset_histogram(
        figure_object=figure_object,
        bin_centers=mean_predictions,
        bin_counts=example_counts,
        has_predictions=False
    )

    axes_object.set_xlabel('Prediction (K day$^{-1}$)')
    axes_object.set_ylabel(r'Conditional mean observation (K day$^{-1}$)')

    title_string = 'Attributes diagram for Model {0:s}\nREL = {1:.2f}'.format(
        'A' if for_good_model else 'B',
        reliability_k2_day02
    )
    title_string += r' K$^2$ day$^{-2}$'
    axes_object.set_title(title_string)

    output_file_name = '{0:s}/attributes_diagram_{1:s}_model.jpg'.format(
        output_dir_name,
        'good' if for_good_model else 'poor'
    )
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name,
        output_file_name=output_file_name,
        border_width_pixels=TITLE_FONT_SIZE + 75
    )
    _overlay_text(
        image_file_name=output_file_name,
        x_offset_from_left_px=TITLE_FONT_SIZE + 50,
        y_offset_from_top_px=TITLE_FONT_SIZE + 200,
        text_string='({0:s})'.format(panel_letter)
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name,
        output_file_name=output_file_name
    )

    return output_file_name


def _plot_spread_vs_skill_1model(output_dir_name, for_good_model, panel_letter):
    """Creates spread-skill plot for one model.

    :param output_dir_name: Name of output directory.  Figure will be saved
        here.
    :param for_good_model: Boolean flag.
    :param panel_letter: Letter used to label panel.
    :return: output_file_name: Full path to image file where figure was saved.
    """

    if for_good_model:
        actual_heating_rates_k_day01, predicted_hr_matrix_k_day01 = (
            _create_data_good_model()
        )
    else:
        actual_heating_rates_k_day01, predicted_hr_matrix_k_day01 = (
            _create_data_poor_model()
        )

    these_bin_edges = numpy.linspace(
        0, 5, num=NUM_SPREAD_BINS + 1, dtype=float
    )[1:-1]

    result_dict = ss_utils.get_results_one_var(
        target_values=actual_heating_rates_k_day01,
        prediction_matrix=predicted_hr_matrix_k_day01,
        bin_edge_prediction_stdevs=these_bin_edges
    )

    these_dim_keys_no_bins = (
        ss_utils.VECTOR_FIELD_DIM, ss_utils.WAVELENGTH_DIM
    )
    these_dim_keys_no_edge = (
        ss_utils.VECTOR_FIELD_DIM, ss_utils.WAVELENGTH_DIM,
        ss_utils.HEATING_RATE_BIN_DIM
    )
    these_dim_keys_with_edge = (
        ss_utils.VECTOR_FIELD_DIM, ss_utils.WAVELENGTH_DIM,
        ss_utils.HEATING_RATE_BIN_EDGE_DIM
    )
    rdict = result_dict

    for this_key in [
            ss_utils.MEAN_PREDICTION_STDEVS_KEY,
            ss_utils.BIN_EDGE_PREDICTION_STDEVS_KEY,
            ss_utils.RMSE_VALUES_KEY,
            ss_utils.EXAMPLE_COUNTS_KEY,
            ss_utils.MEAN_MEAN_PREDICTIONS_KEY,
            ss_utils.MEAN_TARGET_VALUES_KEY
    ]:
        rdict[this_key] = numpy.expand_dims(rdict[this_key], axis=0)
        rdict[this_key] = numpy.expand_dims(rdict[this_key], axis=0)

    for this_key in [
            ss_utils.SPREAD_SKILL_RELIABILITY_KEY,
            ss_utils.SPREAD_SKILL_RATIO_KEY
    ]:
        rdict[this_key] = numpy.expand_dims(
            numpy.array([rdict[this_key]]), axis=0
        )

    main_data_dict = {
        ss_utils.VECTOR_FLAT_MEAN_STDEV_KEY: (
            these_dim_keys_no_edge,
            rdict[ss_utils.MEAN_PREDICTION_STDEVS_KEY]
        ),
        ss_utils.VECTOR_FLAT_BIN_EDGE_KEY: (
            these_dim_keys_with_edge,
            rdict[ss_utils.BIN_EDGE_PREDICTION_STDEVS_KEY]
        ),
        ss_utils.VECTOR_FLAT_RMSE_KEY: (
            these_dim_keys_no_edge,
            rdict[ss_utils.RMSE_VALUES_KEY]
        ),
        ss_utils.VECTOR_FLAT_SSREL_KEY: (
            these_dim_keys_no_bins,
            rdict[ss_utils.SPREAD_SKILL_RELIABILITY_KEY]
        ),
        ss_utils.VECTOR_FLAT_SSRAT_KEY: (
            these_dim_keys_no_bins,
            rdict[ss_utils.SPREAD_SKILL_RATIO_KEY]
        ),
        ss_utils.VECTOR_FLAT_EXAMPLE_COUNT_KEY: (
            these_dim_keys_no_edge,
            rdict[ss_utils.EXAMPLE_COUNTS_KEY]
        ),
        ss_utils.VECTOR_FLAT_MEAN_MEAN_PREDICTION_KEY: (
            these_dim_keys_no_edge,
            rdict[ss_utils.MEAN_MEAN_PREDICTIONS_KEY]
        ),
        ss_utils.VECTOR_FLAT_MEAN_TARGET_KEY: (
            these_dim_keys_no_edge,
            rdict[ss_utils.MEAN_TARGET_VALUES_KEY]
        )
    }

    bin_indices = numpy.linspace(
        0, NUM_SPREAD_BINS - 1, num=NUM_SPREAD_BINS, dtype=int
    )
    metadata_dict = {
        ss_utils.SCALAR_FIELD_DIM: [],
        ss_utils.HEIGHT_DIM: numpy.array([1.]),
        ss_utils.WAVELENGTH_DIM:
            numpy.array([example_utils.DUMMY_BROADBAND_WAVELENGTH_METRES]),
        ss_utils.VECTOR_FIELD_DIM: [example_utils.SHORTWAVE_HEATING_RATE_NAME],
        ss_utils.HEATING_RATE_BIN_DIM: bin_indices
    }
    result_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict
    )

    figure_object, axes_object = uq_eval_plotting.plot_spread_vs_skill(
        result_table_xarray=result_table_xarray,
        target_var_name=example_utils.SHORTWAVE_HEATING_RATE_NAME,
        target_height_m_agl=None,
        target_wavelength_metres=example_utils.DUMMY_BROADBAND_WAVELENGTH_METRES
    )

    title_string = 'Spread vs. skill for Model {0:s}\nSSREL = {1:.3f}'.format(
        'A' if for_good_model else 'B',
        result_dict[ss_utils.SPREAD_SKILL_RELIABILITY_KEY][0, 0]
    )
    title_string += r' K day$^{-1}$'
    title_string += '; SSRAT = {0:.3f}'.format(
        result_dict[ss_utils.SPREAD_SKILL_RATIO_KEY][0, 0]
    )
    axes_object.set_title(title_string)

    output_file_name = '{0:s}/spread_skill_plot_{1:s}_model.jpg'.format(
        output_dir_name,
        'good' if for_good_model else 'poor'
    )
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name,
        output_file_name=output_file_name,
        border_width_pixels=TITLE_FONT_SIZE + 75
    )
    _overlay_text(
        image_file_name=output_file_name,
        x_offset_from_left_px=TITLE_FONT_SIZE + 50,
        y_offset_from_top_px=TITLE_FONT_SIZE + 200,
        text_string='({0:s})'.format(panel_letter)
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name,
        output_file_name=output_file_name
    )

    return output_file_name


def _plot_discard_test_1model(output_dir_name, for_good_model, panel_letter):
    """Plots discard test for one model.

    :param output_dir_name: Name of output directory.  Figure will be saved
        here.
    :param for_good_model: Boolean flag.
    :param panel_letter: Letter used to label panel.
    :return: output_file_name: Full path to image file where figure was saved.
    """

    if for_good_model:
        actual_heating_rates_k_day01, predicted_hr_matrix_k_day01 = (
            _create_data_good_model()
        )
    else:
        actual_heating_rates_k_day01, predicted_hr_matrix_k_day01 = (
            _create_data_poor_model()
        )

    uncertainty_values = numpy.std(predicted_hr_matrix_k_day01, axis=1, ddof=1)
    uncertainty_thresholds = numpy.percentile(
        uncertainty_values,
        numpy.linspace(5, 100, num=20, dtype=float)[::-1]
    )
    num_thresholds = len(uncertainty_thresholds)

    these_dim_keys_2d = (dt_utils.VECTOR_FIELD_DIM, dt_utils.WAVELENGTH_DIM)
    these_dim_keys_3d = (
        dt_utils.VECTOR_FIELD_DIM, dt_utils.WAVELENGTH_DIM,
        dt_utils.UNCERTAINTY_THRESHOLD_DIM
    )
    these_dim_2d = (1, 1)
    these_dim_3d = (1, 1, num_thresholds)

    main_data_dict = {
        dt_utils.POST_DISCARD_ERROR_KEY: (
            (dt_utils.UNCERTAINTY_THRESHOLD_DIM,),
            numpy.full(num_thresholds, numpy.nan)
        ),
        dt_utils.EXAMPLE_FRACTION_KEY: (
            (dt_utils.UNCERTAINTY_THRESHOLD_DIM,),
            numpy.full(num_thresholds, numpy.nan)
        ),
        dt_utils.VECTOR_FLAT_POST_DISCARD_ERROR_KEY: (
            these_dim_keys_3d, numpy.full(these_dim_3d, numpy.nan)
        ),
        dt_utils.VECTOR_FLAT_MEAN_MEAN_PREDICTION_KEY: (
            these_dim_keys_3d, numpy.full(these_dim_3d, numpy.nan)
        ),
        dt_utils.VECTOR_FLAT_MEAN_TARGET_KEY: (
            these_dim_keys_3d, numpy.full(these_dim_3d, numpy.nan)
        ),
        dt_utils.VECTOR_FLAT_MONO_FRACTION_KEY: (
            these_dim_keys_2d, numpy.full(these_dim_2d, numpy.nan)
        ),
        dt_utils.VECTOR_FLAT_MEAN_DI_KEY: (
            these_dim_keys_2d, numpy.full(these_dim_2d, numpy.nan)
        )
    }

    metadata_dict = {
        dt_utils.SCALAR_FIELD_DIM: [],
        dt_utils.HEIGHT_DIM: numpy.array([1.]),
        ss_utils.WAVELENGTH_DIM:
            numpy.array([example_utils.DUMMY_BROADBAND_WAVELENGTH_METRES]),
        dt_utils.VECTOR_FIELD_DIM: [example_utils.SHORTWAVE_HEATING_RATE_NAME],
        dt_utils.UNCERTAINTY_THRESHOLD_DIM: uncertainty_thresholds
    }
    result_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict
    )

    use_example_flags = numpy.full(len(uncertainty_values), True, dtype=bool)
    error_function = uq_evaluation.make_error_function_dwmse_1height()

    for i in range(len(uncertainty_thresholds)):
        this_inverted_mask = uncertainty_values > uncertainty_thresholds[i]
        use_example_flags[this_inverted_mask] = False
        rtx = result_table_xarray

        rtx[dt_utils.EXAMPLE_FRACTION_KEY].values[i] = numpy.mean(
            use_example_flags
        )
        rtx[dt_utils.POST_DISCARD_ERROR_KEY].values[i] = error_function(
            numpy.expand_dims(actual_heating_rates_k_day01, axis=-1),
            numpy.expand_dims(predicted_hr_matrix_k_day01, axis=-2),
            use_example_flags
        )

        this_mean_pred_by_example = numpy.mean(
            predicted_hr_matrix_k_day01[use_example_flags, :], axis=-1
        )
        rtx[dt_utils.VECTOR_FLAT_MEAN_MEAN_PREDICTION_KEY].values[0, 0, i] = (
            numpy.mean(this_mean_pred_by_example)
        )
        rtx[dt_utils.VECTOR_FLAT_MEAN_TARGET_KEY].values[0, 0, i] = numpy.mean(
            actual_heating_rates_k_day01[use_example_flags]
        )
        rtx[dt_utils.VECTOR_FLAT_POST_DISCARD_ERROR_KEY].values[0, 0, i] = (
            error_function(
                numpy.expand_dims(actual_heating_rates_k_day01, axis=-1),
                numpy.expand_dims(predicted_hr_matrix_k_day01, axis=-2),
                use_example_flags
            )
        )
        result_table_xarray = rtx

    rtx = result_table_xarray

    discard_fractions = 1. - rtx[dt_utils.EXAMPLE_FRACTION_KEY].values
    rtx[dt_utils.VECTOR_FLAT_MONO_FRACTION_KEY].values[0, 0] = numpy.mean(
        numpy.diff(
            rtx[dt_utils.VECTOR_FLAT_POST_DISCARD_ERROR_KEY].values[0, 0, :]
        ) < 0
    )
    rtx[dt_utils.VECTOR_FLAT_MEAN_DI_KEY].values[0, 0] = numpy.mean(
        -1 * numpy.diff(
            rtx[dt_utils.VECTOR_FLAT_POST_DISCARD_ERROR_KEY].values[0, 0, :]
        )
        / numpy.diff(discard_fractions)
    )

    result_table_xarray = rtx

    figure_object, axes_object = uq_eval_plotting.plot_discard_test(
        result_table_xarray=result_table_xarray,
        target_var_name=example_utils.SHORTWAVE_HEATING_RATE_NAME,
        target_height_m_agl=None,
        target_wavelength_metres=example_utils.DUMMY_BROADBAND_WAVELENGTH_METRES
    )

    title_string = 'Discard test for Model {0:s}\nMF = {1:.1f}%'.format(
        'A' if for_good_model else 'B',
        100 * rtx[dt_utils.VECTOR_FLAT_MONO_FRACTION_KEY].values[0, 0]
    )
    axes_object.set_title(title_string)

    output_file_name = '{0:s}/discard_test_{1:s}_model.jpg'.format(
        'good' if for_good_model else 'poor',
        output_dir_name
    )
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name,
        output_file_name=output_file_name,
        border_width_pixels=TITLE_FONT_SIZE + 75
    )
    _overlay_text(
        image_file_name=output_file_name,
        x_offset_from_left_px=TITLE_FONT_SIZE + 50,
        y_offset_from_top_px=TITLE_FONT_SIZE + 200,
        text_string='({0:s})'.format(panel_letter)
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name,
        output_file_name=output_file_name
    )

    return output_file_name


def _plot_pit_histogram_1model(output_dir_name, for_good_model, panel_letter):
    """Plots PIT histogram for one model.

    :param output_dir_name: Name of output directory.  Figure will be saved
        here.
    :param for_good_model: Boolean flag.
    :param panel_letter: Letter used to label panel.
    :return: output_file_name: Full path to image file where figure was saved.
    """

    if for_good_model:
        actual_heating_rates_k_day01, predicted_hr_matrix_k_day01 = (
            _create_data_good_model()
        )
    else:
        actual_heating_rates_k_day01, predicted_hr_matrix_k_day01 = (
            _create_data_poor_model()
        )

    result_dict = pit_utils._get_histogram_one_var(
        target_values=actual_heating_rates_k_day01,
        prediction_matrix=predicted_hr_matrix_k_day01,
        num_bins=NUM_PIT_HISTOGRAM_BINS
    )
    rdict = result_dict

    for this_key in [pit_utils.BIN_COUNTS_KEY]:
        rdict[this_key] = numpy.expand_dims(rdict[this_key], axis=0)
        rdict[this_key] = numpy.expand_dims(rdict[this_key], axis=0)

    for this_key in [
            pit_utils.PITD_KEY,
            pit_utils.PERFECT_PITD_KEY,
            pit_utils.LOW_BIN_BIAS_KEY,
            pit_utils.MIDDLE_BIN_BIAS_KEY,
            pit_utils.HIGH_BIN_BIAS_KEY,
            pit_utils.EXTREME_PIT_FREQ_KEY
    ]:
        rdict[this_key] = numpy.expand_dims(
            numpy.array([rdict[this_key]]), axis=0
        )

    these_dim_keys_2d = (pit_utils.VECTOR_FIELD_DIM, pit_utils.WAVELENGTH_DIM)
    these_dim_keys_3d = (
        pit_utils.VECTOR_FIELD_DIM, pit_utils.WAVELENGTH_DIM,
        pit_utils.BIN_CENTER_DIM
    )

    main_data_dict = {
        pit_utils.VECTOR_FLAT_PITD_KEY: (
            these_dim_keys_2d, rdict[pit_utils.PITD_KEY]
        ),
        pit_utils.VECTOR_FLAT_PERFECT_PITD_KEY: (
            these_dim_keys_2d, rdict[pit_utils.PERFECT_PITD_KEY]
        ),
        pit_utils.VECTOR_FLAT_BIN_COUNT_KEY: (
            these_dim_keys_3d, rdict[pit_utils.BIN_COUNTS_KEY]
        ),
        pit_utils.VECTOR_FLAT_LOW_BIN_BIAS_KEY: (
            these_dim_keys_2d, rdict[pit_utils.LOW_BIN_BIAS_KEY]
        ),
        pit_utils.VECTOR_FLAT_MIDDLE_BIN_BIAS_KEY: (
            these_dim_keys_2d, rdict[pit_utils.MIDDLE_BIN_BIAS_KEY]
        ),
        pit_utils.VECTOR_FLAT_HIGH_BIN_BIAS_KEY: (
            these_dim_keys_2d, rdict[pit_utils.HIGH_BIN_BIAS_KEY]
        ),
        pit_utils.VECTOR_FLAT_EXTREME_PIT_FREQ_KEY: (
            these_dim_keys_2d, rdict[pit_utils.EXTREME_PIT_FREQ_KEY]
        )
    }

    bin_edges = numpy.linspace(
        0, 1, num=NUM_PIT_HISTOGRAM_BINS + 1, dtype=float
    )
    bin_centers = bin_edges[:-1] + numpy.diff(bin_edges) / 2

    metadata_dict = {
        pit_utils.SCALAR_FIELD_DIM: [],
        pit_utils.HEIGHT_DIM: numpy.array([1.]),
        pit_utils.WAVELENGTH_DIM:
            numpy.array([example_utils.DUMMY_BROADBAND_WAVELENGTH_METRES]),
        pit_utils.VECTOR_FIELD_DIM: [example_utils.SHORTWAVE_HEATING_RATE_NAME],
        pit_utils.BIN_CENTER_DIM: bin_centers,
        pit_utils.BIN_EDGE_DIM: bin_edges
    }
    result_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict
    )

    figure_object, axes_object = uq_eval_plotting.plot_pit_histogram(
        result_table_xarray=result_table_xarray,
        target_var_name=example_utils.SHORTWAVE_HEATING_RATE_NAME,
        target_height_m_agl=None,
        target_wavelength_metres=example_utils.DUMMY_BROADBAND_WAVELENGTH_METRES
    )

    title_string = 'PIT histogram for Model {0:s}\nPITD = {1:.4f}'.format(
        'A' if for_good_model else 'B',
        result_dict[pit_utils.PITD_KEY][0, 0]
    )
    axes_object.set_title(title_string)

    output_file_name = '{0:s}/pit_histogram_{1:s}_model.jpg'.format(
        'good' if for_good_model else 'poor',
        output_dir_name
    )
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name,
        output_file_name=output_file_name,
        border_width_pixels=TITLE_FONT_SIZE + 75
    )
    _overlay_text(
        image_file_name=output_file_name,
        x_offset_from_left_px=TITLE_FONT_SIZE + 50,
        y_offset_from_top_px=TITLE_FONT_SIZE + 200,
        text_string='({0:s})'.format(panel_letter)
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name,
        output_file_name=output_file_name
    )

    return output_file_name


def _run(output_dir_name):
    """Makes schematic to explain evaluation methods.

    This is effectively the main method.

    :param output_dir_name: See documentation at top of file.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    panel_file_names = []
    panel_file_names.append(_make_scatterplot_1model(
        output_dir_name=output_dir_name, for_good_model=True, panel_letter='a'
    ))
    panel_file_names.append(_make_scatterplot_1model(
        output_dir_name=output_dir_name, for_good_model=False, panel_letter='b'
    ))
    panel_file_names.append(_plot_attr_diagram_1model(
        output_dir_name=output_dir_name, for_good_model=True, panel_letter='c'
    ))
    panel_file_names.append(_plot_attr_diagram_1model(
        output_dir_name=output_dir_name, for_good_model=False, panel_letter='d'
    ))
    panel_file_names.append(_plot_spread_vs_skill_1model(
        output_dir_name=output_dir_name, for_good_model=True, panel_letter='e'
    ))
    panel_file_names.append(_plot_spread_vs_skill_1model(
        output_dir_name=output_dir_name, for_good_model=False, panel_letter='f'
    ))
    panel_file_names.append(_plot_discard_test_1model(
        output_dir_name=output_dir_name, for_good_model=True, panel_letter='g'
    ))
    panel_file_names.append(_plot_discard_test_1model(
        output_dir_name=output_dir_name, for_good_model=False, panel_letter='h'
    ))
    panel_file_names.append(_plot_pit_histogram_1model(
        output_dir_name=output_dir_name, for_good_model=True, panel_letter='i'
    ))
    panel_file_names.append(_plot_pit_histogram_1model(
        output_dir_name=output_dir_name, for_good_model=False, panel_letter='j'
    ))

    for this_file_name in panel_file_names:
        imagemagick_utils.resize_image(
            input_file_name=this_file_name, output_file_name=this_file_name,
            output_size_pixels=PANEL_SIZE_PX
        )

    concat_figure_file_name = '{0:s}/evaluation_schematic.jpg'.format(
        output_dir_name
    )

    print('Concatenating panels to: "{0:s}"...'.format(concat_figure_file_name))
    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=concat_figure_file_name,
        num_panel_rows=4, num_panel_columns=3
    )
    imagemagick_utils.resize_image(
        input_file_name=concat_figure_file_name,
        output_file_name=concat_figure_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
