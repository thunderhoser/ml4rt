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
from ml4rt.utils import example_utils
from ml4rt.utils import pit_utils
from ml4rt.utils import spread_skill_utils as ss_utils
from ml4rt.plotting import uq_evaluation_plotting as uq_eval_plotting

SAMPLE_SIZE = int(1e6)
ENSEMBLE_SIZE = 50

MIN_HEATING_RATE_K_DAY01 = 0.
MAX_HEATING_RATE_K_DAY01 = 41.
NUM_SPREAD_BINS = 51
NUM_PIT_HISTOGRAM_BINS = 51

CONVERT_EXE_NAME = 'convert'
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

    actual_heating_rates_k_day01 = numpy.random.gamma(
        shape=4, scale=4, size=SAMPLE_SIZE
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

        these_predictions = numpy.random.normal(
            loc=actual_heating_rates_k_day01[i],
            scale=0.1 * actual_heating_rates_k_day01[i],
            size=ENSEMBLE_SIZE + 1
        )
        numpy.random.shuffle(these_predictions)
        actual_heating_rates_k_day01[i] = these_predictions[0]
        predicted_hr_matrix_k_day01[i, :] = these_predictions[1:]

    print('Have created ensemble for all {0:d} examples!'.format(SAMPLE_SIZE))
    # predicted_hr_matrix_k_day01 = numpy.maximum(
    #     predicted_hr_matrix_k_day01, MIN_HEATING_RATE_K_DAY01
    # )
    # predicted_hr_matrix_k_day01 = numpy.minimum(
    #     predicted_hr_matrix_k_day01, MAX_HEATING_RATE_K_DAY01
    # )

    # stdev_predicted_heating_rates_k_day01 = numpy.std(
    #     predicted_hr_matrix_k_day01, axis=1, ddof=1
    # )
    # good_indices = numpy.where(stdev_predicted_heating_rates_k_day01 <= 5)[0]
    #
    # predicted_hr_matrix_k_day01 = predicted_hr_matrix_k_day01[good_indices, :]
    # actual_heating_rates_k_day01 = actual_heating_rates_k_day01[good_indices]

    return actual_heating_rates_k_day01, predicted_hr_matrix_k_day01


def _create_data_poor_model():
    """Creates data (predictions and targets) for poor model.

    :return: actual_heating_rates_k_day01: Same.
    :return: predicted_hr_matrix_k_day01: Same.
    """

    actual_heating_rates_k_day01 = numpy.random.gamma(
        shape=4, scale=4, size=SAMPLE_SIZE
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

        this_mean = numpy.random.normal(
            loc=actual_heating_rates_k_day01[i],
            scale=0.1 * actual_heating_rates_k_day01[i],
            size=1
        )[0]
        predicted_hr_matrix_k_day01[i, :] = numpy.random.normal(
            loc=this_mean, scale=1., size=ENSEMBLE_SIZE
        )

    print('Have created ensemble for all {0:d} examples!'.format(SAMPLE_SIZE))

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
    axes_object.legend(
        legend_handles, legend_strings, loc='upper left',
        bbox_to_anchor=(0, 0.99), fancybox=True, shadow=False,
        facecolor='white', edgecolor='k', framealpha=1., ncol=1,
        fontsize=36, markerscale=4
    )

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
        target_wavelength_metres=example_utils.DUMMY_BROADBAND_WAVELENGTH_METRES,
        plot_inset=False
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
    panel_file_names.append(_plot_spread_vs_skill_1model(
        output_dir_name=output_dir_name, for_good_model=True, panel_letter='c'
    ))
    panel_file_names.append(_plot_spread_vs_skill_1model(
        output_dir_name=output_dir_name, for_good_model=False, panel_letter='d'
    ))
    panel_file_names.append(_plot_pit_histogram_1model(
        output_dir_name=output_dir_name, for_good_model=True, panel_letter='e'
    ))
    panel_file_names.append(_plot_pit_histogram_1model(
        output_dir_name=output_dir_name, for_good_model=False, panel_letter='f'
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
        num_panel_rows=3, num_panel_columns=2
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
