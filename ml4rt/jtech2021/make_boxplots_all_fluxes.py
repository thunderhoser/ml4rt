"""Makes boxplots for various evaluation scores."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from scipy.stats import percentileofscore
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4rt.utils import evaluation
from ml4rt.utils import example_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
TOLERANCE = 1e-6

FLUX_NAMES = [
    evaluation.SHORTWAVE_NET_FLUX_NAME,
    example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME,
    example_utils.SHORTWAVE_TOA_UP_FLUX_NAME
]

FLUX_NAMES_VERBOSE = [
    r'$F_{net}$', r'$F_{down}^{sfc}$', r'$F_{up}^{TOA}$'
]

METRES_TO_KM = 0.001
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

HEATING_RATE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
FLUX_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
REFERENCE_LINE_COLOUR = numpy.full(3, 152. / 255)

FONT_SIZE = 30
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

OVERALL_EVAL_FILES_ARG_NAME = 'input_overall_eval_file_names'
MULTICLOUD_EVAL_FILES_ARG_NAME = 'input_multicloud_eval_file_names'
DESCRIPTIONS_ARG_NAME = 'description_strings'
OVERALL_HEIGHTS_ARG_NAME = 'overall_heights_m_agl'
MULTICLOUD_HEIGHTS_ARG_NAME = 'multicloud_heights_m_agl'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

OVERALL_EVAL_FILES_HELP_STRING = (
    'List of paths to files with overall evaluation scores (one per model).  '
    'Each will be read by `evaluation.read_file`.'
)
MULTICLOUD_EVAL_FILES_HELP_STRING = (
    'List of paths to files with evaluation scores for multi-layer cloud (one '
    'per model).  Each will be read by `evaluation.read_file`.'
)
DESCRIPTIONS_HELP_STRING = (
    'List of descriptions (one per model).  This must be a space-separated '
    'list; underscores will be replaced by spaces.'
)
OVERALL_HEIGHTS_HELP_STRING = (
    'Will plot overall evaluation scores for heating rate at these heights '
    '(metres above ground level).'
)
MULTICLOUD_HEIGHTS_HELP_STRING = (
    'Will plot multi-cloud-layer evaluation scores for heating rate at these '
    'heights (metres above ground level).'
)
OUTPUT_DIR_HELP_STRING = 'Name of output directory.  Figures will be saved here.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + OVERALL_EVAL_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=OVERALL_EVAL_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MULTICLOUD_EVAL_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=MULTICLOUD_EVAL_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DESCRIPTIONS_ARG_NAME, type=str, nargs='+', required=True,
    help=DESCRIPTIONS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OVERALL_HEIGHTS_ARG_NAME, type=int, nargs='+', required=True,
    help=OVERALL_HEIGHTS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MULTICLOUD_HEIGHTS_ARG_NAME, type=int, nargs='+', required=True,
    help=MULTICLOUD_HEIGHTS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _read_one_file(evaluation_file_name, heights_m_agl):
    """Reads one evaluation file.

    B = number of bootstrap replicates
    H = number of heights for heating rate
    F = number of flux components

    :param evaluation_file_name: Path to input file (will be read by
        `evaluation.read_file`).
    :param heights_m_agl: See documentation at top of file.
    :return: flux_msess_matrix: F-by-B numpy array of MSE skill scores for flux.
    :return: heating_rate_msess_matrix: H-by-B numpy array of MSE skill scores
        for heating rate.
    :return: flux_bias_matrix_w_m02: F-by-B numpy array of biases (W m^-2) for
        flux.
    :return: heating_rate_bias_matrix_k_day01: H-by-B numpy array of biases
        (K day^-1) for heating rate.
    """

    print('Reading data from: "{0:s}"...'.format(evaluation_file_name))
    result_table_xarray = evaluation.read_file(evaluation_file_name)

    net_flux_index = numpy.where(
        result_table_xarray.coords[evaluation.AUX_TARGET_FIELD_DIM].values ==
        evaluation.SHORTWAVE_NET_FLUX_NAME
    )[0][0]

    other_flux_indices = numpy.array([
        numpy.where(
            result_table_xarray.coords[evaluation.SCALAR_FIELD_DIM].values == n
        )[0][0]
        for n in FLUX_NAMES[1:]
    ], dtype=int)

    t = result_table_xarray

    flux_msess_matrix = numpy.concatenate((
        t[evaluation.AUX_MSE_SKILL_KEY].values[[net_flux_index], :],
        t[evaluation.SCALAR_MSE_SKILL_KEY].values[other_flux_indices, :]
    ))

    flux_bias_matrix_w_m02 = numpy.concatenate((
        t[evaluation.AUX_BIAS_KEY].values[[net_flux_index], :],
        t[evaluation.SCALAR_BIAS_KEY].values[other_flux_indices, :]
    ))

    heating_rate_index = numpy.where(
        result_table_xarray.coords[evaluation.VECTOR_FIELD_DIM].values ==
        example_utils.SHORTWAVE_HEATING_RATE_NAME
    )[0][0]

    num_heights = len(heights_m_agl)
    num_bootstrap_reps = flux_bias_matrix_w_m02.shape[1]
    heating_rate_msess_matrix = numpy.full(
        (num_heights, num_bootstrap_reps), numpy.nan
    )
    heating_rate_bias_matrix_k_day01 = numpy.full(
        (num_heights, num_bootstrap_reps), numpy.nan
    )

    for k in range(num_heights):
        these_diffs = numpy.absolute(
            result_table_xarray.coords[evaluation.HEIGHT_DIM].values -
            heights_m_agl[k]
        )
        this_height_index = numpy.where(these_diffs <= TOLERANCE)[0][0]

        heating_rate_msess_matrix[k, :] = (
            result_table_xarray[evaluation.VECTOR_MSE_SKILL_KEY].values[
                this_height_index, heating_rate_index, :
            ]
        )

        heating_rate_bias_matrix_k_day01[k, :] = (
            result_table_xarray[evaluation.VECTOR_BIAS_KEY].values[
                this_height_index, heating_rate_index, :
            ]
        )

    return (
        flux_msess_matrix, heating_rate_msess_matrix,
        flux_bias_matrix_w_m02, heating_rate_bias_matrix_k_day01
    )


def _make_bias_boxplot(
        overall_flux_bias_matrix_w_m02,
        multicloud_flux_bias_matrix_w_m02,
        overall_heating_rate_bias_matrix_k_day01,
        multicloud_heating_rate_bias_matrix_k_day01,
        model_description_strings, overall_heights_m_agl,
        multicloud_heights_m_agl, output_file_name):
    """Makes boxplot for bias.

    M = number of models
    H_o = number of heights for overall scores
    H_m = number of heights for multi-cloud-layer scores
    F = number of flux components
    B = number of bootstrap replicates

    :param overall_flux_bias_matrix_w_m02: M-by-F-by-B numpy array of biases
        (W m^-2) for flux.
    :param multicloud_flux_bias_matrix_w_m02: M-by-F-by-B numpy array of biases
        (W m^-2) for flux.
    :param overall_heating_rate_bias_matrix_k_day01: numpy array (M x H_o x B)
        of biases (K day^-1) for heating rate.
    :param multicloud_heating_rate_bias_matrix_k_day01: numpy array
        (M x H_m x B) of biases (K day^-1) for heating rate.
    :param model_description_strings: length-M list of strings.
    :param overall_heights_m_agl: numpy array (length H_o) of heights (metres
        above ground level).
    :param multicloud_heights_m_agl: numpy array (length H_m) of heights (metres
        above ground level).
    :param output_file_name: Path to output file.  Figure will be saved here.
    """

    # Housekeeping.
    figure_object, flux_axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    flux_axes_object.yaxis.label.set_color(FLUX_COLOUR)
    flux_axes_object.tick_params(axis='y', colors=FLUX_COLOUR)

    heating_rate_axes_object = flux_axes_object.twinx()
    heating_rate_axes_object.yaxis.set_ticks_position('right')
    heating_rate_axes_object.yaxis.set_label_position('right')
    heating_rate_axes_object.yaxis.label.set_color(HEATING_RATE_COLOUR)
    heating_rate_axes_object.tick_params(axis='y', colors=HEATING_RATE_COLOUR)

    # heating_rate_axes_object.set_zorder(flux_axes_object.get_zorder() + 1)
    # heating_rate_axes_object.patch.set_visible(False)

    num_models = len(model_description_strings)
    num_overall_heights = len(overall_heights_m_agl)
    num_multicloud_heights = len(multicloud_heights_m_agl)
    num_flux_components = overall_flux_bias_matrix_w_m02.shape[1]
    num_boxes = num_models * (
        num_overall_heights + num_multicloud_heights + 2 * num_flux_components
    )

    x_tick_values = numpy.linspace(0, num_boxes - 1, num=num_boxes, dtype=float)
    x_cutoff_values = x_tick_values[::num_models] - 0.5
    x_cutoff_values = x_cutoff_values[1:]

    # Plot boxplots.
    boxplot_style_dict = {
        'color': FLUX_COLOUR,
        'linewidth': 2
    }

    box_index = -1
    x_label_strings = [''] * num_boxes

    for j in range(num_flux_components):
        for i in range(num_models):
            box_index += 1
            x_label_strings[box_index] = (
                model_description_strings[i] + ': ' + FLUX_NAMES_VERBOSE[j]
            )

            flux_axes_object.boxplot(
                numpy.absolute(overall_flux_bias_matrix_w_m02[i, j, :]),
                widths=1., notch=False, sym='', whis=(0.5, 99.5),
                medianprops=boxplot_style_dict, boxprops=boxplot_style_dict,
                whiskerprops=boxplot_style_dict, capprops=boxplot_style_dict,
                positions=x_tick_values[[box_index]]
            )

            for k in range(i + 1, num_models):
                these_diffs = (
                    numpy.absolute(overall_flux_bias_matrix_w_m02[i, j, :]) -
                    numpy.absolute(overall_flux_bias_matrix_w_m02[k, j, :])
                )
                this_percentile = percentileofscore(
                    a=these_diffs, score=0., kind='mean'
                )

                print((
                    '{0:s} bias of model "{1:s}" better than {0:s} bias of '
                    'model "{2:s}" with p-value = {3:.4f}'
                ).format(
                    FLUX_NAMES[j], model_description_strings[k],
                    model_description_strings[i], this_percentile / 100
                ))

    for j in range(num_flux_components):
        for i in range(num_models):
            box_index += 1
            x_label_strings[box_index] = (
                model_description_strings[i] + ', MLC: ' + FLUX_NAMES_VERBOSE[j]
            )

            flux_axes_object.boxplot(
                numpy.absolute(multicloud_flux_bias_matrix_w_m02[i, j, :]),
                widths=1., notch=False, sym='', whis=(0.5, 99.5),
                medianprops=boxplot_style_dict, boxprops=boxplot_style_dict,
                whiskerprops=boxplot_style_dict, capprops=boxplot_style_dict,
                positions=x_tick_values[[box_index]]
            )

            for k in range(i + 1, num_models):
                these_diffs = (
                    numpy.absolute(multicloud_flux_bias_matrix_w_m02[i, j, :]) -
                    numpy.absolute(multicloud_flux_bias_matrix_w_m02[k, j, :])
                )
                this_percentile = percentileofscore(
                    a=these_diffs, score=0., kind='mean'
                )

                print((
                    'MLC {0:s} bias of model "{1:s}" better than MLC {0:s} bias'
                    ' of model "{2:s}" with p-value = {3:.4f}'
                ).format(
                    FLUX_NAMES[j], model_description_strings[k],
                    model_description_strings[i], this_percentile / 100
                ))

    boxplot_style_dict = {
        'color': HEATING_RATE_COLOUR,
        'linewidth': 2
    }

    for j in range(num_overall_heights):
        for i in range(num_models):
            box_index += 1
            this_height_km = int(numpy.round(
                overall_heights_m_agl[j] * METRES_TO_KM
            ))
            x_label_strings[box_index] = '{0:s}: {1:d}-km HR'.format(
                model_description_strings[i], this_height_km
            )

            heating_rate_axes_object.boxplot(
                numpy.absolute(
                    overall_heating_rate_bias_matrix_k_day01[i, j, :]
                ),
                widths=1., notch=False, sym='', whis=(0.5, 99.5),
                medianprops=boxplot_style_dict, boxprops=boxplot_style_dict,
                whiskerprops=boxplot_style_dict, capprops=boxplot_style_dict,
                positions=x_tick_values[[box_index]]
            )

            for k in range(i + 1, num_models):
                these_diffs = (
                    numpy.absolute(
                        overall_heating_rate_bias_matrix_k_day01[i, j, :]
                    )
                    -
                    numpy.absolute(
                        overall_heating_rate_bias_matrix_k_day01[k, j, :]
                    )
                )
                this_percentile = percentileofscore(
                    a=these_diffs, score=0., kind='mean'
                )

                print((
                    '{0:d}-km-HR bias of model "{1:s}" better than {0:d}-km-HR '
                    'bias of model "{2:s}" with p-value = {3:.4f}'
                ).format(
                    this_height_km,
                    model_description_strings[k], model_description_strings[i],
                    this_percentile / 100
                ))

    for j in range(num_multicloud_heights):
        for i in range(num_models):
            box_index += 1
            this_height_km = int(numpy.round(
                multicloud_heights_m_agl[j] * METRES_TO_KM
            ))
            x_label_strings[box_index] = '{0:s}, MLC: {1:d}-km HR'.format(
                model_description_strings[i], this_height_km
            )

            heating_rate_axes_object.boxplot(
                numpy.absolute(
                    multicloud_heating_rate_bias_matrix_k_day01[i, j, :]
                ),
                widths=1., notch=False, sym='', whis=(0.5, 99.5),
                medianprops=boxplot_style_dict, boxprops=boxplot_style_dict,
                whiskerprops=boxplot_style_dict, capprops=boxplot_style_dict,
                positions=x_tick_values[[box_index]]
            )

            for k in range(i + 1, num_models):
                these_diffs = (
                    numpy.absolute(
                        multicloud_heating_rate_bias_matrix_k_day01[i, j, :]
                    )
                    -
                    numpy.absolute(
                        multicloud_heating_rate_bias_matrix_k_day01[k, j, :]
                    )
                )
                this_percentile = percentileofscore(
                    a=these_diffs, score=0., kind='mean'
                )

                print((
                    'MLC {0:d}-km-HR bias of model "{1:s}" better than MLC '
                    '{0:d}-km-HR bias of model "{2:s}" with p-value = {3:.4f}'
                ).format(
                    this_height_km,
                    model_description_strings[k], model_description_strings[i],
                    this_percentile / 100
                ))

    flux_axes_object.set_xticklabels(x_label_strings, rotation=90.)
    heating_rate_axes_object.set_ylabel(
        r'Absolute bias for heating rate (K day$^{-1}$)'
    )
    flux_axes_object.set_ylabel(r'Absolute bias for flux (W m$^{-2}$)')

    y_limits = flux_axes_object.get_ylim()
    heating_rate_y_limits = heating_rate_axes_object.get_ylim()

    for this_cutoff_value in x_cutoff_values:
        flux_axes_object.plot(
            numpy.full(2, this_cutoff_value), y_limits,
            color=REFERENCE_LINE_COLOUR, linestyle='dashed', linewidth=2,
            zorder=-1e10
        )

    flux_axes_object.set_ylim(y_limits)
    heating_rate_axes_object.set_ylim(heating_rate_y_limits)

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _make_msess_boxplot(
        overall_flux_msess_matrix, multicloud_flux_msess_matrix,
        overall_heating_rate_msess_matrix, multicloud_heating_rate_msess_matrix,
        model_description_strings, overall_heights_m_agl,
        multicloud_heights_m_agl, output_file_name):
    """Makes boxplot for MSE skill score.

    M = number of models
    H_o = number of heights for overall scores
    H_m = number of heights for multi-cloud-layer scores
    F = number of flux components
    B = number of bootstrap replicates

    :param overall_flux_msess_matrix: M-by-F-by-B numpy array of MSE skill
        scores for flux.
    :param multicloud_flux_msess_matrix: M-by-F-by-B numpy array of MSE skill
        scores for flux.
    :param overall_heating_rate_msess_matrix: numpy array (M x H_o x B) of MSE
        skill scores for heating rate.
    :param multicloud_heating_rate_msess_matrix: numpy array (M x H_m x B) of
        MSE skill scores for heating rate.
    :param model_description_strings: length-M list of strings.
    :param overall_heights_m_agl: numpy array (length H_o) of heights (metres
        above ground level).
    :param multicloud_heights_m_agl: numpy array (length H_m) of heights (metres
        above ground level).
    :param output_file_name: Path to output file.  Figure will be saved here.
    """

    # Housekeeping.
    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    num_models = len(model_description_strings)
    num_overall_heights = len(overall_heights_m_agl)
    num_multicloud_heights = len(multicloud_heights_m_agl)
    num_flux_components = overall_flux_msess_matrix.shape[1]
    num_boxes = num_models * (
        num_overall_heights + num_multicloud_heights + 2 * num_flux_components
    )

    x_tick_values = numpy.linspace(0, num_boxes - 1, num=num_boxes, dtype=float)
    x_cutoff_values = x_tick_values[::num_models] - 0.5
    x_cutoff_values = x_cutoff_values[1:]

    # Plot boxplots.
    boxplot_style_dict = {
        'color': FLUX_COLOUR,
        'linewidth': 2
    }

    box_index = -1
    x_label_strings = [''] * num_boxes

    for j in range(num_flux_components):
        for i in range(num_models):
            box_index += 1
            x_label_strings[box_index] = (
                model_description_strings[i] + ': ' + FLUX_NAMES_VERBOSE[j]
            )

            axes_object.boxplot(
                overall_flux_msess_matrix[i, j, :],
                widths=1., notch=False, sym='', whis=(0.5, 99.5),
                medianprops=boxplot_style_dict, boxprops=boxplot_style_dict,
                whiskerprops=boxplot_style_dict, capprops=boxplot_style_dict,
                positions=x_tick_values[[box_index]]
            )

            for k in range(i + 1, num_models):
                these_diffs = (
                    overall_flux_msess_matrix[i, j, :] -
                    overall_flux_msess_matrix[k, j, :]
                )
                this_percentile = percentileofscore(
                    a=these_diffs, score=0., kind='mean'
                )

                print((
                    '{0:s} MSESS of model "{1:s}" better than {0:s} MSESS of '
                    'model "{2:s}" with p-value = {3:.4f}'
                ).format(
                    FLUX_NAMES[j], model_description_strings[i],
                    model_description_strings[k], this_percentile / 100
                ))

    for j in range(num_flux_components):
        for i in range(num_models):
            box_index += 1
            x_label_strings[box_index] = (
                model_description_strings[i] + r', MLC: ' +
                FLUX_NAMES_VERBOSE[j]
            )

            axes_object.boxplot(
                multicloud_flux_msess_matrix[i, j, :],
                widths=1., notch=False, sym='', whis=(0.5, 99.5),
                medianprops=boxplot_style_dict, boxprops=boxplot_style_dict,
                whiskerprops=boxplot_style_dict, capprops=boxplot_style_dict,
                positions=x_tick_values[[box_index]]
            )

            for k in range(i + 1, num_models):
                these_diffs = (
                    multicloud_flux_msess_matrix[i, j, :] -
                    multicloud_flux_msess_matrix[k, j, :]
                )
                this_percentile = percentileofscore(
                    a=these_diffs, score=0., kind='mean'
                )

                print((
                    'MLC {0:s} MSESS of model "{1:s}" better than MLC {0:s} '
                    'MSESS of model "{2:s}" with p-value = {3:.4f}'
                ).format(
                    FLUX_NAMES[j], model_description_strings[i],
                    model_description_strings[k], this_percentile / 100
                ))

    boxplot_style_dict = {
        'color': HEATING_RATE_COLOUR,
        'linewidth': 2
    }

    for j in range(num_overall_heights):
        for i in range(num_models):
            box_index += 1
            this_height_km = int(numpy.round(
                overall_heights_m_agl[j] * METRES_TO_KM
            ))
            x_label_strings[box_index] = '{0:s}: {1:d}-km HR'.format(
                model_description_strings[i], this_height_km
            )

            axes_object.boxplot(
                overall_heating_rate_msess_matrix[i, j, :],
                widths=1., notch=False, sym='', whis=(0.5, 99.5),
                medianprops=boxplot_style_dict, boxprops=boxplot_style_dict,
                whiskerprops=boxplot_style_dict, capprops=boxplot_style_dict,
                positions=x_tick_values[[box_index]]
            )

            for k in range(i + 1, num_models):
                these_diffs = (
                    overall_heating_rate_msess_matrix[i, j, :] -
                    overall_heating_rate_msess_matrix[k, j, :]
                )
                this_percentile = percentileofscore(
                    a=these_diffs, score=0., kind='mean'
                )

                print((
                    '{0:d}-km-HR MSESS of model "{1:s}" better than {0:d}-km-HR'
                    ' MSESS of model "{2:s}" with p-value = {3:.4f}'
                ).format(
                    this_height_km,
                    model_description_strings[i], model_description_strings[k],
                    this_percentile / 100
                ))

    for j in range(num_multicloud_heights):
        for i in range(num_models):
            box_index += 1
            this_height_km = int(numpy.round(
                multicloud_heights_m_agl[j] * METRES_TO_KM
            ))
            x_label_strings[box_index] = '{0:s}, MLC: {1:d}-km HR'.format(
                model_description_strings[i], this_height_km
            )

            axes_object.boxplot(
                multicloud_heating_rate_msess_matrix[i, j, :],
                widths=1., notch=False, sym='', whis=(0.5, 99.5),
                medianprops=boxplot_style_dict, boxprops=boxplot_style_dict,
                whiskerprops=boxplot_style_dict, capprops=boxplot_style_dict,
                positions=x_tick_values[[box_index]]
            )

            for k in range(i + 1, num_models):
                these_diffs = (
                    multicloud_heating_rate_msess_matrix[i, j, :] -
                    multicloud_heating_rate_msess_matrix[k, j, :]
                )
                this_percentile = percentileofscore(
                    a=these_diffs, score=0., kind='mean'
                )

                print((
                    'MLC {0:d}-km-HR MSESS of model "{1:s}" better than MLC '
                    '{0:d}-km-HR MSESS of model "{2:s}" with p-value = {3:.4f}'
                ).format(
                    this_height_km,
                    model_description_strings[i], model_description_strings[k],
                    this_percentile / 100
                ))

    axes_object.set_xticklabels(x_label_strings, rotation=90.)
    axes_object.set_ylabel('MSE skill score')

    y_limits = axes_object.get_ylim()

    for this_cutoff_value in x_cutoff_values:
        axes_object.plot(
            numpy.full(2, this_cutoff_value), y_limits,
            color=REFERENCE_LINE_COLOUR, linestyle='dashed', linewidth=2,
            zorder=-1e10
        )

    axes_object.set_ylim(y_limits)

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run(overall_eval_file_names, multicloud_eval_file_names,
         description_strings, overall_heights_m_agl, multicloud_heights_m_agl,
         output_dir_name):
    """Makes boxplots for various evaluation scores.

    This is effectively the main method.

    :param overall_eval_file_names: See documentation at top of file.
    :param multicloud_eval_file_names: Same.
    :param description_strings: Same.
    :param overall_heights_m_agl: Same.
    :param multicloud_heights_m_agl: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    num_models = len(overall_eval_file_names)
    expected_dim = numpy.array([num_models], dtype=int)

    error_checking.assert_is_numpy_array(
        numpy.array(multicloud_eval_file_names), exact_dimensions=expected_dim
    )
    error_checking.assert_is_numpy_array(
        numpy.array(description_strings), exact_dimensions=expected_dim
    )

    description_strings = [s.replace('_', ' ') for s in description_strings]

    overall_flux_msess_matrix = None
    multicloud_flux_msess_matrix = None
    overall_flux_bias_matrix_w_m02 = None
    multicloud_flux_bias_matrix_w_m02 = None
    overall_heating_rate_msess_matrix = None
    multicloud_heating_rate_msess_matrix = None
    overall_heating_rate_bias_matrix_k_day01 = None
    multicloud_heating_rate_bias_matrix_k_day01 = None

    for i in range(num_models):
        if i == 0:
            this_flux_msess_matrix = _read_one_file(
                evaluation_file_name=overall_eval_file_names[i],
                heights_m_agl=overall_heights_m_agl
            )[0]

            num_bootstrap_reps = this_flux_msess_matrix.shape[1]
            these_dim = (num_models, len(FLUX_NAMES), num_bootstrap_reps)

            overall_flux_msess_matrix = numpy.full(these_dim, numpy.nan)
            multicloud_flux_msess_matrix = numpy.full(these_dim, numpy.nan)
            overall_flux_bias_matrix_w_m02 = numpy.full(these_dim, numpy.nan)
            multicloud_flux_bias_matrix_w_m02 = numpy.full(these_dim, numpy.nan)

            these_dim = (
                num_models, len(overall_heights_m_agl), num_bootstrap_reps
            )
            overall_heating_rate_msess_matrix = numpy.full(these_dim, numpy.nan)
            overall_heating_rate_bias_matrix_k_day01 = numpy.full(
                these_dim, numpy.nan
            )

            these_dim = (
                num_models, len(multicloud_heights_m_agl), num_bootstrap_reps
            )
            multicloud_heating_rate_msess_matrix = numpy.full(
                these_dim, numpy.nan
            )
            multicloud_heating_rate_bias_matrix_k_day01 = numpy.full(
                these_dim, numpy.nan
            )

        (
            overall_flux_msess_matrix[i, ...],
            overall_heating_rate_msess_matrix[i, ...],
            overall_flux_bias_matrix_w_m02[i, ...],
            overall_heating_rate_bias_matrix_k_day01[i, ...]
        ) = _read_one_file(
            evaluation_file_name=overall_eval_file_names[i],
            heights_m_agl=overall_heights_m_agl
        )

        (
            multicloud_flux_msess_matrix[i, ...],
            multicloud_heating_rate_msess_matrix[i, ...],
            multicloud_flux_bias_matrix_w_m02[i, ...],
            multicloud_heating_rate_bias_matrix_k_day01[i, ...]
        ) = _read_one_file(
            evaluation_file_name=multicloud_eval_file_names[i],
            heights_m_agl=multicloud_heights_m_agl
        )

    print(SEPARATOR_STRING)

    _make_msess_boxplot(
        overall_flux_msess_matrix=overall_flux_msess_matrix,
        multicloud_flux_msess_matrix=multicloud_flux_msess_matrix,
        overall_heating_rate_msess_matrix=overall_heating_rate_msess_matrix,
        multicloud_heating_rate_msess_matrix=
        multicloud_heating_rate_msess_matrix,
        model_description_strings=description_strings,
        overall_heights_m_agl=overall_heights_m_agl,
        multicloud_heights_m_agl=multicloud_heights_m_agl,
        output_file_name='{0:s}/mse_skill_score.jpg'.format(output_dir_name)
    )

    _make_bias_boxplot(
        overall_flux_bias_matrix_w_m02=overall_flux_bias_matrix_w_m02,
        multicloud_flux_bias_matrix_w_m02=multicloud_flux_bias_matrix_w_m02,
        overall_heating_rate_bias_matrix_k_day01=
        overall_heating_rate_bias_matrix_k_day01,
        multicloud_heating_rate_bias_matrix_k_day01=
        multicloud_heating_rate_bias_matrix_k_day01,
        model_description_strings=description_strings,
        overall_heights_m_agl=overall_heights_m_agl,
        multicloud_heights_m_agl=multicloud_heights_m_agl,
        output_file_name='{0:s}/bias.jpg'.format(output_dir_name)
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        overall_eval_file_names=getattr(
            INPUT_ARG_OBJECT, OVERALL_EVAL_FILES_ARG_NAME
        ),
        multicloud_eval_file_names=getattr(
            INPUT_ARG_OBJECT, MULTICLOUD_EVAL_FILES_ARG_NAME
        ),
        description_strings=getattr(INPUT_ARG_OBJECT, DESCRIPTIONS_ARG_NAME),
        overall_heights_m_agl=numpy.array(
            getattr(INPUT_ARG_OBJECT, OVERALL_HEIGHTS_ARG_NAME), dtype=int
        ),
        multicloud_heights_m_agl=numpy.array(
            getattr(INPUT_ARG_OBJECT, MULTICLOUD_HEIGHTS_ARG_NAME), dtype=int
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
