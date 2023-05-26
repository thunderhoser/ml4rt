"""Plots hyperparameter grids for experiment with BNN-only UQ method."""

import os
import sys
import glob
import argparse
import numpy
from scipy.stats import rankdata
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import evaluation
import pit_utils
import spread_skill_utils as ss_utils
import discard_test_utils as dt_utils
import file_system_utils
import imagemagick_utils
import gg_plotting_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

BAYESIAN_DENSE_LAYER_COUNTS_AXIS1 = numpy.array([1, 2, 3], dtype=int)
BAYESIAN_SKIP_LAYER_COUNTS_AXIS2 = numpy.array([1, 1, 2, 2], dtype=int)
BAYESIAN_UPCONV_LAYER_COUNTS_AXIS2 = numpy.array([1, 2, 1, 2], dtype=int)
SPECTRAL_COMPLEXITIES_AXIS3 = numpy.array([64, 64, 128, 128], dtype=int)
BAYESIAN_LAYER_TYPE_STRINGS_AXIS3 = [
    'flipout', 'reparameterization', 'flipout', 'reparameterization'
]

HEATING_RATE_MAE_NAME = 'heating_rate_mae'
HEATING_RATE_REL_NAME = 'heating_rate_rel'
HEATING_RATE_SSREL_NAME = 'heating_rate_ssrel'
HEATING_RATE_SSRAT_NAME = 'heating_rate_ssrat'
HEATING_RATE_PITD_NAME = 'heating_rate_pitd'
HEATING_RATE_MF_NAME = 'heating_rate_mf'

ALL_FLUX_MAE_NAME = 'all_flux_mae'
ALL_FLUX_REL_NAME = 'all_flux_rel'
ALL_FLUX_SSREL_NAME = 'all_flux_ssrel'
ALL_FLUX_SSRAT_NAME = 'all_flux_ssrat'
ALL_FLUX_PITD_NAME = 'all_flux_pitd'
ALL_FLUX_MF_NAME = 'all_flux_mf'

METRIC_NAMES = [
    HEATING_RATE_MAE_NAME, HEATING_RATE_REL_NAME,
    HEATING_RATE_SSREL_NAME, HEATING_RATE_SSRAT_NAME,
    HEATING_RATE_PITD_NAME, HEATING_RATE_MF_NAME,
    ALL_FLUX_MAE_NAME, ALL_FLUX_REL_NAME,
    ALL_FLUX_SSREL_NAME, ALL_FLUX_SSRAT_NAME,
    ALL_FLUX_PITD_NAME, ALL_FLUX_MF_NAME
]

METRIC_NAMES_FANCY = [
    'heating-rate MAE', 'heating-rate REL',
    'heating-rate SSREL', 'heating-rate SSRAT',
    'heating-rate PITD', 'heating-rate MF',
    'all-flux MAE', 'all-flux REL',
    'all-flux SSREL', 'all-flux SSRAT',
    'all-flux PITD', 'all-flux MF'
]

METRIC_UNITS = [
    r'K day$^{-1}$', r'K$^{2}$ day$^{-2}$',
    r'K day$^{-1}$', 'unitless',
    'unitless', 'unitless',
    r'W m$^{-2}$', r'W$^{2}$ m$^{-4}$',
    r'W m$^{-2}$', 'unitless',
    'unitless', 'unitless'
]

BEST_MARKER_TYPE = '*'
BEST_MARKER_SIZE_GRID_CELLS = 0.175
WHITE_COLOUR = numpy.full(3, 1.)
BLACK_COLOUR = numpy.full(3, 0.)

SELECTED_MARKER_TYPE = 'o'
SELECTED_MARKER_SIZE_GRID_CELLS = 0.175
SELECTED_MARKER_INDICES_CLEAN_TRAINED = numpy.array([2, 3, 1], dtype=int)
SELECTED_MARKER_INDICES_LP_TRAINED = numpy.array([1, 1, 3], dtype=int)

MAIN_COLOUR_MAP_OBJECT = pyplot.get_cmap(name='viridis', lut=20)
MONO_FRACTION_COLOUR_MAP_OBJECT = pyplot.get_cmap(name='cividis', lut=20)
SSRAT_COLOUR_MAP_NAME = 'seismic'

MAIN_COLOUR_MAP_OBJECT.set_bad(numpy.full(3, 152. / 255))
MONO_FRACTION_COLOUR_MAP_OBJECT.set_bad(numpy.full(3, 152. / 255))

FONT_SIZE = 26
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

EXPERIMENT_DIR_ARG_NAME = 'experiment_dir_name'
TRAINED_WITH_CLEAN_DATA_ARG_NAME = 'trained_with_clean_data'

EXPERIMENT_DIR_HELP_STRING = 'Name of top-level directory with models.'
TRAINED_WITH_CLEAN_DATA_HELP_STRING = (
    'Boolean flag.  If True (False), models were trained with clean (lightly '
    'perturbed) data.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EXPERIMENT_DIR_ARG_NAME, type=str, required=True,
    help=EXPERIMENT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TRAINED_WITH_CLEAN_DATA_ARG_NAME, type=int, required=True,
    help=TRAINED_WITH_CLEAN_DATA_HELP_STRING
)


def _finite_percentile(input_array, percentile_level):
    """Takes percentile of input array, considering only finite values.

    :param input_array: numpy array.
    :param percentile_level: Percentile level, ranging from 0...100.
    :return: output_percentile: Percentile value.
    """

    return numpy.percentile(
        input_array[numpy.isfinite(input_array)], percentile_level
    )


def _get_ssrat_colour_scheme(max_colour_value):
    """Returns colour scheme for spread-skill ratio (SSRAT).

    :param max_colour_value: Max value in colour scheme.
    :return: colour_map_object: Colour map (instance of `matplotlib.pyplot.cm`).
    :return: colour_norm_object: Colour-normalizer (maps from data space to
        colour-bar space, which goes from 0...1).  This is an instance of
        `matplotlib.colors.Normalize`.
    """

    orig_colour_map_object = pyplot.get_cmap(SSRAT_COLOUR_MAP_NAME)

    negative_values = numpy.linspace(0, 1, num=1001, dtype=float)
    positive_values = numpy.linspace(1, max_colour_value, num=1001, dtype=float)
    bias_values = numpy.concatenate((negative_values, positive_values))

    normalized_values = numpy.linspace(0, 1, num=len(bias_values), dtype=float)
    rgb_matrix = orig_colour_map_object(normalized_values)[:, :-1]

    colour_map_object = matplotlib.colors.ListedColormap(rgb_matrix)
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        bias_values, colour_map_object.N
    )

    return colour_map_object, colour_norm_object


def _plot_scores_2d(
        score_matrix, colour_map_object, colour_norm_object, x_tick_labels,
        y_tick_labels):
    """Plots scores on 2-D grid.

    M = number of rows in grid
    N = number of columns in grid

    :param score_matrix: M-by-N numpy array of scores.
    :param colour_map_object: Colour map (instance of `matplotlib.pyplot.cm`).
    :param colour_norm_object: Colour-normalizer (maps from data space to
        colour-bar space, which goes from 0...1).  This is an instance of
        `matplotlib.colors.Normalize`.
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
        orientation_string='vertical', extend_min=False, extend_max=False,
        font_size=FONT_SIZE, fraction_of_axis_length=0.6
    )

    tick_values = colour_bar_object.get_ticks()
    tick_strings = ['{0:.2g}'.format(v) for v in tick_values]
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    return figure_object, axes_object


def _read_metrics_one_model(model_dir_name):
    """Reads metrics for one model.

    :param model_dir_name: Name of directory with trained model and validation
        data.
    :return: metric_dict: Dictionary, where each key is a string from the list
        `METRIC_NAMES` and each value is a scalar.
    """

    metric_dict = {}
    for this_metric_name in METRIC_NAMES:
        metric_dict[this_metric_name] = numpy.nan

    model_file_pattern = '{0:s}/model*.h5'.format(model_dir_name)
    model_file_names = glob.glob(model_file_pattern)

    if len(model_file_names) == 0:
        return metric_dict

    model_file_names.sort()
    model_file_name = model_file_names[-1]
    this_file_name = '{0:s}/validation_perturbed_for_uq/evaluation.nc'.format(
        model_file_name[:-3]
    )

    if not os.path.isfile(this_file_name):
        return metric_dict

    print('Reading data from: "{0:s}"...'.format(this_file_name))
    this_eval_table_xarray = evaluation.read_file(this_file_name)

    this_mae_matrix = numpy.concatenate((
        this_eval_table_xarray[evaluation.SCALAR_MAE_KEY].values,
        this_eval_table_xarray[evaluation.AUX_MAE_KEY].values
    ), axis=0)

    assert this_mae_matrix.shape[0] == 3
    metric_dict[ALL_FLUX_MAE_NAME] = numpy.mean(
        numpy.nanmean(this_mae_matrix, axis=1)
    )

    this_mae_matrix = (
        this_eval_table_xarray[evaluation.VECTOR_MAE_KEY].values
    )
    assert this_mae_matrix.shape[1] == 1
    this_mae_matrix = this_mae_matrix[:, 0, :]
    metric_dict[HEATING_RATE_MAE_NAME] = numpy.mean(
        numpy.nanmean(this_mae_matrix, axis=1)
    )

    this_reliability_matrix = numpy.concatenate((
        this_eval_table_xarray[evaluation.SCALAR_RELIABILITY_KEY].values,
        this_eval_table_xarray[evaluation.AUX_RELIABILITY_KEY].values
    ), axis=0)

    metric_dict[ALL_FLUX_REL_NAME] = numpy.mean(
        numpy.nanmean(this_reliability_matrix, axis=1)
    )

    this_reliability_matrix = this_eval_table_xarray[
        evaluation.VECTOR_FLAT_RELIABILITY_KEY
    ].values
    metric_dict[HEATING_RATE_REL_NAME] = numpy.nanmean(
        this_reliability_matrix
    )

    this_file_name = (
        '{0:s}/validation_perturbed_for_uq/spread_vs_skill.nc'
    ).format(model_file_name[:-3])

    print('Reading data from: "{0:s}"...'.format(this_file_name))
    this_ss_table_xarray = ss_utils.read_results(this_file_name)

    these_ssrel = numpy.concatenate((
        this_ss_table_xarray[ss_utils.SCALAR_SSREL_KEY].values,
        this_ss_table_xarray[ss_utils.AUX_SSREL_KEY].values
    ))
    metric_dict[ALL_FLUX_SSREL_NAME] = numpy.mean(these_ssrel)

    these_ssrat = numpy.concatenate((
        this_ss_table_xarray[ss_utils.SCALAR_SSRAT_KEY].values,
        this_ss_table_xarray[ss_utils.AUX_SSRAT_KEY].values
    ))
    metric_dict[ALL_FLUX_SSRAT_NAME] = numpy.mean(these_ssrat)

    metric_dict[HEATING_RATE_SSREL_NAME] = (
        this_ss_table_xarray[ss_utils.VECTOR_FLAT_SSREL_KEY].values[0]
    )
    metric_dict[HEATING_RATE_SSRAT_NAME] = (
        this_ss_table_xarray[ss_utils.VECTOR_FLAT_SSRAT_KEY].values[0]
    )

    this_file_name = (
        '{0:s}/validation_perturbed_for_uq/pit_histograms.nc'
    ).format(model_file_name[:-3])

    print('Reading data from: "{0:s}"...'.format(this_file_name))
    this_pit_table_xarray = pit_utils.read_results(this_file_name)

    these_pitd = numpy.concatenate((
        this_pit_table_xarray[pit_utils.SCALAR_PITD_KEY].values,
        this_pit_table_xarray[pit_utils.AUX_PITD_KEY].values
    ))
    metric_dict[ALL_FLUX_PITD_NAME] = numpy.mean(these_pitd)
    metric_dict[HEATING_RATE_PITD_NAME] = (
        this_pit_table_xarray[pit_utils.VECTOR_FLAT_PITD_KEY].values[0]
    )

    this_file_name = (
        '{0:s}/validation_perturbed_for_uq/discard_test_for_heating_rates.nc'
    ).format(model_file_name[:-3])

    print('Reading data from: "{0:s}"...'.format(this_file_name))
    this_dt_table_xarray = dt_utils.read_results(this_file_name)

    these_mf = numpy.concatenate((
        this_dt_table_xarray[dt_utils.SCALAR_MONO_FRACTION_KEY].values,
        this_dt_table_xarray[dt_utils.AUX_MONO_FRACTION_KEY].values
    ))
    metric_dict[ALL_FLUX_MF_NAME] = numpy.mean(these_mf)
    metric_dict[HEATING_RATE_MF_NAME] = this_dt_table_xarray[
        dt_utils.VECTOR_FLAT_MONO_FRACTION_KEY
    ].values[0]

    return metric_dict


def _print_ranking_all_metrics(metric_matrix, main_metric_name):
    """Prints ranking for all metrics.

    A = length of first hyperparam axis
    B = length of second hyperparam axis
    C = length of third hyperparam axis
    M = number of metrics

    :param metric_matrix: A-by-B-by-C-by-M numpy array of metric values.
    :param main_metric_name: Name of main metric.
    """

    main_metric_index = METRIC_NAMES.index(main_metric_name)
    values_1d = numpy.ravel(metric_matrix[..., main_metric_index])

    if '_mf' in main_metric_name:
        values_1d[numpy.isnan(values_1d)] = -numpy.inf
        sort_indices_1d = numpy.argsort(-values_1d)
    elif '_ssrat' in main_metric_name:
        values_1d[numpy.isnan(values_1d)] = numpy.inf
        sort_indices_1d = numpy.argsort(numpy.absolute(1. - values_1d))
    else:
        values_1d[numpy.isnan(values_1d)] = numpy.inf
        sort_indices_1d = numpy.argsort(values_1d)

    i_sort_indices, j_sort_indices, k_sort_indices = numpy.unravel_index(
        sort_indices_1d, metric_matrix.shape[:-1]
    )

    metric_rank_matrix = numpy.full(metric_matrix.shape, numpy.nan)

    for m in range(len(METRIC_NAMES)):
        these_values = numpy.ravel(metric_matrix[..., m])

        if '_mf' in main_metric_name:
            these_values = -1 * these_values
            these_values[numpy.isnan(these_values)] = -numpy.inf
        elif '_ssrat' in main_metric_name:
            these_values = numpy.absolute(1. - these_values)
            these_values[numpy.isnan(these_values)] = numpy.inf
        else:
            these_values[numpy.isnan(these_values)] = numpy.inf

        metric_rank_matrix[..., m] = numpy.reshape(
            rankdata(these_values, method='average'),
            metric_rank_matrix.shape[:-1]
        )

    names = METRIC_NAMES
    mrm = metric_rank_matrix

    for m in range(len(i_sort_indices)):
        i = i_sort_indices[m]
        j = j_sort_indices[m]
        k = k_sort_indices[m]

        print((
            'Num Bayesian skip/upconv/dense layers = {0:d}/{1:d}/{2:d} ... '
            'Bayesian layer type = {3:s} ... spectral complexity = {4:d} ... '
            'deterministic HR ranks (MAE, REL) = {5:.1f}, {6:.1f} ... '
            'UQ-based HR ranks (SSREL, SSRAT, PITD, MF) = '
            '{7:.1f}, {8:.1f}, {9:.1f}, {10:.1f} ... '
            'deterministic flux ranks (MAE, REL) = {11:.1f}, {12:.1f} ... '
            'UQ-based flux ranks (SSREL, SSRAT, PITD, MF) = '
            '{13:.1f}, {14:.1f}, {15:.1f}, {16:.1f}'
        ).format(
            BAYESIAN_SKIP_LAYER_COUNTS_AXIS2[j],
            BAYESIAN_UPCONV_LAYER_COUNTS_AXIS2[j],
            BAYESIAN_DENSE_LAYER_COUNTS_AXIS1[i],
            BAYESIAN_LAYER_TYPE_STRINGS_AXIS3[k],
            SPECTRAL_COMPLEXITIES_AXIS3[k],
            mrm[i, j, k, names.index(HEATING_RATE_MAE_NAME)],
            mrm[i, j, k, names.index(HEATING_RATE_REL_NAME)],
            mrm[i, j, k, names.index(HEATING_RATE_SSREL_NAME)],
            mrm[i, j, k, names.index(HEATING_RATE_SSRAT_NAME)],
            mrm[i, j, k, names.index(HEATING_RATE_PITD_NAME)],
            mrm[i, j, k, names.index(HEATING_RATE_MF_NAME)],
            mrm[i, j, k, names.index(ALL_FLUX_MAE_NAME)],
            mrm[i, j, k, names.index(ALL_FLUX_REL_NAME)],
            mrm[i, j, k, names.index(ALL_FLUX_SSREL_NAME)],
            mrm[i, j, k, names.index(ALL_FLUX_SSRAT_NAME)],
            mrm[i, j, k, names.index(ALL_FLUX_PITD_NAME)],
            mrm[i, j, k, names.index(ALL_FLUX_MF_NAME)]
        ))


def _print_ranking_one_metric(metric_matrix, metric_index):
    """Prints ranking for one metric.

    A = length of first hyperparam axis
    B = length of second hyperparam axis
    C = length of third hyperparam axis
    M = number of metrics

    :param metric_matrix: A-by-B-by-C-by-M numpy array of metric values.
    :param metric_index: Will print ranking for [i]th metric, where
        i = `metric_index`.
    """

    values_1d = numpy.ravel(metric_matrix[..., metric_index])

    if '_mf' in METRIC_NAMES[metric_index]:
        values_1d[numpy.isnan(values_1d)] = -numpy.inf
        sort_indices_1d = numpy.argsort(-values_1d)
    elif '_ssrat' in METRIC_NAMES[metric_index]:
        values_1d[numpy.isnan(values_1d)] = numpy.inf
        sort_indices_1d = numpy.argsort(numpy.absolute(1. - values_1d))
    else:
        values_1d[numpy.isnan(values_1d)] = numpy.inf
        sort_indices_1d = numpy.argsort(values_1d)

    i_sort_indices, j_sort_indices, k_sort_indices = numpy.unravel_index(
        sort_indices_1d, metric_matrix.shape[:-1]
    )

    for m in range(len(i_sort_indices)):
        i = i_sort_indices[m]
        j = j_sort_indices[m]
        k = k_sort_indices[m]

        print((
            '{0:d}th-best {1:s} = {2:.3g} {3:s} ... '
            'num Bayesian skip/upconv/dense layers = {4:d}/{5:d}/{6:d} ... '
            'Bayesian layer type = {7:s} ... spectral complexity = {8:d}'
        ).format(
            m + 1, METRIC_NAMES_FANCY[metric_index],
            metric_matrix[i, j, k, metric_index], METRIC_UNITS[metric_index],
            BAYESIAN_SKIP_LAYER_COUNTS_AXIS2[j],
            BAYESIAN_UPCONV_LAYER_COUNTS_AXIS2[j],
            BAYESIAN_DENSE_LAYER_COUNTS_AXIS1[i],
            BAYESIAN_LAYER_TYPE_STRINGS_AXIS3[k],
            SPECTRAL_COMPLEXITIES_AXIS3[k]
        ))


def _run(experiment_dir_name, trained_with_clean_data):
    """Plots hyperparameter grids for experiment with BNN-only UQ method.

    This is effectively the main method.

    :param experiment_dir_name: See documentation at top of file.
    :param trained_with_clean_data: Same.
    """

    length_axis1 = len(BAYESIAN_DENSE_LAYER_COUNTS_AXIS1)
    length_axis2 = len(BAYESIAN_SKIP_LAYER_COUNTS_AXIS2)
    length_axis3 = len(SPECTRAL_COMPLEXITIES_AXIS3)
    num_metrics = len(METRIC_NAMES)

    y_tick_labels = [
        '{0:d}'.format(c) for c in BAYESIAN_DENSE_LAYER_COUNTS_AXIS1
    ]
    x_tick_labels = [
        '{0:d}/{1:d}'.format(s, u) for s, u in zip(
            BAYESIAN_SKIP_LAYER_COUNTS_AXIS2, BAYESIAN_UPCONV_LAYER_COUNTS_AXIS2
        )
    ]

    y_axis_label = 'Number of Bayesian dense layers'
    x_axis_label = 'Number of Bayesian skip/upconv layers'

    metric_matrix = numpy.full(
        (length_axis1, length_axis2, length_axis3, num_metrics),
        numpy.nan
    )

    for i in range(length_axis1):
        for j in range(length_axis2):
            for k in range(length_axis3):
                this_model_dir_name = (
                    '{0:s}/num-first-layer-channels={1:03d}_'
                    'num-bayesian-skip-layers={2:d}_'
                    'num-bayesian-upconv-layers={3:d}_'
                    'num-bayesian-dense-layers={4:d}_bayesian-layer-type={5:s}'
                ).format(
                    experiment_dir_name,
                    SPECTRAL_COMPLEXITIES_AXIS3[k],
                    BAYESIAN_SKIP_LAYER_COUNTS_AXIS2[j],
                    BAYESIAN_UPCONV_LAYER_COUNTS_AXIS2[j],
                    BAYESIAN_DENSE_LAYER_COUNTS_AXIS1[i],
                    BAYESIAN_LAYER_TYPE_STRINGS_AXIS3[k]
                )

                this_metric_dict = _read_metrics_one_model(this_model_dir_name)
                for m in range(num_metrics):
                    metric_matrix[i, j, k, m] = this_metric_dict[
                        METRIC_NAMES[m]
                    ]

    print(SEPARATOR_STRING)

    for m in range(num_metrics):
        _print_ranking_one_metric(metric_matrix=metric_matrix, metric_index=m)
        print(SEPARATOR_STRING)

    _print_ranking_all_metrics(
        metric_matrix=metric_matrix, main_metric_name=HEATING_RATE_MAE_NAME
    )
    print(SEPARATOR_STRING)

    _print_ranking_all_metrics(
        metric_matrix=metric_matrix, main_metric_name=HEATING_RATE_SSRAT_NAME
    )
    print(SEPARATOR_STRING)

    output_dir_name = '{0:s}/hyperparam_grids'.format(experiment_dir_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    for m in range(num_metrics):
        if numpy.all(numpy.isnan(metric_matrix[..., m])):
            continue

        panel_file_names = [''] * length_axis3

        for k in range(length_axis3):
            if '_mf' in METRIC_NAMES[m]:
                max_colour_value = _finite_percentile(
                    numpy.absolute(metric_matrix[..., m]), 100
                )
                min_colour_value = _finite_percentile(
                    numpy.absolute(metric_matrix[..., m]), 5
                )

                colour_norm_object = matplotlib.colors.Normalize(
                    vmin=min_colour_value, vmax=max_colour_value, clip=False
                )
                colour_map_object = MONO_FRACTION_COLOUR_MAP_OBJECT

                best_linear_index = numpy.nanargmax(
                    numpy.ravel(metric_matrix[..., m])
                )
                marker_colour = BLACK_COLOUR

            elif '_ssrat' in METRIC_NAMES[m]:
                if not numpy.any(metric_matrix[..., m] > 1):
                    max_colour_value = _finite_percentile(
                        numpy.absolute(metric_matrix[..., m]), 100
                    )
                    min_colour_value = _finite_percentile(
                        numpy.absolute(metric_matrix[..., m]), 5
                    )

                    colour_norm_object = matplotlib.colors.Normalize(
                        vmin=min_colour_value, vmax=max_colour_value, clip=False
                    )
                    colour_map_object = MONO_FRACTION_COLOUR_MAP_OBJECT
                else:
                    this_offset = _finite_percentile(
                        numpy.absolute(metric_matrix[..., m] - 1.), 97.5
                    )
                    colour_map_object, colour_norm_object = (
                        _get_ssrat_colour_scheme(
                            max_colour_value=1. + this_offset
                        )
                    )

                best_linear_index = numpy.nanargmin(
                    numpy.absolute(numpy.ravel(metric_matrix[..., m]) - 1.)
                )
                marker_colour = BLACK_COLOUR

            else:
                max_colour_value = _finite_percentile(
                    numpy.absolute(metric_matrix[..., m]), 95
                )
                min_colour_value = _finite_percentile(
                    numpy.absolute(metric_matrix[..., m]), 0
                )

                colour_norm_object = matplotlib.colors.Normalize(
                    vmin=min_colour_value, vmax=max_colour_value, clip=False
                )
                colour_map_object = MAIN_COLOUR_MAP_OBJECT

                best_linear_index = numpy.nanargmin(
                    numpy.ravel(metric_matrix[..., m])
                )
                marker_colour = WHITE_COLOUR

            figure_object, axes_object = _plot_scores_2d(
                score_matrix=metric_matrix[..., k, m],
                colour_map_object=colour_map_object,
                colour_norm_object=colour_norm_object,
                x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
            )

            best_indices = numpy.unravel_index(
                best_linear_index, metric_matrix[..., m].shape
            )

            figure_width_px = (
                figure_object.get_size_inches()[0] * figure_object.dpi
            )
            marker_size_px = figure_width_px * (
                BEST_MARKER_SIZE_GRID_CELLS / metric_matrix.shape[2]
            )

            if best_indices[2] == k:
                axes_object.plot(
                    best_indices[1], best_indices[0],
                    linestyle='None', marker=BEST_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=marker_colour,
                    markeredgecolor=marker_colour
                )

            if trained_with_clean_data:
                selected_marker_indices = SELECTED_MARKER_INDICES_CLEAN_TRAINED
            else:
                selected_marker_indices = SELECTED_MARKER_INDICES_LP_TRAINED

            if selected_marker_indices[2] == k:
                axes_object.plot(
                    selected_marker_indices[1], selected_marker_indices[0],
                    linestyle='None', marker=SELECTED_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=marker_colour,
                    markeredgecolor=marker_colour
                )

            axes_object.set_xlabel(x_axis_label)
            axes_object.set_ylabel(y_axis_label)
            axes_object.set_title(
                'Spectral complexity = {0:d}\nTraining method = {1:s}'.format(
                    SPECTRAL_COMPLEXITIES_AXIS3[k],
                    BAYESIAN_LAYER_TYPE_STRINGS_AXIS3[k]
                )
            )

            panel_file_names[k] = (
                '{0:s}/{1:s}_{2:s}_spectral-complexity={3:03d}.jpg'
            ).format(
                output_dir_name,
                METRIC_NAMES[m].replace('_', '-'),
                BAYESIAN_LAYER_TYPE_STRINGS_AXIS3[k],
                SPECTRAL_COMPLEXITIES_AXIS3[k]
            )

            print('Saving figure to: "{0:s}"...'.format(panel_file_names[k]))
            figure_object.savefig(
                panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
                pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(figure_object)

        num_panel_rows = int(numpy.floor(
            numpy.sqrt(length_axis3)
        ))
        num_panel_columns = int(numpy.ceil(
            float(length_axis3) / num_panel_rows
        ))
        concat_figure_file_name = '{0:s}/{1:s}.jpg'.format(
            output_dir_name, METRIC_NAMES[m]
        )

        print('Concatenating panels to: "{0:s}"...'.format(
            concat_figure_file_name
        ))
        imagemagick_utils.concatenate_images(
            input_file_names=panel_file_names,
            output_file_name=concat_figure_file_name,
            num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
        )
        imagemagick_utils.resize_image(
            input_file_name=concat_figure_file_name,
            output_file_name=concat_figure_file_name,
            output_size_pixels=int(1e7)
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        experiment_dir_name=getattr(INPUT_ARG_OBJECT, EXPERIMENT_DIR_ARG_NAME),
        trained_with_clean_data=bool(
            getattr(INPUT_ARG_OBJECT, TRAINED_WITH_CLEAN_DATA_ARG_NAME)
        )
    )
