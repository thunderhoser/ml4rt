"""Evaluation methods for uncertainty quantification (UQ)."""

import os
import numpy
import xarray
from scipy.integrate import simps
from scipy.stats import percentileofscore
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4rt.io import example_io
from ml4rt.io import prediction_io
from ml4rt.utils import example_utils
from ml4rt.machine_learning import neural_net

# MAX_NUM_CLIMO_EXAMPLES = 10000
NUM_EXAMPLES_PER_BATCH = 1000

SHORTWAVE_NET_FLUX_NAME = 'net_shortwave_flux_w_m02'
SHORTWAVE_RAW_FLUX_NAMES = [
    example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME,
    example_utils.SHORTWAVE_TOA_UP_FLUX_NAME
]

LONGWAVE_NET_FLUX_NAME = 'net_longwave_flux_w_m02'
LONGWAVE_RAW_FLUX_NAMES = [
    example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME,
    example_utils.LONGWAVE_TOA_UP_FLUX_NAME
]

MEAN_PREDICTION_STDEVS_KEY = 'mean_prediction_stdevs'
BIN_EDGE_PREDICTION_STDEVS_KEY = 'bin_edge_prediction_stdevs'
RMSE_VALUES_KEY = 'rmse_values'
SPREAD_SKILL_RELIABILITY_KEY = 'spread_skill_reliability'
SPREAD_SKILL_RATIO_KEY = 'spread_skill_ratio'
EXAMPLE_COUNTS_KEY = 'example_counts'
MEAN_MEAN_PREDICTIONS_KEY = 'mean_mean_predictions'
MEAN_TARGET_VALUES_KEY = 'mean_target_values'

AUX_TARGET_NAMES_KEY = 'aux_target_field_names'
AUX_PREDICTED_NAMES_KEY = 'aux_predicted_field_names'
AUX_TARGET_VALS_KEY = 'aux_target_matrix'
AUX_PREDICTED_VALS_KEY = 'aux_prediction_matrix'
SHORTWAVE_SURFACE_DOWN_FLUX_INDEX_KEY = 'shortwave_surface_down_flux_index'
SHORTWAVE_TOA_UP_FLUX_INDEX_KEY = 'shortwave_toa_up_flux_index'
LONGWAVE_SURFACE_DOWN_FLUX_INDEX_KEY = 'longwave_surface_down_flux_index'
LONGWAVE_TOA_UP_FLUX_INDEX_KEY = 'longwave_toa_up_flux_index'

SCALAR_FIELD_DIM = 'scalar_field'
VECTOR_FIELD_DIM = 'vector_field'
HEIGHT_DIM = 'height_m_agl'
AUX_TARGET_FIELD_DIM = 'aux_target_field'
AUX_PREDICTED_FIELD_DIM = 'aux_predicted_field'
RAW_FLUX_BIN_DIM = 'raw_flux_stdev_bin'
RAW_FLUX_BIN_EDGE_DIM = 'raw_flux_stdev_bin_edge'
NET_FLUX_BIN_DIM = 'net_flux_stdev_bin'
NET_FLUX_BIN_EDGE_DIM = 'net_flux_stdev_bin_edge'
HEATING_RATE_BIN_DIM = 'heating_rate_stdev_bin'
HEATING_RATE_BIN_EDGE_DIM = 'heating_rate_stdev_bin_edge'

SCALAR_MEAN_STDEV_KEY = 'scalar_mean_prediction_stdev'
SCALAR_BIN_EDGE_KEY = 'scalar_bin_edge_prediction_stdev'
SCALAR_RMSE_KEY = 'scalar_rmse'
SCALAR_SSREL_KEY = 'scalar_spread_skill_reliability'
SCALAR_SSRAT_KEY = 'scalar_spread_skill_ratio'
SCALAR_EXAMPLE_COUNT_KEY = 'scalar_example_count'
SCALAR_MEAN_MEAN_PREDICTION_KEY = 'scalar_mean_mean_prediction'
SCALAR_MEAN_TARGET_KEY = 'scalar_mean_target_value'
VECTOR_MEAN_STDEV_KEY = 'vector_mean_prediction_stdev'
VECTOR_BIN_EDGE_KEY = 'vector_bin_edge_prediction_stdev'
VECTOR_RMSE_KEY = 'vector_rmse'
VECTOR_SSREL_KEY = 'vector_spread_skill_reliability'
VECTOR_SSRAT_KEY = 'vector_spread_skill_ratio'
VECTOR_EXAMPLE_COUNT_KEY = 'vector_example_count'
VECTOR_MEAN_MEAN_PREDICTION_KEY = 'vector_mean_mean_prediction'
VECTOR_MEAN_TARGET_KEY = 'vector_mean_target_value'
AUX_MEAN_STDEV_KEY = 'aux_mean_prediction_stdev'
AUX_BIN_EDGE_KEY = 'aux_bin_edge_prediction_stdev'
AUX_RMSE_KEY = 'aux_rmse'
AUX_SSREL_KEY = 'aux_spread_skill_reliability'
AUX_SSRAT_KEY = 'aux_spread_skill_ratio'
AUX_EXAMPLE_COUNT_KEY = 'aux_example_count'
AUX_MEAN_MEAN_PREDICTION_KEY = 'aux_mean_mean_prediction'
AUX_MEAN_TARGET_KEY = 'aux_mean_target_value'

MODEL_FILE_KEY = 'model_file_name'
PREDICTION_FILE_KEY = 'prediction_file_name'

DISCARD_FRACTION_DIM = 'discard_fraction'
POST_DISCARD_ERROR_KEY = 'post_discard_error'
EXAMPLE_FRACTION_KEY = 'example_fraction'
MONOTONICITY_FRACTION_KEY = 'monotonicity_fraction'
MEAN_DISCARD_IMPROVEMENT_KEY = 'mean_discard_improvement'
SCALAR_POST_DISCARD_ERROR_KEY = 'scalar_post_discard_error'
SCALAR_MONOTONICITY_FRACTION_KEY = 'scalar_monotonicity_fraction'
SCALAR_MEAN_DISCARD_IMPROVEMENT_KEY = 'scalar_mean_discard_improvement'
VECTOR_POST_DISCARD_ERROR_KEY = 'vector_post_discard_error'
VECTOR_MONOTONICITY_FRACTION_KEY = 'vector_monotonicity_fraction'
VECTOR_MEAN_DISCARD_IMPROVEMENT_KEY = 'vector_mean_discard_improvement'
AUX_POST_DISCARD_ERROR_KEY = 'aux_post_discard_error'
AUX_MONOTONICITY_FRACTION_KEY = 'aux_monotonicity_fraction'
AUX_MEAN_DISCARD_IMPROVEMENT_KEY = 'aux_mean_discard_improvement'

SCALAR_CRPS_KEY = 'scalar_crps'
VECTOR_CRPS_KEY = 'vector_crps'
AUX_CRPS_KEY = 'aux_crps'
SCALAR_CRPSS_KEY = 'scalar_crpss'
VECTOR_CRPSS_KEY = 'vector_crpss'
AUX_CRPSS_KEY = 'aux_crpss'

PIT_HISTOGRAM_BIN_DIM = 'pit_histogram_bin_center'
PIT_HISTOGRAM_BIN_EDGE_DIM = 'pit_histogram_bin_edge'

SCALAR_PITD_KEY = 'scalar_pitd'
SCALAR_PERFECT_PITD_KEY = 'scalar_perfect_pitd'
SCALAR_PIT_BIN_COUNT_KEY = 'scalar_pit_hist_bin_count'
VECTOR_PITD_KEY = 'vector_pitd'
VECTOR_PERFECT_PITD_KEY = 'vector_perfect_pitd'
VECTOR_PIT_BIN_COUNT_KEY = 'vector_pit_hist_bin_count'
AUX_PITD_KEY = 'aux_pitd'
AUX_PERFECT_PITD_KEY = 'aux_perfect_pitd'
AUX_PIT_BIN_COUNT_KEY = 'aux_pit_hist_bin_count'


def _get_aux_fields(prediction_dict, example_dict):
    """Returns auxiliary fields.

    F = number of pairs of auxiliary fields
    E = number of examples
    S = number of ensemble members

    :param prediction_dict: See doc for `prediction_io.read_file`.
    :param example_dict: Dictionary with the following keys (details for each
        key in documentation for `example_io.read_file`).
    example_dict['scalar_target_names']

    :return: aux_prediction_dict: Dictionary with the following keys.
    aux_prediction_dict['aux_target_field_names']: length-F list with names of
        target fields.
    aux_prediction_dict['aux_predicted_field_names']: length-F list with names
        of predicted fields.
    aux_prediction_dict['aux_target_matrix']: E-by-F numpy array of target
        (actual) values.
    aux_prediction_dict['aux_prediction_matrix']: E-by-F numpy array of
        predicted values.
    aux_prediction_dict['shortwave_surface_down_flux_index']: Array index of
        shortwave surface downwelling flux in `mean_training_example_dict`.  If
        surface downwelling flux is not available, this is -1.
    aux_prediction_dict['longwave_surface_down_flux_index']: Same but for
        longwave.
    aux_prediction_dict['shortwave_toa_up_flux_index']: Array index of shortwave
        TOA upwelling flux in `mean_training_example_dict`.  If TOA upwelling
        flux is not available, this is -1.
    aux_prediction_dict['longwave_toa_up_flux_index']: Same but for longwave.
    """

    scalar_target_matrix = prediction_dict[prediction_io.SCALAR_TARGETS_KEY]
    scalar_prediction_matrix = (
        prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY]
    )

    num_examples = scalar_prediction_matrix.shape[0]
    num_ensemble_members = scalar_prediction_matrix.shape[-1]

    aux_target_matrix = numpy.full((num_examples, 0), numpy.nan)
    aux_prediction_matrix = numpy.full(
        (num_examples, 0, num_ensemble_members), numpy.nan
    )
    aux_target_field_names = []
    aux_predicted_field_names = []

    shortwave_surface_down_flux_index = -1
    shortwave_toa_up_flux_index = -1
    longwave_surface_down_flux_index = -1
    longwave_toa_up_flux_index = -1

    scalar_target_names = example_dict[example_utils.SCALAR_TARGET_NAMES_KEY]
    these_flux_names = [
        example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME,
        example_utils.SHORTWAVE_TOA_UP_FLUX_NAME
    ]

    if all([n in scalar_target_names for n in these_flux_names]):
        shortwave_surface_down_flux_index = scalar_target_names.index(
            example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME
        )
        shortwave_toa_up_flux_index = scalar_target_names.index(
            example_utils.SHORTWAVE_TOA_UP_FLUX_NAME
        )

        aux_target_field_names.append(SHORTWAVE_NET_FLUX_NAME)
        aux_predicted_field_names.append(SHORTWAVE_NET_FLUX_NAME)

        this_target_matrix = (
            scalar_target_matrix[:, [shortwave_surface_down_flux_index]] -
            scalar_target_matrix[:, [shortwave_toa_up_flux_index]]
        )
        aux_target_matrix = numpy.concatenate(
            (aux_target_matrix, this_target_matrix), axis=1
        )

        this_prediction_matrix = (
            scalar_prediction_matrix[:, [shortwave_surface_down_flux_index], :]
            - scalar_prediction_matrix[:, [shortwave_toa_up_flux_index], :]
        )
        aux_prediction_matrix = numpy.concatenate(
            (aux_prediction_matrix, this_prediction_matrix), axis=1
        )

    these_flux_names = [
        example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME,
        example_utils.LONGWAVE_TOA_UP_FLUX_NAME
    ]

    if all([n in scalar_target_names for n in these_flux_names]):
        longwave_surface_down_flux_index = scalar_target_names.index(
            example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME
        )
        longwave_toa_up_flux_index = scalar_target_names.index(
            example_utils.LONGWAVE_TOA_UP_FLUX_NAME
        )

        aux_target_field_names.append(LONGWAVE_NET_FLUX_NAME)
        aux_predicted_field_names.append(LONGWAVE_NET_FLUX_NAME)

        this_target_matrix = (
            scalar_target_matrix[:, [longwave_surface_down_flux_index]] -
            scalar_target_matrix[:, [longwave_toa_up_flux_index]]
        )
        aux_target_matrix = numpy.concatenate(
            (aux_target_matrix, this_target_matrix), axis=1
        )

        this_prediction_matrix = (
            scalar_prediction_matrix[:, [longwave_surface_down_flux_index], :] -
            scalar_prediction_matrix[:, [longwave_toa_up_flux_index], :]
        )
        aux_prediction_matrix = numpy.concatenate(
            (aux_prediction_matrix, this_prediction_matrix), axis=1
        )

    return {
        AUX_TARGET_NAMES_KEY: aux_target_field_names,
        AUX_PREDICTED_NAMES_KEY: aux_predicted_field_names,
        AUX_TARGET_VALS_KEY: aux_target_matrix,
        AUX_PREDICTED_VALS_KEY: aux_prediction_matrix,
        SHORTWAVE_SURFACE_DOWN_FLUX_INDEX_KEY:
            shortwave_surface_down_flux_index,
        SHORTWAVE_TOA_UP_FLUX_INDEX_KEY: shortwave_toa_up_flux_index,
        LONGWAVE_SURFACE_DOWN_FLUX_INDEX_KEY: longwave_surface_down_flux_index,
        LONGWAVE_TOA_UP_FLUX_INDEX_KEY: longwave_toa_up_flux_index
    }


def _get_crps_one_var(target_values, prediction_matrix, num_integration_levels):
    """Computes CRPS for one variable.

    CRPS = continuous ranked probability score

    E = number of examples
    S = number of ensemble members

    :param target_values: length-E numpy array of actual values.
    :param prediction_matrix: E-by-S numpy array of predicted values.
    :param num_integration_levels: Will use this many integration levels to
        compute CRPS.
    :return: crps_value: CRPS (scalar).
    """

    num_examples = len(target_values)
    crps_numerator = 0.
    crps_denominator = 0.

    prediction_by_integ_level = numpy.linspace(
        numpy.min(prediction_matrix), numpy.max(prediction_matrix),
        num=num_integration_levels, dtype=float
    )

    for i in range(0, num_examples, NUM_EXAMPLES_PER_BATCH):
        first_index = i
        last_index = min([
            i + NUM_EXAMPLES_PER_BATCH, num_examples
        ])

        cdf_matrix = numpy.stack([
            numpy.mean(
                prediction_matrix[first_index:last_index, :] <= l, axis=-1
            )
            for l in prediction_by_integ_level
        ], axis=-1)

        this_prediction_matrix = numpy.repeat(
            numpy.expand_dims(prediction_by_integ_level, axis=0),
            axis=0, repeats=last_index - first_index
        )
        this_target_matrix = numpy.repeat(
            numpy.expand_dims(target_values[first_index:last_index], axis=-1),
            axis=1, repeats=num_integration_levels
        )
        heaviside_matrix = (
            (this_prediction_matrix >= this_target_matrix).astype(int)
        )

        integrated_cdf_matrix = simps(
            y=(cdf_matrix - heaviside_matrix) ** 2,
            x=prediction_by_integ_level, axis=-1
        )
        crps_numerator += numpy.sum(integrated_cdf_matrix)
        crps_denominator += integrated_cdf_matrix.size

    return crps_numerator / crps_denominator


def _get_climo_crps_one_var(
        new_target_values, training_target_values, num_integration_levels,
        max_ensemble_size):
    """Computes CRPS of climatological model for one variable.

    CRPS = continuous ranked probability score

    E_n = number of examples in new (evaluation) dataset
    E_t = number of examples in training dataset

    :param new_target_values: numpy array (length E_n) of target values.
    :param training_target_values: numpy array (length E_t) of target values.
    :param num_integration_levels: Will use this many integration levels to
        compute CRPS.
    :param max_ensemble_size: Will use this max ensemble size to compute CRPS.
    :return: climo_crps: Climo CRPS (scalar).
    """

    num_new_examples = len(new_target_values)
    num_training_examples = len(training_target_values)
    ensemble_size = min([num_training_examples, max_ensemble_size])

    crps_numerator = 0.
    crps_denominator = 0.

    prediction_by_integ_level = numpy.linspace(
        numpy.min(training_target_values), numpy.max(training_target_values),
        num=num_integration_levels, dtype=float
    )

    for i in range(0, num_new_examples, NUM_EXAMPLES_PER_BATCH):
        first_index = i
        last_index = min([
            i + NUM_EXAMPLES_PER_BATCH, num_new_examples
        ])

        this_num_examples = last_index - first_index
        this_actual_prediction_matrix = numpy.random.choice(
            training_target_values, size=(this_num_examples, ensemble_size),
            replace=True
        )

        cdf_matrix = numpy.stack([
            numpy.mean(this_actual_prediction_matrix <= l, axis=-1)
            for l in prediction_by_integ_level
        ], axis=-1)

        this_integrand_prediction_matrix = numpy.repeat(
            numpy.expand_dims(prediction_by_integ_level, axis=0),
            axis=0, repeats=last_index - first_index
        )
        this_target_matrix = numpy.repeat(
            numpy.expand_dims(
                new_target_values[first_index:last_index], axis=-1
            ),
            axis=1, repeats=num_integration_levels
        )
        heaviside_matrix = (
            (this_integrand_prediction_matrix >= this_target_matrix).astype(int)
        )

        integrated_cdf_matrix = simps(
            y=(cdf_matrix - heaviside_matrix) ** 2,
            x=prediction_by_integ_level, axis=-1
        )
        crps_numerator += numpy.sum(integrated_cdf_matrix)
        crps_denominator += integrated_cdf_matrix.size

    return crps_numerator / crps_denominator


def _get_spread_vs_skill_one_var(
        target_values, prediction_matrix, bin_edge_prediction_stdevs):
    """Computes spread-skill relationship for one target variable.

    E = number of examples
    S = number of ensemble members
    B = number of bins

    :param target_values: length-E numpy array of actual values.
    :param prediction_matrix: E-by-S numpy array of predicted values.
    :param bin_edge_prediction_stdevs: length-(B - 1) numpy array of bin
        cutoffs.  Each is a standard deviation for the predictive distribution.
        Ultimately, there will be B + 1 edges; this method will use 0 as the
        lowest edge and inf as the highest edge.
    :return: result_dict: Dictionary with the following keys.
    result_dict['mean_prediction_stdevs']: length-B numpy array, where the [i]th
        entry is the mean standard deviation of predictive distributions in the
        [i]th bin.
    result_dict['bin_edge_prediction_stdevs']: length-(B + 1) numpy array,
        where the [i]th and [i + 1]th entries are the edges for the [i]th bin.
    result_dict['rmse_values']: length-B numpy array, where the [i]th
        entry is the root mean squared error of mean predictions in the [i]th
        bin.
    result_dict['spread_skill_reliability']: Spread-skill reliability (SSREL).
    result_dict['spread_skill_ratio']: Spread-skill ratio (SSRAT).
    result_dict['example_counts']: length-B numpy array of corresponding example
        counts.
    result_dict['mean_mean_predictions']: length-B numpy array, where the
        [i]th entry is the mean mean prediction for the [i]th bin.
    result_dict['mean_target_values']: length-B numpy array, where the [i]th
        entry is the mean target value for the [i]th bin.
    """

    # Check input args.
    error_checking.assert_is_numpy_array_without_nan(target_values)
    error_checking.assert_is_numpy_array(target_values, num_dimensions=1)

    error_checking.assert_is_numpy_array_without_nan(prediction_matrix)
    error_checking.assert_is_numpy_array(prediction_matrix, num_dimensions=2)

    num_examples = len(target_values)
    num_ensemble_members = prediction_matrix.shape[1]
    error_checking.assert_is_greater(num_ensemble_members, 1)

    these_dim = numpy.array([num_examples, num_ensemble_members], dtype=int)
    error_checking.assert_is_numpy_array(
        prediction_matrix, exact_dimensions=these_dim
    )

    error_checking.assert_is_numpy_array(
        bin_edge_prediction_stdevs, num_dimensions=1
    )
    error_checking.assert_is_greater_numpy_array(bin_edge_prediction_stdevs, 0.)
    error_checking.assert_is_greater_numpy_array(
        numpy.diff(bin_edge_prediction_stdevs), 0.
    )

    bin_edge_prediction_stdevs = numpy.concatenate((
        numpy.array([0.]),
        bin_edge_prediction_stdevs,
        numpy.array([numpy.inf])
    ))

    num_bins = len(bin_edge_prediction_stdevs) - 1
    assert num_bins >= 2

    # Do actual stuff.
    mean_predictions = numpy.mean(prediction_matrix, axis=1)
    predictive_stdevs = numpy.std(prediction_matrix, axis=1, ddof=1)
    squared_errors = (mean_predictions - target_values) ** 2

    mean_prediction_stdevs = numpy.full(num_bins, numpy.nan)
    rmse_values = numpy.full(num_bins, numpy.nan)
    example_counts = numpy.full(num_bins, 0, dtype=int)
    mean_mean_predictions = numpy.full(num_bins, numpy.nan)
    mean_target_values = numpy.full(num_bins, numpy.nan)

    for k in range(num_bins):
        these_indices = numpy.where(numpy.logical_and(
            predictive_stdevs >= bin_edge_prediction_stdevs[k],
            predictive_stdevs < bin_edge_prediction_stdevs[k + 1]
        ))[0]

        mean_prediction_stdevs[k] = numpy.sqrt(numpy.mean(
            predictive_stdevs[these_indices] ** 2
        ))
        rmse_values[k] = numpy.sqrt(numpy.mean(
            squared_errors[these_indices]
        ))

        example_counts[k] = len(these_indices)
        mean_mean_predictions[k] = numpy.mean(mean_predictions[these_indices])
        mean_target_values[k] = numpy.mean(target_values[these_indices])

    these_diffs = numpy.absolute(mean_prediction_stdevs - rmse_values)
    these_diffs[numpy.isnan(these_diffs)] = 0.
    spread_skill_reliability = numpy.average(
        these_diffs, weights=example_counts
    )

    this_numer = numpy.sqrt(numpy.mean(predictive_stdevs ** 2))
    this_denom = numpy.sqrt(numpy.mean(squared_errors))
    spread_skill_ratio = this_numer / this_denom

    return {
        MEAN_PREDICTION_STDEVS_KEY: mean_prediction_stdevs,
        BIN_EDGE_PREDICTION_STDEVS_KEY: bin_edge_prediction_stdevs,
        RMSE_VALUES_KEY: rmse_values,
        SPREAD_SKILL_RELIABILITY_KEY: spread_skill_reliability,
        SPREAD_SKILL_RATIO_KEY: spread_skill_ratio,
        EXAMPLE_COUNTS_KEY: example_counts,
        MEAN_MEAN_PREDICTIONS_KEY: mean_mean_predictions,
        MEAN_TARGET_VALUES_KEY: mean_target_values
    }


def _get_pit_histogram_one_var(target_values, prediction_matrix, num_bins):
    """Computes PIT (probability integral transform) histogram for one variable.

    E = number of examples
    S = number of ensemble members
    B = number of bins in histogram

    :param target_values: length-E numpy array of actual values.
    :param prediction_matrix: E-by-S numpy array of predicted values.
    :param num_bins: Number of bins in histogram.
    :return: bin_edges: length-(B + 1) numpy array of bin edges (ranging from
        0...1, because PIT ranges from 0...1).
    :return: bin_counts: length-B numpy array with number of examples in each
        bin.
    :return: pitd_value: Value of the calibration-deviation metric (PITD).
    :return: perfect_pitd_value: Minimum expected PITD value.
    """

    num_examples = len(target_values)
    pit_values = numpy.full(num_examples, numpy.nan)

    for i in range(num_examples):
        pit_values[i] = 0.01 * percentileofscore(
            a=prediction_matrix[i, :], score=target_values[i], kind='mean'
        )

    bin_edges = numpy.linspace(0, 1, num=num_bins + 1, dtype=float)
    indices_example_to_bin = numpy.digitize(
        x=pit_values, bins=bin_edges, right=False
    ) - 1
    indices_example_to_bin[indices_example_to_bin < 0] = 0
    indices_example_to_bin[indices_example_to_bin >= num_bins] = num_bins - 1

    used_bin_indices, used_bin_counts = numpy.unique(
        indices_example_to_bin, return_counts=True
    )
    bin_counts = numpy.full(num_bins, 0, dtype=int)
    bin_counts[used_bin_indices] = used_bin_counts

    bin_frequencies = bin_counts.astype(float) / num_examples
    perfect_bin_frequency = 1. / num_bins

    pitd_value = numpy.sqrt(
        numpy.mean((bin_frequencies - perfect_bin_frequency) ** 2)
    )
    perfect_pitd_value = numpy.sqrt(
        (1. - perfect_bin_frequency) / (num_examples * num_bins)
    )

    return bin_edges, bin_counts, pitd_value, perfect_pitd_value


def make_heating_rate_stdev_function():
    """Makes function to compute stdev uncertainty of heating rates.

    :return: uncertainty_function: Function handle.
    """

    def uncertainty_function(prediction_dict):
        """Computes stdev uncertainty of heating rates for each example.

        E = number of examples

        :param prediction_dict: Dictionary in format returned by
            `prediction_io.read_file`.
        :return: uncertainty_values: length-E numpy array of uncertainty values.
        """

        model_file_name = prediction_dict[prediction_io.MODEL_FILE_KEY]
        model_metafile_name = neural_net.find_metafile(
            model_dir_name=os.path.split(model_file_name)[0],
            raise_error_if_missing=True
        )

        model_metadata_dict = neural_net.read_metafile(model_metafile_name)
        generator_option_dict = (
            model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
        )
        vector_target_names = (
            generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY]
        )

        try:
            hr_index = vector_target_names.index(
                example_utils.SHORTWAVE_HEATING_RATE_NAME
            )
        except ValueError:
            hr_index = vector_target_names.index(
                example_utils.LONGWAVE_HEATING_RATE_NAME
            )

        predicted_hr_matrix_k_day01 = prediction_dict[
            prediction_io.VECTOR_PREDICTIONS_KEY
        ][..., hr_index, :]

        num_ensemble_members = predicted_hr_matrix_k_day01.shape[-1]
        assert num_ensemble_members > 1

        pixelwise_stdev_matrix_k_day01 = numpy.std(
            predicted_hr_matrix_k_day01, ddof=1, axis=-1
        )
        return numpy.sqrt(
            numpy.mean(pixelwise_stdev_matrix_k_day01 ** 2, axis=-1)
        )

    return uncertainty_function


def make_flux_stdev_function():
    """Makes function to compute stdev uncertainty of fluxes.

    :return: uncertainty_function: Function handle.
    """

    def uncertainty_function(prediction_dict):
        """Computes stdev uncertainty of fluxes for each example.

        E = number of examples

        :param prediction_dict: Dictionary in format returned by
            `prediction_io.read_file`.
        :return: uncertainty_values: length-E numpy array of uncertainty values.
        """

        num_ensemble_members = (
            prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY].shape[-1]
        )
        assert num_ensemble_members > 1

        model_file_name = prediction_dict[prediction_io.MODEL_FILE_KEY]
        model_metafile_name = neural_net.find_metafile(
            model_dir_name=os.path.split(model_file_name)[0],
            raise_error_if_missing=True
        )

        model_metadata_dict = neural_net.read_metafile(model_metafile_name)
        generator_option_dict = (
            model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
        )
        scalar_target_names = (
            generator_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY]
        )

        num_examples = (
            prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY].shape[0]
        )
        predicted_flux_matrix_w_m02 = numpy.full(
            (num_examples, 0, num_ensemble_members), numpy.nan
        )

        for this_name in SHORTWAVE_RAW_FLUX_NAMES + LONGWAVE_RAW_FLUX_NAMES:
            if this_name not in scalar_target_names:
                continue

            j = scalar_target_names.index(this_name)

            predicted_flux_matrix_w_m02 = numpy.concatenate((
                predicted_flux_matrix_w_m02,
                prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY][:, [j], :]
            ), axis=1)

        if all([n in scalar_target_names for n in SHORTWAVE_RAW_FLUX_NAMES]):
            down_index = scalar_target_names.index(
                example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME
            )
            up_index = scalar_target_names.index(
                example_utils.SHORTWAVE_TOA_UP_FLUX_NAME
            )
            this_matrix = prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY]

            predicted_flux_matrix_w_m02 = numpy.concatenate((
                predicted_flux_matrix_w_m02,
                this_matrix[:, [down_index], :] - this_matrix[:, [up_index], :]
            ), axis=1)

        elementwise_stdev_matrix_k_day01 = numpy.std(
            predicted_flux_matrix_w_m02, ddof=1, axis=-1
        )
        return numpy.sqrt(
            numpy.mean(elementwise_stdev_matrix_k_day01 ** 2, axis=-1)
        )

    return uncertainty_function


def make_error_function_dwmse_1height():
    """Makes function to compute DWMSE for heating rate at one height.

    DWMSE = dual-weighted mean squared error

    :return: error_function: Function handle.
    """

    def error_function(
            actual_heating_rates_k_day01, predicted_hr_matrix_k_day01,
            use_example_flags):
        """Computes DWMSE.

        E = number of examples
        S = ensemble size

        :param actual_heating_rates_k_day01: length-E numpy array of actual
            heating rates at one height.
        :param predicted_hr_matrix_k_day01: E-by-S numpy array of predicted
            heating rates at the same height.
        :param use_example_flags: length-E numpy array of Boolean flags,
            indicating which examples to use.
        :return: dwmse_k3_day03: Scalar DWMSE value.
        """

        mean_pred_heating_rates_k_day01 = numpy.mean(
            predicted_hr_matrix_k_day01[use_example_flags, :], axis=-1
        )

        weights_k_day01 = numpy.maximum(
            numpy.absolute(mean_pred_heating_rates_k_day01),
            numpy.absolute(actual_heating_rates_k_day01[use_example_flags])
        )
        squared_errors = (
            mean_pred_heating_rates_k_day01 -
            actual_heating_rates_k_day01[use_example_flags]
        ) ** 2

        return numpy.mean(weights_k_day01 * squared_errors)

    return error_function


def make_error_function_flux_mse_1var():
    """Makes function to compute MSE for one flux variable.

    :return: error_function: Function handle.
    """

    def error_function(actual_fluxes_w_m02, predicted_flux_matrix_w_m02,
                       use_example_flags):
        """Computes MSE.

        E = number of examples
        S = ensemble size

        :param actual_fluxes_w_m02: length-E numpy array of actual values for
            one flux variable.
        :param predicted_flux_matrix_w_m02: E-by-S numpy array of predicted
            values for the same flux variable.
        :param use_example_flags: length-E numpy array of Boolean flags,
            indicating which examples to use.
        :return: mse_w2_m04: Scalar MSE value.
        """

        mean_pred_fluxes_w_m02 = numpy.mean(
            predicted_flux_matrix_w_m02[use_example_flags, :], axis=-1
        )
        return numpy.mean(
            (mean_pred_fluxes_w_m02 - actual_fluxes_w_m02[use_example_flags])
            ** 2
        )

    return error_function


def make_error_function_dwmse_plus_flux_mse(scaling_factor_for_dwmse,
                                            scaling_factor_for_flux_mse):
    """Makes function to compute total error.

    Total error = (scaling_factor_for_dwmse * DWMSE) +
                  (scaling_factor_for_flux_mse * MSE_flux)

    DWMSE = dual-weighted mean squared error for heating rates

    :param scaling_factor_for_dwmse: See above.
    :param scaling_factor_for_flux_mse: See above.
    :return: error_function: Function handle.
    """

    error_checking.assert_is_geq(scaling_factor_for_dwmse, 0.)
    error_checking.assert_is_geq(scaling_factor_for_flux_mse, 0.)

    def error_function(prediction_dict, use_example_flags):
        """Computes total error.

        E = number of examples

        :param prediction_dict: Dictionary in format returned by
            `prediction_io.read_file`.
        :param use_example_flags: length-E numpy array of Boolean flags,
            indicating which examples to use.
        :return: total_error: Scalar error value.
        """

        predicted_flux_matrix_w_m02 = numpy.mean(
            prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY][
                use_example_flags, ...
            ],
            axis=-1
        )
        actual_flux_matrix_w_m02 = prediction_dict[
            prediction_io.SCALAR_TARGETS_KEY
        ][use_example_flags, :]

        predicted_net_flux_matrix_w_m02 = (
            predicted_flux_matrix_w_m02[:, 0] -
            predicted_flux_matrix_w_m02[:, 1]
        )
        actual_net_flux_matrix_w_m02 = (
            actual_flux_matrix_w_m02[:, 0] -
            actual_flux_matrix_w_m02[:, 1]
        )

        net_flux_sse_w2_m04 = numpy.sum(
            (predicted_net_flux_matrix_w_m02 - actual_net_flux_matrix_w_m02)
            ** 2
        )
        raw_flux_sse_w2_m04 = numpy.sum(
            (predicted_flux_matrix_w_m02 - actual_flux_matrix_w_m02) ** 2
        )

        num_examples = actual_flux_matrix_w_m02.shape[0]
        flux_mse_w_m02 = (
            (net_flux_sse_w2_m04 + raw_flux_sse_w2_m04) / (3 * num_examples)
        )

        predicted_hr_matrix_k_day01 = numpy.mean(
            prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY][
                use_example_flags, ...
            ],
            axis=-1
        )
        actual_hr_matrix_k_day01 = prediction_dict[
            prediction_io.VECTOR_TARGETS_KEY
        ][use_example_flags, ...]

        weight_matrix_k_day01 = numpy.maximum(
            numpy.absolute(predicted_hr_matrix_k_day01),
            numpy.absolute(actual_hr_matrix_k_day01)
        )
        heating_rate_dwmse_k3_day03 = numpy.mean(
            weight_matrix_k_day01 *
            (predicted_hr_matrix_k_day01 - actual_hr_matrix_k_day01) ** 2
        )

        return (
            scaling_factor_for_dwmse * heating_rate_dwmse_k3_day03 +
            scaling_factor_for_flux_mse * flux_mse_w_m02
        )

    return error_function


def get_crps_all_vars(prediction_file_name, num_integration_levels,
                      ensemble_size_for_climo):
    """Computes continuous ranked probability (CRPS) for all target variables.

    :param prediction_file_name: Path to input file (will be read by
        `prediction_io.read_file`).
    :param num_integration_levels: See doc for `_get_crps_one_var`.
    :param ensemble_size_for_climo: Ensemble size used to compute CRPS of
        climatological model.
    :return: result_table_xarray: xarray table with results (variable and
        dimension names should make the table self-explanatory).
    """

    error_checking.assert_is_integer(num_integration_levels)
    error_checking.assert_is_geq(num_integration_levels, 100)
    error_checking.assert_is_leq(num_integration_levels, 100000)

    error_checking.assert_is_integer(ensemble_size_for_climo)
    error_checking.assert_is_geq(ensemble_size_for_climo, 100)
    error_checking.assert_is_leq(ensemble_size_for_climo, 10000)

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(prediction_file_name)

    model_file_name = prediction_dict[prediction_io.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    generator_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    normalization_file_name = (
        generator_option_dict[neural_net.NORMALIZATION_FILE_KEY]
    )

    print('Reading data from: "{0:s}"...'.format(normalization_file_name))
    training_example_dict = example_io.read_file(normalization_file_name)

    # num_training_examples = len(
    #     training_example_dict[example_utils.EXAMPLE_IDS_KEY]
    # )
    #
    # if num_training_examples > MAX_NUM_CLIMO_EXAMPLES:
    #     these_indices = numpy.linspace(
    #         0, num_training_examples - 1, num=num_training_examples, dtype=int
    #     )
    #     these_indices = numpy.random.choice(
    #         these_indices, size=MAX_NUM_CLIMO_EXAMPLES, replace=False
    #     )
    #     training_example_dict = example_utils.subset_by_index(
    #         example_dict=training_example_dict, desired_indices=these_indices
    #     )

    heights_m_agl = prediction_dict[prediction_io.HEIGHTS_KEY]
    example_dict = {
        example_utils.SCALAR_TARGET_NAMES_KEY:
            generator_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY],
        example_utils.VECTOR_TARGET_NAMES_KEY:
            generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY],
        example_utils.SCALAR_PREDICTOR_NAMES_KEY:
            generator_option_dict[neural_net.SCALAR_PREDICTOR_NAMES_KEY],
        example_utils.VECTOR_PREDICTOR_NAMES_KEY:
            generator_option_dict[neural_net.VECTOR_PREDICTOR_NAMES_KEY],
        example_utils.HEIGHTS_KEY: heights_m_agl
    }

    aux_prediction_dict = _get_aux_fields(
        prediction_dict=prediction_dict, example_dict=example_dict
    )
    aux_target_field_names = aux_prediction_dict[AUX_TARGET_NAMES_KEY]
    aux_predicted_field_names = aux_prediction_dict[AUX_PREDICTED_NAMES_KEY]
    aux_target_matrix = aux_prediction_dict[AUX_TARGET_VALS_KEY]
    aux_prediction_matrix = aux_prediction_dict[AUX_PREDICTED_VALS_KEY]

    scalar_target_matrix = prediction_dict[prediction_io.SCALAR_TARGETS_KEY]
    scalar_prediction_matrix = (
        prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY]
    )
    vector_target_matrix = prediction_dict[prediction_io.VECTOR_TARGETS_KEY]
    vector_prediction_matrix = (
        prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY]
    )

    num_heights = vector_target_matrix.shape[1]
    num_vector_targets = vector_target_matrix.shape[2]
    num_scalar_targets = scalar_target_matrix.shape[1]
    num_aux_targets = len(aux_target_field_names)

    main_data_dict = {
        SCALAR_CRPS_KEY: (
            (SCALAR_FIELD_DIM,), numpy.full(num_scalar_targets, numpy.nan)
        ),
        SCALAR_CRPSS_KEY: (
            (SCALAR_FIELD_DIM,), numpy.full(num_scalar_targets, numpy.nan)
        ),
        VECTOR_CRPS_KEY: (
            (VECTOR_FIELD_DIM, HEIGHT_DIM),
            numpy.full((num_vector_targets, num_heights), numpy.nan)
        ),
        VECTOR_CRPSS_KEY: (
            (VECTOR_FIELD_DIM, HEIGHT_DIM),
            numpy.full((num_vector_targets, num_heights), numpy.nan)
        )
    }

    if num_aux_targets > 0:
        main_data_dict.update({
            AUX_CRPS_KEY: (
                (AUX_TARGET_FIELD_DIM,), numpy.full(num_aux_targets, numpy.nan)
            ),
            AUX_CRPSS_KEY: (
                (AUX_TARGET_FIELD_DIM,), numpy.full(num_aux_targets, numpy.nan)
            )
        })

    metadata_dict = {
        SCALAR_FIELD_DIM: example_dict[example_utils.SCALAR_TARGET_NAMES_KEY],
        HEIGHT_DIM: heights_m_agl,
        VECTOR_FIELD_DIM: example_dict[example_utils.VECTOR_TARGET_NAMES_KEY]
    }

    if num_aux_targets > 0:
        metadata_dict[AUX_TARGET_FIELD_DIM] = aux_target_field_names
        metadata_dict[AUX_PREDICTED_FIELD_DIM] = aux_predicted_field_names

    result_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict
    )
    result_table_xarray.attrs[MODEL_FILE_KEY] = model_file_name
    result_table_xarray.attrs[PREDICTION_FILE_KEY] = prediction_file_name

    for j in range(num_scalar_targets):
        print('Computing CRPS for {0:s}...'.format(
            example_dict[example_utils.SCALAR_TARGET_NAMES_KEY][j]
        ))

        result_table_xarray[SCALAR_CRPS_KEY].values[j] = _get_crps_one_var(
            target_values=scalar_target_matrix[:, j],
            prediction_matrix=scalar_prediction_matrix[:, j, :],
            num_integration_levels=num_integration_levels
        )

        print('Computing CRPSS for {0:s}...'.format(
            example_dict[example_utils.SCALAR_TARGET_NAMES_KEY][j]
        ))

        these_training_values = example_utils.get_field_from_dict(
            example_dict=training_example_dict,
            field_name=example_dict[example_utils.SCALAR_TARGET_NAMES_KEY][j]
        )
        this_climo_crps = _get_climo_crps_one_var(
            new_target_values=scalar_target_matrix[:, j],
            training_target_values=these_training_values,
            num_integration_levels=num_integration_levels,
            max_ensemble_size=ensemble_size_for_climo
        )
        this_climo_crps = max([this_climo_crps, 1e-9])

        result_table_xarray[SCALAR_CRPSS_KEY].values[j] = (
            1. - result_table_xarray[SCALAR_CRPS_KEY].values[j] /
            this_climo_crps
        )

    for j in range(num_vector_targets):
        for k in range(num_heights):
            print('Computing CRPS for {0:s} at {1:d} m AGL...'.format(
                example_dict[example_utils.VECTOR_TARGET_NAMES_KEY][j],
                int(numpy.round(heights_m_agl[k]))
            ))

            result_table_xarray[VECTOR_CRPS_KEY].values[j, k] = (
                _get_crps_one_var(
                    target_values=vector_target_matrix[:, k, j],
                    prediction_matrix=vector_prediction_matrix[:, k, j, :],
                    num_integration_levels=num_integration_levels
                )
            )

            print('Computing CRPSS for {0:s} at {1:d} m AGL...'.format(
                example_dict[example_utils.VECTOR_TARGET_NAMES_KEY][j],
                int(numpy.round(heights_m_agl[k]))
            ))

            these_training_values = example_utils.get_field_from_dict(
                example_dict=training_example_dict,
                field_name=
                example_dict[example_utils.VECTOR_TARGET_NAMES_KEY][j],
                height_m_agl=heights_m_agl[k]
            )
            this_climo_crps = _get_climo_crps_one_var(
                new_target_values=vector_target_matrix[:, k, j],
                training_target_values=these_training_values,
                num_integration_levels=num_integration_levels,
                max_ensemble_size=ensemble_size_for_climo
            )
            this_climo_crps = max([this_climo_crps, 1e-9])

            result_table_xarray[VECTOR_CRPSS_KEY].values[j, k] = (
                1. - result_table_xarray[VECTOR_CRPS_KEY].values[j, k] /
                this_climo_crps
            )

    for j in range(num_aux_targets):
        print('Computing CRPS for {0:s}...'.format(aux_target_field_names[j]))

        result_table_xarray[AUX_CRPS_KEY].values[j] = _get_crps_one_var(
            target_values=aux_target_matrix[:, j],
            prediction_matrix=aux_prediction_matrix[:, j, :],
            num_integration_levels=num_integration_levels
        )

        print('Computing CRPSS for {0:s}...'.format(aux_target_field_names[j]))

        if aux_target_field_names[j] == SHORTWAVE_NET_FLUX_NAME:
            these_down_fluxes_w_m02 = example_utils.get_field_from_dict(
                example_dict=training_example_dict,
                field_name=example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME
            )
            these_up_fluxes_w_m02 = example_utils.get_field_from_dict(
                example_dict=training_example_dict,
                field_name=example_utils.SHORTWAVE_TOA_UP_FLUX_NAME
            )
            these_training_values = (
                these_down_fluxes_w_m02 - these_up_fluxes_w_m02
            )
        elif aux_target_field_names[j] == LONGWAVE_NET_FLUX_NAME:
            these_down_fluxes_w_m02 = example_utils.get_field_from_dict(
                example_dict=training_example_dict,
                field_name=example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME
            )
            these_up_fluxes_w_m02 = example_utils.get_field_from_dict(
                example_dict=training_example_dict,
                field_name=example_utils.LONGWAVE_TOA_UP_FLUX_NAME
            )
            these_training_values = (
                these_down_fluxes_w_m02 - these_up_fluxes_w_m02
            )

        this_climo_crps = _get_climo_crps_one_var(
            new_target_values=aux_target_matrix[:, j],
            training_target_values=these_training_values,
            num_integration_levels=num_integration_levels,
            max_ensemble_size=ensemble_size_for_climo
        )
        this_climo_crps = max([this_climo_crps, 1e-9])

        result_table_xarray[AUX_CRPSS_KEY].values[j] = (
            1. - result_table_xarray[AUX_CRPS_KEY].values[j] / this_climo_crps
        )

    return result_table_xarray


def get_pit_histogram_all_vars(prediction_file_name, num_bins):
    """Computes PIT (prob integral transform) histo for all target variables.

    :param prediction_file_name: Path to input file (will be read by
        `prediction_io.read_file`).
    :param num_bins: Number of bins per histogram.
    :return: result_table_xarray: xarray table with results (variable and
        dimension names should make the table self-explanatory).
    """

    error_checking.assert_is_integer(num_bins)
    error_checking.assert_is_geq(num_bins, 10)
    error_checking.assert_is_leq(num_bins, 1000)

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(prediction_file_name)

    model_file_name = prediction_dict[prediction_io.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    generator_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    heights_m_agl = prediction_dict[prediction_io.HEIGHTS_KEY]

    example_dict = {
        example_utils.SCALAR_TARGET_NAMES_KEY:
            generator_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY],
        example_utils.VECTOR_TARGET_NAMES_KEY:
            generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY],
        example_utils.SCALAR_PREDICTOR_NAMES_KEY:
            generator_option_dict[neural_net.SCALAR_PREDICTOR_NAMES_KEY],
        example_utils.VECTOR_PREDICTOR_NAMES_KEY:
            generator_option_dict[neural_net.VECTOR_PREDICTOR_NAMES_KEY],
        example_utils.HEIGHTS_KEY: heights_m_agl
    }

    aux_prediction_dict = _get_aux_fields(
        prediction_dict=prediction_dict, example_dict=example_dict
    )
    aux_target_field_names = aux_prediction_dict[AUX_TARGET_NAMES_KEY]
    aux_predicted_field_names = aux_prediction_dict[AUX_PREDICTED_NAMES_KEY]
    aux_target_matrix = aux_prediction_dict[AUX_TARGET_VALS_KEY]
    aux_prediction_matrix = aux_prediction_dict[AUX_PREDICTED_VALS_KEY]

    scalar_target_matrix = prediction_dict[prediction_io.SCALAR_TARGETS_KEY]
    scalar_prediction_matrix = (
        prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY]
    )
    vector_target_matrix = prediction_dict[prediction_io.VECTOR_TARGETS_KEY]
    vector_prediction_matrix = (
        prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY]
    )

    num_heights = vector_target_matrix.shape[1]
    num_vector_targets = vector_target_matrix.shape[2]
    num_scalar_targets = scalar_target_matrix.shape[1]
    num_aux_targets = len(aux_target_field_names)

    main_data_dict = {
        SCALAR_PITD_KEY: (
            (SCALAR_FIELD_DIM,), numpy.full(num_scalar_targets, numpy.nan)
        ),
        SCALAR_PERFECT_PITD_KEY: (
            (SCALAR_FIELD_DIM,), numpy.full(num_scalar_targets, numpy.nan)
        ),
        SCALAR_PIT_BIN_COUNT_KEY: (
            (SCALAR_FIELD_DIM, PIT_HISTOGRAM_BIN_DIM),
            numpy.full((num_scalar_targets, num_bins), -1, dtype=int)
        ),
        VECTOR_PITD_KEY: (
            (VECTOR_FIELD_DIM, HEIGHT_DIM),
            numpy.full((num_vector_targets, num_heights), numpy.nan)
        ),
        VECTOR_PERFECT_PITD_KEY: (
            (VECTOR_FIELD_DIM, HEIGHT_DIM),
            numpy.full((num_vector_targets, num_heights), numpy.nan)
        ),
        VECTOR_PIT_BIN_COUNT_KEY: (
            (VECTOR_FIELD_DIM, HEIGHT_DIM, PIT_HISTOGRAM_BIN_DIM),
            numpy.full(
                (num_vector_targets, num_heights, num_bins), -1, dtype=int
            )
        ),
    }

    if num_aux_targets > 0:
        main_data_dict.update({
            AUX_PITD_KEY: (
                (AUX_TARGET_FIELD_DIM,), numpy.full(num_aux_targets, numpy.nan)
            ),
            AUX_PERFECT_PITD_KEY: (
                (AUX_TARGET_FIELD_DIM,), numpy.full(num_aux_targets, numpy.nan)
            ),
            AUX_PIT_BIN_COUNT_KEY: (
                (AUX_TARGET_FIELD_DIM, PIT_HISTOGRAM_BIN_DIM),
                numpy.full((num_aux_targets, num_bins), -1, dtype=int)
            )
        })

    bin_edges = numpy.linspace(0, 1, num=num_bins + 1, dtype=float)
    bin_centers = bin_edges[:-1] + numpy.diff(bin_edges) / 2

    metadata_dict = {
        SCALAR_FIELD_DIM: example_dict[example_utils.SCALAR_TARGET_NAMES_KEY],
        HEIGHT_DIM: heights_m_agl,
        VECTOR_FIELD_DIM: example_dict[example_utils.VECTOR_TARGET_NAMES_KEY],
        PIT_HISTOGRAM_BIN_DIM: bin_centers,
        PIT_HISTOGRAM_BIN_EDGE_DIM: bin_edges
    }

    if num_aux_targets > 0:
        metadata_dict[AUX_TARGET_FIELD_DIM] = aux_target_field_names
        metadata_dict[AUX_PREDICTED_FIELD_DIM] = aux_predicted_field_names

    result_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict
    )
    result_table_xarray.attrs[MODEL_FILE_KEY] = model_file_name
    result_table_xarray.attrs[PREDICTION_FILE_KEY] = prediction_file_name

    for j in range(num_scalar_targets):
        print('Computing PIT histogram for {0:s}...'.format(
            example_dict[example_utils.SCALAR_TARGET_NAMES_KEY][j]
        ))

        (
            _,
            result_table_xarray[SCALAR_PIT_BIN_COUNT_KEY].values[j, :],
            result_table_xarray[SCALAR_PITD_KEY].values[j],
            result_table_xarray[SCALAR_PERFECT_PITD_KEY].values[j]
        ) = _get_pit_histogram_one_var(
            target_values=scalar_target_matrix[:, j],
            prediction_matrix=scalar_prediction_matrix[:, j, :],
            num_bins=num_bins
        )

    for j in range(num_vector_targets):
        for k in range(num_heights):
            print('Computing PIT histogram for {0:s} at {1:d} m AGL...'.format(
                example_dict[example_utils.VECTOR_TARGET_NAMES_KEY][j],
                int(numpy.round(heights_m_agl[k]))
            ))

            (
                _,
                result_table_xarray[VECTOR_PIT_BIN_COUNT_KEY].values[j, k, :],
                result_table_xarray[VECTOR_PITD_KEY].values[j, k],
                result_table_xarray[VECTOR_PERFECT_PITD_KEY].values[j, k]
            ) = _get_pit_histogram_one_var(
                target_values=vector_target_matrix[:, k, j],
                prediction_matrix=vector_prediction_matrix[:, k, j, :],
                num_bins=num_bins
            )

    for j in range(num_aux_targets):
        print('Computing PIT histogram for {0:s}...'.format(
            aux_target_field_names[j]
        ))

        (
            _,
            result_table_xarray[AUX_PIT_BIN_COUNT_KEY].values[j, :],
            result_table_xarray[AUX_PITD_KEY].values[j],
            result_table_xarray[AUX_PERFECT_PITD_KEY].values[j]
        ) = _get_pit_histogram_one_var(
            target_values=aux_target_matrix[:, j],
            prediction_matrix=aux_prediction_matrix[:, j, :],
            num_bins=num_bins
        )

    return result_table_xarray


def run_discard_test_all_vars(
        prediction_file_name, discard_fractions, error_function,
        uncertainty_function, is_error_pos_oriented,
        error_function_for_hr_1height, error_function_for_flux_1var):
    """Runs discard test for all target variables.

    E = number of examples
    F = number of discard fractions
    S = ensemble size

    :param prediction_file_name: Path to input file (will be read by
        `prediction_io.read_file`).
    :param discard_fractions: length-(F - 1) numpy array of discard fractions,
        ranging from (0, 1).  This method will use 0 as the lowest discard
        fraction.

    :param error_function: Function with the following inputs and outputs...
    Input: prediction_dict: See above.
    Input: use_example_flags: length-E numpy array of Boolean flags,
        indicating which examples to use.
    Output: error_value: Scalar value of error metric.

    :param uncertainty_function: Function with the following inputs and
        outputs...
    Input: prediction_dict: See above.
    Output: uncertainty_values: length-E numpy array with values of uncertainty
        metric.  The metric must be oriented so that higher value = more
        uncertainty.

    :param is_error_pos_oriented: Boolean flag.  If True (False), error function
        is positively (negatively) oriented.

    :param error_function_for_hr_1height: Function with the following inputs and
        outputs...
    Input: actual_heating_rates_k_day01: length-E numpy array of actual heating
        rates at one height.
    Input: predicted_hr_matrix_k_day01: E-by-S numpy array of predicted heating
        rates at the same height.
    Input: use_example_flags: length-E numpy array of Boolean flags,
        indicating which examples to use.
    Output: error_value: Scalar value of error metric.

    :param error_function_for_flux_1var: Function with the following inputs and
        outputs...
    Input: actual_fluxes_w_m02: length-E numpy array of actual values for one
        flux variable.
    Input: predicted_fluxes_w_m02: E-by-S numpy array of predicted values for
        one flux variable.
    Input: use_example_flags: length-E numpy array of Boolean flags,
        indicating which examples to use.
    Output: error_value: Scalar value of error metric.

    :return: result_table_xarray: xarray table with results (variable and
        dimension names should make the table self-explanatory).
    """

    # Check input args.
    error_checking.assert_is_boolean(is_error_pos_oriented)

    error_checking.assert_is_numpy_array(discard_fractions, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(discard_fractions, 0.)
    error_checking.assert_is_less_than_numpy_array(discard_fractions, 1.)

    discard_fractions = numpy.concatenate((
        numpy.array([0.]),
        discard_fractions
    ))
    discard_fractions = numpy.sort(discard_fractions)

    num_fractions = len(discard_fractions)
    assert num_fractions >= 2

    # Do actual stuff.
    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(prediction_file_name)

    model_file_name = prediction_dict[prediction_io.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    generator_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    heights_m_agl = prediction_dict[prediction_io.HEIGHTS_KEY]

    example_dict = {
        example_utils.SCALAR_TARGET_NAMES_KEY:
            generator_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY],
        example_utils.VECTOR_TARGET_NAMES_KEY:
            generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY],
        example_utils.SCALAR_PREDICTOR_NAMES_KEY:
            generator_option_dict[neural_net.SCALAR_PREDICTOR_NAMES_KEY],
        example_utils.VECTOR_PREDICTOR_NAMES_KEY:
            generator_option_dict[neural_net.VECTOR_PREDICTOR_NAMES_KEY],
        example_utils.HEIGHTS_KEY: heights_m_agl
    }

    aux_prediction_dict = _get_aux_fields(
        prediction_dict=prediction_dict, example_dict=example_dict
    )
    aux_target_field_names = aux_prediction_dict[AUX_TARGET_NAMES_KEY]
    aux_predicted_field_names = aux_prediction_dict[AUX_PREDICTED_NAMES_KEY]
    aux_target_matrix = aux_prediction_dict[AUX_TARGET_VALS_KEY]
    aux_prediction_matrix = aux_prediction_dict[AUX_PREDICTED_VALS_KEY]

    scalar_target_matrix = prediction_dict[prediction_io.SCALAR_TARGETS_KEY]
    scalar_prediction_matrix = (
        prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY]
    )
    vector_target_matrix = prediction_dict[prediction_io.VECTOR_TARGETS_KEY]
    vector_prediction_matrix = (
        prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY]
    )

    num_heights = vector_target_matrix.shape[1]
    num_vector_targets = vector_target_matrix.shape[2]
    num_scalar_targets = scalar_target_matrix.shape[1]

    main_data_dict = {
        POST_DISCARD_ERROR_KEY: (
            (DISCARD_FRACTION_DIM,), numpy.full(num_fractions, numpy.nan)
        ),
        EXAMPLE_FRACTION_KEY: (
            (DISCARD_FRACTION_DIM,), numpy.full(num_fractions, -1, dtype=int)
        )
    }

    these_dim_keys_1d = (SCALAR_FIELD_DIM,)
    these_dim_keys_2d = (SCALAR_FIELD_DIM, DISCARD_FRACTION_DIM)
    these_dim_1d = num_scalar_targets
    these_dim_2d = (num_scalar_targets, num_fractions)

    main_data_dict.update({
        SCALAR_POST_DISCARD_ERROR_KEY: (
            these_dim_keys_2d, numpy.full(these_dim_2d, numpy.nan)
        ),
        SCALAR_MEAN_MEAN_PREDICTION_KEY: (
            these_dim_keys_2d, numpy.full(these_dim_2d, numpy.nan)
        ),
        SCALAR_MEAN_TARGET_KEY: (
            these_dim_keys_2d, numpy.full(these_dim_2d, numpy.nan)
        ),
        SCALAR_MONOTONICITY_FRACTION_KEY: (
            these_dim_keys_1d, numpy.full(these_dim_1d, numpy.nan)
        ),
        SCALAR_MEAN_DISCARD_IMPROVEMENT_KEY: (
            these_dim_keys_1d, numpy.full(these_dim_1d, numpy.nan)
        )
    })

    these_dim_keys_2d = (VECTOR_FIELD_DIM, HEIGHT_DIM)
    these_dim_keys_3d = (VECTOR_FIELD_DIM, HEIGHT_DIM, DISCARD_FRACTION_DIM)
    these_dim_2d = (num_vector_targets, num_heights)
    these_dim_3d = (num_vector_targets, num_heights, num_fractions)

    main_data_dict.update({
        VECTOR_POST_DISCARD_ERROR_KEY: (
            these_dim_keys_3d, numpy.full(these_dim_3d, numpy.nan)
        ),
        VECTOR_MEAN_MEAN_PREDICTION_KEY: (
            these_dim_keys_3d, numpy.full(these_dim_3d, numpy.nan)
        ),
        VECTOR_MEAN_TARGET_KEY: (
            these_dim_keys_3d, numpy.full(these_dim_3d, numpy.nan)
        ),
        VECTOR_MONOTONICITY_FRACTION_KEY: (
            these_dim_keys_2d, numpy.full(these_dim_2d, numpy.nan)
        ),
        VECTOR_MEAN_DISCARD_IMPROVEMENT_KEY: (
            these_dim_keys_2d, numpy.full(these_dim_2d, numpy.nan)
        )
    })

    num_aux_targets = len(aux_target_field_names)

    if num_aux_targets > 0:
        these_dim_keys_1d = (AUX_TARGET_FIELD_DIM,)
        these_dim_keys_2d = (AUX_TARGET_FIELD_DIM, DISCARD_FRACTION_DIM)
        these_dim_1d = num_aux_targets
        these_dim_2d = (num_aux_targets, num_fractions)

        main_data_dict.update({
            AUX_POST_DISCARD_ERROR_KEY: (
                these_dim_keys_2d, numpy.full(these_dim_2d, numpy.nan)
            ),
            AUX_MEAN_MEAN_PREDICTION_KEY: (
                these_dim_keys_2d, numpy.full(these_dim_2d, numpy.nan)
            ),
            AUX_MEAN_TARGET_KEY: (
                these_dim_keys_2d, numpy.full(these_dim_2d, numpy.nan)
            ),
            AUX_MONOTONICITY_FRACTION_KEY: (
                these_dim_keys_1d, numpy.full(these_dim_1d, numpy.nan)
            ),
            AUX_MEAN_DISCARD_IMPROVEMENT_KEY: (
                these_dim_keys_1d, numpy.full(these_dim_1d, numpy.nan)
            )
        })

    metadata_dict = {
        SCALAR_FIELD_DIM: example_dict[example_utils.SCALAR_TARGET_NAMES_KEY],
        HEIGHT_DIM: heights_m_agl,
        VECTOR_FIELD_DIM: example_dict[example_utils.VECTOR_TARGET_NAMES_KEY],
        DISCARD_FRACTION_DIM: discard_fractions
    }

    if num_aux_targets > 0:
        metadata_dict[AUX_TARGET_FIELD_DIM] = aux_target_field_names
        metadata_dict[AUX_PREDICTED_FIELD_DIM] = aux_predicted_field_names

    result_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict
    )
    result_table_xarray.attrs[MODEL_FILE_KEY] = model_file_name
    result_table_xarray.attrs[PREDICTION_FILE_KEY] = prediction_file_name

    uncertainty_values = uncertainty_function(prediction_dict)
    use_example_flags = numpy.full(len(uncertainty_values), 1, dtype=bool)

    for i in range(num_fractions):
        this_percentile_level = 100 * (1 - discard_fractions[i])
        this_inverted_mask = (
            uncertainty_values >
            numpy.percentile(uncertainty_values, this_percentile_level)
        )
        use_example_flags[this_inverted_mask] = False

        result_table_xarray[EXAMPLE_FRACTION_KEY].values[i] = numpy.mean(
            use_example_flags
        )
        result_table_xarray[POST_DISCARD_ERROR_KEY].values[i] = error_function(
            prediction_dict, use_example_flags
        )

        t = result_table_xarray

        for k in range(num_scalar_targets):
            t[SCALAR_MEAN_MEAN_PREDICTION_KEY].values[k, i] = numpy.mean(
                numpy.mean(
                    scalar_prediction_matrix[:, k, :][use_example_flags, :],
                    axis=-1
                )
            )

            t[SCALAR_MEAN_TARGET_KEY].values[k, i] = numpy.mean(
                scalar_target_matrix[:, k][use_example_flags]
            )

            t[SCALAR_POST_DISCARD_ERROR_KEY].values[k, i] = (
                error_function_for_flux_1var(
                    scalar_target_matrix[:, k],
                    scalar_prediction_matrix[:, k, :],
                    use_example_flags
                )
            )

        for k in range(num_aux_targets):
            t[AUX_MEAN_MEAN_PREDICTION_KEY].values[k, i] = numpy.mean(
                numpy.mean(
                    aux_prediction_matrix[:, k, :][use_example_flags, :],
                    axis=-1
                )
            )

            t[AUX_MEAN_TARGET_KEY].values[k, i] = numpy.mean(
                aux_target_matrix[:, k][use_example_flags]
            )

            t[AUX_POST_DISCARD_ERROR_KEY].values[k, i] = (
                error_function_for_flux_1var(
                    aux_target_matrix[:, k],
                    aux_prediction_matrix[:, k, :],
                    use_example_flags
                )
            )

        for k in range(num_vector_targets):
            for j in range(num_heights):
                this_mean_pred_by_example = numpy.mean(
                    vector_prediction_matrix[:, j, k, :][use_example_flags, :],
                    axis=-1
                )

                t[VECTOR_MEAN_MEAN_PREDICTION_KEY].values[k, j, i] = numpy.mean(
                    this_mean_pred_by_example
                )

                t[VECTOR_MEAN_TARGET_KEY].values[k, j, i] = numpy.mean(
                    vector_target_matrix[:, j, k][use_example_flags]
                )

                t[VECTOR_POST_DISCARD_ERROR_KEY].values[k, j, i] = (
                    error_function_for_hr_1height(
                        vector_target_matrix[:, j, k],
                        vector_prediction_matrix[:, j, k, :],
                        use_example_flags
                    )
                )

        result_table_xarray = t

    t = result_table_xarray

    for k in range(num_scalar_targets):
        if is_error_pos_oriented:
            t[SCALAR_MONOTONICITY_FRACTION_KEY].values[k] = numpy.mean(
                numpy.diff(t[SCALAR_POST_DISCARD_ERROR_KEY].values[k, :]) > 0
            )
            t[SCALAR_MEAN_DISCARD_IMPROVEMENT_KEY].values[k] = numpy.mean(
                numpy.diff(t[SCALAR_POST_DISCARD_ERROR_KEY].values[k, :]) /
                numpy.diff(discard_fractions)
            )
        else:
            t[SCALAR_MONOTONICITY_FRACTION_KEY].values[k] = numpy.mean(
                numpy.diff(t[SCALAR_POST_DISCARD_ERROR_KEY].values[k, :]) < 0
            )
            t[SCALAR_MEAN_DISCARD_IMPROVEMENT_KEY].values[k] = numpy.mean(
                -1 * numpy.diff(t[SCALAR_POST_DISCARD_ERROR_KEY].values[k, :]) /
                numpy.diff(discard_fractions)
            )

    for k in range(num_aux_targets):
        if is_error_pos_oriented:
            t[AUX_MONOTONICITY_FRACTION_KEY].values[k] = numpy.mean(
                numpy.diff(t[AUX_POST_DISCARD_ERROR_KEY].values[k, :]) > 0
            )
            t[AUX_MEAN_DISCARD_IMPROVEMENT_KEY].values[k] = numpy.mean(
                numpy.diff(t[AUX_POST_DISCARD_ERROR_KEY].values[k, :]) /
                numpy.diff(discard_fractions)
            )
        else:
            t[AUX_MONOTONICITY_FRACTION_KEY].values[k] = numpy.mean(
                numpy.diff(t[AUX_POST_DISCARD_ERROR_KEY].values[k, :]) < 0
            )
            t[AUX_MEAN_DISCARD_IMPROVEMENT_KEY].values[k] = numpy.mean(
                -1 * numpy.diff(t[AUX_POST_DISCARD_ERROR_KEY].values[k, :]) /
                numpy.diff(discard_fractions)
            )

    for k in range(num_vector_targets):
        for j in range(num_heights):
            if is_error_pos_oriented:
                t[VECTOR_MONOTONICITY_FRACTION_KEY].values[k, j] = numpy.mean(
                    numpy.diff(
                        t[VECTOR_POST_DISCARD_ERROR_KEY].values[k, j, :]
                    ) > 0
                )

                (
                    t[VECTOR_MEAN_DISCARD_IMPROVEMENT_KEY].values[k, j]
                ) = numpy.mean(
                    numpy.diff(t[VECTOR_POST_DISCARD_ERROR_KEY].values[k, j, :])
                    / numpy.diff(discard_fractions)
                )
            else:
                t[VECTOR_MONOTONICITY_FRACTION_KEY].values[k, j] = numpy.mean(
                    numpy.diff(
                        t[VECTOR_POST_DISCARD_ERROR_KEY].values[k, j, :]
                    ) < 0
                )

                (
                    t[VECTOR_MEAN_DISCARD_IMPROVEMENT_KEY].values[k, j]
                ) = numpy.mean(
                    -1 *
                    numpy.diff(t[VECTOR_POST_DISCARD_ERROR_KEY].values[k, j, :])
                    / numpy.diff(discard_fractions)
                )

    if is_error_pos_oriented:
        t.attrs[MONOTONICITY_FRACTION_KEY] = numpy.mean(
            numpy.diff(t[POST_DISCARD_ERROR_KEY].values) > 0
        )
        t.attrs[MEAN_DISCARD_IMPROVEMENT_KEY] = numpy.mean(
            numpy.diff(t[POST_DISCARD_ERROR_KEY].values) /
            numpy.diff(discard_fractions)
        )
    else:
        t.attrs[MONOTONICITY_FRACTION_KEY] = numpy.mean(
            numpy.diff(t[POST_DISCARD_ERROR_KEY].values) < 0
        )
        t.attrs[MEAN_DISCARD_IMPROVEMENT_KEY] = numpy.mean(
            -1 * numpy.diff(t[POST_DISCARD_ERROR_KEY].values)
            / numpy.diff(discard_fractions)
        )

    result_table_xarray = t
    return result_table_xarray


def get_spread_vs_skill_all_vars(
        prediction_file_name, num_heating_rate_bins,
        min_heating_rate_k_day01, max_heating_rate_k_day01,
        min_heating_rate_percentile, max_heating_rate_percentile,
        num_raw_flux_bins, min_raw_flux_w_m02, max_raw_flux_w_m02,
        min_raw_flux_percentile, max_raw_flux_percentile,
        num_net_flux_bins, min_net_flux_w_m02, max_net_flux_w_m02,
        min_net_flux_percentile, max_net_flux_percentile):
    """Computes spread-skill relationship for each target variable.

    :param prediction_file_name: Path to input file (will be read by
        `prediction_io.read_file`).
    :param num_heating_rate_bins: Number of standard-deviation bins for heating
        rate.
    :param min_heating_rate_k_day01: Minimum heating rate (Kelvins per day) for
        stdev bins.  If you instead want minimum heating rate to be a
        percentile over the data -- chosen independently at each height -- make
        this argument None.
    :param max_heating_rate_k_day01: Same as above but max heating rate.
    :param min_heating_rate_percentile: Determines minimum heating rate for
        stdev bins.  This percentile (ranging from 0...100) will be taken
        independently at each height.
    :param max_heating_rate_percentile: Same as above but max heating rate.
    :param num_raw_flux_bins: Number of stdev bins for raw flux variables
        (surface downwelling and TOA upwelling).
    :param min_raw_flux_w_m02: Min flux for stdev bins on raw flux variables
        (surface downwelling and TOA upwelling).  If you want to specify min/max
        by percentiles instead -- chosen independently for each variable -- make
        this argument None.
    :param max_raw_flux_w_m02: Same as above but for max flux.
    :param min_raw_flux_percentile: Min percentile for stdev bins, taken
        independently for each raw flux variable (surface downwelling and TOA
        upwelling).  If you want to specify min/max by physical values instead,
        make this argument None.
    :param max_raw_flux_percentile: Same as above but for max percentile.
    :param num_net_flux_bins: Number of stdev bins for net flux.
    :param min_net_flux_w_m02: Min net flux for stdev bins.  If you want to
        specify min/max by percentiles instead, make this argument None.
    :param max_net_flux_w_m02: Same as above but for max flux.
    :param min_net_flux_percentile: Min net-flux percentile for stdev bins.  If
        you want to specify min/max by physical values instead, make this
        argument None.
    :param max_net_flux_percentile: Same as above but for max percentile.
    :return: result_table_xarray: xarray table with results (variable and
        dimension names should make the table self-explanatory).
    """

    # Check input args.
    error_checking.assert_is_integer(num_heating_rate_bins)
    error_checking.assert_is_geq(num_heating_rate_bins, 10)
    error_checking.assert_is_leq(num_heating_rate_bins, 1000)
    error_checking.assert_is_integer(num_raw_flux_bins)
    error_checking.assert_is_geq(num_raw_flux_bins, 10)
    error_checking.assert_is_leq(num_raw_flux_bins, 1000)
    error_checking.assert_is_integer(num_net_flux_bins)
    error_checking.assert_is_geq(num_net_flux_bins, 10)
    error_checking.assert_is_leq(num_net_flux_bins, 1000)

    if min_heating_rate_k_day01 is None or max_heating_rate_k_day01 is None:
        error_checking.assert_is_leq(min_heating_rate_percentile, 10.)
        error_checking.assert_is_geq(max_heating_rate_percentile, 90.)
    else:
        error_checking.assert_is_greater(
            max_heating_rate_k_day01, min_heating_rate_k_day01
        )

    if min_raw_flux_w_m02 is None or max_raw_flux_w_m02 is None:
        error_checking.assert_is_leq(min_raw_flux_percentile, 10.)
        error_checking.assert_is_geq(max_raw_flux_percentile, 90.)
    else:
        error_checking.assert_is_geq(min_raw_flux_w_m02, 0.)
        error_checking.assert_is_greater(
            max_raw_flux_w_m02, min_raw_flux_w_m02
        )

    if min_net_flux_w_m02 is None or max_net_flux_w_m02 is None:
        error_checking.assert_is_leq(min_net_flux_percentile, 10.)
        error_checking.assert_is_geq(max_net_flux_percentile, 90.)
    else:
        error_checking.assert_is_greater(
            max_net_flux_w_m02, min_net_flux_w_m02
        )

    # Do actual stuff.
    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(prediction_file_name)

    model_file_name = prediction_dict[prediction_io.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    generator_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    heights_m_agl = prediction_dict[prediction_io.HEIGHTS_KEY]

    example_dict = {
        example_utils.SCALAR_TARGET_NAMES_KEY:
            generator_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY],
        example_utils.VECTOR_TARGET_NAMES_KEY:
            generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY],
        example_utils.SCALAR_PREDICTOR_NAMES_KEY:
            generator_option_dict[neural_net.SCALAR_PREDICTOR_NAMES_KEY],
        example_utils.VECTOR_PREDICTOR_NAMES_KEY:
            generator_option_dict[neural_net.VECTOR_PREDICTOR_NAMES_KEY],
        example_utils.HEIGHTS_KEY: heights_m_agl
    }

    aux_prediction_dict = _get_aux_fields(
        prediction_dict=prediction_dict, example_dict=example_dict
    )
    aux_target_field_names = aux_prediction_dict[AUX_TARGET_NAMES_KEY]
    aux_predicted_field_names = aux_prediction_dict[AUX_PREDICTED_NAMES_KEY]
    aux_target_matrix = aux_prediction_dict[AUX_TARGET_VALS_KEY]
    aux_prediction_matrix = aux_prediction_dict[AUX_PREDICTED_VALS_KEY]

    scalar_target_matrix = prediction_dict[prediction_io.SCALAR_TARGETS_KEY]
    scalar_prediction_matrix = (
        prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY]
    )
    vector_target_matrix = prediction_dict[prediction_io.VECTOR_TARGETS_KEY]
    vector_prediction_matrix = (
        prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY]
    )

    num_heights = vector_target_matrix.shape[1]
    num_vector_targets = vector_target_matrix.shape[2]
    num_scalar_targets = scalar_target_matrix.shape[1]

    these_dim_no_bins = num_scalar_targets
    these_dim_no_edge = (num_scalar_targets, num_raw_flux_bins)
    these_dim_with_edge = (num_scalar_targets, num_raw_flux_bins + 1)

    these_dim_keys_no_bins = (SCALAR_FIELD_DIM,)
    these_dim_keys_no_edge = (SCALAR_FIELD_DIM, RAW_FLUX_BIN_DIM)
    these_dim_keys_with_edge = (SCALAR_FIELD_DIM, RAW_FLUX_BIN_EDGE_DIM)

    main_data_dict = {
        SCALAR_MEAN_STDEV_KEY: (
            these_dim_keys_no_edge, numpy.full(these_dim_no_edge, numpy.nan)
        ),
        SCALAR_BIN_EDGE_KEY: (
            these_dim_keys_with_edge, numpy.full(these_dim_with_edge, numpy.nan)
        ),
        SCALAR_RMSE_KEY: (
            these_dim_keys_no_edge, numpy.full(these_dim_no_edge, numpy.nan)
        ),
        SCALAR_SSREL_KEY: (
            these_dim_keys_no_bins, numpy.full(these_dim_no_bins, numpy.nan)
        ),
        SCALAR_SSRAT_KEY: (
            these_dim_keys_no_bins, numpy.full(these_dim_no_bins, numpy.nan)
        ),
        SCALAR_EXAMPLE_COUNT_KEY: (
            these_dim_keys_no_edge, numpy.full(these_dim_no_edge, -1, dtype=int)
        ),
        SCALAR_MEAN_MEAN_PREDICTION_KEY: (
            these_dim_keys_no_edge, numpy.full(these_dim_no_edge, numpy.nan)
        ),
        SCALAR_MEAN_TARGET_KEY: (
            these_dim_keys_no_edge, numpy.full(these_dim_no_edge, numpy.nan)
        )
    }

    these_dim_no_bins = (num_vector_targets, num_heights)
    these_dim_no_edge = (num_vector_targets, num_heights, num_raw_flux_bins)
    these_dim_with_edge = (
        num_vector_targets, num_heights, num_raw_flux_bins + 1
    )

    these_dim_keys_no_bins = (VECTOR_FIELD_DIM, HEIGHT_DIM)
    these_dim_keys_no_edge = (
        VECTOR_FIELD_DIM, HEIGHT_DIM, HEATING_RATE_BIN_DIM
    )
    these_dim_keys_with_edge = (
        VECTOR_FIELD_DIM, HEIGHT_DIM, HEATING_RATE_BIN_EDGE_DIM
    )

    main_data_dict.update({
        VECTOR_MEAN_STDEV_KEY: (
            these_dim_keys_no_edge, numpy.full(these_dim_no_edge, numpy.nan)
        ),
        VECTOR_BIN_EDGE_KEY: (
            these_dim_keys_with_edge, numpy.full(these_dim_with_edge, numpy.nan)
        ),
        VECTOR_RMSE_KEY: (
            these_dim_keys_no_edge, numpy.full(these_dim_no_edge, numpy.nan)
        ),
        VECTOR_SSREL_KEY: (
            these_dim_keys_no_bins, numpy.full(these_dim_no_bins, numpy.nan)
        ),
        VECTOR_SSRAT_KEY: (
            these_dim_keys_no_bins, numpy.full(these_dim_no_bins, numpy.nan)
        ),
        VECTOR_EXAMPLE_COUNT_KEY: (
            these_dim_keys_no_edge, numpy.full(these_dim_no_edge, -1, dtype=int)
        ),
        VECTOR_MEAN_MEAN_PREDICTION_KEY: (
            these_dim_keys_no_edge, numpy.full(these_dim_no_edge, numpy.nan)
        ),
        VECTOR_MEAN_TARGET_KEY: (
            these_dim_keys_no_edge, numpy.full(these_dim_no_edge, numpy.nan)
        )
    })

    num_aux_targets = len(aux_target_field_names)

    if num_aux_targets > 0:
        these_dim_no_bins = num_aux_targets
        these_dim_no_edge = (num_aux_targets, num_raw_flux_bins)
        these_dim_with_edge = (num_aux_targets, num_raw_flux_bins + 1)

        these_dim_keys_no_bins = (AUX_TARGET_FIELD_DIM,)
        these_dim_keys_no_edge = (AUX_TARGET_FIELD_DIM, NET_FLUX_BIN_DIM)
        these_dim_keys_with_edge = (AUX_TARGET_FIELD_DIM, NET_FLUX_BIN_EDGE_DIM)

        main_data_dict.update({
            AUX_MEAN_STDEV_KEY: (
                these_dim_keys_no_edge, numpy.full(these_dim_no_edge, numpy.nan)
            ),
            AUX_BIN_EDGE_KEY: (
                these_dim_keys_with_edge,
                numpy.full(these_dim_with_edge, numpy.nan)
            ),
            AUX_RMSE_KEY: (
                these_dim_keys_no_edge, numpy.full(these_dim_no_edge, numpy.nan)
            ),
            AUX_SSREL_KEY: (
                these_dim_keys_no_bins, numpy.full(these_dim_no_bins, numpy.nan)
            ),
            AUX_SSRAT_KEY: (
                these_dim_keys_no_bins, numpy.full(these_dim_no_bins, numpy.nan)
            ),
            AUX_EXAMPLE_COUNT_KEY: (
                these_dim_keys_no_edge,
                numpy.full(these_dim_no_edge, -1, dtype=int)
            ),
            AUX_MEAN_MEAN_PREDICTION_KEY: (
                these_dim_keys_no_edge, numpy.full(these_dim_no_edge, numpy.nan)
            ),
            AUX_MEAN_TARGET_KEY: (
                these_dim_keys_no_edge, numpy.full(these_dim_no_edge, numpy.nan)
            )
        })

    raw_flux_bin_indices = numpy.linspace(
        0, num_raw_flux_bins - 1, num=num_raw_flux_bins, dtype=int
    )
    net_flux_bin_indices = numpy.linspace(
        0, num_net_flux_bins - 1, num=num_net_flux_bins, dtype=int
    )
    heating_rate_bin_indices = numpy.linspace(
        0, num_heating_rate_bins - 1, num=num_heating_rate_bins, dtype=int
    )

    metadata_dict = {
        SCALAR_FIELD_DIM: example_dict[example_utils.SCALAR_TARGET_NAMES_KEY],
        HEIGHT_DIM: heights_m_agl,
        VECTOR_FIELD_DIM: example_dict[example_utils.VECTOR_TARGET_NAMES_KEY],
        RAW_FLUX_BIN_DIM: raw_flux_bin_indices,
        NET_FLUX_BIN_DIM: net_flux_bin_indices,
        HEATING_RATE_BIN_DIM: heating_rate_bin_indices
    }

    if num_aux_targets > 0:
        metadata_dict[AUX_TARGET_FIELD_DIM] = aux_target_field_names
        metadata_dict[AUX_PREDICTED_FIELD_DIM] = aux_predicted_field_names

    result_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict
    )
    result_table_xarray.attrs[MODEL_FILE_KEY] = model_file_name
    result_table_xarray.attrs[PREDICTION_FILE_KEY] = prediction_file_name

    for j in range(num_scalar_targets):
        print('Computing spread-skill relationship for {0:s}...'.format(
            example_dict[example_utils.SCALAR_TARGET_NAMES_KEY][j]
        ))

        if min_raw_flux_w_m02 is None or max_raw_flux_w_m02 is None:
            these_stdevs = numpy.std(
                scalar_prediction_matrix[:, j, :], ddof=1, axis=-1
            )

            these_bin_edges = numpy.linspace(
                numpy.percentile(these_stdevs, min_raw_flux_percentile),
                numpy.percentile(these_stdevs, max_raw_flux_percentile),
                num=num_raw_flux_bins + 1, dtype=float
            )[1:-1]
        else:
            these_bin_edges = numpy.linspace(
                min_raw_flux_w_m02, max_raw_flux_w_m02,
                num=num_raw_flux_bins + 1, dtype=float
            )[1:-1]

        this_result_dict = _get_spread_vs_skill_one_var(
            target_values=scalar_target_matrix[:, j],
            prediction_matrix=scalar_prediction_matrix[:, j, :],
            bin_edge_prediction_stdevs=these_bin_edges
        )

        result_table_xarray[SCALAR_MEAN_STDEV_KEY].values[j, :] = (
            this_result_dict[MEAN_PREDICTION_STDEVS_KEY]
        )
        result_table_xarray[SCALAR_BIN_EDGE_KEY].values[j, :] = (
            this_result_dict[BIN_EDGE_PREDICTION_STDEVS_KEY]
        )
        result_table_xarray[SCALAR_RMSE_KEY].values[j, :] = (
            this_result_dict[RMSE_VALUES_KEY]
        )
        result_table_xarray[SCALAR_SSREL_KEY].values[j] = (
            this_result_dict[SPREAD_SKILL_RELIABILITY_KEY]
        )
        result_table_xarray[SCALAR_SSRAT_KEY].values[j] = (
            this_result_dict[SPREAD_SKILL_RATIO_KEY]
        )
        result_table_xarray[SCALAR_EXAMPLE_COUNT_KEY].values[j, :] = (
            this_result_dict[EXAMPLE_COUNTS_KEY]
        )
        result_table_xarray[SCALAR_MEAN_MEAN_PREDICTION_KEY].values[j, :] = (
            this_result_dict[MEAN_MEAN_PREDICTIONS_KEY]
        )
        result_table_xarray[SCALAR_MEAN_TARGET_KEY].values[j, :] = (
            this_result_dict[MEAN_TARGET_VALUES_KEY]
        )

    for j in range(num_aux_targets):
        print('Computing spread-skill relationship for {0:s}...'.format(
            aux_target_field_names[j]
        ))

        if min_net_flux_w_m02 is None or max_net_flux_w_m02 is None:
            these_stdevs = numpy.std(
                aux_prediction_matrix[:, j, :], ddof=1, axis=-1
            )

            these_bin_edges = numpy.linspace(
                numpy.percentile(these_stdevs, min_net_flux_percentile),
                numpy.percentile(these_stdevs, max_net_flux_percentile),
                num=num_net_flux_bins + 1, dtype=float
            )[1:-1]
        else:
            these_bin_edges = numpy.linspace(
                min_net_flux_w_m02, max_net_flux_w_m02,
                num=num_net_flux_bins + 1, dtype=float
            )[1:-1]

        this_result_dict = _get_spread_vs_skill_one_var(
            target_values=aux_target_matrix[:, j],
            prediction_matrix=aux_prediction_matrix[:, j, :],
            bin_edge_prediction_stdevs=these_bin_edges
        )

        result_table_xarray[AUX_MEAN_STDEV_KEY].values[j, :] = (
            this_result_dict[MEAN_PREDICTION_STDEVS_KEY]
        )
        result_table_xarray[AUX_BIN_EDGE_KEY].values[j, :] = (
            this_result_dict[BIN_EDGE_PREDICTION_STDEVS_KEY]
        )
        result_table_xarray[AUX_RMSE_KEY].values[j, :] = (
            this_result_dict[RMSE_VALUES_KEY]
        )
        result_table_xarray[AUX_SSREL_KEY].values[j] = (
            this_result_dict[SPREAD_SKILL_RELIABILITY_KEY]
        )
        result_table_xarray[AUX_SSRAT_KEY].values[j] = (
            this_result_dict[SPREAD_SKILL_RATIO_KEY]
        )
        result_table_xarray[AUX_EXAMPLE_COUNT_KEY].values[j, :] = (
            this_result_dict[EXAMPLE_COUNTS_KEY]
        )
        result_table_xarray[AUX_MEAN_MEAN_PREDICTION_KEY].values[j, :] = (
            this_result_dict[MEAN_MEAN_PREDICTIONS_KEY]
        )
        result_table_xarray[AUX_MEAN_TARGET_KEY].values[j, :] = (
            this_result_dict[MEAN_TARGET_VALUES_KEY]
        )

    for j in range(num_vector_targets):
        for k in range(num_heights):
            print((
                'Computing spread-skill relationship for {0:s} at {1:d} '
                'm AGL...'
            ).format(
                example_dict[example_utils.VECTOR_TARGET_NAMES_KEY][j],
                int(numpy.round(heights_m_agl[k]))
            ))

            if (
                    min_heating_rate_k_day01 is None or
                    max_heating_rate_k_day01 is None
            ):
                these_stdevs = numpy.std(
                    vector_prediction_matrix[:, k, j, :], ddof=1, axis=-1
                )
                this_max_value = numpy.percentile(
                    these_stdevs, max_heating_rate_percentile
                )
                this_max_value = max([this_max_value, 1.])

                these_bin_edges = numpy.linspace(
                    numpy.percentile(these_stdevs, min_heating_rate_percentile),
                    this_max_value,
                    num=num_heating_rate_bins + 1, dtype=float
                )[1:-1]
            else:
                these_bin_edges = numpy.linspace(
                    min_heating_rate_k_day01, max_heating_rate_k_day01,
                    num=num_heating_rate_bins + 1, dtype=float
                )[1:-1]

            this_result_dict = _get_spread_vs_skill_one_var(
                target_values=vector_target_matrix[:, k, j],
                prediction_matrix=vector_prediction_matrix[:, k, j, :],
                bin_edge_prediction_stdevs=these_bin_edges
            )

            result_table_xarray[VECTOR_MEAN_STDEV_KEY].values[j, k, :] = (
                this_result_dict[MEAN_PREDICTION_STDEVS_KEY]
            )
            result_table_xarray[VECTOR_BIN_EDGE_KEY].values[j, k, :] = (
                this_result_dict[BIN_EDGE_PREDICTION_STDEVS_KEY]
            )
            result_table_xarray[VECTOR_RMSE_KEY].values[j, k, :] = (
                this_result_dict[RMSE_VALUES_KEY]
            )
            result_table_xarray[VECTOR_SSREL_KEY].values[j, k] = (
                this_result_dict[SPREAD_SKILL_RELIABILITY_KEY]
            )
            result_table_xarray[VECTOR_SSRAT_KEY].values[j, k] = (
                this_result_dict[SPREAD_SKILL_RATIO_KEY]
            )
            result_table_xarray[VECTOR_EXAMPLE_COUNT_KEY].values[j, k, :] = (
                this_result_dict[EXAMPLE_COUNTS_KEY]
            )
            result_table_xarray[VECTOR_MEAN_MEAN_PREDICTION_KEY].values[
                j, k, :
            ] = this_result_dict[MEAN_MEAN_PREDICTIONS_KEY]

            result_table_xarray[VECTOR_MEAN_TARGET_KEY].values[j, k, :] = (
                this_result_dict[MEAN_TARGET_VALUES_KEY]
            )

    return result_table_xarray


def write_spread_vs_skill(spread_skill_table_xarray, netcdf_file_name):
    """Writes spread-vs.-skill results to NetCDF file.

    :param spread_skill_table_xarray: xarray table in format returned by
        `get_spread_vs_skill_all_vars`.
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)

    spread_skill_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )


def read_spread_vs_skill(netcdf_file_name):
    """Reads spread-vs.-skill results from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: spread_skill_table_xarray: xarray table.  Documentation in the
        xarray table should make values self-explanatory.
    """

    return xarray.open_dataset(netcdf_file_name)


def write_discard_results(discard_test_table_xarray, netcdf_file_name):
    """Writes discard-test results to NetCDF file.

    :param discard_test_table_xarray: xarray table in format returned by
        `run_discard_test_all_vars`.
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)

    discard_test_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )


def read_discard_results(netcdf_file_name):
    """Reads discard-test results from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: discard_test_table_xarray: xarray table.  Documentation in the
        xarray table should make values self-explanatory.
    """

    return xarray.open_dataset(netcdf_file_name)


def write_crps(crps_table_xarray, netcdf_file_name):
    """Writes CRPS for all target variables to NetCDF file.

    :param crps_table_xarray: xarray table in format returned by
        `get_crps_all_vars`.
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)

    crps_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )


def read_crps(netcdf_file_name):
    """Reads CRPS for all target variables from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: crps_table_xarray: xarray table.  Documentation in the
        xarray table should make values self-explanatory.
    """

    return xarray.open_dataset(netcdf_file_name)


def write_pit_histograms(pit_histogram_table_xarray, netcdf_file_name):
    """Writes PIT histogram for all target variables to NetCDF file.

    :param pit_histogram_table_xarray: xarray table in format returned by
        `get_pit_histogram_all_vars`.
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)

    pit_histogram_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )


def read_pit_histograms(netcdf_file_name):
    """Reads PIT histograms for all target variables from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: pit_histogram_table_xarray: xarray table.  Documentation in the
        xarray table should make values self-explanatory.
    """

    return xarray.open_dataset(netcdf_file_name)
