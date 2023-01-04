"""Helper methods for computing spread-skill relationship."""

import os
import copy
import numpy
import xarray
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4rt.io import prediction_io
from ml4rt.utils import example_utils
from ml4rt.utils import uq_evaluation
from ml4rt.machine_learning import neural_net

TOLERANCE = 1e-6

MEAN_PREDICTION_STDEVS_KEY = 'mean_prediction_stdevs'
BIN_EDGE_PREDICTION_STDEVS_KEY = 'bin_edge_prediction_stdevs'
RMSE_VALUES_KEY = 'rmse_values'
SPREAD_SKILL_RELIABILITY_KEY = 'spread_skill_reliability'
SPREAD_SKILL_RATIO_KEY = 'spread_skill_ratio'
EXAMPLE_COUNTS_KEY = 'example_counts'
MEAN_MEAN_PREDICTIONS_KEY = 'mean_mean_predictions'
MEAN_TARGET_VALUES_KEY = 'mean_target_values'

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

VECTOR_FLAT_MEAN_STDEV_KEY = 'vector_flat_mean_prediction_stdev'
VECTOR_FLAT_BIN_EDGE_KEY = 'vector_flat_bin_edge_prediction_stdev'
VECTOR_FLAT_RMSE_KEY = 'vector_flat_rmse'
VECTOR_FLAT_SSREL_KEY = 'vector_flat_spread_skill_reliability'
VECTOR_FLAT_SSRAT_KEY = 'vector_flat_spread_skill_ratio'
VECTOR_FLAT_EXAMPLE_COUNT_KEY = 'vector_flat_example_count'
VECTOR_FLAT_MEAN_MEAN_PREDICTION_KEY = 'vector_flat_mean_mean_prediction'
VECTOR_FLAT_MEAN_TARGET_KEY = 'vector_flat_mean_target_value'

AUX_MEAN_STDEV_KEY = 'aux_mean_prediction_stdev'
AUX_BIN_EDGE_KEY = 'aux_bin_edge_prediction_stdev'
AUX_RMSE_KEY = 'aux_rmse'
AUX_SSREL_KEY = 'aux_spread_skill_reliability'
AUX_SSRAT_KEY = 'aux_spread_skill_ratio'
AUX_EXAMPLE_COUNT_KEY = 'aux_example_count'
AUX_MEAN_MEAN_PREDICTION_KEY = 'aux_mean_mean_prediction'
AUX_MEAN_TARGET_KEY = 'aux_mean_target_value'

SCALAR_FIELD_DIM = uq_evaluation.SCALAR_FIELD_DIM
VECTOR_FIELD_DIM = uq_evaluation.VECTOR_FIELD_DIM
HEIGHT_DIM = uq_evaluation.HEIGHT_DIM
AUX_TARGET_FIELD_DIM = uq_evaluation.AUX_TARGET_FIELD_DIM
AUX_PREDICTED_FIELD_DIM = uq_evaluation.AUX_PREDICTED_FIELD_DIM

MODEL_FILE_KEY = uq_evaluation.MODEL_FILE_KEY
PREDICTION_FILE_KEY = uq_evaluation.PREDICTION_FILE_KEY


def _get_results_one_var(
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


def get_results_all_vars(
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

    aux_prediction_dict = uq_evaluation.get_aux_fields(
        prediction_dict=prediction_dict, example_dict=example_dict
    )
    aux_target_field_names = aux_prediction_dict[
        uq_evaluation.AUX_TARGET_NAMES_KEY
    ]
    aux_predicted_field_names = aux_prediction_dict[
        uq_evaluation.AUX_PREDICTED_NAMES_KEY
    ]
    aux_target_matrix = aux_prediction_dict[uq_evaluation.AUX_TARGET_VALS_KEY]
    aux_prediction_matrix = aux_prediction_dict[
        uq_evaluation.AUX_PREDICTED_VALS_KEY
    ]

    scalar_target_matrix = prediction_dict[prediction_io.SCALAR_TARGETS_KEY]
    scalar_prediction_matrix = (
        prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY]
    )
    vector_target_matrix = prediction_dict[prediction_io.VECTOR_TARGETS_KEY]
    vector_prediction_matrix = (
        prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY]
    )

    del prediction_dict

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

    these_dim_no_bins = num_vector_targets
    these_dim_no_edge = (num_vector_targets, num_raw_flux_bins)
    these_dim_with_edge = (num_vector_targets, num_raw_flux_bins + 1)

    these_dim_keys_no_bins = (VECTOR_FIELD_DIM,)
    these_dim_keys_no_edge = (VECTOR_FIELD_DIM, HEATING_RATE_BIN_DIM)
    these_dim_keys_with_edge = (VECTOR_FIELD_DIM, HEATING_RATE_BIN_EDGE_DIM)

    main_data_dict.update({
        VECTOR_FLAT_MEAN_STDEV_KEY: (
            these_dim_keys_no_edge, numpy.full(these_dim_no_edge, numpy.nan)
        ),
        VECTOR_FLAT_BIN_EDGE_KEY: (
            these_dim_keys_with_edge, numpy.full(these_dim_with_edge, numpy.nan)
        ),
        VECTOR_FLAT_RMSE_KEY: (
            these_dim_keys_no_edge, numpy.full(these_dim_no_edge, numpy.nan)
        ),
        VECTOR_FLAT_SSREL_KEY: (
            these_dim_keys_no_bins, numpy.full(these_dim_no_bins, numpy.nan)
        ),
        VECTOR_FLAT_SSRAT_KEY: (
            these_dim_keys_no_bins, numpy.full(these_dim_no_bins, numpy.nan)
        ),
        VECTOR_FLAT_EXAMPLE_COUNT_KEY: (
            these_dim_keys_no_edge, numpy.full(these_dim_no_edge, -1, dtype=int)
        ),
        VECTOR_FLAT_MEAN_MEAN_PREDICTION_KEY: (
            these_dim_keys_no_edge, numpy.full(these_dim_no_edge, numpy.nan)
        ),
        VECTOR_FLAT_MEAN_TARGET_KEY: (
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

        this_result_dict = _get_results_one_var(
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

        this_result_dict = _get_results_one_var(
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
        print('Computing spread-skill relationship for {0:s}...'.format(
            example_dict[example_utils.VECTOR_TARGET_NAMES_KEY][j]
        ))

        if (
                min_heating_rate_k_day01 is None or
                max_heating_rate_k_day01 is None
        ):
            these_stdevs = numpy.std(
                vector_prediction_matrix[:, :, j, :], ddof=1, axis=-1
            )

            these_bin_edges = numpy.linspace(
                numpy.percentile(these_stdevs, min_heating_rate_percentile),
                numpy.percentile(these_stdevs, max_heating_rate_percentile),
                num=num_heating_rate_bins + 1, dtype=float
            )[1:-1]
        else:
            these_bin_edges = numpy.linspace(
                min_heating_rate_k_day01, max_heating_rate_k_day01,
                num=num_heating_rate_bins + 1, dtype=float
            )[1:-1]

        these_targets = numpy.ravel(vector_target_matrix[..., j])
        this_prediction_matrix = numpy.reshape(
            vector_prediction_matrix[:, :, j, :],
            (len(these_targets), vector_prediction_matrix.shape[-1])
        )

        this_result_dict = _get_results_one_var(
            target_values=these_targets,
            prediction_matrix=this_prediction_matrix,
            bin_edge_prediction_stdevs=these_bin_edges
        )

        result_table_xarray[VECTOR_FLAT_MEAN_STDEV_KEY].values[j, :] = (
            this_result_dict[MEAN_PREDICTION_STDEVS_KEY]
        )
        result_table_xarray[VECTOR_FLAT_BIN_EDGE_KEY].values[j, :] = (
            this_result_dict[BIN_EDGE_PREDICTION_STDEVS_KEY]
        )
        result_table_xarray[VECTOR_FLAT_RMSE_KEY].values[j, :] = (
            this_result_dict[RMSE_VALUES_KEY]
        )
        result_table_xarray[VECTOR_FLAT_SSREL_KEY].values[j] = (
            this_result_dict[SPREAD_SKILL_RELIABILITY_KEY]
        )
        result_table_xarray[VECTOR_FLAT_SSRAT_KEY].values[j] = (
            this_result_dict[SPREAD_SKILL_RATIO_KEY]
        )
        result_table_xarray[VECTOR_FLAT_EXAMPLE_COUNT_KEY].values[j, :] = (
            this_result_dict[EXAMPLE_COUNTS_KEY]
        )
        result_table_xarray[VECTOR_FLAT_MEAN_TARGET_KEY].values[j, :] = (
            this_result_dict[MEAN_TARGET_VALUES_KEY]
        )
        result_table_xarray[VECTOR_FLAT_MEAN_MEAN_PREDICTION_KEY].values[
            j, :
        ] = this_result_dict[MEAN_MEAN_PREDICTIONS_KEY]

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

            this_result_dict = _get_results_one_var(
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


def merge_results_over_examples(result_tables_xarray):
    """Merges spread-vs.-skill results over many examples.

    :param result_tables_xarray: List of xarray tables, each created by
        `get_results_all_vars`, each containing results for a different
        set of examples.
    :return: result_table_xarray: Single xarray table with results for all
        examples (variable and dimension names should make the table
        self-explanatory).
    """

    prediction_file_names, _ = (
        uq_evaluation.check_results_before_merging(result_tables_xarray)
    )

    scalar_target_names = (
        result_tables_xarray[0].coords[SCALAR_FIELD_DIM].values
    )
    vector_target_names = (
        result_tables_xarray[0].coords[VECTOR_FIELD_DIM].values
    )
    heights_m_agl = result_tables_xarray[0].coords[HEIGHT_DIM].values

    try:
        aux_predicted_field_names = (
            result_tables_xarray[0].coords[AUX_PREDICTED_FIELD_DIM].values
        )
    except:
        aux_predicted_field_names = []

    num_tables = len(result_tables_xarray)

    for i in range(1, num_tables):
        assert numpy.allclose(
            result_tables_xarray[i][SCALAR_BIN_EDGE_KEY].values,
            result_tables_xarray[0][SCALAR_BIN_EDGE_KEY].values,
            atol=TOLERANCE
        )
        assert numpy.allclose(
            result_tables_xarray[i][VECTOR_BIN_EDGE_KEY].values,
            result_tables_xarray[0][VECTOR_BIN_EDGE_KEY].values,
            atol=TOLERANCE
        )

        if len(aux_predicted_field_names) == 0:
            continue

        assert numpy.allclose(
            result_tables_xarray[i][AUX_BIN_EDGE_KEY].values,
            result_tables_xarray[0][AUX_BIN_EDGE_KEY].values,
            atol=TOLERANCE
        )

    result_table_xarray = copy.deepcopy(result_tables_xarray[0])
    num_raw_flux_bins = len(result_table_xarray.coords[RAW_FLUX_BIN_DIM].values)

    for j in range(len(scalar_target_names)):
        for i in range(num_raw_flux_bins):
            these_example_counts = numpy.array([
                t[SCALAR_EXAMPLE_COUNT_KEY].values[j, i]
                for t in result_tables_xarray
            ])

            result_table_xarray[SCALAR_EXAMPLE_COUNT_KEY].values[j, i] = (
                numpy.sum(these_example_counts)
            )

            these_mean_stdevs = numpy.array([
                t[SCALAR_MEAN_STDEV_KEY].values[j, i]
                for t in result_tables_xarray
            ])
            these_mean_stdevs[numpy.isfinite(these_mean_stdevs) == False] = 0.

            result_table_xarray[SCALAR_MEAN_STDEV_KEY].values[
                j, i
            ] = numpy.sqrt(numpy.average(
                these_mean_stdevs ** 2, weights=these_example_counts
            ))

            these_rmse = numpy.array([
                t[SCALAR_RMSE_KEY].values[j, i] for t in result_tables_xarray
            ])
            these_rmse[numpy.isfinite(these_rmse) == False] = 0.

            result_table_xarray[SCALAR_RMSE_KEY].values[
                j, i
            ] = numpy.sqrt(numpy.average(
                these_rmse ** 2, weights=these_example_counts
            ))

            these_mean_mean_predictions = numpy.array([
                t[SCALAR_MEAN_MEAN_PREDICTION_KEY].values[j, i]
                for t in result_tables_xarray
            ])
            these_mean_mean_predictions[
                numpy.isfinite(these_mean_mean_predictions) == False
            ] = 0.

            result_table_xarray[SCALAR_MEAN_MEAN_PREDICTION_KEY].values[
                j, i
            ] = numpy.average(
                these_mean_mean_predictions, weights=these_example_counts
            )

            these_mean_targets = numpy.array([
                t[SCALAR_MEAN_TARGET_KEY].values[j, i]
                for t in result_tables_xarray
            ])
            these_mean_targets[numpy.isfinite(these_mean_targets) == False] = 0.

            result_table_xarray[SCALAR_MEAN_TARGET_KEY].values[
                j, i
            ] = numpy.average(these_mean_targets, weights=these_example_counts)

        mean_stdev_by_bin = (
            result_table_xarray[SCALAR_MEAN_STDEV_KEY].values[j, :] + 0.
        )
        rmse_by_bin = result_table_xarray[SCALAR_RMSE_KEY].values[j, :] + 0.
        num_examples_by_bin = (
            result_table_xarray[SCALAR_EXAMPLE_COUNT_KEY].values[j, :] + 0
        )
        mean_stdev_by_bin[numpy.isfinite(mean_stdev_by_bin) == False] = 0.
        rmse_by_bin[numpy.isfinite(rmse_by_bin) == False] = 0.

        result_table_xarray[SCALAR_SSREL_KEY].values[j] = numpy.average(
            numpy.absolute(mean_stdev_by_bin - rmse_by_bin),
            weights=num_examples_by_bin
        )

        this_numer = numpy.sqrt(numpy.average(
            mean_stdev_by_bin ** 2, weights=num_examples_by_bin
        ))
        this_denom = numpy.sqrt(numpy.average(
            rmse_by_bin ** 2, weights=num_examples_by_bin
        ))
        result_table_xarray[SCALAR_SSRAT_KEY].values[j] = (
            this_numer / this_denom
        )

    num_net_flux_bins = len(result_table_xarray.coords[NET_FLUX_BIN_DIM].values)

    for j in range(len(aux_predicted_field_names)):
        for i in range(num_net_flux_bins):
            these_example_counts = numpy.array([
                t[AUX_EXAMPLE_COUNT_KEY].values[j, i]
                for t in result_tables_xarray
            ])

            result_table_xarray[AUX_EXAMPLE_COUNT_KEY].values[j, i] = (
                numpy.sum(these_example_counts)
            )

            these_mean_stdevs = numpy.array([
                t[AUX_MEAN_STDEV_KEY].values[j, i]
                for t in result_tables_xarray
            ])
            these_mean_stdevs[numpy.isfinite(these_mean_stdevs) == False] = 0.

            result_table_xarray[AUX_MEAN_STDEV_KEY].values[
                j, i
            ] = numpy.sqrt(numpy.average(
                these_mean_stdevs ** 2, weights=these_example_counts
            ))

            these_rmse = numpy.array([
                t[AUX_RMSE_KEY].values[j, i] for t in result_tables_xarray
            ])
            these_rmse[numpy.isfinite(these_rmse) == False] = 0.

            result_table_xarray[AUX_RMSE_KEY].values[
                j, i
            ] = numpy.sqrt(numpy.average(
                these_rmse ** 2, weights=these_example_counts
            ))

            these_mean_mean_predictions = numpy.array([
                t[AUX_MEAN_MEAN_PREDICTION_KEY].values[j, i]
                for t in result_tables_xarray
            ])
            these_mean_mean_predictions[
                numpy.isfinite(these_mean_mean_predictions) == False
            ] = 0.

            result_table_xarray[AUX_MEAN_MEAN_PREDICTION_KEY].values[
                j, i
            ] = numpy.average(
                these_mean_mean_predictions, weights=these_example_counts
            )

            these_mean_targets = numpy.array([
                t[AUX_MEAN_TARGET_KEY].values[j, i]
                for t in result_tables_xarray
            ])
            these_mean_targets[numpy.isfinite(these_mean_targets) == False] = 0.

            result_table_xarray[AUX_MEAN_TARGET_KEY].values[
                j, i
            ] = numpy.average(these_mean_targets, weights=these_example_counts)

        mean_stdev_by_bin = (
            result_table_xarray[AUX_MEAN_STDEV_KEY].values[j, :]
        )
        rmse_by_bin = result_table_xarray[AUX_RMSE_KEY].values[j, :]
        num_examples_by_bin = (
            result_table_xarray[AUX_EXAMPLE_COUNT_KEY].values[j, :]
        )
        mean_stdev_by_bin[numpy.isfinite(mean_stdev_by_bin) == False] = 0.
        rmse_by_bin[numpy.isfinite(rmse_by_bin) == False] = 0.

        result_table_xarray[AUX_SSREL_KEY].values[j] = numpy.average(
            numpy.absolute(mean_stdev_by_bin - rmse_by_bin),
            weights=num_examples_by_bin
        )

        this_numer = numpy.sqrt(numpy.average(
            mean_stdev_by_bin ** 2, weights=num_examples_by_bin
        ))
        this_denom = numpy.sqrt(numpy.average(
            rmse_by_bin ** 2, weights=num_examples_by_bin
        ))
        result_table_xarray[AUX_SSRAT_KEY].values[j] = this_numer / this_denom

    num_heating_rate_bins = len(
        result_table_xarray.coords[HEATING_RATE_BIN_DIM].values
    )

    for j in range(len(vector_target_names)):
        for i in range(num_heating_rate_bins):
            these_example_counts = numpy.array([
                t[VECTOR_FLAT_EXAMPLE_COUNT_KEY].values[j, i]
                for t in result_tables_xarray
            ])

            result_table_xarray[VECTOR_FLAT_EXAMPLE_COUNT_KEY].values[j, i] = (
                numpy.sum(these_example_counts)
            )

            these_mean_stdevs = numpy.array([
                t[VECTOR_FLAT_MEAN_STDEV_KEY].values[j, i]
                for t in result_tables_xarray
            ])
            these_mean_stdevs[numpy.isfinite(these_mean_stdevs) == False] = 0.

            result_table_xarray[VECTOR_FLAT_MEAN_STDEV_KEY].values[
                j, i
            ] = numpy.sqrt(numpy.average(
                these_mean_stdevs ** 2, weights=these_example_counts
            ))

            these_rmse = numpy.array([
                t[VECTOR_FLAT_RMSE_KEY].values[j, i]
                for t in result_tables_xarray
            ])
            these_rmse[numpy.isfinite(these_rmse) == False] = 0.

            result_table_xarray[VECTOR_FLAT_RMSE_KEY].values[
                j, i
            ] = numpy.sqrt(numpy.average(
                these_rmse ** 2, weights=these_example_counts
            ))

            these_mean_mean_predictions = numpy.array([
                t[VECTOR_FLAT_MEAN_MEAN_PREDICTION_KEY].values[j, i]
                for t in result_tables_xarray
            ])
            these_mean_mean_predictions[
                numpy.isfinite(these_mean_mean_predictions) == False
            ] = 0.

            result_table_xarray[VECTOR_FLAT_MEAN_MEAN_PREDICTION_KEY].values[
                j, i
            ] = numpy.average(
                these_mean_mean_predictions, weights=these_example_counts
            )

            these_mean_targets = numpy.array([
                t[VECTOR_FLAT_MEAN_TARGET_KEY].values[j, i]
                for t in result_tables_xarray
            ])
            these_mean_targets[numpy.isfinite(these_mean_targets) == False] = 0.

            result_table_xarray[VECTOR_FLAT_MEAN_TARGET_KEY].values[
                j, i
            ] = numpy.average(these_mean_targets, weights=these_example_counts)

        mean_stdev_by_bin = (
            result_table_xarray[VECTOR_FLAT_MEAN_STDEV_KEY].values[j, :]
        )
        rmse_by_bin = result_table_xarray[VECTOR_FLAT_RMSE_KEY].values[j, :]
        num_examples_by_bin = (
            result_table_xarray[VECTOR_FLAT_EXAMPLE_COUNT_KEY].values[j, :]
        )
        mean_stdev_by_bin[numpy.isfinite(mean_stdev_by_bin) == False] = 0.
        rmse_by_bin[numpy.isfinite(rmse_by_bin) == False] = 0.

        result_table_xarray[VECTOR_FLAT_SSREL_KEY].values[j] = numpy.average(
            numpy.absolute(mean_stdev_by_bin - rmse_by_bin),
            weights=num_examples_by_bin
        )

        this_numer = numpy.sqrt(numpy.average(
            mean_stdev_by_bin ** 2, weights=num_examples_by_bin
        ))
        this_denom = numpy.sqrt(numpy.average(
            rmse_by_bin ** 2, weights=num_examples_by_bin
        ))
        result_table_xarray[VECTOR_FLAT_SSRAT_KEY].values[j] = (
            this_numer / this_denom
        )

    for j in range(len(vector_target_names)):
        for k in range(len(heights_m_agl)):
            for i in range(num_heating_rate_bins):
                these_example_counts = numpy.array([
                    t[VECTOR_EXAMPLE_COUNT_KEY].values[j, k, i]
                    for t in result_tables_xarray
                ])

                result_table_xarray[VECTOR_EXAMPLE_COUNT_KEY].values[
                    j, k, i
                ] = numpy.sum(these_example_counts)

                these_mean_stdevs = numpy.array([
                    t[VECTOR_MEAN_STDEV_KEY].values[j, k, i]
                    for t in result_tables_xarray
                ])
                these_mean_stdevs[
                    numpy.isfinite(these_mean_stdevs) == False
                ] = 0.

                result_table_xarray[VECTOR_MEAN_STDEV_KEY].values[
                    j, k, i
                ] = numpy.sqrt(numpy.average(
                    these_mean_stdevs ** 2, weights=these_example_counts
                ))

                these_rmse = numpy.array([
                    t[VECTOR_RMSE_KEY].values[j, k, i]
                    for t in result_tables_xarray
                ])
                these_rmse[numpy.isfinite(these_rmse) == False] = 0.

                result_table_xarray[VECTOR_RMSE_KEY].values[
                    j, k, i
                ] = numpy.sqrt(numpy.average(
                    these_rmse ** 2, weights=these_example_counts
                ))

                these_mean_mean_predictions = numpy.array([
                    t[VECTOR_MEAN_MEAN_PREDICTION_KEY].values[j, k, i]
                    for t in result_tables_xarray
                ])
                these_mean_mean_predictions[
                    numpy.isfinite(these_mean_mean_predictions) == False
                ] = 0.

                result_table_xarray[VECTOR_MEAN_MEAN_PREDICTION_KEY].values[
                    j, k, i
                ] = numpy.average(
                    these_mean_mean_predictions, weights=these_example_counts
                )

                these_mean_targets = numpy.array([
                    t[VECTOR_MEAN_TARGET_KEY].values[j, k, i]
                    for t in result_tables_xarray
                ])
                these_mean_targets[
                    numpy.isfinite(these_mean_targets) == False
                ] = 0.

                result_table_xarray[VECTOR_MEAN_TARGET_KEY].values[
                    j, k, i
                ] = numpy.average(
                    these_mean_targets, weights=these_example_counts
                )

            mean_stdev_by_bin = (
                result_table_xarray[VECTOR_MEAN_STDEV_KEY].values[j, k, :]
            )
            rmse_by_bin = result_table_xarray[VECTOR_RMSE_KEY].values[j, k, :]
            num_examples_by_bin = (
                result_table_xarray[VECTOR_EXAMPLE_COUNT_KEY].values[j, k, :]
            )
            mean_stdev_by_bin[numpy.isfinite(mean_stdev_by_bin) == False] = 0.
            rmse_by_bin[numpy.isfinite(rmse_by_bin) == False] = 0.

            result_table_xarray[VECTOR_SSREL_KEY].values[j, k] = numpy.average(
                numpy.absolute(mean_stdev_by_bin - rmse_by_bin),
                weights=num_examples_by_bin
            )

            this_numer = numpy.sqrt(numpy.average(
                mean_stdev_by_bin ** 2, weights=num_examples_by_bin
            ))
            this_denom = numpy.sqrt(numpy.average(
                rmse_by_bin ** 2, weights=num_examples_by_bin
            ))
            result_table_xarray[VECTOR_SSRAT_KEY].values[j, k] = (
                this_numer / this_denom
            )

    result_table_xarray.attrs[PREDICTION_FILE_KEY] = ' '.join([
        '{0:s}'.format(f) for f in prediction_file_names
    ])
    return result_table_xarray


def write_results(result_table_xarray, netcdf_file_name):
    """Writes spread-vs.-skill results to NetCDF file.

    :param result_table_xarray: xarray table in format returned by
        `get_results_all_vars`.
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)

    result_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )


def read_results(netcdf_file_name):
    """Reads spread-vs.-skill results from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: result_table_xarray: xarray table.  Documentation in the
        xarray table should make values self-explanatory.
    """

    return xarray.open_dataset(netcdf_file_name)
