"""Methods for calibrating uncertainty estimates."""

import os
import sys
import numpy
import xarray
from scipy.interpolate import interp1d

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import error_checking
import prediction_io
import spread_skill_utils as ss_utils
import uq_evaluation
import example_utils
import neural_net

TOLERANCE = 1e-6

SCALAR_BIN_EDGE_KEY = 'scalar_bin_edge_prediction_stdev'
SCALAR_STDEV_INFLATION_KEY = 'scalar_stdev_inflation_factor'
VECTOR_BIN_EDGE_KEY = 'vector_bin_edge_prediction_stdev'
VECTOR_STDEV_INFLATION_KEY = 'vector_stdev_inflation_factor'

BIN_DIM = 'bin'
BIN_EDGE_DIM = 'bin_edge'
SCALAR_FIELD_DIM = uq_evaluation.SCALAR_FIELD_DIM
VECTOR_FIELD_DIM = uq_evaluation.VECTOR_FIELD_DIM
HEIGHT_DIM = uq_evaluation.HEIGHT_DIM

MODEL_FILE_KEY = uq_evaluation.MODEL_FILE_KEY
PREDICTION_FILE_KEY = uq_evaluation.PREDICTION_FILE_KEY


def _train_model_one_variable(
        target_values, prediction_matrix, bin_edge_prediction_stdevs):
    """Trains uncertainty-calibration model for one target variable.

    E = number of examples
    S = ensemble size
    B = number of bins

    :param target_values: length-E numpy array of target values.
    :param prediction_matrix: E-by-S numpy array of predictions.
    :param bin_edge_prediction_stdevs: length-(B + 1) numpy array of bin
        cutoffs.  These cutoffs will be applied to the standard deviation of the
        predictive distribution (one stdev per example).
    :return: bin_edge_prediction_stdevs: length-(B + 1) numpy array, where the
        [i]th and [i + 1]th entries are the edges for the [i]th bin.
    :return: stdev_inflation_factors: length-B numpy array of inflation factors
        for standard deviation.
    """

    spread_skill_result_dict = ss_utils.get_results_one_var(
        target_values=target_values, prediction_matrix=prediction_matrix,
        bin_edge_prediction_stdevs=bin_edge_prediction_stdevs
    )

    bin_edge_prediction_stdevs = (
        spread_skill_result_dict[ss_utils.BIN_EDGE_PREDICTION_STDEVS_KEY]
    )
    stdev_inflation_factors = (
        spread_skill_result_dict[ss_utils.RMSE_VALUES_KEY] /
        spread_skill_result_dict[ss_utils.MEAN_PREDICTION_STDEVS_KEY]
    )

    nan_flags = numpy.isnan(stdev_inflation_factors)
    if not numpy.any(nan_flags):
        return bin_edge_prediction_stdevs, stdev_inflation_factors

    nan_indices = numpy.where(nan_flags)[0]
    real_indices = numpy.where(numpy.invert(nan_flags))[0]
    bin_center_prediction_stdevs = 0.5 * (
        bin_edge_prediction_stdevs[:-1] + bin_edge_prediction_stdevs[1:]
    )

    interp_object = interp1d(
        x=bin_center_prediction_stdevs[real_indices],
        y=stdev_inflation_factors[real_indices],
        kind='linear', bounds_error=False, assume_sorted=True,
        fill_value=(
            stdev_inflation_factors[real_indices[0]],
            stdev_inflation_factors[real_indices[-1]]
        )
    )

    stdev_inflation_factors[nan_indices] = interp_object(
        bin_center_prediction_stdevs[nan_indices]
    )

    return bin_edge_prediction_stdevs, stdev_inflation_factors


def _apply_model_one_variable(
        prediction_matrix, bin_edge_prediction_stdevs, stdev_inflation_factors):
    """Applies uncertainty-calibration model to one target variable.

    :param prediction_matrix: See doc for `_train_model_one_variable`.
    :param bin_edge_prediction_stdevs: Same.
    :param stdev_inflation_factors: Same.
    :return: prediction_matrix: Same as input but with adjusted values.
    """

    ensemble_size = prediction_matrix.shape[-1]
    error_checking.assert_is_greater(ensemble_size, 1)

    mean_prediction_matrix = numpy.mean(prediction_matrix, axis=-1)
    mean_prediction_matrix = numpy.repeat(
        numpy.expand_dims(mean_prediction_matrix, -1),
        axis=-1, repeats=ensemble_size
    )

    prediction_stdev_matrix = numpy.std(prediction_matrix, ddof=1, axis=-1)
    prediction_stdev_matrix = numpy.repeat(
        numpy.expand_dims(prediction_stdev_matrix, -1),
        axis=-1, repeats=ensemble_size
    )

    num_bins = len(bin_edge_prediction_stdevs) - 1
    new_prediction_matrix = prediction_matrix + 0.

    for k in range(num_bins):
        idx = numpy.where(numpy.logical_and(
            prediction_stdev_matrix >= bin_edge_prediction_stdevs[k],
            prediction_stdev_matrix < bin_edge_prediction_stdevs[k + 1]
        ))

        new_prediction_matrix[idx] = (
            mean_prediction_matrix[idx] +
            stdev_inflation_factors[k] *
            (prediction_matrix[idx] - mean_prediction_matrix[idx])
        )

    assert not numpy.any(numpy.isnan(new_prediction_matrix))

    return new_prediction_matrix


def train_models_all_vars(
        prediction_file_name, num_spread_bins, min_spread_percentile,
        max_spread_percentile):
    """Trains uncertainty-calibration model for each target variable.

    T_v = number of vector target variables
    T_s = number of scalar target variables
    H = number of heights
    B = number of spread bins

    :param prediction_file_name: Path to input file (will be read by
        `prediction_io.read_file`).
    :param num_spread_bins: Number of spread bins for each target variable.
    :param min_spread_percentile: Minimum spread percentile, applied to each
        target variable separately to establish the lower edge of the lowest
        spread bin.
    :param max_spread_percentile: Max spread percentile, applied to each target
        variable separately to establish the upper edge of the highest spread
        bin.
    :return: result_table_xarray: xarray table with results.  Metadata and
        variable names in this table should make it self-explanatory.
    """

    error_checking.assert_is_integer(num_spread_bins)
    error_checking.assert_is_geq(num_spread_bins, 10)
    error_checking.assert_is_leq(num_spread_bins, 1000)
    error_checking.assert_is_geq(min_spread_percentile, 0.)
    error_checking.assert_is_leq(min_spread_percentile, 10.)
    error_checking.assert_is_geq(max_spread_percentile, 90.)
    error_checking.assert_is_leq(max_spread_percentile, 100.)

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

    these_dim_no_edge = (num_scalar_targets, num_spread_bins)
    these_dim_keys_no_edge = (SCALAR_FIELD_DIM, BIN_DIM)
    these_dim_with_edge = (num_scalar_targets, num_spread_bins + 1)
    these_dim_keys_with_edge = (SCALAR_FIELD_DIM, BIN_EDGE_DIM)

    main_data_dict = {
        SCALAR_BIN_EDGE_KEY: (
            these_dim_keys_with_edge, numpy.full(these_dim_with_edge, numpy.nan)
        ),
        SCALAR_STDEV_INFLATION_KEY: (
            these_dim_keys_no_edge, numpy.full(these_dim_no_edge, numpy.nan)
        )
    }

    these_dim_no_edge = (num_vector_targets, num_heights, num_spread_bins)
    these_dim_keys_no_edge = (VECTOR_FIELD_DIM, HEIGHT_DIM, BIN_DIM)
    these_dim_with_edge = (num_vector_targets, num_heights, num_spread_bins + 1)
    these_dim_keys_with_edge = (VECTOR_FIELD_DIM, HEIGHT_DIM, BIN_EDGE_DIM)

    main_data_dict.update({
        VECTOR_BIN_EDGE_KEY: (
            these_dim_keys_with_edge, numpy.full(these_dim_with_edge, numpy.nan)
        ),
        VECTOR_STDEV_INFLATION_KEY: (
            these_dim_keys_no_edge, numpy.full(these_dim_no_edge, numpy.nan)
        )
    })

    metadata_dict = {
        SCALAR_FIELD_DIM:
            generator_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY],
        HEIGHT_DIM: heights_m_agl,
        VECTOR_FIELD_DIM:
            generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY],
        BIN_DIM: numpy.linspace(
            0, num_spread_bins - 1, num=num_spread_bins, dtype=int
        )
    }

    result_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict
    )
    result_table_xarray.attrs[MODEL_FILE_KEY] = model_file_name
    result_table_xarray.attrs[PREDICTION_FILE_KEY] = prediction_file_name
    t = result_table_xarray

    for k in range(num_scalar_targets):
        print('Training uncertainty calibration for {0:s}...'.format(
            generator_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY][k]
        ))

        these_stdevs = numpy.std(
            scalar_prediction_matrix[:, k, :], ddof=1, axis=-1
        )
        these_bin_edges = numpy.linspace(
            numpy.percentile(these_stdevs, min_spread_percentile),
            numpy.percentile(these_stdevs, max_spread_percentile),
            num=num_spread_bins + 1, dtype=float
        )[1:-1]

        these_fudge_factors = TOLERANCE * numpy.linspace(
            1, len(these_bin_edges), num=len(these_bin_edges), dtype=float
        )
        these_bin_edges = these_bin_edges + these_fudge_factors

        (
            t[SCALAR_BIN_EDGE_KEY].values[k, :],
            t[SCALAR_STDEV_INFLATION_KEY].values[k, :]
        ) = _train_model_one_variable(
            target_values=scalar_target_matrix[:, k],
            prediction_matrix=scalar_prediction_matrix[:, k, :],
            bin_edge_prediction_stdevs=these_bin_edges
        )

    for k in range(num_vector_targets):
        for j in range(num_heights):
            print((
                'Training uncertainty calibration for {0:s} at {1:d} m AGL...'
            ).format(
                generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY][k],
                int(numpy.round(heights_m_agl[j]))
            ))

            these_stdevs = numpy.std(
                vector_prediction_matrix[:, j, k, :], ddof=1, axis=-1
            )
            these_bin_edges = numpy.linspace(
                numpy.percentile(these_stdevs, min_spread_percentile),
                numpy.percentile(these_stdevs, max_spread_percentile),
                num=num_spread_bins + 1, dtype=float
            )[1:-1]

            these_fudge_factors = TOLERANCE * numpy.linspace(
                1, len(these_bin_edges), num=len(these_bin_edges), dtype=float
            )
            these_bin_edges = these_bin_edges + these_fudge_factors

            (
                t[VECTOR_BIN_EDGE_KEY].values[k, j, :],
                t[VECTOR_STDEV_INFLATION_KEY].values[k, j, :]
            ) = _train_model_one_variable(
                target_values=vector_target_matrix[:, j, k],
                prediction_matrix=vector_prediction_matrix[:, j, k, :],
                bin_edge_prediction_stdevs=these_bin_edges
            )

    result_table_xarray = t
    return result_table_xarray


def apply_models_all_vars(prediction_file_name, uncertainty_calib_table_xarray):
    """Applies uncertainty-calibration model for each target variable.

    :param prediction_file_name: Path to input file (will be read by
        `prediction_io.read_file`).
    :param uncertainty_calib_table_xarray: xarray table in format returned by
        `train_models_all_vars`.
    :return: prediction_dict: Dictionary in format returned by
        `prediction_io.read_file`, except containing predictions with
        calibrated uncertainty.
    """

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
    scalar_target_names = (
        generator_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY]
    )
    vector_target_names = (
        generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY]
    )

    uct = uncertainty_calib_table_xarray
    pd = prediction_dict

    for k in range(len(scalar_target_names)):
        print('Applying uncertainty calibration for {0:s}...'.format(
            scalar_target_names[k]
        ))

        var_idx = uct.coords[SCALAR_FIELD_DIM].values.tolist().index(
            scalar_target_names[k]
        )

        pd[prediction_io.SCALAR_PREDICTIONS_KEY][:, k, :] = (
            _apply_model_one_variable(
                prediction_matrix=
                pd[prediction_io.SCALAR_PREDICTIONS_KEY][:, k, :],
                bin_edge_prediction_stdevs=
                uct[SCALAR_BIN_EDGE_KEY].values[var_idx, :],
                stdev_inflation_factors=
                uct[SCALAR_STDEV_INFLATION_KEY].values[var_idx, :]
            )
        )

        pd[prediction_io.SCALAR_PREDICTIONS_KEY][:, k, :] = numpy.maximum(
            pd[prediction_io.SCALAR_PREDICTIONS_KEY][:, k, :],
            0.
        )

    for k in range(len(vector_target_names)):
        for j in range(len(heights_m_agl)):
            print((
                'Applying uncertainty calibration for {0:s} at {1:d} m AGL...'
            ).format(
                vector_target_names[k],
                int(numpy.round(heights_m_agl[j]))
            ))

            var_idx = uct.coords[VECTOR_FIELD_DIM].values.tolist().index(
                vector_target_names[k]
            )
            hgt_idx = numpy.where(
                numpy.absolute(uct.coords[HEIGHT_DIM].values - heights_m_agl[j])
                < TOLERANCE
            )[0][0]

            pd[prediction_io.VECTOR_PREDICTIONS_KEY][:, j, k, :] = (
                _apply_model_one_variable(
                    prediction_matrix=
                    pd[prediction_io.VECTOR_PREDICTIONS_KEY][:, j, k, :],
                    bin_edge_prediction_stdevs=
                    uct[SCALAR_BIN_EDGE_KEY].values[var_idx, hgt_idx, :],
                    stdev_inflation_factors=
                    uct[SCALAR_STDEV_INFLATION_KEY].values[var_idx, hgt_idx, :]
                )
            )

            if (
                    vector_target_names[k] ==
                    example_utils.LONGWAVE_HEATING_RATE_NAME
            ):
                continue

            pd[prediction_io.VECTOR_PREDICTIONS_KEY][:, j, k, :] = (
                numpy.maximum(
                    pd[prediction_io.VECTOR_PREDICTIONS_KEY][:, j, k, :],
                    0.
                )
            )

    prediction_dict = pd
    return prediction_dict


def write_file(result_table_xarray, netcdf_file_name):
    """Writes uncertainty-calibration models to NetCDF file.

    :param result_table_xarray: xarray table in format returned by
        `train_models_all_vars`.
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)

    result_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )


def read_results(netcdf_file_name):
    """Reads uncertainty-calibration models from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: result_table_xarray: xarray table.  Documentation in the
        xarray table should make values self-explanatory.
    """

    return xarray.open_dataset(netcdf_file_name)
