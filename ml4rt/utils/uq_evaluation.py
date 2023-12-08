"""Evaluation methods for uncertainty quantification (UQ)."""

import os
import copy
import numpy
import xarray
from gewittergefahr.gg_utils import error_checking
from ml4rt.io import prediction_io
from ml4rt.utils import example_utils
from ml4rt.machine_learning import neural_net

TOLERANCE = 1e-6

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
WAVELENGTH_DIM = 'wavelength_metres'
AUX_TARGET_FIELD_DIM = 'aux_target_field'
AUX_PREDICTED_FIELD_DIM = 'aux_predicted_field'

MODEL_FILE_KEY = 'model_file_name'
PREDICTION_FILE_KEY = 'prediction_file_name'


def get_aux_fields(prediction_dict, example_dict):
    """Returns auxiliary fields.

    E = number of examples
    W = number of wavelengths
    F = number of pairs of auxiliary fields
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
    aux_prediction_dict['aux_target_matrix']: E-by-W-by-F numpy array of target
        (actual) values.
    aux_prediction_dict['aux_prediction_matrix']: E-by-W-by-F-by-S numpy array
        of predicted values.
    aux_prediction_dict['shortwave_surface_down_flux_index']: Array index of
        shortwave surface downwelling flux in `mean_training_example_dict`.  If
        not available, this is -1.
    aux_prediction_dict['longwave_surface_down_flux_index']: Same but for
        longwave.
    aux_prediction_dict['shortwave_toa_up_flux_index']: Array index of shortwave
        TOA upwelling flux in `mean_training_example_dict`.  If not available,
        this is -1.
    aux_prediction_dict['longwave_toa_up_flux_index']: Same but for longwave.
    """

    scalar_target_matrix = prediction_dict[prediction_io.SCALAR_TARGETS_KEY]
    scalar_prediction_matrix = (
        prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY]
    )

    num_examples = scalar_prediction_matrix.shape[0]
    num_wavelengths = scalar_prediction_matrix.shape[1]
    num_ensemble_members = scalar_prediction_matrix.shape[3]

    aux_target_matrix = numpy.full(
        (num_examples, num_wavelengths, 0), numpy.nan
    )
    aux_prediction_matrix = numpy.full(
        (num_examples, num_wavelengths, 0, num_ensemble_members), numpy.nan
    )
    aux_target_field_names = []
    aux_predicted_field_names = []

    scalar_target_names = example_dict[example_utils.SCALAR_TARGET_NAMES_KEY]
    these_flux_names = [
        example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME,
        example_utils.SHORTWAVE_TOA_UP_FLUX_NAME
    ]

    aux_prediction_dict = {
        SHORTWAVE_SURFACE_DOWN_FLUX_INDEX_KEY: -1,
        SHORTWAVE_TOA_UP_FLUX_INDEX_KEY: -1,
        LONGWAVE_SURFACE_DOWN_FLUX_INDEX_KEY: -1,
        LONGWAVE_TOA_UP_FLUX_INDEX_KEY: -1
    }

    if all([n in scalar_target_names for n in these_flux_names]):
        d_idx = scalar_target_names.index(
            example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME
        )
        u_idx = scalar_target_names.index(
            example_utils.SHORTWAVE_TOA_UP_FLUX_NAME
        )

        aux_target_field_names.append(SHORTWAVE_NET_FLUX_NAME)
        aux_predicted_field_names.append(SHORTWAVE_NET_FLUX_NAME)

        this_target_matrix = (
            scalar_target_matrix[..., [d_idx]] -
            scalar_target_matrix[..., [u_idx]]
        )
        aux_target_matrix = numpy.concatenate(
            (aux_target_matrix, this_target_matrix), axis=-1
        )

        this_prediction_matrix = (
            scalar_prediction_matrix[..., [d_idx], :]
            - scalar_prediction_matrix[..., [u_idx], :]
        )
        aux_prediction_matrix = numpy.concatenate(
            (aux_prediction_matrix, this_prediction_matrix), axis=-2
        )

        aux_prediction_dict[SHORTWAVE_SURFACE_DOWN_FLUX_INDEX_KEY] = d_idx + 0
        aux_prediction_dict[SHORTWAVE_TOA_UP_FLUX_INDEX_KEY] = u_idx + 0

    these_flux_names = [
        example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME,
        example_utils.LONGWAVE_TOA_UP_FLUX_NAME
    ]

    if all([n in scalar_target_names for n in these_flux_names]):
        d_idx = scalar_target_names.index(
            example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME
        )
        u_idx = scalar_target_names.index(
            example_utils.LONGWAVE_TOA_UP_FLUX_NAME
        )

        aux_target_field_names.append(LONGWAVE_NET_FLUX_NAME)
        aux_predicted_field_names.append(LONGWAVE_NET_FLUX_NAME)

        this_target_matrix = (
            scalar_target_matrix[..., [d_idx]] -
            scalar_target_matrix[..., [u_idx]]
        )
        aux_target_matrix = numpy.concatenate(
            (aux_target_matrix, this_target_matrix), axis=-1
        )

        this_prediction_matrix = (
            scalar_prediction_matrix[..., [d_idx], :] -
            scalar_prediction_matrix[..., [u_idx], :]
        )
        aux_prediction_matrix = numpy.concatenate(
            (aux_prediction_matrix, this_prediction_matrix), axis=-2
        )

        aux_prediction_dict[LONGWAVE_SURFACE_DOWN_FLUX_INDEX_KEY] = d_idx + 0
        aux_prediction_dict[LONGWAVE_TOA_UP_FLUX_INDEX_KEY] = u_idx + 0

    aux_prediction_dict.update({
        AUX_TARGET_NAMES_KEY: aux_target_field_names,
        AUX_PREDICTED_NAMES_KEY: aux_predicted_field_names,
        AUX_TARGET_VALS_KEY: aux_target_matrix,
        AUX_PREDICTED_VALS_KEY: aux_prediction_matrix
    })

    return aux_prediction_dict


def check_results_before_merging(result_tables_xarray):
    """Before merging results over many examples, checks input args.

    T = number of result tables = number of example sets over which to merge

    :param result_tables_xarray: length-T of xarray tables, each created by
        the same method (crps_utils.get_crps_related_scores_all_vars or
        spread_skill_utils.get_results_all_vars or
        discard_test_utils.run_discard_test or
        pit_utils.get_histogram_all_vars),
        each containing results for a different set of examples.
    :return: prediction_file_names: length-T list of paths to prediction files.
    :return: num_examples_by_table: length-T numpy array of example counts.
    """

    scalar_target_names = (
        result_tables_xarray[0].coords[SCALAR_FIELD_DIM].values.tolist()
    )
    vector_target_names = (
        result_tables_xarray[0].coords[VECTOR_FIELD_DIM].values.tolist()
    )
    heights_m_agl = result_tables_xarray[0].coords[HEIGHT_DIM].values
    wavelengths_metres = result_tables_xarray[0].coords[WAVELENGTH_DIM].values

    try:
        aux_target_field_names = (
            result_tables_xarray[0].coords[AUX_TARGET_FIELD_DIM].values.tolist()
        )
        aux_predicted_field_names = result_tables_xarray[0].coords[
            AUX_PREDICTED_FIELD_DIM
        ].values.tolist()
    except:
        aux_target_field_names = []
        aux_predicted_field_names = []

    num_tables = len(result_tables_xarray)

    for i in range(1, num_tables):
        assert (
            result_tables_xarray[i].coords[SCALAR_FIELD_DIM].values.tolist() ==
            scalar_target_names
        )
        assert (
            result_tables_xarray[i].coords[VECTOR_FIELD_DIM].values.tolist() ==
            vector_target_names
        )
        assert numpy.allclose(
            result_tables_xarray[i].coords[HEIGHT_DIM].values,
            heights_m_agl,
            atol=TOLERANCE
        )
        assert numpy.allclose(
            result_tables_xarray[i].coords[WAVELENGTH_DIM].values,
            wavelengths_metres,
            atol=TOLERANCE
        )

        try:
            these_target_field_names = result_tables_xarray[i].coords[
                AUX_TARGET_FIELD_DIM
            ].values.tolist()

            these_predicted_field_names = result_tables_xarray[i].coords[
                AUX_PREDICTED_FIELD_DIM
            ].values.tolist()
        except:
            these_target_field_names = []
            these_predicted_field_names = []

        assert these_target_field_names == aux_target_field_names
        assert these_predicted_field_names == aux_predicted_field_names
        assert (
            result_tables_xarray[i].attrs[MODEL_FILE_KEY] ==
            result_tables_xarray[0].attrs[MODEL_FILE_KEY]
        )

    prediction_file_names = [
        t.attrs[PREDICTION_FILE_KEY] for t in result_tables_xarray
    ]
    assert len(set(prediction_file_names)) == len(prediction_file_names)

    num_examples_by_table = numpy.full(num_tables, -1, dtype=int)

    for i in range(num_tables):
        print('Reading data from: "{0:s}"...'.format(
            result_tables_xarray[i].attrs[PREDICTION_FILE_KEY]
        ))
        this_prediction_dict = prediction_io.read_file(
            result_tables_xarray[i].attrs[PREDICTION_FILE_KEY]
        )
        num_examples_by_table[i] = len(
            this_prediction_dict[prediction_io.EXAMPLE_IDS_KEY]
        )

    return prediction_file_names, num_examples_by_table


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

        elementwise_stdev_matrix_k_day01 = numpy.std(
            predicted_hr_matrix_k_day01, ddof=1, axis=-1
        )

        all_axes_except0 = tuple(
            range(1, elementwise_stdev_matrix_k_day01.ndim)
        )
        return numpy.sqrt(numpy.mean(
            elementwise_stdev_matrix_k_day01 ** 2, axis=all_axes_except0
        ))

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

        pdict = prediction_dict
        num_ensemble_members = (
            pdict[prediction_io.VECTOR_PREDICTIONS_KEY].shape[-1]
        )
        assert num_ensemble_members > 1

        model_file_name = pdict[prediction_io.MODEL_FILE_KEY]
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

        num_examples = pdict[prediction_io.VECTOR_PREDICTIONS_KEY].shape[0]
        num_wavelengths = pdict[prediction_io.VECTOR_PREDICTIONS_KEY].shape[2]
        predicted_flux_matrix_w_m02 = numpy.full(
            (num_examples, num_wavelengths, 0, num_ensemble_members), numpy.nan
        )

        for this_name in SHORTWAVE_RAW_FLUX_NAMES + LONGWAVE_RAW_FLUX_NAMES:
            if this_name not in scalar_target_names:
                continue

            j = scalar_target_names.index(this_name)

            predicted_flux_matrix_w_m02 = numpy.concatenate((
                predicted_flux_matrix_w_m02,
                pdict[prediction_io.SCALAR_PREDICTIONS_KEY][..., [j], :]
            ), axis=-2)

        if all([n in scalar_target_names for n in SHORTWAVE_RAW_FLUX_NAMES]):
            d_idx = scalar_target_names.index(
                example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME
            )
            u_idx = scalar_target_names.index(
                example_utils.SHORTWAVE_TOA_UP_FLUX_NAME
            )
            net_flux_matrix_w_m02 = (
                pdict[prediction_io.SCALAR_PREDICTIONS_KEY][..., [d_idx], :] -
                pdict[prediction_io.SCALAR_PREDICTIONS_KEY][..., [u_idx], :]
            )
            predicted_flux_matrix_w_m02 = numpy.concatenate(
                (predicted_flux_matrix_w_m02, net_flux_matrix_w_m02), axis=-2
            )

        elementwise_stdev_matrix_w_m02 = numpy.std(
            predicted_flux_matrix_w_m02, ddof=1, axis=-1
        )

        all_axes_except0 = tuple(
            range(1, elementwise_stdev_matrix_w_m02.ndim)
        )
        return numpy.sqrt(numpy.mean(
            elementwise_stdev_matrix_w_m02 ** 2, axis=all_axes_except0
        ))

    return uncertainty_function


def make_error_function_dwmse_1height():
    """Makes function to compute DWMSE for heating rate at one height.

    DWMSE = dual-weighted mean squared error

    :return: error_function: Function handle.
    """

    def error_function(
            actual_hr_matrix_k_day01, predicted_hr_matrix_k_day01,
            use_example_flags):
        """Computes DWMSE.

        E = number of examples
        W = number of wavelengths
        S = ensemble size

        :param actual_hr_matrix_k_day01: E-by-W numpy array of actual heating
            rates at one height.
        :param predicted_hr_matrix_k_day01: E-by-W-by-S numpy array of predicted
            heating rates at the same height.
        :param use_example_flags: length-E numpy array of Boolean flags,
            indicating which examples to use.
        :return: dwmse_k3_day03: Scalar DWMSE value.
        """

        mean_pred_hr_matrix_k_day01 = numpy.mean(
            predicted_hr_matrix_k_day01[use_example_flags, ...], axis=-1
        )

        weight_matrix_k_day01 = numpy.maximum(
            numpy.absolute(mean_pred_hr_matrix_k_day01),
            numpy.absolute(actual_hr_matrix_k_day01[use_example_flags, :])
        )
        squared_error_matrix_k2_day02 = (
            mean_pred_hr_matrix_k_day01 -
            actual_hr_matrix_k_day01[use_example_flags, :]
        ) ** 2

        return numpy.mean(weight_matrix_k_day01 * squared_error_matrix_k2_day02)

    return error_function


def make_error_function_flux_mse_1var():
    """Makes function to compute MSE for one flux variable.

    :return: error_function: Function handle.
    """

    def error_function(actual_flux_matrix_w_m02, predicted_flux_matrix_w_m02,
                       use_example_flags):
        """Computes MSE.

        E = number of examples
        W = number of wavelengths
        S = ensemble size

        :param actual_flux_matrix_w_m02: E-by-W numpy array of actual values for
            one flux variable.
        :param predicted_flux_matrix_w_m02: E-by-W-by-S numpy array of predicted
            values for the same flux variable.
        :param use_example_flags: length-E numpy array of Boolean flags,
            indicating which examples to use.
        :return: mse_w2_m04: Scalar MSE value.
        """

        mean_pred_flux_matrix_w_m02 = numpy.mean(
            predicted_flux_matrix_w_m02[use_example_flags, ...], axis=-1
        )
        return numpy.mean(
            (mean_pred_flux_matrix_w_m02 -
             actual_flux_matrix_w_m02[use_example_flags, :])
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

        pdict = prediction_dict
        predicted_flux_matrix_w_m02 = numpy.mean(
            pdict[prediction_io.SCALAR_PREDICTIONS_KEY][use_example_flags, ...],
            axis=-1
        )
        actual_flux_matrix_w_m02 = (
            pdict[prediction_io.SCALAR_TARGETS_KEY][use_example_flags, ...]
        )

        predicted_net_flux_matrix_w_m02 = (
            predicted_flux_matrix_w_m02[..., 0] -
            predicted_flux_matrix_w_m02[..., 1]
        )
        actual_net_flux_matrix_w_m02 = (
            actual_flux_matrix_w_m02[..., 0] -
            actual_flux_matrix_w_m02[..., 1]
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
            pdict[prediction_io.VECTOR_PREDICTIONS_KEY][use_example_flags, ...],
            axis=-1
        )
        actual_hr_matrix_k_day01 = (
            pdict[prediction_io.VECTOR_TARGETS_KEY][use_example_flags, ...]
        )

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


def add_wavelength_dim_to_table(evaluation_table_xarray):
    """Adds wavelength dimension to UQ-evaluation table.

    :param evaluation_table_xarray: xarray table in format created by
        `crps_utils.get_crps_related_scores_all_vars` or
        `spread_skill_utils.get_results_all_vars` or
        `discard_test_utils.run_discard_test` or
        `pit_utils.get_histogram_all_vars`.
    :return: evaluation_table_xarray: Same but including wavelength dimension.
    """

    if WAVELENGTH_DIM in evaluation_table_xarray.coords:
        return evaluation_table_xarray

    data_dict = {}
    coord_dict = {}

    for var_name in evaluation_table_xarray.data_vars:
        dimension_keys = list(evaluation_table_xarray[var_name].dims)
        data_matrix = evaluation_table_xarray[var_name].values

        if SCALAR_FIELD_DIM in dimension_keys:
            i = dimension_keys.index(SCALAR_FIELD_DIM)
            dimension_keys.insert(i + 1, WAVELENGTH_DIM)
            data_matrix = numpy.expand_dims(data_matrix, axis=i + 1)

        if VECTOR_FIELD_DIM in dimension_keys:
            i = dimension_keys.index(VECTOR_FIELD_DIM)

            if (
                    HEIGHT_DIM in dimension_keys and
                    dimension_keys.index(HEIGHT_DIM) == i + 1
            ):
                dimension_keys.insert(i + 2, WAVELENGTH_DIM)
                data_matrix = numpy.expand_dims(data_matrix, axis=i + 2)
            else:
                dimension_keys.insert(i + 1, WAVELENGTH_DIM)
                data_matrix = numpy.expand_dims(data_matrix, axis=i + 1)

        if AUX_TARGET_FIELD_DIM in dimension_keys:
            i = dimension_keys.index(AUX_TARGET_FIELD_DIM)
            dimension_keys.insert(i + 1, WAVELENGTH_DIM)
            data_matrix = numpy.expand_dims(data_matrix, axis=i + 1)

        data_dict[var_name] = (copy.deepcopy(dimension_keys), data_matrix + 0.)

    for coord_name in evaluation_table_xarray.coords:
        coord_dict[coord_name] = (
            evaluation_table_xarray.coords[coord_name].values
        )

    coord_dict[WAVELENGTH_DIM] = numpy.array(
        [example_utils.DUMMY_BROADBAND_WAVELENGTH_METRES]
    )

    return xarray.Dataset(
        data_vars=data_dict, coords=coord_dict,
        attrs=evaluation_table_xarray.attrs
    )
