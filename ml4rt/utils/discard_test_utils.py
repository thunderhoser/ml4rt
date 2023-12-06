"""Helper methods for running discard test."""

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

UNCERTAINTY_THRESHOLD_DIM = 'uncertainty_threshold'
POST_DISCARD_ERROR_KEY = 'post_discard_error'
EXAMPLE_FRACTION_KEY = 'example_fraction'
MONO_FRACTION_KEY = 'mono_fraction'
MEAN_DI_KEY = 'mean_discard_improvement'

SCALAR_POST_DISCARD_ERROR_KEY = 'scalar_post_discard_error'
SCALAR_MONO_FRACTION_KEY = 'scalar_mono_fraction'
SCALAR_MEAN_DI_KEY = 'scalar_mean_discard_improvement'
SCALAR_MEAN_MEAN_PREDICTION_KEY = 'scalar_mean_mean_prediction'
SCALAR_MEAN_TARGET_KEY = 'scalar_mean_target_value'

VECTOR_POST_DISCARD_ERROR_KEY = 'vector_post_discard_error'
VECTOR_MONO_FRACTION_KEY = 'vector_mono_fraction'
VECTOR_MEAN_DI_KEY = 'vector_mean_discard_improvement'
VECTOR_MEAN_MEAN_PREDICTION_KEY = 'vector_mean_mean_prediction'
VECTOR_MEAN_TARGET_KEY = 'vector_mean_target_value'

VECTOR_FLAT_POST_DISCARD_ERROR_KEY = 'vector_flat_post_discard_error'
VECTOR_FLAT_MONO_FRACTION_KEY = 'vector_flat_mono_fraction'
VECTOR_FLAT_MEAN_DI_KEY = (
    'vector_flat_mean_discard_improvement'
)
VECTOR_FLAT_MEAN_MEAN_PREDICTION_KEY = 'vector_flat_mean_mean_prediction'
VECTOR_FLAT_MEAN_TARGET_KEY = 'vector_flat_mean_target_value'

AUX_POST_DISCARD_ERROR_KEY = 'aux_post_discard_error'
AUX_MONO_FRACTION_KEY = 'aux_mono_fraction'
AUX_MEAN_DI_KEY = 'aux_mean_discard_improvement'
AUX_MEAN_MEAN_PREDICTION_KEY = 'aux_mean_mean_prediction'
AUX_MEAN_TARGET_KEY = 'aux_mean_target_value'

SCALAR_FIELD_DIM = uq_evaluation.SCALAR_FIELD_DIM
VECTOR_FIELD_DIM = uq_evaluation.VECTOR_FIELD_DIM
HEIGHT_DIM = uq_evaluation.HEIGHT_DIM
WAVELENGTH_DIM = uq_evaluation.WAVELENGTH_DIM
AUX_TARGET_FIELD_DIM = uq_evaluation.AUX_TARGET_FIELD_DIM
AUX_PREDICTED_FIELD_DIM = uq_evaluation.AUX_PREDICTED_FIELD_DIM

MODEL_FILE_KEY = uq_evaluation.MODEL_FILE_KEY
PREDICTION_FILE_KEY = uq_evaluation.PREDICTION_FILE_KEY


def _compute_mf_and_di(result_table_xarray, is_error_pos_oriented):
    """Computes MF and mean DI for each target variable.

    MF = monotonicity fraction
    DI = discard improvement

    :param result_table_xarray: xarray table created by
        `run_discard_test`.
    :param is_error_pos_oriented: Boolean flag.
    :return: result_table_xarray: Same as input but with overall stats
        (monotonicity fraction and mean discard improvement) updated.
    """

    rtx = result_table_xarray
    num_scalar_targets = len(rtx.coords[SCALAR_FIELD_DIM].values)
    num_vector_targets = len(rtx.coords[VECTOR_FIELD_DIM].values)
    num_heights = len(rtx.coords[HEIGHT_DIM].values)
    num_wavelengths = len(rtx.coords[WAVELENGTH_DIM].values)

    try:
        num_aux_targets = len(rtx.coords[AUX_PREDICTED_FIELD_DIM].values)
    except:
        num_aux_targets = 0

    discard_fractions = 1. - rtx[EXAMPLE_FRACTION_KEY].values

    for t in range(num_scalar_targets):
        for w in range(num_wavelengths):
            rtx[SCALAR_MONO_FRACTION_KEY].values[t, w] = numpy.mean(
                numpy.diff(rtx[SCALAR_POST_DISCARD_ERROR_KEY].values[t, w, :]) > 0
            )
            rtx[SCALAR_MEAN_DI_KEY].values[t, w] = numpy.mean(
                numpy.diff(rtx[SCALAR_POST_DISCARD_ERROR_KEY].values[t, w, :]) /
                numpy.diff(discard_fractions)
            )

            if not is_error_pos_oriented:
                rtx[SCALAR_MONO_FRACTION_KEY].values[t, w] = (
                    1. - rtx[SCALAR_MONO_FRACTION_KEY].values[t, w]
                )
                rtx[SCALAR_MEAN_DI_KEY].values[t, w] *= -1.

    for t in range(num_aux_targets):
        for w in range(num_wavelengths):
            rtx[AUX_MONO_FRACTION_KEY].values[t, w] = numpy.mean(
                numpy.diff(rtx[AUX_POST_DISCARD_ERROR_KEY].values[t, w, :]) > 0
            )
            rtx[AUX_MEAN_DI_KEY].values[t, w] = numpy.mean(
                numpy.diff(rtx[AUX_POST_DISCARD_ERROR_KEY].values[t, w, :]) /
                numpy.diff(discard_fractions)
            )

            if not is_error_pos_oriented:
                rtx[AUX_MONO_FRACTION_KEY].values[t, w] = (
                    1. - rtx[AUX_MONO_FRACTION_KEY].values[t, w]
                )
                rtx[AUX_MEAN_DI_KEY].values[t, w] *= -1.

    for t in range(num_vector_targets):
        for w in range(num_wavelengths):
            rtx[VECTOR_FLAT_MONO_FRACTION_KEY].values[t, w] = numpy.mean(
                numpy.diff(rtx[VECTOR_FLAT_POST_DISCARD_ERROR_KEY].values[t, w, :])
                > 0
            )
            rtx[VECTOR_FLAT_MEAN_DI_KEY].values[t, w] = numpy.mean(
                numpy.diff(rtx[VECTOR_FLAT_POST_DISCARD_ERROR_KEY].values[t, w, :]) /
                numpy.diff(discard_fractions)
            )

            if not is_error_pos_oriented:
                rtx[VECTOR_FLAT_MONO_FRACTION_KEY].values[t, w] = (
                    1. - rtx[VECTOR_FLAT_MONO_FRACTION_KEY].values[t, w]
                )
                rtx[VECTOR_FLAT_MEAN_DI_KEY].values[t, w] *= -1.

            for h in range(num_heights):
                rtx[VECTOR_MONO_FRACTION_KEY].values[t, h, w] = numpy.mean(
                    numpy.diff(rtx[VECTOR_POST_DISCARD_ERROR_KEY].values[t, h, w, :]) > 0
                )
                rtx[VECTOR_MEAN_DI_KEY].values[t, h, w] = numpy.mean(
                    numpy.diff(rtx[VECTOR_POST_DISCARD_ERROR_KEY].values[t, h, w, :])
                    / numpy.diff(discard_fractions)
                )

                if not is_error_pos_oriented:
                    rtx[VECTOR_MONO_FRACTION_KEY].values[t, h, w] = (
                        1. - rtx[VECTOR_MONO_FRACTION_KEY].values[t, h, w]
                    )
                    rtx[VECTOR_MEAN_DI_KEY].values[t, h, w] *= -1.

    rtx.attrs[MONO_FRACTION_KEY] = numpy.mean(
        numpy.diff(rtx[POST_DISCARD_ERROR_KEY].values) > 0
    )
    rtx.attrs[MEAN_DI_KEY] = numpy.mean(
        numpy.diff(rtx[POST_DISCARD_ERROR_KEY].values) /
        numpy.diff(discard_fractions)
    )

    if not is_error_pos_oriented:
        rtx.attrs[MONO_FRACTION_KEY] = 1. - rtx.attrs[MONO_FRACTION_KEY]
        rtx.attrs[MEAN_DI_KEY] *= -1.

    result_table_xarray = rtx
    return result_table_xarray


def run_discard_test(
        prediction_file_name, uncertainty_thresholds, error_function,
        uncertainty_function, is_error_pos_oriented,
        error_function_for_hr_1height, error_function_for_flux_1var):
    """Runs the discard test.

    E = number of examples
    W = number of wavelengths
    F = number of discard fractions
    S = ensemble size

    :param prediction_file_name: Path to input file (will be read by
        `prediction_io.read_file`).
    :param uncertainty_thresholds: length-F numpy array of uncertainty
        thresholds, one for each discard fraction.

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
    Input: actual_hr_matrix_k_day01: E-by-W numpy array of actual heating rates
        at one height.
    Input: predicted_hr_matrix_k_day01: E-by-W-by-S numpy array of predicted
        heating rates at the same height.
    Input: use_example_flags: length-E numpy array of Boolean flags,
        indicating which examples to use.
    Output: error_value: Scalar value of error metric.

    :param error_function_for_flux_1var: Function with the following inputs and
        outputs...
    Input: actual_flux_matrix_w_m02: E-by-W numpy array of actual values for one
        flux variable.
    Input: predicted_flux_matrix_w_m02: E-by-W-by-S numpy array of predicted
        values for one flux variable.
    Input: use_example_flags: length-E numpy array of Boolean flags,
        indicating which examples to use.
    Output: error_value: Scalar value of error metric.

    :return: result_table_xarray: xarray table with results (variable and
        dimension names should make the table self-explanatory).
    """

    # Check input args.
    error_checking.assert_is_boolean(is_error_pos_oriented)
    error_checking.assert_is_numpy_array_without_nan(uncertainty_thresholds)
    error_checking.assert_is_numpy_array(
        uncertainty_thresholds, num_dimensions=1
    )

    uncertainty_thresholds = numpy.sort(uncertainty_thresholds)[::-1]
    num_thresholds = len(uncertainty_thresholds)
    assert num_thresholds >= 2

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
    wavelengths_metres = prediction_dict[prediction_io.TARGET_WAVELENGTHS_KEY]

    example_dict = {
        example_utils.SCALAR_TARGET_NAMES_KEY:
            generator_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY],
        example_utils.VECTOR_TARGET_NAMES_KEY:
            generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY],
        example_utils.SCALAR_PREDICTOR_NAMES_KEY:
            generator_option_dict[neural_net.SCALAR_PREDICTOR_NAMES_KEY],
        example_utils.VECTOR_PREDICTOR_NAMES_KEY:
            generator_option_dict[neural_net.VECTOR_PREDICTOR_NAMES_KEY],
        example_utils.HEIGHTS_KEY: heights_m_agl,
        example_utils.TARGET_WAVELENGTHS_KEY: wavelengths_metres
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

    num_heights = vector_target_matrix.shape[1]
    num_wavelengths = vector_target_matrix.shape[2]
    num_vector_targets = vector_target_matrix.shape[3]
    num_scalar_targets = scalar_target_matrix.shape[2]
    ensemble_size = vector_prediction_matrix.shape[-1]

    main_data_dict = {
        POST_DISCARD_ERROR_KEY: (
            (UNCERTAINTY_THRESHOLD_DIM,), numpy.full(num_thresholds, numpy.nan)
        ),
        EXAMPLE_FRACTION_KEY: (
            (UNCERTAINTY_THRESHOLD_DIM,), numpy.full(num_thresholds, numpy.nan)
        )
    }

    these_dim_keys_1d = (SCALAR_FIELD_DIM, WAVELENGTH_DIM)
    these_dim_keys_2d = (
        SCALAR_FIELD_DIM, WAVELENGTH_DIM, UNCERTAINTY_THRESHOLD_DIM
    )
    these_dim_1d = (num_scalar_targets, num_wavelengths)
    these_dim_2d = (num_scalar_targets, num_wavelengths, num_thresholds)

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
        SCALAR_MONO_FRACTION_KEY: (
            these_dim_keys_1d, numpy.full(these_dim_1d, numpy.nan)
        ),
        SCALAR_MEAN_DI_KEY: (
            these_dim_keys_1d, numpy.full(these_dim_1d, numpy.nan)
        )
    })

    these_dim_keys_2d = (VECTOR_FIELD_DIM, HEIGHT_DIM, WAVELENGTH_DIM)
    these_dim_keys_3d = (
        VECTOR_FIELD_DIM, HEIGHT_DIM, WAVELENGTH_DIM, UNCERTAINTY_THRESHOLD_DIM
    )
    these_dim_2d = (num_vector_targets, num_heights, num_wavelengths)
    these_dim_3d = (
        num_vector_targets, num_heights, num_wavelengths, num_thresholds
    )

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
        VECTOR_MONO_FRACTION_KEY: (
            these_dim_keys_2d, numpy.full(these_dim_2d, numpy.nan)
        ),
        VECTOR_MEAN_DI_KEY: (
            these_dim_keys_2d, numpy.full(these_dim_2d, numpy.nan)
        )
    })

    these_dim_keys_2d = (VECTOR_FIELD_DIM, WAVELENGTH_DIM)
    these_dim_keys_3d = (
        VECTOR_FIELD_DIM, WAVELENGTH_DIM, UNCERTAINTY_THRESHOLD_DIM
    )
    these_dim_2d = (num_vector_targets, num_wavelengths)
    these_dim_3d = (num_vector_targets, num_wavelengths, num_thresholds)

    main_data_dict.update({
        VECTOR_FLAT_POST_DISCARD_ERROR_KEY: (
            these_dim_keys_3d, numpy.full(these_dim_3d, numpy.nan)
        ),
        VECTOR_FLAT_MEAN_MEAN_PREDICTION_KEY: (
            these_dim_keys_3d, numpy.full(these_dim_3d, numpy.nan)
        ),
        VECTOR_FLAT_MEAN_TARGET_KEY: (
            these_dim_keys_3d, numpy.full(these_dim_3d, numpy.nan)
        ),
        VECTOR_FLAT_MONO_FRACTION_KEY: (
            these_dim_keys_2d, numpy.full(these_dim_2d, numpy.nan)
        ),
        VECTOR_FLAT_MEAN_DI_KEY: (
            these_dim_keys_2d, numpy.full(these_dim_2d, numpy.nan)
        )
    })

    num_aux_targets = len(aux_target_field_names)

    if num_aux_targets > 0:
        these_dim_keys_1d = (AUX_TARGET_FIELD_DIM, WAVELENGTH_DIM)
        these_dim_keys_2d = (
            AUX_TARGET_FIELD_DIM, WAVELENGTH_DIM, UNCERTAINTY_THRESHOLD_DIM
        )
        these_dim_1d = (num_aux_targets, num_wavelengths)
        these_dim_2d = (num_aux_targets, num_wavelengths, num_thresholds)

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
            AUX_MONO_FRACTION_KEY: (
                these_dim_keys_1d, numpy.full(these_dim_1d, numpy.nan)
            ),
            AUX_MEAN_DI_KEY: (
                these_dim_keys_1d, numpy.full(these_dim_1d, numpy.nan)
            )
        })

    metadata_dict = {
        SCALAR_FIELD_DIM: example_dict[example_utils.SCALAR_TARGET_NAMES_KEY],
        VECTOR_FIELD_DIM: example_dict[example_utils.VECTOR_TARGET_NAMES_KEY],
        HEIGHT_DIM: heights_m_agl,
        WAVELENGTH_DIM: wavelengths_metres,
        UNCERTAINTY_THRESHOLD_DIM: uncertainty_thresholds
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

    for i in range(num_thresholds):
        this_inverted_mask = uncertainty_values > uncertainty_thresholds[i]
        use_example_flags[this_inverted_mask] = False

        result_table_xarray[EXAMPLE_FRACTION_KEY].values[i] = numpy.mean(
            use_example_flags
        )
        result_table_xarray[POST_DISCARD_ERROR_KEY].values[i] = error_function(
            prediction_dict, use_example_flags
        )

        rtx = result_table_xarray

        for t in range(num_scalar_targets):
            for w in range(num_wavelengths):
                rtx[SCALAR_MEAN_MEAN_PREDICTION_KEY].values[
                    t, w, i
                ] = numpy.mean(numpy.mean(
                    scalar_prediction_matrix[:, w, t, :][
                        use_example_flags, ...
                    ],
                    axis=-1
                ))

                rtx[SCALAR_MEAN_TARGET_KEY].values[t, w, i] = numpy.mean(
                    scalar_target_matrix[:, w, t][use_example_flags]
                )

                rtx[SCALAR_POST_DISCARD_ERROR_KEY].values[t, w, i] = (
                    error_function_for_flux_1var(
                        scalar_target_matrix[:, [w], t],
                        scalar_prediction_matrix[:, [w], t, :],
                        use_example_flags
                    )
                )

        for t in range(num_aux_targets):
            for w in range(num_wavelengths):
                rtx[AUX_MEAN_MEAN_PREDICTION_KEY].values[
                    t, w, i
                ] = numpy.mean(numpy.mean(
                    aux_prediction_matrix[:, w, t, :][use_example_flags, ...],
                    axis=-1
                ))

                rtx[AUX_MEAN_TARGET_KEY].values[t, w, i] = numpy.mean(
                    aux_target_matrix[:, w, t][use_example_flags]
                )

                rtx[AUX_POST_DISCARD_ERROR_KEY].values[t, w, i] = (
                    error_function_for_flux_1var(
                        aux_target_matrix[:, [w], t],
                        aux_prediction_matrix[:, [w], t, :],
                        use_example_flags
                    )
                )

        for t in range(num_vector_targets):
            for w in range(num_wavelengths):
                this_mean_pred_by_example = numpy.mean(
                    vector_prediction_matrix[..., w, t, :][
                        use_example_flags, ...
                    ],
                    axis=-1
                )

                rtx[VECTOR_FLAT_MEAN_MEAN_PREDICTION_KEY].values[t, w, i] = (
                    numpy.mean(this_mean_pred_by_example)
                )
                rtx[VECTOR_FLAT_MEAN_TARGET_KEY].values[t, w, i] = numpy.mean(
                    vector_target_matrix[..., w, t][use_example_flags, ...]
                )

                this_target_matrix = numpy.expand_dims(
                    numpy.ravel(vector_target_matrix[..., w, t]),
                    axis=-1
                )
                this_prediction_matrix = numpy.reshape(
                    vector_prediction_matrix[..., w, t, :],
                    (this_target_matrix.shape[0], ensemble_size)
                )
                this_prediction_matrix = numpy.expand_dims(
                    this_prediction_matrix, axis=-2
                )
                these_flags = numpy.repeat(
                    use_example_flags, axis=0, repeats=num_heights
                )

                rtx[VECTOR_FLAT_POST_DISCARD_ERROR_KEY].values[t, w, i] = (
                    error_function_for_hr_1height(
                        this_target_matrix, this_prediction_matrix, these_flags
                    )
                )

                for h in range(num_heights):
                    this_mean_pred_by_example = numpy.mean(
                        vector_prediction_matrix[:, h, w, t, :][
                            use_example_flags, ...
                        ],
                        axis=-1
                    )

                    rtx[VECTOR_MEAN_MEAN_PREDICTION_KEY].values[t, h, w, i] = (
                        numpy.mean(this_mean_pred_by_example)
                    )
                    rtx[VECTOR_MEAN_TARGET_KEY].values[t, h, w, i] = numpy.mean(
                        vector_target_matrix[:, h, w, t][use_example_flags]
                    )
                    rtx[VECTOR_POST_DISCARD_ERROR_KEY].values[t, h, w, i] = (
                        error_function_for_hr_1height(
                            vector_target_matrix[:, h, [w], t],
                            vector_prediction_matrix[:, h, [w], t, :],
                            use_example_flags
                        )
                    )

        result_table_xarray = rtx

    return _compute_mf_and_di(
        result_table_xarray=result_table_xarray,
        is_error_pos_oriented=is_error_pos_oriented
    )


def merge_results_over_examples(result_tables_xarray):
    """Merges discard-test results over many examples.

    :param result_tables_xarray: List of xarray tables, each created by
        `run_discard_test`, each containing results for a different
        set of examples.
    :return: result_table_xarray: Single xarray table with results for all
        examples (variable and dimension names should make the table
        self-explanatory).
    """

    prediction_file_names, num_examples_by_table = (
        uq_evaluation.check_results_before_merging(result_tables_xarray)
    )

    num_tables = len(result_tables_xarray)

    for i in range(1, num_tables):
        assert numpy.allclose(
            result_tables_xarray[i].coords[UNCERTAINTY_THRESHOLD_DIM].values,
            result_tables_xarray[0].coords[UNCERTAINTY_THRESHOLD_DIM].values,
            atol=TOLERANCE
        )

    scalar_target_names = (
        result_tables_xarray[0].coords[SCALAR_FIELD_DIM].values
    )
    vector_target_names = (
        result_tables_xarray[0].coords[VECTOR_FIELD_DIM].values
    )
    heights_m_agl = result_tables_xarray[0].coords[HEIGHT_DIM].values
    wavelengths_metres = result_tables_xarray[0].coords[WAVELENGTH_DIM].values

    try:
        aux_predicted_field_names = (
            result_tables_xarray[0].coords[AUX_PREDICTED_FIELD_DIM].values
        )
    except:
        aux_predicted_field_names = []

    this_result_table_xarray = _compute_mf_and_di(
        result_table_xarray=copy.deepcopy(result_tables_xarray[0]),
        is_error_pos_oriented=True
    )
    is_error_pos_oriented = True

    these_keys = [
        SCALAR_MONO_FRACTION_KEY, VECTOR_FLAT_MONO_FRACTION_KEY,
        VECTOR_MONO_FRACTION_KEY
    ]
    if len(aux_predicted_field_names) > 0:
        these_keys.append(AUX_MONO_FRACTION_KEY)

    for this_key in these_keys:
        is_error_pos_oriented &= numpy.allclose(
            this_result_table_xarray[this_key].values,
            result_tables_xarray[0][this_key].values,
            atol=TOLERANCE
        )

    rtx = copy.deepcopy(result_tables_xarray[0])
    num_thresholds = len(
        rtx.coords[UNCERTAINTY_THRESHOLD_DIM].values
    )

    for i in range(num_thresholds):
        example_fraction_by_table_this_bin = numpy.array([
            this_tbl[EXAMPLE_FRACTION_KEY].values[i]
            for this_tbl in result_tables_xarray
        ])
        num_examples_by_table_this_bin = (
            example_fraction_by_table_this_bin * num_examples_by_table
        )
        rtx[EXAMPLE_FRACTION_KEY].values[i] = (
            float(numpy.sum(num_examples_by_table_this_bin)) /
            numpy.sum(num_examples_by_table)
        )

        these_errors = numpy.array([
            this_tbl[POST_DISCARD_ERROR_KEY].values[i]
            for this_tbl in result_tables_xarray
        ])
        rtx[POST_DISCARD_ERROR_KEY].values[i] = numpy.average(
            these_errors, weights=num_examples_by_table_this_bin
        )

        for t in range(len(scalar_target_names)):
            for w in range(len(wavelengths_metres)):
                these_errors = numpy.array([
                    this_tbl[SCALAR_POST_DISCARD_ERROR_KEY].values[t, w, i]
                    for this_tbl in result_tables_xarray
                ])
                rtx[SCALAR_POST_DISCARD_ERROR_KEY].values[t, w, i] = (
                    numpy.average(
                        these_errors, weights=num_examples_by_table_this_bin
                    )
                )

                these_mean_mean_predictions = numpy.array([
                    this_tbl[SCALAR_MEAN_MEAN_PREDICTION_KEY].values[t, w, i]
                    for this_tbl in result_tables_xarray
                ])
                rtx[SCALAR_MEAN_MEAN_PREDICTION_KEY].values[t, w, i] = (
                    numpy.average(
                        these_mean_mean_predictions,
                        weights=num_examples_by_table_this_bin
                    )
                )

                these_mean_targets = numpy.array([
                    this_tbl[SCALAR_MEAN_TARGET_KEY].values[t, w, i]
                    for this_tbl in result_tables_xarray
                ])
                rtx[SCALAR_MEAN_TARGET_KEY].values[t, w, i] = numpy.average(
                    these_mean_targets, weights=num_examples_by_table_this_bin
                )

        for t in range(len(aux_predicted_field_names)):
            for w in range(len(wavelengths_metres)):
                these_errors = numpy.array([
                    this_tbl[AUX_POST_DISCARD_ERROR_KEY].values[t, w, i]
                    for this_tbl in result_tables_xarray
                ])
                rtx[AUX_POST_DISCARD_ERROR_KEY].values[t, w, i] = numpy.average(
                    these_errors, weights=num_examples_by_table_this_bin
                )

                these_mean_mean_predictions = numpy.array([
                    this_tbl[AUX_MEAN_MEAN_PREDICTION_KEY].values[t, w, i]
                    for this_tbl in result_tables_xarray
                ])
                rtx[AUX_MEAN_MEAN_PREDICTION_KEY].values[t, w, i] = (
                    numpy.average(
                        these_mean_mean_predictions,
                        weights=num_examples_by_table_this_bin
                    )
                )

                these_mean_targets = numpy.array([
                    this_tbl[AUX_MEAN_TARGET_KEY].values[t, w, i]
                    for this_tbl in result_tables_xarray
                ])
                rtx[AUX_MEAN_TARGET_KEY].values[t, w, i] = numpy.average(
                    these_mean_targets, weights=num_examples_by_table_this_bin
                )

        for t in range(len(vector_target_names)):
            for w in range(len(wavelengths_metres)):
                these_errors = numpy.array([
                    this_tbl[VECTOR_FLAT_POST_DISCARD_ERROR_KEY].values[t, w, i]
                    for this_tbl in result_tables_xarray
                ])
                rtx[VECTOR_FLAT_POST_DISCARD_ERROR_KEY].values[t, w, i] = (
                    numpy.average(
                        these_errors, weights=num_examples_by_table_this_bin
                    )
                )

                these_mean_mean_predictions = numpy.array([
                    this_tbl[VECTOR_FLAT_MEAN_MEAN_PREDICTION_KEY].values[t, w, i]
                    for this_tbl in result_tables_xarray
                ])
                rtx[VECTOR_FLAT_MEAN_MEAN_PREDICTION_KEY].values[t, w, i] = (
                    numpy.average(
                        these_mean_mean_predictions,
                        weights=num_examples_by_table_this_bin
                    )
                )

                these_mean_targets = numpy.array([
                    this_tbl[VECTOR_FLAT_MEAN_TARGET_KEY].values[t, w, i]
                    for this_tbl in result_tables_xarray
                ])
                rtx[VECTOR_FLAT_MEAN_TARGET_KEY].values[t, w, i] = (
                    numpy.average(
                        these_mean_targets,
                        weights=num_examples_by_table_this_bin
                    )
                )

                for h in range(len(heights_m_agl)):
                    these_errors = numpy.array([
                        this_tbl[VECTOR_POST_DISCARD_ERROR_KEY].values[t, h, w, i]
                        for this_tbl in result_tables_xarray
                    ])
                    rtx[VECTOR_POST_DISCARD_ERROR_KEY].values[t, h, w, i] = (
                        numpy.average(
                            these_errors, weights=num_examples_by_table_this_bin
                        )
                    )

                    these_mean_mean_predictions = numpy.array([
                        this_tbl[VECTOR_MEAN_MEAN_PREDICTION_KEY].values[t, h, w, i]
                        for this_tbl in result_tables_xarray
                    ])
                    rtx[VECTOR_MEAN_MEAN_PREDICTION_KEY].values[t, h, w, i] = (
                        numpy.average(
                            these_mean_mean_predictions,
                            weights=num_examples_by_table_this_bin
                        )
                    )

                    these_mean_targets = numpy.array([
                        this_tbl[VECTOR_MEAN_TARGET_KEY].values[t, h, w, i]
                        for this_tbl in result_tables_xarray
                    ])
                    rtx[VECTOR_MEAN_TARGET_KEY].values[t, h, w, i] = (
                        numpy.average(
                            these_mean_targets,
                            weights=num_examples_by_table_this_bin
                        )
                    )

    rtx.attrs[PREDICTION_FILE_KEY] = ' '.join([
        '{0:s}'.format(f) for f in prediction_file_names
    ])

    return _compute_mf_and_di(
        result_table_xarray=rtx,
        is_error_pos_oriented=is_error_pos_oriented
    )


def write_results(result_table_xarray, netcdf_file_name):
    """Writes discard-test results to NetCDF file.

    :param result_table_xarray: xarray table in format returned by
        `run_discard_test`.
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)

    result_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )


def read_results(netcdf_file_name):
    """Reads discard-test results from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: result_table_xarray: xarray table.  Documentation in the
        xarray table should make values self-explanatory.
    """

    return xarray.open_dataset(netcdf_file_name)
