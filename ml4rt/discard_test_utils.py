"""Helper methods for running discard test."""

import os
import sys
import copy
import numpy
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import error_checking
import prediction_io
import example_utils
import uq_evaluation
import neural_net

TOLERANCE = 1e-6

DISCARD_FRACTION_DIM = 'discard_fraction'
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

    num_scalar_targets = len(
        result_table_xarray.coords[SCALAR_FIELD_DIM].values
    )
    num_vector_targets = len(
        result_table_xarray.coords[VECTOR_FIELD_DIM].values
    )
    num_heights = len(result_table_xarray.coords[HEIGHT_DIM].values)

    try:
        num_aux_targets = len(
            result_table_xarray.coords[AUX_PREDICTED_FIELD_DIM].values
        )
    except:
        num_aux_targets = 0

    discard_fractions = result_table_xarray.coords[DISCARD_FRACTION_DIM].values
    t = result_table_xarray

    for k in range(num_scalar_targets):
        if is_error_pos_oriented:
            t[SCALAR_MONO_FRACTION_KEY].values[k] = numpy.mean(
                numpy.diff(t[SCALAR_POST_DISCARD_ERROR_KEY].values[k, :]) > 0
            )
            t[SCALAR_MEAN_DI_KEY].values[k] = numpy.mean(
                numpy.diff(t[SCALAR_POST_DISCARD_ERROR_KEY].values[k, :]) /
                numpy.diff(discard_fractions)
            )
        else:
            t[SCALAR_MONO_FRACTION_KEY].values[k] = numpy.mean(
                numpy.diff(t[SCALAR_POST_DISCARD_ERROR_KEY].values[k, :]) < 0
            )
            t[SCALAR_MEAN_DI_KEY].values[k] = numpy.mean(
                -1 * numpy.diff(t[SCALAR_POST_DISCARD_ERROR_KEY].values[k, :]) /
                numpy.diff(discard_fractions)
            )

    for k in range(num_aux_targets):
        if is_error_pos_oriented:
            t[AUX_MONO_FRACTION_KEY].values[k] = numpy.mean(
                numpy.diff(t[AUX_POST_DISCARD_ERROR_KEY].values[k, :]) > 0
            )
            t[AUX_MEAN_DI_KEY].values[k] = numpy.mean(
                numpy.diff(t[AUX_POST_DISCARD_ERROR_KEY].values[k, :]) /
                numpy.diff(discard_fractions)
            )
        else:
            t[AUX_MONO_FRACTION_KEY].values[k] = numpy.mean(
                numpy.diff(t[AUX_POST_DISCARD_ERROR_KEY].values[k, :]) < 0
            )
            t[AUX_MEAN_DI_KEY].values[k] = numpy.mean(
                -1 * numpy.diff(t[AUX_POST_DISCARD_ERROR_KEY].values[k, :]) /
                numpy.diff(discard_fractions)
            )

    for k in range(num_vector_targets):
        if is_error_pos_oriented:
            t[VECTOR_FLAT_MONO_FRACTION_KEY].values[k] = numpy.mean(
                numpy.diff(t[VECTOR_FLAT_POST_DISCARD_ERROR_KEY].values[k, :])
                > 0
            )
            t[VECTOR_FLAT_MEAN_DI_KEY].values[k] = numpy.mean(
                numpy.diff(t[VECTOR_FLAT_POST_DISCARD_ERROR_KEY].values[k, :]) /
                numpy.diff(discard_fractions)
            )
        else:
            t[VECTOR_FLAT_MONO_FRACTION_KEY].values[k] = numpy.mean(
                numpy.diff(t[VECTOR_FLAT_POST_DISCARD_ERROR_KEY].values[k, :])
                < 0
            )
            t[VECTOR_FLAT_MEAN_DI_KEY].values[k] = numpy.mean(
                -1 *
                numpy.diff(t[VECTOR_FLAT_POST_DISCARD_ERROR_KEY].values[k, :]) /
                numpy.diff(discard_fractions)
            )

        for j in range(num_heights):
            if is_error_pos_oriented:
                t[VECTOR_MONO_FRACTION_KEY].values[k, j] = numpy.mean(
                    numpy.diff(
                        t[VECTOR_POST_DISCARD_ERROR_KEY].values[k, j, :]
                    ) > 0
                )

                t[VECTOR_MEAN_DI_KEY].values[k, j] = numpy.mean(
                    numpy.diff(t[VECTOR_POST_DISCARD_ERROR_KEY].values[k, j, :])
                    / numpy.diff(discard_fractions)
                )
            else:
                t[VECTOR_MONO_FRACTION_KEY].values[k, j] = numpy.mean(
                    numpy.diff(t[VECTOR_POST_DISCARD_ERROR_KEY].values[k, j, :])
                    < 0
                )

                t[VECTOR_MEAN_DI_KEY].values[k, j] = numpy.mean(
                    -1 *
                    numpy.diff(t[VECTOR_POST_DISCARD_ERROR_KEY].values[k, j, :])
                    / numpy.diff(discard_fractions)
                )

    if is_error_pos_oriented:
        t.attrs[MONO_FRACTION_KEY] = numpy.mean(
            numpy.diff(t[POST_DISCARD_ERROR_KEY].values) > 0
        )
        t.attrs[MEAN_DI_KEY] = numpy.mean(
            numpy.diff(t[POST_DISCARD_ERROR_KEY].values) /
            numpy.diff(discard_fractions)
        )
    else:
        t.attrs[MONO_FRACTION_KEY] = numpy.mean(
            numpy.diff(t[POST_DISCARD_ERROR_KEY].values) < 0
        )
        t.attrs[MEAN_DI_KEY] = numpy.mean(
            -1 * numpy.diff(t[POST_DISCARD_ERROR_KEY].values)
            / numpy.diff(discard_fractions)
        )

    result_table_xarray = t
    return result_table_xarray


def run_discard_test(
        prediction_file_name, discard_fractions, error_function,
        uncertainty_function, is_error_pos_oriented,
        error_function_for_hr_1height, error_function_for_flux_1var):
    """Runs the discard test.

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
        SCALAR_MONO_FRACTION_KEY: (
            these_dim_keys_1d, numpy.full(these_dim_1d, numpy.nan)
        ),
        SCALAR_MEAN_DI_KEY: (
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
        VECTOR_MONO_FRACTION_KEY: (
            these_dim_keys_2d, numpy.full(these_dim_2d, numpy.nan)
        ),
        VECTOR_MEAN_DI_KEY: (
            these_dim_keys_2d, numpy.full(these_dim_2d, numpy.nan)
        )
    })

    these_dim_keys_2d = (VECTOR_FIELD_DIM,)
    these_dim_keys_3d = (VECTOR_FIELD_DIM, DISCARD_FRACTION_DIM)
    these_dim_2d = num_vector_targets
    these_dim_3d = (num_vector_targets, num_fractions)

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
            AUX_MONO_FRACTION_KEY: (
                these_dim_keys_1d, numpy.full(these_dim_1d, numpy.nan)
            ),
            AUX_MEAN_DI_KEY: (
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
            this_mean_pred_by_example = numpy.mean(
                vector_prediction_matrix[:, :, k, :][use_example_flags, ...],
                axis=-1
            )

            t[VECTOR_FLAT_MEAN_MEAN_PREDICTION_KEY].values[k, i] = numpy.mean(
                this_mean_pred_by_example
            )

            t[VECTOR_FLAT_MEAN_TARGET_KEY].values[k, i] = numpy.mean(
                vector_target_matrix[:, :, k][use_example_flags, :]
            )

            these_targets = numpy.ravel(vector_target_matrix[:, :, k])
            this_prediction_matrix = numpy.reshape(
                vector_prediction_matrix[:, :, k, :],
                (len(these_targets), vector_prediction_matrix.shape[-1])
            )

            t[VECTOR_FLAT_POST_DISCARD_ERROR_KEY].values[k, i] = (
                error_function_for_hr_1height(
                    these_targets, this_prediction_matrix, use_example_flags
                )
            )

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
            result_tables_xarray[i].coords[DISCARD_FRACTION_DIM].values,
            result_tables_xarray[0].coords[DISCARD_FRACTION_DIM].values,
            atol=TOLERANCE
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

    result_table_xarray = copy.deepcopy(result_tables_xarray[0])
    num_discard_fractions = len(
        result_table_xarray.coords[DISCARD_FRACTION_DIM].values
    )

    for i in range(num_discard_fractions):
        example_fraction_by_table_this_bin = numpy.array([
            t[EXAMPLE_FRACTION_KEY].values[i] for t in result_tables_xarray
        ])
        num_examples_by_table_this_bin = (
            example_fraction_by_table_this_bin * num_examples_by_table
        )

        result_table_xarray[EXAMPLE_FRACTION_KEY].values[i] = (
            float(numpy.sum(num_examples_by_table_this_bin)) /
            numpy.sum(num_examples_by_table)
        )

        these_errors = numpy.array([
            t[POST_DISCARD_ERROR_KEY].values[i] for t in result_tables_xarray
        ])
        result_table_xarray[POST_DISCARD_ERROR_KEY].values[i] = numpy.average(
            these_errors, weights=num_examples_by_table_this_bin
        )

        for j in range(len(scalar_target_names)):
            these_errors = numpy.array([
                t[SCALAR_POST_DISCARD_ERROR_KEY].values[j, i]
                for t in result_tables_xarray
            ])
            result_table_xarray[SCALAR_POST_DISCARD_ERROR_KEY].values[
                j, i
            ] = numpy.average(
                these_errors, weights=num_examples_by_table_this_bin
            )

            these_mean_mean_predictions = numpy.array([
                t[SCALAR_MEAN_MEAN_PREDICTION_KEY].values[j, i]
                for t in result_tables_xarray
            ])
            result_table_xarray[SCALAR_MEAN_MEAN_PREDICTION_KEY].values[
                j, i
            ] = numpy.average(
                these_mean_mean_predictions,
                weights=num_examples_by_table_this_bin
            )

            these_mean_targets = numpy.array([
                t[SCALAR_MEAN_TARGET_KEY].values[j, i]
                for t in result_tables_xarray
            ])
            result_table_xarray[SCALAR_MEAN_TARGET_KEY].values[
                j, i
            ] = numpy.average(
                these_mean_targets, weights=num_examples_by_table_this_bin
            )

        for j in range(len(aux_predicted_field_names)):
            these_errors = numpy.array([
                t[AUX_POST_DISCARD_ERROR_KEY].values[j, i]
                for t in result_tables_xarray
            ])
            result_table_xarray[AUX_POST_DISCARD_ERROR_KEY].values[
                j, i
            ] = numpy.average(
                these_errors, weights=num_examples_by_table_this_bin
            )

            these_mean_mean_predictions = numpy.array([
                t[AUX_MEAN_MEAN_PREDICTION_KEY].values[j, i]
                for t in result_tables_xarray
            ])
            result_table_xarray[AUX_MEAN_MEAN_PREDICTION_KEY].values[
                j, i
            ] = numpy.average(
                these_mean_mean_predictions,
                weights=num_examples_by_table_this_bin
            )

            these_mean_targets = numpy.array([
                t[AUX_MEAN_TARGET_KEY].values[j, i]
                for t in result_tables_xarray
            ])
            result_table_xarray[AUX_MEAN_TARGET_KEY].values[
                j, i
            ] = numpy.average(
                these_mean_targets, weights=num_examples_by_table_this_bin
            )

        for j in range(len(vector_target_names)):
            these_errors = numpy.array([
                t[VECTOR_FLAT_POST_DISCARD_ERROR_KEY].values[j, i]
                for t in result_tables_xarray
            ])
            result_table_xarray[VECTOR_FLAT_POST_DISCARD_ERROR_KEY].values[
                j, i
            ] = numpy.average(
                these_errors, weights=num_examples_by_table_this_bin
            )

            these_mean_mean_predictions = numpy.array([
                t[VECTOR_FLAT_MEAN_MEAN_PREDICTION_KEY].values[j, i]
                for t in result_tables_xarray
            ])
            result_table_xarray[VECTOR_FLAT_MEAN_MEAN_PREDICTION_KEY].values[
                j, i
            ] = numpy.average(
                these_mean_mean_predictions,
                weights=num_examples_by_table_this_bin
            )

            these_mean_targets = numpy.array([
                t[VECTOR_FLAT_MEAN_TARGET_KEY].values[j, i]
                for t in result_tables_xarray
            ])
            result_table_xarray[VECTOR_FLAT_MEAN_TARGET_KEY].values[
                j, i
            ] = numpy.average(
                these_mean_targets, weights=num_examples_by_table_this_bin
            )

            for k in range(len(heights_m_agl)):
                these_errors = numpy.array([
                    t[VECTOR_POST_DISCARD_ERROR_KEY].values[j, k, i]
                    for t in result_tables_xarray
                ])
                result_table_xarray[VECTOR_POST_DISCARD_ERROR_KEY].values[
                    j, k, i
                ] = numpy.average(
                    these_errors, weights=num_examples_by_table_this_bin
                )

                these_mean_mean_predictions = numpy.array([
                    t[VECTOR_MEAN_MEAN_PREDICTION_KEY].values[j, k, i]
                    for t in result_tables_xarray
                ])
                result_table_xarray[VECTOR_MEAN_MEAN_PREDICTION_KEY].values[
                    j, k, i
                ] = numpy.average(
                    these_mean_mean_predictions,
                    weights=num_examples_by_table_this_bin
                )

                these_mean_targets = numpy.array([
                    t[VECTOR_MEAN_TARGET_KEY].values[j, k, i]
                    for t in result_tables_xarray
                ])
                result_table_xarray[VECTOR_MEAN_TARGET_KEY].values[
                    j, k, i
                ] = numpy.average(
                    these_mean_targets, weights=num_examples_by_table_this_bin
                )

    result_table_xarray.attrs[PREDICTION_FILE_KEY] = ' '.join([
        '{0:s}'.format(f) for f in prediction_file_names
    ])

    return _compute_mf_and_di(
        result_table_xarray=result_table_xarray,
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
