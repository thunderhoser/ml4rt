"""Helper methods for computing continuous ranked probability score (CRPS)."""

import os
import sys
import copy
import numpy
import xarray
from scipy.integrate import simps

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import error_checking
import example_io
import prediction_io
import example_utils
import uq_evaluation
import neural_net

TOLERANCE = 1e-6
NUM_EXAMPLES_PER_BATCH = 1000
# MAX_NUM_CLIMO_EXAMPLES = 10000

SCALAR_CRPS_KEY = 'scalar_crps'
VECTOR_CRPS_KEY = 'vector_crps'
AUX_CRPS_KEY = 'aux_crps'
SCALAR_DWCRPS_KEY = 'scalar_dwcrps'
VECTOR_DWCRPS_KEY = 'vector_dwcrps'
AUX_DWCRPS_KEY = 'aux_dwcrps'
SCALAR_CRPSS_KEY = 'scalar_crpss'
VECTOR_CRPSS_KEY = 'vector_crpss'
AUX_CRPSS_KEY = 'aux_crpss'

SCALAR_FIELD_DIM = uq_evaluation.SCALAR_FIELD_DIM
VECTOR_FIELD_DIM = uq_evaluation.VECTOR_FIELD_DIM
HEIGHT_DIM = uq_evaluation.HEIGHT_DIM
AUX_TARGET_FIELD_DIM = uq_evaluation.AUX_TARGET_FIELD_DIM
AUX_PREDICTED_FIELD_DIM = uq_evaluation.AUX_PREDICTED_FIELD_DIM

MODEL_FILE_KEY = uq_evaluation.MODEL_FILE_KEY
PREDICTION_FILE_KEY = uq_evaluation.PREDICTION_FILE_KEY
SHORTWAVE_NET_FLUX_NAME = uq_evaluation.SHORTWAVE_NET_FLUX_NAME
LONGWAVE_NET_FLUX_NAME = uq_evaluation.LONGWAVE_NET_FLUX_NAME


def __get_approx_crps_one_var(target_values, prediction_matrix):
    """Computes approximate (ensemble formulation of) CRPS for one variable.

    E = number of examples
    S = number of ensemble members

    :param target_values: length-E numpy array of actual values.
    :param prediction_matrix: E-by-S numpy array of predicted values.
    :return: approx_crps_value: Approx CRPS (scalar).
    """

    # TODO(thunderhoser): I am not really using this method.

    abs_error_matrix = numpy.absolute(
        prediction_matrix - numpy.expand_dims(target_values, axis=-1)
    )
    mae_by_example = numpy.mean(abs_error_matrix, axis=-1)

    num_examples = len(target_values)
    mean_pairwise_diff_by_example = numpy.full(num_examples, numpy.nan)

    for i in range(num_examples):
        mean_pairwise_diff_by_example[i] = numpy.mean(numpy.absolute(
            numpy.expand_dims(prediction_matrix[i, :], axis=-1) -
            numpy.expand_dims(prediction_matrix[i, :], axis=-2)
        ))

    return numpy.mean(mae_by_example - 0.5 * mean_pairwise_diff_by_example)


def __get_approx_dwcrps_one_var(target_values, prediction_matrix):
    """Computes approximate (ensemble formulation of) DWCRPS for one variable.

    DWCRPS = dual-weighted CRPS

    E = number of examples
    S = number of ensemble members

    :param target_values: length-E numpy array of actual values.
    :param prediction_matrix: E-by-S numpy array of predicted values.
    :return: approx_dwcrps_value: Approx DWCRPS (scalar).
    """

    # TODO(thunderhoser): I am not really using this method.

    weight_matrix = numpy.maximum(
        numpy.absolute(prediction_matrix),
        numpy.absolute(numpy.expand_dims(target_values, axis=-1))
    )
    abs_error_matrix = numpy.absolute(
        prediction_matrix - numpy.expand_dims(target_values, axis=-1)
    )
    mae_by_example = numpy.mean(weight_matrix * abs_error_matrix, axis=-1)

    num_examples = len(target_values)
    mean_pairwise_diff_by_example = numpy.full(num_examples, numpy.nan)

    for i in range(num_examples):
        first_prediction_matrix = numpy.expand_dims(
            prediction_matrix[i, :], axis=-1
        )
        second_prediction_matrix = numpy.expand_dims(
            prediction_matrix[i, :], axis=-2
        )

        mean_pairwise_diff_by_example[i] = numpy.mean(
            numpy.maximum(
                numpy.absolute(first_prediction_matrix),
                numpy.absolute(second_prediction_matrix)
            )
            * numpy.absolute(
                first_prediction_matrix -
                second_prediction_matrix
            )
        )

    return numpy.mean(mae_by_example - 0.5 * mean_pairwise_diff_by_example)


def _get_crps_and_dwcrps_one_var(target_values, prediction_matrix,
                                 num_integration_levels):
    """Computes CRPS and DWCRPS for one variable.

    E = number of examples
    S = number of ensemble members

    :param target_values: length-E numpy array of actual values.
    :param prediction_matrix: E-by-S numpy array of predicted values.
    :param num_integration_levels: Will use this many integration levels to
        compute CRPS and DWCRPS.
    :return: crps_value: CRPS (scalar).
    :return: dwcrps_value: DWCRPS (scalar).
    """

    num_examples = len(target_values)
    crps_numerator = 0.
    dwcrps_numerator = 0.
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
        weight_matrix = numpy.maximum(
            numpy.absolute(this_prediction_matrix),
            numpy.absolute(this_target_matrix)
        )

        integrated_cdf_matrix = simps(
            y=(cdf_matrix - heaviside_matrix) ** 2,
            x=prediction_by_integ_level, axis=-1
        )
        crps_numerator += numpy.sum(integrated_cdf_matrix)
        crps_denominator += integrated_cdf_matrix.size

        integrated_cdf_matrix = simps(
            y=weight_matrix * (cdf_matrix - heaviside_matrix) ** 2,
            x=prediction_by_integ_level, axis=-1
        )
        dwcrps_numerator += numpy.sum(integrated_cdf_matrix)

    return (
        crps_numerator / crps_denominator,
        dwcrps_numerator / crps_denominator
    )


def _get_climo_crps_one_var(
        new_target_values, training_target_values, num_integration_levels,
        max_ensemble_size):
    """Computes CRPS of climatological model for one variable.

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


def get_crps_related_scores_all_vars(
        prediction_file_name, num_integration_levels, ensemble_size_for_climo):
    """Computes CRPS and related scores for all target variables.

    "Related scores" = DWCRPS and CRPSS (continuous ranked probability *skill*
    score)

    :param prediction_file_name: Path to input file (will be read by
        `prediction_io.read_file`).
    :param num_integration_levels: See doc for `_get_crps_and_dwcrps_one_var`.
    :param ensemble_size_for_climo: Ensemble size used to compute CRPS of
        climatological model.
    :return: result_table_xarray: xarray table with results (variable and
        dimension names should make the table self-explanatory).
    """

    error_checking.assert_is_integer(num_integration_levels)
    error_checking.assert_is_geq(num_integration_levels, 100)
    error_checking.assert_is_leq(num_integration_levels, 100000)

    error_checking.assert_is_integer(ensemble_size_for_climo)
    error_checking.assert_is_geq(ensemble_size_for_climo, 10)
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
    num_aux_targets = len(aux_target_field_names)

    main_data_dict = {
        SCALAR_CRPS_KEY: (
            (SCALAR_FIELD_DIM,), numpy.full(num_scalar_targets, numpy.nan)
        ),
        SCALAR_CRPSS_KEY: (
            (SCALAR_FIELD_DIM,), numpy.full(num_scalar_targets, numpy.nan)
        ),
        SCALAR_DWCRPS_KEY: (
            (SCALAR_FIELD_DIM,), numpy.full(num_scalar_targets, numpy.nan)
        ),
        VECTOR_CRPS_KEY: (
            (VECTOR_FIELD_DIM, HEIGHT_DIM),
            numpy.full((num_vector_targets, num_heights), numpy.nan)
        ),
        VECTOR_CRPSS_KEY: (
            (VECTOR_FIELD_DIM, HEIGHT_DIM),
            numpy.full((num_vector_targets, num_heights), numpy.nan)
        ),
        VECTOR_DWCRPS_KEY: (
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
            ),
            AUX_DWCRPS_KEY: (
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
        print('Computing CRPS and DWCRPS for {0:s}...'.format(
            example_dict[example_utils.SCALAR_TARGET_NAMES_KEY][j]
        ))

        (
            result_table_xarray[SCALAR_CRPS_KEY].values[j],
            result_table_xarray[SCALAR_DWCRPS_KEY].values[j]
        ) = _get_crps_and_dwcrps_one_var(
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
            print((
                'Computing CRPS and DWCRPS for {0:s} at {1:d} m AGL...'
            ).format(
                example_dict[example_utils.VECTOR_TARGET_NAMES_KEY][j],
                int(numpy.round(heights_m_agl[k]))
            ))

            (
                result_table_xarray[VECTOR_CRPS_KEY].values[j, k],
                result_table_xarray[VECTOR_DWCRPS_KEY].values[j, k]
            ) = _get_crps_and_dwcrps_one_var(
                target_values=vector_target_matrix[:, k, j],
                prediction_matrix=vector_prediction_matrix[:, k, j, :],
                num_integration_levels=num_integration_levels
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
        print('Computing CRPS and DWCRPS for {0:s}...'.format(
            aux_target_field_names[j]
        ))

        (
            result_table_xarray[AUX_CRPS_KEY].values[j],
            result_table_xarray[AUX_DWCRPS_KEY].values[j]
        ) = _get_crps_and_dwcrps_one_var(
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


def merge_results_over_examples(result_tables_xarray):
    """Merges CRPS results over many examples.

    :param result_tables_xarray: List of xarray tables, each created by
        `get_crps_related_scores_all_vars`, each containing results for a
        different set of examples.
    :return: result_table_xarray: Single xarray table with results for all
        examples (variable and dimension names should make the table
        self-explanatory).
    """

    prediction_file_names, num_examples_by_table = (
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

    result_table_xarray = copy.deepcopy(result_tables_xarray[0])

    for j in range(len(scalar_target_names)):
        these_crps_values = numpy.array([
            t[SCALAR_CRPS_KEY].values[j] for t in result_tables_xarray
        ])
        result_table_xarray[SCALAR_CRPS_KEY].values[j] = numpy.average(
            these_crps_values, weights=num_examples_by_table
        )

        these_dwcrps_values = numpy.array([
            t[SCALAR_DWCRPS_KEY].values[j] for t in result_tables_xarray
        ])
        result_table_xarray[SCALAR_DWCRPS_KEY].values[j] = numpy.average(
            these_dwcrps_values, weights=num_examples_by_table
        )

        these_crpss_values = numpy.array([
            t[SCALAR_CRPSS_KEY].values[j] for t in result_tables_xarray
        ])
        this_climo_crps = numpy.average(
            these_crps_values / (1. - these_crpss_values),
            weights=num_examples_by_table
        )
        result_table_xarray[SCALAR_CRPSS_KEY].values[j] = (
            1. - result_table_xarray[SCALAR_CRPS_KEY].values[j] /
            this_climo_crps
        )

    for j in range(len(aux_predicted_field_names)):
        these_crps_values = numpy.array([
            t[AUX_CRPS_KEY].values[j] for t in result_tables_xarray
        ])
        result_table_xarray[AUX_CRPS_KEY].values[j] = numpy.average(
            these_crps_values, weights=num_examples_by_table
        )

        these_dwcrps_values = numpy.array([
            t[AUX_DWCRPS_KEY].values[j] for t in result_tables_xarray
        ])
        result_table_xarray[AUX_DWCRPS_KEY].values[j] = numpy.average(
            these_dwcrps_values, weights=num_examples_by_table
        )

        these_crpss_values = numpy.array([
            t[AUX_CRPSS_KEY].values[j] for t in result_tables_xarray
        ])
        this_climo_crps = numpy.average(
            these_crps_values / (1. - these_crpss_values),
            weights=num_examples_by_table
        )
        result_table_xarray[AUX_CRPSS_KEY].values[j] = (
            1. - result_table_xarray[AUX_CRPS_KEY].values[j] / this_climo_crps
        )

    for j in range(len(vector_target_names)):
        for k in range(len(heights_m_agl)):
            these_crps_values = numpy.array([
                t[VECTOR_CRPS_KEY].values[j, k] for t in result_tables_xarray
            ])
            result_table_xarray[VECTOR_CRPS_KEY].values[j, k] = numpy.average(
                these_crps_values, weights=num_examples_by_table
            )

            these_dwcrps_values = numpy.array([
                t[VECTOR_DWCRPS_KEY].values[j, k] for t in result_tables_xarray
            ])
            result_table_xarray[VECTOR_DWCRPS_KEY].values[j, k] = numpy.average(
                these_dwcrps_values, weights=num_examples_by_table
            )

            these_crpss_values = numpy.array([
                t[VECTOR_CRPSS_KEY].values[j, k] for t in result_tables_xarray
            ])
            this_climo_crps = numpy.average(
                these_crps_values / (1. - these_crpss_values),
                weights=num_examples_by_table
            )
            result_table_xarray[VECTOR_CRPSS_KEY].values[j, k] = (
                1. - result_table_xarray[VECTOR_CRPS_KEY].values[j, k] /
                this_climo_crps
            )

    result_table_xarray.attrs[PREDICTION_FILE_KEY] = ' '.join([
        '{0:s}'.format(f) for f in prediction_file_names
    ])
    return result_table_xarray


def write_results(result_table_xarray, netcdf_file_name):
    """Writes CRPS, DWCRPS, and CRPSS for all target variables to NetCDF file.

    :param result_table_xarray: xarray table in format returned by
        `get_crps_related_scores_all_vars`.
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)

    result_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )


def read_results(netcdf_file_name):
    """Reads CRPS, DWCRPS, and CRPSS for all target variables from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: result_table_xarray: xarray table.  Documentation in the
        xarray table should make values self-explanatory.
    """

    return xarray.open_dataset(netcdf_file_name)
