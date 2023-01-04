"""Helper methods for computing PIT and PIT histograms.

PIT = probability integral transform
"""

import os
import copy
import numpy
import xarray
from scipy.stats import percentileofscore
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4rt.io import prediction_io
from ml4rt.utils import example_utils
from ml4rt.utils import uq_evaluation
from ml4rt.machine_learning import neural_net

TOLERANCE = 1e-6

BIN_CENTER_DIM = 'bin_center'
BIN_EDGE_DIM = 'bin_edge'

SCALAR_PITD_KEY = 'scalar_pitd'
SCALAR_PERFECT_PITD_KEY = 'scalar_perfect_pitd'
SCALAR_BIN_COUNT_KEY = 'scalar_bin_count'
VECTOR_PITD_KEY = 'vector_pitd'
VECTOR_PERFECT_PITD_KEY = 'vector_perfect_pitd'
VECTOR_BIN_COUNT_KEY = 'vector_bin_count'
VECTOR_FLAT_PITD_KEY = 'vector_flat_pitd'
VECTOR_FLAT_PERFECT_PITD_KEY = 'vector_flat_perfect_pitd'
VECTOR_FLAT_BIN_COUNT_KEY = 'vector_flat_bin_count'
AUX_PITD_KEY = 'aux_pitd'
AUX_PERFECT_PITD_KEY = 'aux_perfect_pitd'
AUX_BIN_COUNT_KEY = 'aux_bin_count'

SCALAR_FIELD_DIM = uq_evaluation.SCALAR_FIELD_DIM
VECTOR_FIELD_DIM = uq_evaluation.VECTOR_FIELD_DIM
HEIGHT_DIM = uq_evaluation.HEIGHT_DIM
AUX_TARGET_FIELD_DIM = uq_evaluation.AUX_TARGET_FIELD_DIM
AUX_PREDICTED_FIELD_DIM = uq_evaluation.AUX_PREDICTED_FIELD_DIM

MODEL_FILE_KEY = uq_evaluation.MODEL_FILE_KEY
PREDICTION_FILE_KEY = uq_evaluation.PREDICTION_FILE_KEY


def _get_histogram_one_var(target_values, prediction_matrix, num_bins):
    """Computes PIT histogram for one variable.

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


def get_histogram_all_vars(prediction_file_name, num_bins):
    """Computes PIT histogram for each target variable.

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
        SCALAR_PITD_KEY: (
            (SCALAR_FIELD_DIM,), numpy.full(num_scalar_targets, numpy.nan)
        ),
        SCALAR_PERFECT_PITD_KEY: (
            (SCALAR_FIELD_DIM,), numpy.full(num_scalar_targets, numpy.nan)
        ),
        SCALAR_BIN_COUNT_KEY: (
            (SCALAR_FIELD_DIM, BIN_CENTER_DIM),
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
        VECTOR_BIN_COUNT_KEY: (
            (VECTOR_FIELD_DIM, HEIGHT_DIM, BIN_CENTER_DIM),
            numpy.full(
                (num_vector_targets, num_heights, num_bins), -1, dtype=int
            )
        ),
        VECTOR_FLAT_PITD_KEY: (
            (VECTOR_FIELD_DIM,), numpy.full(num_vector_targets, numpy.nan)
        ),
        VECTOR_FLAT_PERFECT_PITD_KEY: (
            (VECTOR_FIELD_DIM,), numpy.full(num_vector_targets, numpy.nan)
        ),
        VECTOR_FLAT_BIN_COUNT_KEY: (
            (VECTOR_FIELD_DIM, BIN_CENTER_DIM),
            numpy.full((num_vector_targets, num_bins), -1, dtype=int)
        )
    }

    if num_aux_targets > 0:
        main_data_dict.update({
            AUX_PITD_KEY: (
                (AUX_TARGET_FIELD_DIM,), numpy.full(num_aux_targets, numpy.nan)
            ),
            AUX_PERFECT_PITD_KEY: (
                (AUX_TARGET_FIELD_DIM,), numpy.full(num_aux_targets, numpy.nan)
            ),
            AUX_BIN_COUNT_KEY: (
                (AUX_TARGET_FIELD_DIM, BIN_CENTER_DIM),
                numpy.full((num_aux_targets, num_bins), -1, dtype=int)
            )
        })

    bin_edges = numpy.linspace(0, 1, num=num_bins + 1, dtype=float)
    bin_centers = bin_edges[:-1] + numpy.diff(bin_edges) / 2

    metadata_dict = {
        SCALAR_FIELD_DIM: example_dict[example_utils.SCALAR_TARGET_NAMES_KEY],
        HEIGHT_DIM: heights_m_agl,
        VECTOR_FIELD_DIM: example_dict[example_utils.VECTOR_TARGET_NAMES_KEY],
        BIN_CENTER_DIM: bin_centers,
        BIN_EDGE_DIM: bin_edges
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
            result_table_xarray[SCALAR_BIN_COUNT_KEY].values[j, :],
            result_table_xarray[SCALAR_PITD_KEY].values[j],
            result_table_xarray[SCALAR_PERFECT_PITD_KEY].values[j]
        ) = _get_histogram_one_var(
            target_values=scalar_target_matrix[:, j],
            prediction_matrix=scalar_prediction_matrix[:, j, :],
            num_bins=num_bins
        )

    for j in range(num_vector_targets):
        print('Computing PIT histogram for {0:s}...'.format(
            example_dict[example_utils.VECTOR_TARGET_NAMES_KEY][j]
        ))

        these_targets = numpy.ravel(vector_target_matrix[:, :, j])
        this_prediction_matrix = numpy.reshape(
            vector_prediction_matrix[:, :, j, :],
            (len(these_targets), vector_prediction_matrix.shape[-1])
        )

        (
            _,
            result_table_xarray[VECTOR_FLAT_BIN_COUNT_KEY].values[j, :],
            result_table_xarray[VECTOR_FLAT_PITD_KEY].values[j],
            result_table_xarray[VECTOR_FLAT_PERFECT_PITD_KEY].values[j]
        ) = _get_histogram_one_var(
            target_values=these_targets,
            prediction_matrix=this_prediction_matrix, num_bins=num_bins
        )

        for k in range(num_heights):
            print('Computing PIT histogram for {0:s} at {1:d} m AGL...'.format(
                example_dict[example_utils.VECTOR_TARGET_NAMES_KEY][j],
                int(numpy.round(heights_m_agl[k]))
            ))

            (
                _,
                result_table_xarray[VECTOR_BIN_COUNT_KEY].values[j, k, :],
                result_table_xarray[VECTOR_PITD_KEY].values[j, k],
                result_table_xarray[VECTOR_PERFECT_PITD_KEY].values[j, k]
            ) = _get_histogram_one_var(
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
            result_table_xarray[AUX_BIN_COUNT_KEY].values[j, :],
            result_table_xarray[AUX_PITD_KEY].values[j],
            result_table_xarray[AUX_PERFECT_PITD_KEY].values[j]
        ) = _get_histogram_one_var(
            target_values=aux_target_matrix[:, j],
            prediction_matrix=aux_prediction_matrix[:, j, :],
            num_bins=num_bins
        )

    return result_table_xarray


def merge_results_over_examples(result_tables_xarray):
    """Merges PIT-histogram results over many examples.

    :param result_tables_xarray: List of xarray tables, each created by
        `get_histogram_all_vars`, each containing results for a different
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
            result_tables_xarray[i].coords[BIN_CENTER_DIM].values,
            result_tables_xarray[0].coords[BIN_CENTER_DIM].values,
            atol=TOLERANCE
        )
        assert numpy.allclose(
            result_tables_xarray[i].coords[BIN_EDGE_DIM].values,
            result_tables_xarray[0].coords[BIN_EDGE_DIM].values,
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

    result_table_xarray = copy.deepcopy(result_tables_xarray[0])

    num_examples_total = numpy.sum(num_examples_by_table)
    num_bins = len(result_table_xarray.coords[BIN_CENTER_DIM].values)
    perfect_bin_frequency = 1. / num_bins

    for j in range(len(scalar_target_names)):
        for i in range(num_bins):
            result_table_xarray[SCALAR_BIN_COUNT_KEY].values[
                j, i
            ] = numpy.sum([
                t[SCALAR_BIN_COUNT_KEY].values[j, i]
                for t in result_tables_xarray
            ])

        error_checking.assert_equals(
            num_examples_total,
            numpy.sum(result_table_xarray[SCALAR_BIN_COUNT_KEY].values[j, :])
        )

        these_frequencies = (
            result_table_xarray[SCALAR_BIN_COUNT_KEY].values[j, :].astype(float)
            / num_examples_total
        )

        result_table_xarray[SCALAR_PITD_KEY].values[j] = numpy.sqrt(
            numpy.mean((these_frequencies - perfect_bin_frequency) ** 2)
        )
        result_table_xarray[SCALAR_PERFECT_PITD_KEY].values[j] = numpy.sqrt(
            (1. - perfect_bin_frequency) / (num_examples_total * num_bins)
        )

    for j in range(len(aux_predicted_field_names)):
        for i in range(num_bins):
            result_table_xarray[AUX_BIN_COUNT_KEY].values[
                j, i
            ] = numpy.sum([
                t[AUX_BIN_COUNT_KEY].values[j, i]
                for t in result_tables_xarray
            ])

        error_checking.assert_equals(
            num_examples_total,
            numpy.sum(result_table_xarray[AUX_BIN_COUNT_KEY].values[j, :])
        )

        these_frequencies = (
            result_table_xarray[AUX_BIN_COUNT_KEY].values[j, :].astype(float)
            / num_examples_total
        )

        result_table_xarray[AUX_PITD_KEY].values[j] = numpy.sqrt(
            numpy.mean((these_frequencies - perfect_bin_frequency) ** 2)
        )
        result_table_xarray[AUX_PERFECT_PITD_KEY].values[j] = numpy.sqrt(
            (1. - perfect_bin_frequency) / (num_examples_total * num_bins)
        )

    for j in range(len(vector_target_names)):
        for i in range(num_bins):
            result_table_xarray[VECTOR_FLAT_BIN_COUNT_KEY].values[
                j, i
            ] = numpy.sum([
                t[VECTOR_FLAT_BIN_COUNT_KEY].values[j, i]
                for t in result_tables_xarray
            ])

        these_counts = (
            result_table_xarray[VECTOR_FLAT_BIN_COUNT_KEY].values[j, :]
        )
        these_frequencies = these_counts.astype(float) / numpy.sum(these_counts)

        result_table_xarray[VECTOR_FLAT_PITD_KEY].values[j] = numpy.sqrt(
            numpy.mean((these_frequencies - perfect_bin_frequency) ** 2)
        )
        result_table_xarray[VECTOR_FLAT_PERFECT_PITD_KEY].values[
            j
        ] = numpy.sqrt(
            (1. - perfect_bin_frequency) / (num_examples_total * num_bins)
        )

        for k in range(len(heights_m_agl)):
            for i in range(num_bins):
                result_table_xarray[VECTOR_BIN_COUNT_KEY].values[
                    j, k, i
                ] = numpy.sum([
                    t[VECTOR_BIN_COUNT_KEY].values[j, k, i]
                    for t in result_tables_xarray
                ])

            error_checking.assert_equals(
                num_examples_total,
                numpy.sum(
                    result_table_xarray[VECTOR_BIN_COUNT_KEY].values[j, k, :]
                )
            )

            these_frequencies = (
                result_table_xarray[VECTOR_BIN_COUNT_KEY].values[
                    j, k, :
                ].astype(float)
                / num_examples_total
            )

            result_table_xarray[VECTOR_PITD_KEY].values[j, k] = numpy.sqrt(
                numpy.mean((these_frequencies - perfect_bin_frequency) ** 2)
            )
            result_table_xarray[VECTOR_PERFECT_PITD_KEY].values[
                j, k
            ] = numpy.sqrt(
                (1. - perfect_bin_frequency) / (num_examples_total * num_bins)
            )

    result_table_xarray.attrs[PREDICTION_FILE_KEY] = ' '.join([
        '{0:s}'.format(f) for f in prediction_file_names
    ])
    return result_table_xarray


def write_results(result_table_xarray, netcdf_file_name):
    """Writes PIT histogram for each target variable to NetCDF file.

    :param result_table_xarray: xarray table in format returned by
        `get_histogram_all_vars`.
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)

    result_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )


def read_results(netcdf_file_name):
    """Reads PIT histogram for each target variable from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: result_table_xarray: xarray table.  Documentation in the
        xarray table should make values self-explanatory.
    """

    return xarray.open_dataset(netcdf_file_name)
