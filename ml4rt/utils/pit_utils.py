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
METRES_TO_MICRONS = 1e6

MAX_PIT_FOR_LOW_BINS = 0.3
MIN_PIT_FOR_HIGH_BINS = 0.7
CONFIDENCE_LEVEL_FOR_NONEXTREME_PIT = 0.95

BIN_EDGES_KEY = 'bin_edges'
BIN_COUNTS_KEY = 'bin_counts'
PITD_KEY = 'pitd_value'
PERFECT_PITD_KEY = 'perfect_pitd_value'
LOW_BIN_BIAS_KEY = 'low_bin_pit_bias'
MIDDLE_BIN_BIAS_KEY = 'middle_bin_pit_bias'
HIGH_BIN_BIAS_KEY = 'high_bin_pit_bias'
EXTREME_PIT_FREQ_KEY = 'extreme_pit_frequency'

BIN_CENTER_DIM = 'bin_center'
BIN_EDGE_DIM = 'bin_edge'

SCALAR_PITD_KEY = 'scalar_pitd'
SCALAR_PERFECT_PITD_KEY = 'scalar_perfect_pitd'
SCALAR_BIN_COUNT_KEY = 'scalar_bin_count'
SCALAR_LOW_BIN_BIAS_KEY = 'scalar_low_bin_pit_bias'
SCALAR_MIDDLE_BIN_BIAS_KEY = 'scalar_middle_bin_pit_bias'
SCALAR_HIGH_BIN_BIAS_KEY = 'scalar_high_bin_pit_bias'
SCALAR_EXTREME_PIT_FREQ_KEY = 'scalar_extreme_pit_frequency'

VECTOR_PITD_KEY = 'vector_pitd'
VECTOR_PERFECT_PITD_KEY = 'vector_perfect_pitd'
VECTOR_BIN_COUNT_KEY = 'vector_bin_count'
VECTOR_LOW_BIN_BIAS_KEY = 'vector_low_bin_pit_bias'
VECTOR_MIDDLE_BIN_BIAS_KEY = 'vector_middle_bin_pit_bias'
VECTOR_HIGH_BIN_BIAS_KEY = 'vector_high_bin_pit_bias'
VECTOR_EXTREME_PIT_FREQ_KEY = 'vector_extreme_pit_frequency'

VECTOR_FLAT_PITD_KEY = 'vector_flat_pitd'
VECTOR_FLAT_PERFECT_PITD_KEY = 'vector_flat_perfect_pitd'
VECTOR_FLAT_BIN_COUNT_KEY = 'vector_flat_bin_count'
VECTOR_FLAT_LOW_BIN_BIAS_KEY = 'vector_flat_low_bin_pit_bias'
VECTOR_FLAT_MIDDLE_BIN_BIAS_KEY = 'vector_flat_middle_bin_pit_bias'
VECTOR_FLAT_HIGH_BIN_BIAS_KEY = 'vector_flat_high_bin_pit_bias'
VECTOR_FLAT_EXTREME_PIT_FREQ_KEY = 'vector_flat_extreme_pit_frequency'

AUX_PITD_KEY = 'aux_pitd'
AUX_PERFECT_PITD_KEY = 'aux_perfect_pitd'
AUX_BIN_COUNT_KEY = 'aux_bin_count'
AUX_LOW_BIN_BIAS_KEY = 'aux_low_bin_pit_bias'
AUX_MIDDLE_BIN_BIAS_KEY = 'aux_middle_bin_pit_bias'
AUX_HIGH_BIN_BIAS_KEY = 'aux_high_bin_pit_bias'
AUX_EXTREME_PIT_FREQ_KEY = 'aux_extreme_pit_frequency'

SCALAR_FIELD_DIM = uq_evaluation.SCALAR_FIELD_DIM
VECTOR_FIELD_DIM = uq_evaluation.VECTOR_FIELD_DIM
HEIGHT_DIM = uq_evaluation.HEIGHT_DIM
WAVELENGTH_DIM = uq_evaluation.WAVELENGTH_DIM
AUX_TARGET_FIELD_DIM = uq_evaluation.AUX_TARGET_FIELD_DIM
AUX_PREDICTED_FIELD_DIM = uq_evaluation.AUX_PREDICTED_FIELD_DIM

MODEL_FILE_KEY = uq_evaluation.MODEL_FILE_KEY
PREDICTION_FILE_KEY = uq_evaluation.PREDICTION_FILE_KEY


def _get_low_mid_hi_bins(bin_edges):
    """Returns indices for low-PIT, medium-PIT, and high-PIT bins.

    B = number of bins

    :param bin_edges: length-(B + 1) numpy array of bin edges, sorted in
        ascending order.
    :return: low_bin_indices: 1-D numpy array with array indices for low-PIT
        bins.
    :return: middle_bin_indices: 1-D numpy array with array indices for
        medium-PIT bins.
    :return: high_bin_indices: 1-D numpy array with array indices for high-PIT
        bins.
    """

    num_bins = len(bin_edges) - 1

    these_diffs = bin_edges - MAX_PIT_FOR_LOW_BINS
    these_diffs[these_diffs > TOLERANCE] = numpy.inf
    max_index_for_low_bins = numpy.argmin(numpy.absolute(these_diffs)) - 1
    max_index_for_low_bins = max([max_index_for_low_bins, 0])

    low_bin_indices = numpy.linspace(
        0, max_index_for_low_bins, num=max_index_for_low_bins + 1, dtype=int
    )

    these_diffs = MIN_PIT_FOR_HIGH_BINS - bin_edges
    these_diffs[these_diffs > TOLERANCE] = numpy.inf
    min_index_for_high_bins = numpy.argmin(numpy.absolute(these_diffs))
    min_index_for_high_bins = min([min_index_for_high_bins, num_bins - 1])

    high_bin_indices = numpy.linspace(
        min_index_for_high_bins, num_bins - 1,
        num=num_bins - min_index_for_high_bins, dtype=int
    )

    middle_bin_indices = numpy.linspace(
        0, num_bins - 1, num=num_bins, dtype=int
    )
    middle_bin_indices = numpy.array(list(
        set(middle_bin_indices.tolist())
        - set(low_bin_indices.tolist())
        - set(high_bin_indices.tolist())
    ))

    return low_bin_indices, middle_bin_indices, high_bin_indices


def _get_histogram_one_var(target_values, prediction_matrix, num_bins):
    """Computes PIT histogram for one variable.

    E = number of examples
    S = number of ensemble members
    B = number of bins in histogram

    :param target_values: length-E numpy array of actual values.
    :param prediction_matrix: E-by-S numpy array of predicted values.
    :param num_bins: Number of bins in histogram.
    :return: result_dict: Dictionary with the following keys.
    result_dict["bin_edges"]: length-(B + 1) numpy array of bin edges (ranging
        from 0...1, because PIT ranges from 0...1).
    result_dict["bin_counts"]: length-B numpy array with number of examples in
        each bin.
    result_dict["pitd_value"]: Value of the calibration-deviation metric (PITD).
    result_dict["perfect_pitd_value"]: Minimum expected PITD value.
    result_dict["low_bin_pit_bias"]: PIT bias for low bins, i.e., PIT values of
        [0, 0.3).
    result_dict["middle_bin_pit_bias"]: PIT bias for middle bins, i.e., PIT
        values of [0.3, 0.7).
    result_dict["high_bin_pit_bias"]: PIT bias for high bins, i.e., PIT values
        of [0.7, 1.0].
    result_dict["extreme_pit_frequency"]: Frequency of extreme PIT values, i.e.,
        below 0.025 or above 0.975.
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

    low_bin_indices, middle_bin_indices, high_bin_indices = (
        _get_low_mid_hi_bins(bin_edges)
    )

    low_bin_pit_bias = numpy.mean(
        bin_frequencies[low_bin_indices] - perfect_bin_frequency
    )
    middle_bin_pit_bias = numpy.mean(
        bin_frequencies[middle_bin_indices] - perfect_bin_frequency
    )
    high_bin_pit_bias = numpy.mean(
        bin_frequencies[high_bin_indices] - perfect_bin_frequency
    )
    extreme_pit_frequency = numpy.mean(numpy.logical_or(
        pit_values < 0.5 * (1. - CONFIDENCE_LEVEL_FOR_NONEXTREME_PIT),
        pit_values > 0.5 * (1. + CONFIDENCE_LEVEL_FOR_NONEXTREME_PIT)
    ))

    return {
        BIN_EDGES_KEY: bin_edges,
        BIN_COUNTS_KEY: bin_counts,
        PITD_KEY: pitd_value,
        PERFECT_PITD_KEY: perfect_pitd_value,
        LOW_BIN_BIAS_KEY: low_bin_pit_bias,
        MIDDLE_BIN_BIAS_KEY: middle_bin_pit_bias,
        HIGH_BIN_BIAS_KEY: high_bin_pit_bias,
        EXTREME_PIT_FREQ_KEY: extreme_pit_frequency
    }


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

    del prediction_dict

    num_heights = vector_target_matrix.shape[1]
    num_wavelengths = vector_target_matrix.shape[2]
    num_vector_targets = vector_target_matrix.shape[3]
    num_scalar_targets = scalar_target_matrix.shape[2]
    num_aux_targets = len(aux_target_field_names)

    main_data_dict = {
        SCALAR_PITD_KEY: (
            (SCALAR_FIELD_DIM, WAVELENGTH_DIM),
            numpy.full((num_scalar_targets, num_wavelengths), numpy.nan)
        ),
        SCALAR_PERFECT_PITD_KEY: (
            (SCALAR_FIELD_DIM, WAVELENGTH_DIM),
            numpy.full((num_scalar_targets, num_wavelengths), numpy.nan)
        ),
        SCALAR_BIN_COUNT_KEY: (
            (SCALAR_FIELD_DIM, WAVELENGTH_DIM, BIN_CENTER_DIM),
            numpy.full(
                (num_scalar_targets, num_wavelengths, num_bins), -1, dtype=int
            )
        ),
        SCALAR_LOW_BIN_BIAS_KEY: (
            (SCALAR_FIELD_DIM, WAVELENGTH_DIM),
            numpy.full((num_scalar_targets, num_wavelengths), numpy.nan)
        ),
        SCALAR_MIDDLE_BIN_BIAS_KEY: (
            (SCALAR_FIELD_DIM, WAVELENGTH_DIM),
            numpy.full((num_scalar_targets, num_wavelengths), numpy.nan)
        ),
        SCALAR_HIGH_BIN_BIAS_KEY: (
            (SCALAR_FIELD_DIM, WAVELENGTH_DIM),
            numpy.full((num_scalar_targets, num_wavelengths), numpy.nan)
        ),
        SCALAR_EXTREME_PIT_FREQ_KEY: (
            (SCALAR_FIELD_DIM, WAVELENGTH_DIM),
            numpy.full((num_scalar_targets, num_wavelengths), numpy.nan)
        ),
        VECTOR_PITD_KEY: (
            (VECTOR_FIELD_DIM, HEIGHT_DIM, WAVELENGTH_DIM),
            numpy.full(
                (num_vector_targets, num_heights, num_wavelengths), numpy.nan
            )
        ),
        VECTOR_PERFECT_PITD_KEY: (
            (VECTOR_FIELD_DIM, HEIGHT_DIM, WAVELENGTH_DIM),
            numpy.full(
                (num_vector_targets, num_heights, num_wavelengths), numpy.nan
            )
        ),
        VECTOR_BIN_COUNT_KEY: (
            (VECTOR_FIELD_DIM, HEIGHT_DIM, WAVELENGTH_DIM, BIN_CENTER_DIM),
            numpy.full(
                (num_vector_targets, num_heights, num_wavelengths, num_bins),
                -1, dtype=int
            )
        ),
        VECTOR_LOW_BIN_BIAS_KEY: (
            (VECTOR_FIELD_DIM, HEIGHT_DIM, WAVELENGTH_DIM),
            numpy.full(
                (num_vector_targets, num_heights, num_wavelengths), numpy.nan
            )
        ),
        VECTOR_MIDDLE_BIN_BIAS_KEY: (
            (VECTOR_FIELD_DIM, HEIGHT_DIM, WAVELENGTH_DIM),
            numpy.full(
                (num_vector_targets, num_heights, num_wavelengths), numpy.nan
            )
        ),
        VECTOR_HIGH_BIN_BIAS_KEY: (
            (VECTOR_FIELD_DIM, HEIGHT_DIM, WAVELENGTH_DIM),
            numpy.full(
                (num_vector_targets, num_heights, num_wavelengths), numpy.nan
            )
        ),
        VECTOR_EXTREME_PIT_FREQ_KEY: (
            (VECTOR_FIELD_DIM, HEIGHT_DIM, WAVELENGTH_DIM),
            numpy.full(
                (num_vector_targets, num_heights, num_wavelengths), numpy.nan
            )
        ),
        VECTOR_FLAT_PITD_KEY: (
            (VECTOR_FIELD_DIM, WAVELENGTH_DIM),
            numpy.full((num_vector_targets, num_wavelengths), numpy.nan)
        ),
        VECTOR_FLAT_PERFECT_PITD_KEY: (
            (VECTOR_FIELD_DIM, WAVELENGTH_DIM),
            numpy.full((num_vector_targets, num_wavelengths), numpy.nan)
        ),
        VECTOR_FLAT_BIN_COUNT_KEY: (
            (VECTOR_FIELD_DIM, WAVELENGTH_DIM, BIN_CENTER_DIM),
            numpy.full(
                (num_vector_targets, num_wavelengths, num_bins), -1, dtype=int
            )
        ),
        VECTOR_FLAT_LOW_BIN_BIAS_KEY: (
            (VECTOR_FIELD_DIM, WAVELENGTH_DIM),
            numpy.full((num_vector_targets, num_wavelengths), numpy.nan)
        ),
        VECTOR_FLAT_MIDDLE_BIN_BIAS_KEY: (
            (VECTOR_FIELD_DIM, WAVELENGTH_DIM),
            numpy.full((num_vector_targets, num_wavelengths), numpy.nan)
        ),
        VECTOR_FLAT_HIGH_BIN_BIAS_KEY: (
            (VECTOR_FIELD_DIM, WAVELENGTH_DIM),
            numpy.full((num_vector_targets, num_wavelengths), numpy.nan)
        ),
        VECTOR_FLAT_EXTREME_PIT_FREQ_KEY: (
            (VECTOR_FIELD_DIM, WAVELENGTH_DIM),
            numpy.full((num_vector_targets, num_wavelengths), numpy.nan)
        )
    }

    if num_aux_targets > 0:
        main_data_dict.update({
            AUX_PITD_KEY: (
                (AUX_TARGET_FIELD_DIM, WAVELENGTH_DIM),
                numpy.full((num_aux_targets, num_wavelengths), numpy.nan)
            ),
            AUX_PERFECT_PITD_KEY: (
                (AUX_TARGET_FIELD_DIM, WAVELENGTH_DIM),
                numpy.full((num_aux_targets, num_wavelengths), numpy.nan)
            ),
            AUX_BIN_COUNT_KEY: (
                (AUX_TARGET_FIELD_DIM, WAVELENGTH_DIM, BIN_CENTER_DIM),
                numpy.full(
                    (num_aux_targets, num_wavelengths, num_bins), -1, dtype=int
                )
            ),
            AUX_LOW_BIN_BIAS_KEY: (
                (AUX_TARGET_FIELD_DIM, WAVELENGTH_DIM),
                numpy.full((num_aux_targets, num_wavelengths), numpy.nan)
            ),
            AUX_MIDDLE_BIN_BIAS_KEY: (
                (AUX_TARGET_FIELD_DIM, WAVELENGTH_DIM),
                numpy.full((num_aux_targets, num_wavelengths), numpy.nan)
            ),
            AUX_HIGH_BIN_BIAS_KEY: (
                (AUX_TARGET_FIELD_DIM, WAVELENGTH_DIM),
                numpy.full((num_aux_targets, num_wavelengths), numpy.nan)
            ),
            AUX_EXTREME_PIT_FREQ_KEY: (
                (AUX_TARGET_FIELD_DIM, WAVELENGTH_DIM),
                numpy.full((num_aux_targets, num_wavelengths), numpy.nan)
            )
        })

    bin_edges = numpy.linspace(0, 1, num=num_bins + 1, dtype=float)
    bin_centers = bin_edges[:-1] + numpy.diff(bin_edges) / 2

    metadata_dict = {
        SCALAR_FIELD_DIM: example_dict[example_utils.SCALAR_TARGET_NAMES_KEY],
        VECTOR_FIELD_DIM: example_dict[example_utils.VECTOR_TARGET_NAMES_KEY],
        HEIGHT_DIM: heights_m_agl,
        WAVELENGTH_DIM: wavelengths_metres,
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

    for t in range(num_scalar_targets):
        for w in range(num_wavelengths):
            print((
                'Computing PIT histogram for {0:s} at {1:.2f} microns...'
            ).format(
                example_dict[example_utils.SCALAR_TARGET_NAMES_KEY][t],
                METRES_TO_MICRONS * wavelengths_metres[w]
            ))

            this_result_dict = _get_histogram_one_var(
                target_values=scalar_target_matrix[:, w, t],
                prediction_matrix=scalar_prediction_matrix[:, w, t, :],
                num_bins=num_bins
            )
            result_table_xarray[SCALAR_BIN_COUNT_KEY].values[t, w, :] = (
                this_result_dict[BIN_COUNTS_KEY]
            )
            result_table_xarray[SCALAR_PITD_KEY].values[t, w] = (
                this_result_dict[PITD_KEY]
            )
            result_table_xarray[SCALAR_PERFECT_PITD_KEY].values[t, w] = (
                this_result_dict[PERFECT_PITD_KEY]
            )
            result_table_xarray[SCALAR_LOW_BIN_BIAS_KEY].values[t, w] = (
                this_result_dict[LOW_BIN_BIAS_KEY]
            )
            result_table_xarray[SCALAR_MIDDLE_BIN_BIAS_KEY].values[t, w] = (
                this_result_dict[MIDDLE_BIN_BIAS_KEY]
            )
            result_table_xarray[SCALAR_HIGH_BIN_BIAS_KEY].values[t, w] = (
                this_result_dict[HIGH_BIN_BIAS_KEY]
            )
            result_table_xarray[SCALAR_EXTREME_PIT_FREQ_KEY].values[t, w] = (
                this_result_dict[EXTREME_PIT_FREQ_KEY]
            )

    for t in range(num_vector_targets):
        for w in range(num_wavelengths):
            print((
                'Computing PIT histogram for {0:s} at {1:.2f} microns...'
            ).format(
                example_dict[example_utils.VECTOR_TARGET_NAMES_KEY][t],
                METRES_TO_MICRONS * wavelengths_metres[w]
            ))

            these_targets = numpy.ravel(vector_target_matrix[..., w, t])
            this_prediction_matrix = numpy.reshape(
                vector_prediction_matrix[..., w, t, :],
                (len(these_targets), vector_prediction_matrix.shape[-1])
            )

            this_result_dict = _get_histogram_one_var(
                target_values=these_targets,
                prediction_matrix=this_prediction_matrix, num_bins=num_bins
            )
            result_table_xarray[VECTOR_FLAT_BIN_COUNT_KEY].values[t, w, :] = (
                this_result_dict[BIN_COUNTS_KEY]
            )
            result_table_xarray[VECTOR_FLAT_PITD_KEY].values[t, w] = (
                this_result_dict[PITD_KEY]
            )
            result_table_xarray[VECTOR_FLAT_PERFECT_PITD_KEY].values[t, w] = (
                this_result_dict[PERFECT_PITD_KEY]
            )
            result_table_xarray[VECTOR_FLAT_LOW_BIN_BIAS_KEY].values[t, w] = (
                this_result_dict[LOW_BIN_BIAS_KEY]
            )
            result_table_xarray[VECTOR_FLAT_MIDDLE_BIN_BIAS_KEY].values[t, w] = (
                this_result_dict[MIDDLE_BIN_BIAS_KEY]
            )
            result_table_xarray[VECTOR_FLAT_HIGH_BIN_BIAS_KEY].values[t, w] = (
                this_result_dict[HIGH_BIN_BIAS_KEY]
            )
            result_table_xarray[VECTOR_FLAT_EXTREME_PIT_FREQ_KEY].values[t, w] = (
                this_result_dict[EXTREME_PIT_FREQ_KEY]
            )

            for h in range(num_heights):
                print((
                    'Computing PIT histogram for {0:s} at {1:.2f} microns and '
                    '{2:d} m AGL...'
                ).format(
                    example_dict[example_utils.VECTOR_TARGET_NAMES_KEY][t],
                    METRES_TO_MICRONS * wavelengths_metres[w],
                    int(numpy.round(heights_m_agl[h]))
                ))

                this_result_dict = _get_histogram_one_var(
                    target_values=vector_target_matrix[:, h, w, t],
                    prediction_matrix=vector_prediction_matrix[:, h, w, t, :],
                    num_bins=num_bins
                )
                result_table_xarray[VECTOR_BIN_COUNT_KEY].values[t, h, w, :] = (
                    this_result_dict[BIN_COUNTS_KEY]
                )
                result_table_xarray[VECTOR_PITD_KEY].values[t, h, w] = (
                    this_result_dict[PITD_KEY]
                )
                result_table_xarray[VECTOR_PERFECT_PITD_KEY].values[t, h, w] = (
                    this_result_dict[PERFECT_PITD_KEY]
                )
                result_table_xarray[VECTOR_LOW_BIN_BIAS_KEY].values[t, h, w] = (
                    this_result_dict[LOW_BIN_BIAS_KEY]
                )
                result_table_xarray[VECTOR_MIDDLE_BIN_BIAS_KEY].values[t, h, w] = (
                    this_result_dict[MIDDLE_BIN_BIAS_KEY]
                )
                result_table_xarray[VECTOR_HIGH_BIN_BIAS_KEY].values[t, h, w] = (
                    this_result_dict[HIGH_BIN_BIAS_KEY]
                )
                result_table_xarray[VECTOR_EXTREME_PIT_FREQ_KEY].values[t, h, w] = (
                    this_result_dict[EXTREME_PIT_FREQ_KEY]
                )

    for t in range(num_aux_targets):
        for w in range(num_wavelengths):
            print((
                'Computing PIT histogram for {0:s} at {1:.2f} microns...'
            ).format(
                aux_target_field_names[t],
                METRES_TO_MICRONS * wavelengths_metres[w]
            ))

            this_result_dict = _get_histogram_one_var(
                target_values=aux_target_matrix[:, w, t],
                prediction_matrix=aux_prediction_matrix[:, w, t, :],
                num_bins=num_bins
            )
            result_table_xarray[AUX_BIN_COUNT_KEY].values[t, w, :] = (
                this_result_dict[BIN_COUNTS_KEY]
            )
            result_table_xarray[AUX_PITD_KEY].values[t, w] = (
                this_result_dict[PITD_KEY]
            )
            result_table_xarray[AUX_PERFECT_PITD_KEY].values[t, w] = (
                this_result_dict[PERFECT_PITD_KEY]
            )
            result_table_xarray[AUX_LOW_BIN_BIAS_KEY].values[t, w] = (
                this_result_dict[LOW_BIN_BIAS_KEY]
            )
            result_table_xarray[AUX_MIDDLE_BIN_BIAS_KEY].values[t, w] = (
                this_result_dict[MIDDLE_BIN_BIAS_KEY]
            )
            result_table_xarray[AUX_HIGH_BIN_BIAS_KEY].values[t, w] = (
                this_result_dict[HIGH_BIN_BIAS_KEY]
            )
            result_table_xarray[AUX_EXTREME_PIT_FREQ_KEY].values[t, w] = (
                this_result_dict[EXTREME_PIT_FREQ_KEY]
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
    wavelengths_metres = result_tables_xarray[0].coords[WAVELENGTH_DIM].values

    try:
        aux_predicted_field_names = (
            result_tables_xarray[0].coords[AUX_PREDICTED_FIELD_DIM].values
        )
    except:
        aux_predicted_field_names = []

    result_table_xarray = copy.deepcopy(result_tables_xarray[0])
    rtx = result_table_xarray

    num_examples_total = numpy.sum(num_examples_by_table)
    num_bins = len(rtx.coords[BIN_CENTER_DIM].values)
    perfect_bin_frequency = 1. / num_bins

    low_bin_indices, middle_bin_indices, high_bin_indices = (
        _get_low_mid_hi_bins(rtx.coords[BIN_EDGE_DIM].values)
    )

    for t in range(len(scalar_target_names)):
        for w in range(len(wavelengths_metres)):
            for b in range(num_bins):
                rtx[SCALAR_BIN_COUNT_KEY].values[t, w, b] = numpy.sum([
                    this_tbl[SCALAR_BIN_COUNT_KEY].values[t, w, b]
                    for this_tbl in result_tables_xarray
                ])

            extreme_freq_by_table = numpy.array([
                this_tbl[SCALAR_EXTREME_PIT_FREQ_KEY].values[t, w]
                for this_tbl in result_tables_xarray
            ])
            rtx[SCALAR_EXTREME_PIT_FREQ_KEY].values[t, w] = numpy.average(
                a=extreme_freq_by_table[num_examples_by_table > 0],
                weights=num_examples_by_table[num_examples_by_table > 0]
            )

            error_checking.assert_equals(
                num_examples_total,
                numpy.sum(rtx[SCALAR_BIN_COUNT_KEY].values[t, w, :])
            )
            these_frequencies = (
                rtx[SCALAR_BIN_COUNT_KEY].values[t, w, :].astype(float)
                / num_examples_total
            )

            rtx[SCALAR_PITD_KEY].values[t, w] = numpy.sqrt(
                numpy.mean((these_frequencies - perfect_bin_frequency) ** 2)
            )
            rtx[SCALAR_PERFECT_PITD_KEY].values[t, w] = numpy.sqrt(
                (1. - perfect_bin_frequency) /
                (num_examples_total * num_bins)
            )
            rtx[SCALAR_LOW_BIN_BIAS_KEY].values[t, w] = numpy.mean(
                these_frequencies[low_bin_indices] - perfect_bin_frequency
            )
            rtx[SCALAR_MIDDLE_BIN_BIAS_KEY].values[t, w] = numpy.mean(
                these_frequencies[middle_bin_indices] - perfect_bin_frequency
            )
            rtx[SCALAR_HIGH_BIN_BIAS_KEY].values[t, w] = numpy.mean(
                these_frequencies[high_bin_indices] - perfect_bin_frequency
            )

    for t in range(len(aux_predicted_field_names)):
        for w in range(len(wavelengths_metres)):
            for b in range(num_bins):
                rtx[AUX_BIN_COUNT_KEY].values[t, w, b] = numpy.sum([
                    this_tbl[AUX_BIN_COUNT_KEY].values[t, w, b]
                    for this_tbl in result_tables_xarray
                ])

            extreme_freq_by_table = numpy.array([
                this_tbl[AUX_EXTREME_PIT_FREQ_KEY].values[t, w]
                for this_tbl in result_tables_xarray
            ])
            rtx[AUX_EXTREME_PIT_FREQ_KEY].values[t, w] = numpy.average(
                a=extreme_freq_by_table[num_examples_by_table > 0],
                weights=num_examples_by_table[num_examples_by_table > 0]
            )

            error_checking.assert_equals(
                num_examples_total,
                numpy.sum(rtx[AUX_BIN_COUNT_KEY].values[t, w, :])
            )
            these_frequencies = (
                rtx[AUX_BIN_COUNT_KEY].values[t, w, :].astype(float)
                / num_examples_total
            )

            rtx[AUX_PITD_KEY].values[t, w] = numpy.sqrt(
                numpy.mean((these_frequencies - perfect_bin_frequency) ** 2)
            )
            rtx[AUX_PERFECT_PITD_KEY].values[t, w] = numpy.sqrt(
                (1. - perfect_bin_frequency) /
                (num_examples_total * num_bins)
            )
            rtx[AUX_LOW_BIN_BIAS_KEY].values[t, w] = numpy.mean(
                these_frequencies[low_bin_indices] - perfect_bin_frequency
            )
            rtx[AUX_MIDDLE_BIN_BIAS_KEY].values[t, w] = numpy.mean(
                these_frequencies[middle_bin_indices] - perfect_bin_frequency
            )
            rtx[AUX_HIGH_BIN_BIAS_KEY].values[t, w] = numpy.mean(
                these_frequencies[high_bin_indices] - perfect_bin_frequency
            )

    for t in range(len(vector_target_names)):
        for w in range(len(wavelengths_metres)):
            for b in range(num_bins):
                rtx[VECTOR_FLAT_BIN_COUNT_KEY].values[t, w, b] = numpy.sum([
                    this_tbl[VECTOR_FLAT_BIN_COUNT_KEY].values[t, w, b]
                    for this_tbl in result_tables_xarray
                ])

            extreme_freq_by_table = numpy.array([
                this_tbl[VECTOR_FLAT_EXTREME_PIT_FREQ_KEY].values[t, w]
                for this_tbl in result_tables_xarray
            ])
            rtx[VECTOR_FLAT_EXTREME_PIT_FREQ_KEY].values[t, w] = numpy.average(
                a=extreme_freq_by_table[num_examples_by_table > 0],
                weights=num_examples_by_table[num_examples_by_table > 0]
            )

            these_counts = rtx[VECTOR_FLAT_BIN_COUNT_KEY].values[t, w, :]
            these_frequencies = (
                these_counts.astype(float) / numpy.sum(these_counts)
            )

            rtx[VECTOR_FLAT_PITD_KEY].values[t, w] = numpy.sqrt(
                numpy.mean((these_frequencies - perfect_bin_frequency) ** 2)
            )
            rtx[VECTOR_FLAT_PERFECT_PITD_KEY].values[t, w] = numpy.sqrt(
                (1. - perfect_bin_frequency) /
                (num_examples_total * num_bins)
            )
            rtx[VECTOR_FLAT_LOW_BIN_BIAS_KEY].values[t, w] = numpy.mean(
                these_frequencies[low_bin_indices] - perfect_bin_frequency
            )
            rtx[VECTOR_FLAT_MIDDLE_BIN_BIAS_KEY].values[t, w] = numpy.mean(
                these_frequencies[middle_bin_indices] - perfect_bin_frequency
            )
            rtx[VECTOR_FLAT_HIGH_BIN_BIAS_KEY].values[t, w] = numpy.mean(
                these_frequencies[high_bin_indices] - perfect_bin_frequency
            )

            for h in range(len(heights_m_agl)):
                for b in range(num_bins):
                    rtx[VECTOR_BIN_COUNT_KEY].values[t, h, w, b] = numpy.sum([
                        this_tbl[VECTOR_BIN_COUNT_KEY].values[t, h, w, b]
                        for this_tbl in result_tables_xarray
                    ])

                extreme_freq_by_table = numpy.array([
                    this_tbl[VECTOR_EXTREME_PIT_FREQ_KEY].values[t, h, w]
                    for this_tbl in result_tables_xarray
                ])
                rtx[VECTOR_EXTREME_PIT_FREQ_KEY].values[t, h, w] = (
                    numpy.average(
                        a=extreme_freq_by_table[num_examples_by_table > 0],
                        weights=num_examples_by_table[num_examples_by_table > 0]
                    )
                )

                error_checking.assert_equals(
                    num_examples_total,
                    numpy.sum(rtx[VECTOR_BIN_COUNT_KEY].values[t, h, w, :])
                )
                these_frequencies = (
                    rtx[VECTOR_BIN_COUNT_KEY].values[t, h, w, :].astype(float)
                    / num_examples_total
                )

                rtx[VECTOR_PITD_KEY].values[t, h, w] = numpy.sqrt(
                    numpy.mean((these_frequencies - perfect_bin_frequency) ** 2)
                )
                rtx[VECTOR_PERFECT_PITD_KEY].values[t, h, w] = numpy.sqrt(
                    (1. - perfect_bin_frequency) /
                    (num_examples_total * num_bins)
                )
                rtx[VECTOR_LOW_BIN_BIAS_KEY].values[t, h, w] = numpy.mean(
                    these_frequencies[low_bin_indices] - perfect_bin_frequency
                )
                rtx[VECTOR_MIDDLE_BIN_BIAS_KEY].values[t, h, w] = numpy.mean(
                    these_frequencies[middle_bin_indices] -
                    perfect_bin_frequency
                )
                rtx[VECTOR_HIGH_BIN_BIAS_KEY].values[t, h, w] = numpy.mean(
                    these_frequencies[high_bin_indices] - perfect_bin_frequency
                )

    rtx.attrs[PREDICTION_FILE_KEY] = ' '.join([
        '{0:s}'.format(f) for f in prediction_file_names
    ])
    return rtx


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
