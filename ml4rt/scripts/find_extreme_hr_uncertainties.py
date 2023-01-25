"""Finds extreme HR uncertainties (largest, smallest, best/worst predicted).

HR = heating rate
"""

import os
import copy
import argparse
import numpy
from scipy.stats import percentileofscore
from gewittergefahr.gg_utils import error_checking
from ml4rt.io import prediction_io
from ml4rt.utils import misc as misc_utils
from ml4rt.utils import example_utils
from ml4rt.machine_learning import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_FILE_ARG_NAME = 'input_prediction_file_name'
FOR_SHORTWAVE_ARG_NAME = 'for_shortwave'
AVERAGE_OVER_HEIGHT_ARG_NAME = 'average_over_height'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
NUM_EXAMPLES_ARG_NAME = 'num_examples_per_set'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing actual and predicted values.  Will be read '
    'by `prediction_io.write_file`.'
)
FOR_SHORTWAVE_HELP_STRING = (
    'Boolean flag.  If 1 (0), will find extreme uncertainties for shortwave '
    '(longwave) heating rate.'
)
AVERAGE_OVER_HEIGHT_HELP_STRING = (
    'Boolean flag.  If 1 (0), will find height-averaged (single-height) '
    'extremes.'
)
CONFIDENCE_LEVEL_HELP_STRING = (
    'Confidence level (range 0...1) used to compute total uncertainty.'
)
NUM_EXAMPLES_HELP_STRING = (
    'Number of examples (profiles) in each set of extremes.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Each set of extremes will be written here by '
    '`prediction_io.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FOR_SHORTWAVE_ARG_NAME, type=int, required=True,
    help=FOR_SHORTWAVE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + AVERAGE_OVER_HEIGHT_ARG_NAME, type=int, required=True,
    help=AVERAGE_OVER_HEIGHT_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=False, default=0.95,
    help=CONFIDENCE_LEVEL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=False, default=100,
    help=NUM_EXAMPLES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_prediction_file_name, for_shortwave, average_over_height,
         confidence_level, num_examples_per_set, output_dir_name):
    """Finds extreme HR uncertainties (largest, smallest, best/worst predicted).

    This is effectively the main method.

    :param input_prediction_file_name: See documentation at top of file.
    :param for_shortwave: Same.
    :param average_over_height: Same.
    :param confidence_level: Same.
    :param num_examples_per_set: Same.
    :param output_dir_name: Same.
    """

    error_checking.assert_is_greater(num_examples_per_set, 0)

    print('Reading data from: "{0:s}"...'.format(input_prediction_file_name))
    prediction_dict = prediction_io.read_file(input_prediction_file_name)

    model_file_name = prediction_dict[prediction_io.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0]
    )

    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    generator_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]

    vector_target_names = (
        generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY]
    )
    hr_index = vector_target_names.index(
        example_utils.SHORTWAVE_HEATING_RATE_NAME if for_shortwave
        else example_utils.LONGWAVE_HEATING_RATE_NAME
    )

    target_matrix_k_day01 = (
        prediction_dict[prediction_io.VECTOR_TARGETS_KEY][..., hr_index]
    )
    prediction_matrix_k_day01 = (
        prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY][..., hr_index, :]
    )

    num_examples = target_matrix_k_day01.shape[0]
    num_heights = target_matrix_k_day01.shape[1]
    pit_matrix = numpy.full((num_examples, num_heights), numpy.nan)

    for j in range(num_heights):
        print('Computing PIT values for {0:d}th of {1:d} heights...'.format(
            j + 1, num_heights
        ))

        for i in range(num_examples):
            if (
                    target_matrix_k_day01[i, j] >
                    numpy.max(prediction_matrix_k_day01[i, j, :])
            ):
                pit_matrix[i, j] = (
                    1. + target_matrix_k_day01[i, j] -
                    numpy.max(prediction_matrix_k_day01[i, j, :])
                )
                continue

            if (
                    target_matrix_k_day01[i, j] <
                    numpy.min(prediction_matrix_k_day01[i, j, :])
            ):
                pit_matrix[i, j] = (
                    target_matrix_k_day01[i, j] -
                    numpy.min(prediction_matrix_k_day01[i, j, :])
                )
                continue

            pit_matrix[i, j] = 0.01 * percentileofscore(
                a=prediction_matrix_k_day01[i, j, :],
                score=target_matrix_k_day01[i, j], kind='mean'
            )

    min_prediction_matrix_k_day01 = numpy.percentile(
        prediction_matrix_k_day01, 50 * (1. - confidence_level), axis=-1
    )
    max_prediction_matrix_k_day01 = numpy.percentile(
        prediction_matrix_k_day01, 50 * (1. + confidence_level), axis=-1
    )
    total_uncertainty_matrix_k_day01 = (
        max_prediction_matrix_k_day01 - min_prediction_matrix_k_day01
    )

    if average_over_height:
        pit_matrix = numpy.mean(pit_matrix, axis=1, keepdims=True)
        total_uncertainty_matrix_k_day01 = numpy.mean(
            total_uncertainty_matrix_k_day01, axis=1, keepdims=True
        )

    print(SEPARATOR_STRING)
    high_bias_indices, low_bias_indices, low_abs_error_indices = (
        misc_utils.find_best_and_worst_predictions(
            bias_matrix=0.5 - pit_matrix,
            absolute_error_matrix=numpy.absolute(pit_matrix - 0.5),
            num_examples_per_set=num_examples_per_set
        )
    )
    print(SEPARATOR_STRING)

    high_bias_prediction_dict = prediction_io.subset_by_index(
        prediction_dict=copy.deepcopy(prediction_dict),
        desired_indices=high_bias_indices
    )
    high_bias_file_name = (
        '{0:s}/predictions_high-bias.nc'.format(output_dir_name)
    )

    print('Writing examples with greatest positive bias to: "{0:s}"...'.format(
        high_bias_file_name
    ))
    prediction_io.write_file(
        netcdf_file_name=high_bias_file_name,
        scalar_target_matrix=
        high_bias_prediction_dict[prediction_io.SCALAR_TARGETS_KEY],
        vector_target_matrix=
        high_bias_prediction_dict[prediction_io.VECTOR_TARGETS_KEY],
        scalar_prediction_matrix=
        high_bias_prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY],
        vector_prediction_matrix=
        high_bias_prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY],
        heights_m_agl=high_bias_prediction_dict[prediction_io.HEIGHTS_KEY],
        example_id_strings=
        high_bias_prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
        model_file_name=high_bias_prediction_dict[prediction_io.MODEL_FILE_KEY],
        isotonic_model_file_name=
        high_bias_prediction_dict[prediction_io.ISOTONIC_MODEL_FILE_KEY],
        uncertainty_calib_model_file_name=high_bias_prediction_dict[
            prediction_io.UNCERTAINTY_CALIB_MODEL_FILE_KEY
        ],
        normalization_file_name=
        high_bias_prediction_dict[prediction_io.NORMALIZATION_FILE_KEY]
    )

    low_bias_prediction_dict = prediction_io.subset_by_index(
        prediction_dict=copy.deepcopy(prediction_dict),
        desired_indices=low_bias_indices
    )
    low_bias_file_name = (
        '{0:s}/predictions_low-bias.nc'.format(output_dir_name)
    )

    print('Writing examples with greatest negative bias to: "{0:s}"...'.format(
        low_bias_file_name
    ))
    prediction_io.write_file(
        netcdf_file_name=low_bias_file_name,
        scalar_target_matrix=
        low_bias_prediction_dict[prediction_io.SCALAR_TARGETS_KEY],
        vector_target_matrix=
        low_bias_prediction_dict[prediction_io.VECTOR_TARGETS_KEY],
        scalar_prediction_matrix=
        low_bias_prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY],
        vector_prediction_matrix=
        low_bias_prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY],
        heights_m_agl=low_bias_prediction_dict[prediction_io.HEIGHTS_KEY],
        example_id_strings=
        low_bias_prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
        model_file_name=low_bias_prediction_dict[prediction_io.MODEL_FILE_KEY],
        isotonic_model_file_name=
        low_bias_prediction_dict[prediction_io.ISOTONIC_MODEL_FILE_KEY],
        uncertainty_calib_model_file_name=low_bias_prediction_dict[
            prediction_io.UNCERTAINTY_CALIB_MODEL_FILE_KEY
        ],
        normalization_file_name=
        low_bias_prediction_dict[prediction_io.NORMALIZATION_FILE_KEY]
    )

    low_abs_error_prediction_dict = prediction_io.subset_by_index(
        prediction_dict=copy.deepcopy(prediction_dict),
        desired_indices=low_abs_error_indices
    )
    low_abs_error_file_name = (
        '{0:s}/predictions_low-absolute-error.nc'.format(output_dir_name)
    )

    print('Writing examples with smallest absolute error to: "{0:s}"...'.format(
        low_abs_error_file_name
    ))
    prediction_io.write_file(
        netcdf_file_name=low_abs_error_file_name,
        scalar_target_matrix=
        low_abs_error_prediction_dict[prediction_io.SCALAR_TARGETS_KEY],
        vector_target_matrix=
        low_abs_error_prediction_dict[prediction_io.VECTOR_TARGETS_KEY],
        scalar_prediction_matrix=
        low_abs_error_prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY],
        vector_prediction_matrix=
        low_abs_error_prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY],
        heights_m_agl=low_abs_error_prediction_dict[prediction_io.HEIGHTS_KEY],
        example_id_strings=
        low_abs_error_prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
        model_file_name=
        low_abs_error_prediction_dict[prediction_io.MODEL_FILE_KEY],
        isotonic_model_file_name=
        low_abs_error_prediction_dict[prediction_io.ISOTONIC_MODEL_FILE_KEY],
        uncertainty_calib_model_file_name=low_abs_error_prediction_dict[
            prediction_io.UNCERTAINTY_CALIB_MODEL_FILE_KEY
        ],
        normalization_file_name=
        low_abs_error_prediction_dict[prediction_io.NORMALIZATION_FILE_KEY]
    )

    max_uncertainty_by_example_k_day01 = numpy.max(
        total_uncertainty_matrix_k_day01, axis=1
    )
    high_uncertainty_indices = numpy.argsort(
        -1 * max_uncertainty_by_example_k_day01
    )[:num_examples_per_set]
    low_uncertainty_indices = numpy.argsort(
        max_uncertainty_by_example_k_day01
    )[:num_examples_per_set]

    print(SEPARATOR_STRING)

    for i in range(num_examples_per_set):
        print('{0:d}th-highest uncertainty = {1:f}'.format(
            i + 1,
            max_uncertainty_by_example_k_day01[high_uncertainty_indices[i]]
        ))

    print(SEPARATOR_STRING)

    for i in range(num_examples_per_set):
        print('{0:d}th-lowest uncertainty = {1:f}'.format(
            i + 1,
            max_uncertainty_by_example_k_day01[low_uncertainty_indices[i]]
        ))

    print(SEPARATOR_STRING)

    low_uncertainty_prediction_dict = prediction_io.subset_by_index(
        prediction_dict=copy.deepcopy(prediction_dict),
        desired_indices=low_uncertainty_indices
    )
    low_uncertainty_file_name = (
        '{0:s}/predictions_low-uncertainty.nc'.format(output_dir_name)
    )

    print('Writing examples with lowest uncertainty to: "{0:s}"...'.format(
        low_uncertainty_file_name
    ))
    prediction_io.write_file(
        netcdf_file_name=low_uncertainty_file_name,
        scalar_target_matrix=
        low_uncertainty_prediction_dict[prediction_io.SCALAR_TARGETS_KEY],
        vector_target_matrix=
        low_uncertainty_prediction_dict[prediction_io.VECTOR_TARGETS_KEY],
        scalar_prediction_matrix=
        low_uncertainty_prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY],
        vector_prediction_matrix=
        low_uncertainty_prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY],
        heights_m_agl=
        low_uncertainty_prediction_dict[prediction_io.HEIGHTS_KEY],
        example_id_strings=
        low_uncertainty_prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
        model_file_name=
        low_uncertainty_prediction_dict[prediction_io.MODEL_FILE_KEY],
        isotonic_model_file_name=
        low_uncertainty_prediction_dict[prediction_io.ISOTONIC_MODEL_FILE_KEY],
        uncertainty_calib_model_file_name=low_uncertainty_prediction_dict[
            prediction_io.UNCERTAINTY_CALIB_MODEL_FILE_KEY
        ],
        normalization_file_name=
        low_uncertainty_prediction_dict[prediction_io.NORMALIZATION_FILE_KEY]
    )

    high_uncertainty_prediction_dict = prediction_io.subset_by_index(
        prediction_dict=copy.deepcopy(prediction_dict),
        desired_indices=high_uncertainty_indices
    )
    high_uncertainty_file_name = (
        '{0:s}/predictions_high-uncertainty.nc'.format(output_dir_name)
    )

    print('Writing examples with highest uncertainty to: "{0:s}"...'.format(
        high_uncertainty_file_name
    ))
    prediction_io.write_file(
        netcdf_file_name=high_uncertainty_file_name,
        scalar_target_matrix=
        high_uncertainty_prediction_dict[prediction_io.SCALAR_TARGETS_KEY],
        vector_target_matrix=
        high_uncertainty_prediction_dict[prediction_io.VECTOR_TARGETS_KEY],
        scalar_prediction_matrix=
        high_uncertainty_prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY],
        vector_prediction_matrix=
        high_uncertainty_prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY],
        heights_m_agl=
        high_uncertainty_prediction_dict[prediction_io.HEIGHTS_KEY],
        example_id_strings=
        high_uncertainty_prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
        model_file_name=
        high_uncertainty_prediction_dict[prediction_io.MODEL_FILE_KEY],
        isotonic_model_file_name=
        high_uncertainty_prediction_dict[prediction_io.ISOTONIC_MODEL_FILE_KEY],
        uncertainty_calib_model_file_name=high_uncertainty_prediction_dict[
            prediction_io.UNCERTAINTY_CALIB_MODEL_FILE_KEY
        ],
        normalization_file_name=
        high_uncertainty_prediction_dict[prediction_io.NORMALIZATION_FILE_KEY]
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_prediction_file_name=getattr(
            INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME
        ),
        for_shortwave=bool(getattr(INPUT_ARG_OBJECT, FOR_SHORTWAVE_ARG_NAME)),
        average_over_height=bool(
            getattr(INPUT_ARG_OBJECT, AVERAGE_OVER_HEIGHT_ARG_NAME)
        ),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        num_examples_per_set=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME),
    )
