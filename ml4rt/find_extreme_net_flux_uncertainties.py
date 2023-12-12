"""Finds extreme NF uncertainties (largest, smallest, best/worst predicted).

NF = net flux
"""

import os
import sys
import copy
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import percentileofscore
import error_checking
import prediction_io
import misc as misc_utils
import example_utils
import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_FILE_ARG_NAME = 'input_prediction_file_name'
FOR_SHORTWAVE_ARG_NAME = 'for_shortwave'
AVERAGE_OVER_WAVELENGTH_ARG_NAME = 'average_over_wavelength'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
NUM_EXAMPLES_ARG_NAME = 'num_examples_per_set'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing actual and predicted values.  Will be read '
    'by `prediction_io.write_file`.'
)
FOR_SHORTWAVE_HELP_STRING = (
    'Boolean flag.  If 1 (0), will find extreme uncertainties for shortwave '
    '(longwave) net flux.'
)
AVERAGE_OVER_WAVELENGTH_HELP_STRING = (
    'Boolean flag.  If 1, will average PIT errors over wavelength for each '
    'example.  If 0, will look for wavelength with worst PIT error in each '
    'example.'
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
    '--' + AVERAGE_OVER_WAVELENGTH_ARG_NAME, type=int, required=True,
    help=AVERAGE_OVER_WAVELENGTH_HELP_STRING
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


def _run(input_prediction_file_name, for_shortwave, average_over_wavelength,
         confidence_level, num_examples_per_set, output_dir_name):
    """Finds extreme NF uncertainties (largest, smallest, best/worst predicted).

    This is effectively the main method.

    :param input_prediction_file_name: See documentation at top of file.
    :param for_shortwave: Same.
    :param average_over_wavelength: Same.
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

    scalar_target_names = (
        generator_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY]
    )
    down_index = scalar_target_names.index(
        example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME if for_shortwave
        else example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME
    )
    up_index = scalar_target_names.index(
        example_utils.SHORTWAVE_TOA_UP_FLUX_NAME if for_shortwave
        else example_utils.LONGWAVE_TOA_UP_FLUX_NAME
    )

    target_matrix_w_m02 = (
        prediction_dict[prediction_io.SCALAR_TARGETS_KEY][..., down_index]
        - prediction_dict[prediction_io.SCALAR_TARGETS_KEY][..., up_index]
    )
    prediction_matrix_w_m02 = (
        prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY][..., down_index, :]
        - prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY][..., up_index, :]
    )

    num_examples = target_matrix_w_m02.shape[0]
    num_wavelengths = target_matrix_w_m02.shape[1]
    pit_matrix = numpy.full((num_examples, num_wavelengths), numpy.nan)

    for i in range(num_examples):
        for w in range(num_wavelengths):
            if (
                    target_matrix_w_m02[i, w] >
                    numpy.max(prediction_matrix_w_m02[i, w, :])
            ):
                pit_matrix[i, w] = (
                    1. + target_matrix_w_m02[i, w] -
                    numpy.max(prediction_matrix_w_m02[i, w, :])
                )
                continue

            if (
                    target_matrix_w_m02[i, w] <
                    numpy.min(prediction_matrix_w_m02[i, w, :])
            ):
                pit_matrix[i, w] = (
                    target_matrix_w_m02[i, w] -
                    numpy.min(prediction_matrix_w_m02[i, w, :])
                )
                continue

            pit_matrix[i, w] = 0.01 * percentileofscore(
                a=prediction_matrix_w_m02[i, w, :],
                score=target_matrix_w_m02[i, w], kind='mean'
            )

    min_prediction_matrix_w_m02 = numpy.percentile(
        prediction_matrix_w_m02, 50 * (1. - confidence_level), axis=-1
    )
    max_prediction_matrix_w_m02 = numpy.percentile(
        prediction_matrix_w_m02, 50 * (1. + confidence_level), axis=-1
    )
    total_uncertainty_matrix_w_m02 = (
        max_prediction_matrix_w_m02 - min_prediction_matrix_w_m02
    )

    if average_over_wavelength:
        pit_matrix = numpy.mean(pit_matrix, axis=1, keepdims=True)
        total_uncertainty_matrix_w_m02 = numpy.mean(
            total_uncertainty_matrix_w_m02, axis=1, keepdims=True
        )

    pit_matrix = numpy.expand_dims(pit_matrix, axis=-1)
    total_uncertainty_matrix_w_m02 = numpy.expand_dims(
        total_uncertainty_matrix_w_m02, axis=-1
    )

    print(SEPARATOR_STRING)
    high_bias_indices, low_bias_indices, low_abs_error_indices = (
        misc_utils.find_best_and_worst_predictions(
            bias_matrix_3d=0.5 - pit_matrix,
            absolute_error_matrix_3d=numpy.absolute(pit_matrix - 0.5),
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
        target_wavelengths_metres=
        high_bias_prediction_dict[prediction_io.TARGET_WAVELENGTHS_KEY],
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
        target_wavelengths_metres=
        low_bias_prediction_dict[prediction_io.TARGET_WAVELENGTHS_KEY],
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
        target_wavelengths_metres=
        low_abs_error_prediction_dict[prediction_io.TARGET_WAVELENGTHS_KEY],
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

    max_uncertainty_by_example_w_m02 = numpy.max(
        total_uncertainty_matrix_w_m02, axis=1
    )
    high_uncertainty_indices = numpy.argsort(
        -1 * max_uncertainty_by_example_w_m02
    )[:num_examples_per_set]
    low_uncertainty_indices = numpy.argsort(
        max_uncertainty_by_example_w_m02
    )[:num_examples_per_set]

    print(SEPARATOR_STRING)

    for i in range(num_examples_per_set):
        print('{0:d}th-highest uncertainty = {1:f}'.format(
            i + 1,
            total_uncertainty_matrix_w_m02[high_uncertainty_indices[i]]
        ))

    print(SEPARATOR_STRING)

    for i in range(num_examples_per_set):
        print('{0:d}th-lowest uncertainty = {1:f}'.format(
            i + 1,
            total_uncertainty_matrix_w_m02[low_uncertainty_indices[i]]
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
        target_wavelengths_metres=
        low_uncertainty_prediction_dict[prediction_io.TARGET_WAVELENGTHS_KEY],
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
        target_wavelengths_metres=
        high_uncertainty_prediction_dict[prediction_io.TARGET_WAVELENGTHS_KEY],
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
        average_over_wavelength=bool(
            getattr(INPUT_ARG_OBJECT, AVERAGE_OVER_WAVELENGTH_ARG_NAME)
        ),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        num_examples_per_set=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME),
    )
