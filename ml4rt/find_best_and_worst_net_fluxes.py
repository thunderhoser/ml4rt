"""Finds best and worst net-flux predictions."""

import os
import sys
import copy
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import prediction_io
import misc as misc_utils
import example_utils
import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_FILE_ARG_NAME = 'input_prediction_file_name'
NUM_EXAMPLES_ARG_NAME = 'num_examples_per_set'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing actual and predicted values.  Will be read '
    'by `prediction_io.write_file`.'
)
NUM_EXAMPLES_HELP_STRING = (
    'Number of examples in each set (of either best or worst predictions).'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Each set (of either best or worst predictions) '
    'will be written here by `prediction_io.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=False, default=100,
    help=NUM_EXAMPLES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_prediction_file_name, num_examples_per_set, output_dir_name):
    """Finds best and worst heating-rate predictions.

    This is effectively the main method.

    :param input_prediction_file_name: See documentation at top of file.
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
        example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME
    )
    up_index = scalar_target_names.index(
        example_utils.SHORTWAVE_TOA_UP_FLUX_NAME
    )

    targets_w_m02 = (
        prediction_dict[prediction_io.SCALAR_TARGETS_KEY][..., down_index] -
        prediction_dict[prediction_io.SCALAR_TARGETS_KEY][..., up_index]
    )
    predictions_w_m02 = (
        prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY][..., down_index] -
        prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY][..., up_index]
    )
    biases_w_m02 = predictions_w_m02 - targets_w_m02

    print(SEPARATOR_STRING)
    high_bias_indices, low_bias_indices, low_abs_error_indices = (
        misc_utils.find_best_and_worst_predictions(
            bias_matrix=numpy.expand_dims(biases_w_m02, axis=1),
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
        model_file_name=high_bias_prediction_dict[prediction_io.MODEL_FILE_KEY]
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
        model_file_name=low_bias_prediction_dict[prediction_io.MODEL_FILE_KEY]
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
        low_abs_error_prediction_dict[prediction_io.MODEL_FILE_KEY]
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_prediction_file_name=getattr(
            INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME
        ),
        num_examples_per_set=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME),
    )
