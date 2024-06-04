"""Trains isotonic-regression models to bias-correct one base ML model."""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import prediction_io
import isotonic_regression

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_FILES_ARG_NAME = 'input_prediction_file_names'
OUTPUT_DIR_ARG_NAME = 'output_model_dir_name'

INPUT_FILES_HELP_STRING = (
    'Paths to input files, each containing model predictions without isotonic '
    'regression.  Will be read by `prediction_io.read_file`.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Isotonic-regression models will be written here'
    ' by `isotonic_regression.write_file`, to a file name determined by '
    '`isotonic_regression.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=INPUT_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(prediction_file_names, output_dir_name):
    """Trains isotonic-regression model.

    This is effectively the main method.

    :param prediction_file_names: See documentation at top of file.
    :param output_dir_name: Same.
    :raises: ValueError: if predictions in `prediction_file_names` were made
        with isotonic regression.
    """

    num_files = len(prediction_file_names)
    prediction_dicts = [dict()] * num_files

    for i in range(num_files):
        print('Reading original predictions from: "{0:s}"...'.format(
            prediction_file_names[i]
        ))
        prediction_dicts[i] = prediction_io.read_file(prediction_file_names[i])

    prediction_dict = prediction_io.concat_predictions(prediction_dicts)

    if prediction_dict[prediction_io.ISOTONIC_MODEL_FILE_KEY] is not None:
        raise ValueError(
            'Predictions used for training isotonic regression must be made'
            ' with base model only (i.e., must not already include isotonic'
            ' regression).'
        )

    orig_vector_prediction_matrix = (
        None if prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY].size == 0
        else prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY]
    )
    vector_target_matrix = (
        None if prediction_dict[prediction_io.VECTOR_TARGETS_KEY].size == 0
        else prediction_dict[prediction_io.VECTOR_TARGETS_KEY]
    )
    orig_scalar_prediction_matrix = (
        None if prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY].size == 0
        else prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY]
    )
    scalar_target_matrix = (
        None if prediction_dict[prediction_io.SCALAR_TARGETS_KEY].size == 0
        else prediction_dict[prediction_io.SCALAR_TARGETS_KEY]
    )

    print(SEPARATOR_STRING)

    # Training for one height/wavelength per method call, to avoid out-of-memory
    # errors.
    num_heights = vector_target_matrix.shape[-3]
    num_wavelengths = vector_target_matrix.shape[-2]
    num_vector_targets = vector_target_matrix.shape[-1]
    vector_model_object_matrix = numpy.full(
        (num_heights, num_wavelengths, num_vector_targets), '', dtype=object
    )

    for h in range(num_heights):
        for w in range(num_wavelengths):
            if h == w == 0:
                scalar_model_object_matrix, this_matrix = (
                    isotonic_regression.train_models(
                        orig_vector_prediction_matrix=
                        orig_vector_prediction_matrix[:, [h], ...][:, :, [w], ...],
                        orig_scalar_prediction_matrix=
                        orig_scalar_prediction_matrix[:, [h], ...],
                        vector_target_matrix=
                        vector_target_matrix[:, [h], ...][:, :, [w], ...],
                        scalar_target_matrix=scalar_target_matrix[:, [h], ...]
                    )
                )

            _, this_matrix = isotonic_regression.train_models(
                orig_vector_prediction_matrix=
                orig_vector_prediction_matrix[:, [h], ...][:, :, [w], ...],
                orig_scalar_prediction_matrix=None,
                vector_target_matrix=
                vector_target_matrix[:, [h], ...][:, :, [w], ...],
                scalar_target_matrix=None
            )
            vector_model_object_matrix[h, w, :] = this_matrix[0, 0, :]

    print(SEPARATOR_STRING)

    output_file_name = isotonic_regression.find_file(
        model_dir_name=output_dir_name, raise_error_if_missing=False
    )

    print('Writing isotonic-regression models to: "{0:s}"...'.format(
        output_file_name
    ))

    isotonic_regression.write_file(
        dill_file_name=output_file_name,
        scalar_model_object_matrix=scalar_model_object_matrix,
        vector_model_object_matrix=vector_model_object_matrix
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_names=getattr(INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
