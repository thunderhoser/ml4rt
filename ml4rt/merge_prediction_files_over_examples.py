"""Merges prediction files for different example sets."""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import prediction_io

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_FILES_ARG_NAME = 'input_file_names'
TAKE_ENSEMBLE_MEAN_ARG_NAME = 'take_ensemble_mean'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILES_HELP_STRING = (
    'List of paths to input files.  Each will be read by '
    '`prediction_io.read_file`.'
)
TAKE_ENSEMBLE_MEAN_HELP_STRING = (
    'Boolean flag.  If 1, will take ensemble mean.  If 0, will keep all '
    'ensemble members.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output (merged) file.  Will be written here by '
    '`prediction_io.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=INPUT_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TAKE_ENSEMBLE_MEAN_ARG_NAME, type=int, required=True,
    help=TAKE_ENSEMBLE_MEAN_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(input_file_names, take_ensemble_mean, output_file_name):
    """Merges prediction files for different example sets.

    This is effectively the main method.

    :param input_file_names: See documentation at top of file.
    :param take_ensemble_mean: Same.
    :param output_file_name: Same.
    """

    num_input_files = len(input_file_names)
    prediction_dicts = [dict()] * num_input_files

    for i in range(num_input_files):
        print('Reading data from: "{0:s}"...'.format(input_file_names[i]))
        prediction_dicts[i] = prediction_io.read_file(input_file_names[i])

        if take_ensemble_mean:
            prediction_dicts[i] = prediction_io.get_ensemble_mean(
                prediction_dicts[i]
            )
            prediction_dicts[i][prediction_io.SCALAR_PREDICTIONS_KEY] = (
                numpy.expand_dims(
                    prediction_dicts[i][prediction_io.SCALAR_PREDICTIONS_KEY],
                    axis=-1
                )
            )
            prediction_dicts[i][prediction_io.VECTOR_PREDICTIONS_KEY] = (
                numpy.expand_dims(
                    prediction_dicts[i][prediction_io.VECTOR_PREDICTIONS_KEY],
                    axis=-1
                )
            )

    prediction_dict = prediction_io.concat_predictions(prediction_dicts)
    del prediction_dicts

    print('Writing data to: "{0:s}"...'.format(output_file_name))
    prediction_io.write_file(
        netcdf_file_name=output_file_name,
        scalar_target_matrix=prediction_dict[prediction_io.SCALAR_TARGETS_KEY],
        vector_target_matrix=prediction_dict[prediction_io.VECTOR_TARGETS_KEY],
        scalar_prediction_matrix=
        prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY],
        vector_prediction_matrix=
        prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY],
        heights_m_agl=prediction_dict[prediction_io.HEIGHTS_KEY],
        example_id_strings=prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
        model_file_name=prediction_dict[prediction_io.MODEL_FILE_KEY],
        isotonic_model_file_name=
        prediction_dict[prediction_io.ISOTONIC_MODEL_FILE_KEY],
        normalization_file_name=
        prediction_dict[prediction_io.NORMALIZATION_FILE_KEY]
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_names=getattr(INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME),
        take_ensemble_mean=bool(
            getattr(INPUT_ARG_OBJECT, TAKE_ENSEMBLE_MEAN_ARG_NAME)
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
