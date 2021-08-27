"""Fixes prediction file by changing path to alleged normalization file."""

import os
import sys
import argparse

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import prediction_io

PREDICTION_FILE_ARG_NAME = 'prediction_file_name'
NEW_GRID_NORM_FILE_ARG_NAME = 'new_grid_norm_file_name'

PREDICTION_FILE_HELP_STRING = (
    'Path to prediction file.  This file will be read by '
    '`prediction_io.read_file`, and the new file, with a different '
    'normalization-file path in the metadata, will be written to the same '
    'place by `prediction_io.write_file`.'
)
NEW_GRID_NORM_FILE_HELP_STRING = (
    'Path to normalization file for new grid.  Will be read by '
    '`example_io.read_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_FILE_ARG_NAME, type=str, required=True,
    help=PREDICTION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NEW_GRID_NORM_FILE_ARG_NAME, type=str, required=True,
    help=NEW_GRID_NORM_FILE_HELP_STRING
)


def _run(prediction_file_name, new_grid_norm_file_name):
    """Fixes prediction file by changing path to alleged normalization file.

    This is effectively the main method.

    :param prediction_file_name: See documentation at top of file.
    :param new_grid_norm_file_name: Same.
    """

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(prediction_file_name)

    print('Writing data to: "{0:s}"...'.format(prediction_file_name))
    prediction_io.write_file(
        netcdf_file_name=prediction_file_name,
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
        normalization_file_name=new_grid_norm_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_FILE_ARG_NAME
        ),
        new_grid_norm_file_name=getattr(
            INPUT_ARG_OBJECT, NEW_GRID_NORM_FILE_ARG_NAME
        )
    )
