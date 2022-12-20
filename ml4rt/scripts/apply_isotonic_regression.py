"""Applies one set of isotonic-regression models to data."""

import os
import argparse

import numpy

from ml4rt.io import prediction_io
from ml4rt.utils import example_utils
from ml4rt.machine_learning import isotonic_regression
from ml4rt.machine_learning import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_PREDICTION_FILE_ARG_NAME = 'input_prediction_file_name'
MODEL_FILE_ARG_NAME = 'input_model_file_name'
OUTPUT_PREDICTION_FILE_ARG_NAME = 'output_prediction_file_name'

INPUT_PREDICTION_FILE_HELP_STRING = (
    'Path to file containing model predictions before isotonic regression.  '
    'Will be read by `prediction_io.read_file`.'
)
MODEL_FILE_HELP_STRING = (
    'Path to file with set of trained isotonic-regression models.  Will be read'
    ' by `isotonic_regression.read_file`.'
)
OUTPUT_PREDICTION_FILE_HELP_STRING = (
    'Path to output file, containing model predictions after isotonic '
    'regression.  Will be written by `prediction_io.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_PREDICTION_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_PREDICTION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_PREDICTION_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_PREDICTION_FILE_HELP_STRING
)


def _run(input_prediction_file_name, model_file_name,
         output_prediction_file_name):
    """Applies one set of isotonic-regression models to data.

    This is effectively the main method.

    :param input_prediction_file_name: See documentation at top of file.
    :param model_file_name: Same.
    :param output_prediction_file_name: Same.
    :raises: ValueError: if predictions in `input_prediction_file_name` were
        made with isotonic regression.
    """

    print('Reading original predictions from: "{0:s}"...'.format(
        input_prediction_file_name
    ))
    prediction_dict = prediction_io.read_file(input_prediction_file_name)
    prediction_dict = prediction_io.get_ensemble_mean(prediction_dict)

    if prediction_dict[prediction_io.ISOTONIC_MODEL_FILE_KEY] is not None:
        raise ValueError(
            'Input predictions must be made with base model only (i.e., must '
            'not already include isotonic regression).'
        )

    orig_vector_prediction_matrix = (
        None if prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY].size == 0
        else prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY]
    )
    orig_scalar_prediction_matrix = (
        None if prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY].size == 0
        else prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY]
    )

    print('Reading isotonic-regression models from: "{0:s}"...'.format(
        model_file_name
    ))
    scalar_model_objects, vector_model_object_matrix = (
        isotonic_regression.read_file(model_file_name)
    )

    print(SEPARATOR_STRING)
    new_vector_prediction_matrix, new_scalar_prediction_matrix = (
        isotonic_regression.apply_models(
            orig_vector_prediction_matrix=orig_vector_prediction_matrix,
            orig_scalar_prediction_matrix=orig_scalar_prediction_matrix,
            scalar_model_objects=scalar_model_objects,
            vector_model_object_matrix=vector_model_object_matrix
        )
    )
    print(SEPARATOR_STRING)

    neural_net_file_name = prediction_dict[prediction_io.MODEL_FILE_KEY]
    metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(neural_net_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(metafile_name))
    metadata_dict = neural_net.read_metafile(metafile_name)
    vector_target_names = metadata_dict[neural_net.TRAINING_OPTIONS_KEY][
        neural_net.VECTOR_TARGET_NAMES_KEY
    ]

    for this_target_name in [
            example_utils.SHORTWAVE_HEATING_RATE_NAME,
            example_utils.LONGWAVE_HEATING_RATE_NAME
    ]:
        try:
            k = vector_target_names.index(this_target_name)
        except ValueError:
            continue

        new_vector_prediction_matrix[:, -1, k] = 0.
        prediction_dict[prediction_io.VECTOR_TARGETS_KEY][:, -1, k] = 0.

    print('Writing new predictions to: "{0:s}"...'.format(
        output_prediction_file_name
    ))
    prediction_io.write_file(
        netcdf_file_name=output_prediction_file_name,
        scalar_target_matrix=prediction_dict[prediction_io.SCALAR_TARGETS_KEY],
        vector_target_matrix=prediction_dict[prediction_io.VECTOR_TARGETS_KEY],
        scalar_prediction_matrix=
        numpy.expand_dims(new_scalar_prediction_matrix, axis=-1),
        vector_prediction_matrix=
        numpy.expand_dims(new_vector_prediction_matrix, axis=-1),
        heights_m_agl=prediction_dict[prediction_io.HEIGHTS_KEY],
        example_id_strings=prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
        model_file_name=prediction_dict[prediction_io.MODEL_FILE_KEY],
        isotonic_model_file_name=model_file_name,
        normalization_file_name=
        prediction_dict[prediction_io.NORMALIZATION_FILE_KEY]
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_prediction_file_name=getattr(
            INPUT_ARG_OBJECT, INPUT_PREDICTION_FILE_ARG_NAME
        ),
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        output_prediction_file_name=getattr(
            INPUT_ARG_OBJECT, OUTPUT_PREDICTION_FILE_ARG_NAME
        )
    )
