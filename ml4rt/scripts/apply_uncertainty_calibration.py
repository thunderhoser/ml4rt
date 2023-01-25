"""Applies uncertainty-calibration model to neural-net predictions."""

import argparse
from ml4rt.io import prediction_io
from ml4rt.machine_learning import uncertainty_calibration as uncertainty_calib

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_PREDICTION_FILE_ARG_NAME = 'input_prediction_file_name'
MODEL_FILE_ARG_NAME = 'input_model_file_name'
OUTPUT_PREDICTION_FILE_ARG_NAME = 'output_prediction_file_name'

INPUT_PREDICTION_FILE_HELP_STRING = (
    'Path to file containing neural-net predictions with no uncertainty '
    'calibration.  Will be read by `prediction_io.read_file`.'
)
MODEL_FILE_HELP_STRING = (
    'Path to file with trained uncertainty-calibration model.  Will be read'
    ' by `uncertainty_calibration.read_file`.'
)
OUTPUT_PREDICTION_FILE_HELP_STRING = (
    'Path to output file, containing model predictions after uncertainty '
    'calibration.  Will be written by `prediction_io.write_file`.'
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
    """Applies uncertainty-calibration model to neural-net predictions.

    This is effectively the main method.

    :param input_prediction_file_name: See documentation at top of file.
    :param model_file_name: Same.
    :param output_prediction_file_name: Same.
    :raises: ValueError: if predictions in `input_prediction_file_name` already
        have calibrated uncertainty.
    """

    print('Reading original predictions from: "{0:s}"...'.format(
        input_prediction_file_name
    ))
    prediction_dict = prediction_io.read_file(input_prediction_file_name)

    if (
            prediction_dict[prediction_io.UNCERTAINTY_CALIB_MODEL_FILE_KEY]
            is not None
    ):
        raise ValueError(
            'Input predictions already have calibrated uncertainty.'
        )

    del prediction_dict

    print('Reading uncertainty-calibration model from: "{0:s}"...'.format(
        model_file_name
    ))
    uncertainty_calib_table_xarray = uncertainty_calib.read_results(
        model_file_name
    )
    prediction_dict = uncertainty_calib.apply_models_all_vars(
        prediction_file_name=input_prediction_file_name,
        uncertainty_calib_table_xarray=uncertainty_calib_table_xarray
    )

    print((
        'Writing predictions with calibrated uncertainty to: "{0:s}"...'
    ).format(
        output_prediction_file_name
    ))

    d = prediction_dict

    prediction_io.write_file(
        netcdf_file_name=output_prediction_file_name,
        scalar_target_matrix=d[prediction_io.SCALAR_TARGETS_KEY],
        vector_target_matrix=d[prediction_io.VECTOR_TARGETS_KEY],
        scalar_prediction_matrix=d[prediction_io.SCALAR_PREDICTIONS_KEY],
        vector_prediction_matrix=d[prediction_io.VECTOR_PREDICTIONS_KEY],
        heights_m_agl=d[prediction_io.HEIGHTS_KEY],
        example_id_strings=d[prediction_io.EXAMPLE_IDS_KEY],
        model_file_name=d[prediction_io.MODEL_FILE_KEY],
        isotonic_model_file_name=d[prediction_io.ISOTONIC_MODEL_FILE_KEY],
        uncertainty_calib_model_file_name=model_file_name,
        normalization_file_name=d[prediction_io.NORMALIZATION_FILE_KEY]
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
