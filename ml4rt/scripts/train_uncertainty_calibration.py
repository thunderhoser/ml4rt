"""Trains uncertainty-calibration model."""

import argparse
from ml4rt.io import prediction_io
from ml4rt.machine_learning import uncertainty_calibration as uncertainty_calib

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

PREDICTION_FILE_ARG_NAME = 'input_prediction_file_name'
NUM_BINS_ARG_NAME = 'num_spread_bins'
MIN_PERCENTILE_ARG_NAME = 'min_spread_percentile'
MAX_PERCENTILE_ARG_NAME = 'max_spread_percentile'
MODEL_FILE_ARG_NAME = 'output_model_file_name'

PREDICTION_FILE_HELP_STRING = (
    'Path to input file, containing predictions to be calibrated.  Will be '
    'read by `prediction_io.read_file`.'
)
NUM_BINS_HELP_STRING = 'Number of spread bins (same for each target variable).'
MIN_PERCENTILE_HELP_STRING = (
    'Minimum spread percentile (same for each target variable), used to '
    'establish the lower edge of the lowest spread bin.'
)
MAX_PERCENTILE_HELP_STRING = (
    'Max spread percentile (same for each target variable), used to establish '
    'the upper edge of the highest spread bin.'
)
MODEL_FILE_HELP_STRING = (
    'Path to output file.  Model will be written here by '
    '`uncertainty_calibration.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_FILE_ARG_NAME, type=str, required=True,
    help=PREDICTION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_BINS_ARG_NAME, type=int, required=True, help=NUM_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_PERCENTILE_ARG_NAME, type=float, required=True,
    help=MIN_PERCENTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=True,
    help=MAX_PERCENTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING
)


def _run(prediction_file_name, num_spread_bins, min_spread_percentile,
         max_spread_percentile, output_file_name):
    """Trains uncertainty-calibration model.

    This is effectively the main method.

    :param prediction_file_name: See documentation at top of file.
    :param num_spread_bins: Same.
    :param min_spread_percentile: Same.
    :param max_spread_percentile: Same.
    :param output_file_name: Same.
    :raises: ValueError: if predictions in `prediction_file_name` already
        have calibrated uncertainty.
    """

    print('Reading original predictions from: "{0:s}"...'.format(
        prediction_file_name
    ))
    prediction_dict = prediction_io.read_file(prediction_file_name)

    if (
            prediction_dict[prediction_io.UNCERTAINTY_CALIB_MODEL_FILE_KEY]
            is not None
    ):
        raise ValueError(
            'Input predictions already have calibrated uncertainty.'
        )

    uncertainty_calib_table_xarray = uncertainty_calib.train_models_all_vars(
        prediction_file_name=prediction_file_name,
        num_spread_bins=num_spread_bins,
        min_spread_percentile=min_spread_percentile,
        max_spread_percentile=max_spread_percentile
    )

    print('Writing uncertainty-calibration model to: "{0:s}"...'.format(
        output_file_name
    ))
    uncertainty_calib.write_file(
        result_table_xarray=uncertainty_calib_table_xarray,
        netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_FILE_ARG_NAME
        ),
        num_spread_bins=getattr(INPUT_ARG_OBJECT, NUM_BINS_ARG_NAME),
        min_spread_percentile=getattr(
            INPUT_ARG_OBJECT, MIN_PERCENTILE_ARG_NAME
        ),
        max_spread_percentile=getattr(
            INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME)
    )
