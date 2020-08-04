"""Averages predicted and target values over many examples."""

import argparse
import numpy
from ml4rt.utils import example_utils
from ml4rt.io import prediction_io

TIME_FORMAT = '%Y-%m-%d-%H%M%S'

INPUT_FILE_ARG_NAME = 'input_prediction_file_name'
USE_PMM_ARG_NAME = 'use_pmm'
MAX_PERCENTILE_ARG_NAME = 'max_pmm_percentile_level'
OUTPUT_FILE_ARG_NAME = 'output_prediction_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file (with predicted and target values for each example).  '
    'Will be read by `prediction_io.read_file`.'
)
USE_PMM_HELP_STRING = (
    'Boolean flag.  If 1 (0), will use probability-matched (arithmetic) means '
    'for vertical profiles.'
)
MAX_PERCENTILE_HELP_STRING = (
    '[used only if `{0:s}` = 1] Max percentile level for probability-matched '
    'means.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file (with mean predicted and target values over all '
    'examples).  Will be written by `prediction_io.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + USE_PMM_ARG_NAME, type=int, required=False, default=1,
    help=USE_PMM_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=False, default=99.,
    help=MAX_PERCENTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(input_file_name, use_pmm, max_pmm_percentile_level, output_file_name):
    """Averages predicted and target values over many examples.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param use_pmm: Same.
    :param max_pmm_percentile_level: Same.
    :param output_file_name: Same.
    """

    print((
        'Reading predicted and target values for each example from: "{0:s}"...'
    ).format(
        input_file_name
    ))
    prediction_dict = prediction_io.read_file(input_file_name)

    num_examples = prediction_dict[prediction_io.VECTOR_TARGETS_KEY].shape[0]

    print('Averaging {0:d} examples...'.format(num_examples))
    mean_prediction_dict = prediction_io.average_predictions(
        prediction_dict=prediction_dict, use_pmm=use_pmm,
        max_pmm_percentile_level=max_pmm_percentile_level
    )

    dummy_scalar_predictor_matrix = numpy.array([0, 0, 0], dtype=float)
    dummy_scalar_predictor_matrix = numpy.expand_dims(
        dummy_scalar_predictor_matrix, axis=0
    )

    dummy_vector_predictor_matrix = numpy.array([0.])
    dummy_vector_predictor_matrix = numpy.expand_dims(
        dummy_vector_predictor_matrix, axis=-1
    )
    dummy_vector_predictor_matrix = numpy.expand_dims(
        dummy_vector_predictor_matrix, axis=-1
    )

    dummy_example_dict = {
        example_utils.SCALAR_PREDICTOR_NAMES_KEY: [
            example_utils.LATITUDE_NAME, example_utils.LONGITUDE_NAME,
            example_utils.ZENITH_ANGLE_NAME
        ],
        example_utils.SCALAR_PREDICTOR_VALS_KEY: dummy_scalar_predictor_matrix,
        example_utils.VECTOR_PREDICTOR_NAMES_KEY:
            [example_utils.TEMPERATURE_NAME],
        example_utils.VECTOR_PREDICTOR_VALS_KEY: dummy_vector_predictor_matrix,
        example_utils.HEIGHTS_KEY: numpy.array([10.]),
        example_utils.VALID_TIMES_KEY: numpy.array([0], dtype=int),
        example_utils.STANDARD_ATMO_FLAGS_KEY: numpy.array([0], dtype=int)
    }

    example_id_strings = example_utils.create_example_ids(dummy_example_dict)
    print(example_id_strings)

    print('Writing mean example to: "{0:s}"...'.format(output_file_name))
    prediction_io.write_file(
        netcdf_file_name=output_file_name,
        scalar_target_matrix=
        mean_prediction_dict[prediction_io.SCALAR_TARGETS_KEY],
        vector_target_matrix=
        mean_prediction_dict[prediction_io.VECTOR_TARGETS_KEY],
        scalar_prediction_matrix=
        mean_prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY],
        vector_prediction_matrix=
        mean_prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY],
        heights_m_agl=mean_prediction_dict[prediction_io.HEIGHTS_KEY],
        example_id_strings=example_id_strings,
        model_file_name=mean_prediction_dict[prediction_io.MODEL_FILE_KEY]
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        use_pmm=bool(getattr(INPUT_ARG_OBJECT, USE_PMM_ARG_NAME)),
        max_pmm_percentile_level=getattr(
            INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
