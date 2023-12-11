"""Creates multi-model ensemble."""

import copy
import argparse
import numpy
from gewittergefahr.gg_utils import error_checking
from ml4rt.io import prediction_io

TOLERANCE = 1e-6

INPUT_FILES_ARG_NAME = 'input_prediction_file_names'
MAX_ENSEMBLE_SIZE_ARG_NAME = 'max_total_ensemble_size'
OUTPUT_FILE_ARG_NAME = 'output_prediction_file_name'

INPUT_FILES_HELP_STRING = (
    'List of paths to input files.  Each file will be read by '
    '`prediction_io.read_file`, and predictions from all these files will be '
    'concatenated along the final (ensemble-member) axis.'
)
MAX_ENSEMBLE_SIZE_HELP_STRING = (
    'Maximum size of total ensemble, after concatenating predictions from all '
    'input files along the final axis.  In other words, the size of the final '
    'axis may not exceed {0:s}.  If it does, {0:s} predictions will be '
    'randomly selected.'
).format(MAX_ENSEMBLE_SIZE_ARG_NAME)

OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  All predictions will be written here by '
    '`prediction_io.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=INPUT_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_ENSEMBLE_SIZE_ARG_NAME, type=int, required=True,
    help=MAX_ENSEMBLE_SIZE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(input_file_names, max_ensemble_size, output_file_name):
    """Creates multi-model ensemble.

    This is effectively the main method.

    :param input_file_names: See documentation at top of file.
    :param max_ensemble_size: Same.
    :param output_file_name: Same.
    """

    error_checking.assert_is_geq(max_ensemble_size, 2)

    num_models = len(input_file_names)
    max_ensemble_size_per_model = int(numpy.ceil(
        float(max_ensemble_size) / num_models
    ))

    ensemble_size_per_model = -1
    prediction_dict = dict()
    model_file_names = [''] * num_models

    for i in range(num_models):
        print('Reading data from: "{0:s}"...'.format(input_file_names[i]))
        this_prediction_dict = prediction_io.read_file(input_file_names[i])

        if i == 0:
            prediction_dict = copy.deepcopy(this_prediction_dict)
            ensemble_size_per_model = (
                prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY].shape[-1]
            )

        this_ensemble_size = (
            this_prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY].shape[-1]
        )
        assert this_ensemble_size == ensemble_size_per_model

        assert numpy.allclose(
            this_prediction_dict[prediction_io.VECTOR_TARGETS_KEY],
            prediction_dict[prediction_io.VECTOR_TARGETS_KEY],
            atol=TOLERANCE
        )
        assert numpy.allclose(
            this_prediction_dict[prediction_io.SCALAR_TARGETS_KEY],
            prediction_dict[prediction_io.SCALAR_TARGETS_KEY],
            atol=TOLERANCE
        )
        assert numpy.allclose(
            this_prediction_dict[prediction_io.HEIGHTS_KEY],
            prediction_dict[prediction_io.HEIGHTS_KEY],
            atol=TOLERANCE
        )
        assert numpy.allclose(
            this_prediction_dict[prediction_io.TARGET_WAVELENGTHS_KEY],
            prediction_dict[prediction_io.TARGET_WAVELENGTHS_KEY],
            atol=TOLERANCE
        )
        assert (
            this_prediction_dict[prediction_io.EXAMPLE_IDS_KEY] ==
            prediction_dict[prediction_io.EXAMPLE_IDS_KEY]
        )
        assert (
            this_prediction_dict[prediction_io.ISOTONIC_MODEL_FILE_KEY] ==
            prediction_dict[prediction_io.ISOTONIC_MODEL_FILE_KEY]
        )
        assert (
            this_prediction_dict[prediction_io.UNCERTAINTY_CALIB_MODEL_FILE_KEY]
            == prediction_dict[prediction_io.UNCERTAINTY_CALIB_MODEL_FILE_KEY]
        )
        assert (
            this_prediction_dict[prediction_io.NORMALIZATION_FILE_KEY] ==
            prediction_dict[prediction_io.NORMALIZATION_FILE_KEY]
        )

        model_file_names[i] = this_prediction_dict[prediction_io.MODEL_FILE_KEY]

        if ensemble_size_per_model > max_ensemble_size_per_model:
            selected_indices = numpy.linspace(
                0, ensemble_size_per_model - 1, num=ensemble_size_per_model,
                dtype=int
            )
            selected_indices = numpy.random.choice(
                selected_indices, size=max_ensemble_size_per_model,
                replace=False
            )

            this_prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY] = (
                this_prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY][
                    ..., selected_indices
                ]
            )
            this_prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY] = (
                this_prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY][
                    ..., selected_indices
                ]
            )

        prediction_dict[
            prediction_io.VECTOR_PREDICTIONS_KEY
        ] = numpy.concatenate((
            prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY],
            this_prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY]
        ), axis=-1)

        prediction_dict[
            prediction_io.SCALAR_PREDICTIONS_KEY
        ] = numpy.concatenate((
            prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY],
            this_prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY]
        ), axis=-1)

    assert len(set(model_file_names)) == len(model_file_names)

    ensemble_size = (
        prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY].shape[-1]
    )

    if ensemble_size > max_ensemble_size:
        selected_indices = numpy.linspace(
            0, ensemble_size - 1, num=ensemble_size, dtype=int
        )
        selected_indices = numpy.random.choice(
            selected_indices, size=max_ensemble_size, replace=False
        )

        prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY] = (
            prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY][
                ..., selected_indices
            ]
        )
        prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY] = (
            prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY][
                ..., selected_indices
            ]
        )

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
        target_wavelengths_metres=
        prediction_dict[prediction_io.TARGET_WAVELENGTHS_KEY],
        example_id_strings=prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
        model_file_name=prediction_dict[prediction_io.MODEL_FILE_KEY],
        isotonic_model_file_name=
        prediction_dict[prediction_io.ISOTONIC_MODEL_FILE_KEY],
        uncertainty_calib_model_file_name=
        prediction_dict[prediction_io.UNCERTAINTY_CALIB_MODEL_FILE_KEY],
        normalization_file_name=
        prediction_dict[prediction_io.NORMALIZATION_FILE_KEY]
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_names=getattr(INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME),
        max_ensemble_size=getattr(INPUT_ARG_OBJECT, MAX_ENSEMBLE_SIZE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
