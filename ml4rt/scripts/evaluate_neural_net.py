"""Evaluates trained neural net."""

import os.path
import argparse
import numpy
from ml4rt.io import example_io
from ml4rt.io import prediction_io
from ml4rt.utils import evaluation
from ml4rt.utils import normalization
from ml4rt.machine_learning import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_FILE_ARG_NAME = 'input_prediction_file_name'
OUTPUT_FILE_ARG_NAME = 'output_eval_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing predicted and actual target values.  Will '
    'be read by `prediction_io.read_file`.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file (will be written by `evaluation.write_file`).'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(prediction_file_name, evaluation_file_name):
    """Evaluates trained neural net.

    This is effectively the main method.

    :param prediction_file_name: See documentation at top of file.
    :param evaluation_file_name: Same.
    """

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(prediction_file_name)

    model_file_name = prediction_dict[prediction_io.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metadata(model_metafile_name)
    generator_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]

    example_dict = {
        example_io.SCALAR_TARGET_NAMES_KEY:
            generator_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY],
        example_io.VECTOR_TARGET_NAMES_KEY:
            generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY],
        example_io.SCALAR_PREDICTOR_NAMES_KEY:
            generator_option_dict[neural_net.SCALAR_PREDICTOR_NAMES_KEY],
        example_io.VECTOR_PREDICTOR_NAMES_KEY:
            generator_option_dict[neural_net.VECTOR_PREDICTOR_NAMES_KEY],
        example_io.HEIGHTS_KEY: generator_option_dict[neural_net.HEIGHTS_KEY]
    }

    normalization_file_name = (
        generator_option_dict[neural_net.NORMALIZATION_FILE_KEY]
    )
    print((
        'Reading training examples (for climatology) from: "{0:s}"...'
    ).format(
        normalization_file_name
    ))
    training_example_dict = example_io.read_file(normalization_file_name)

    mean_training_example_dict = normalization.create_mean_example(
        new_example_dict=example_dict,
        training_example_dict=training_example_dict
    )

    evaluation_dict = evaluation.get_scores_all_variables(
        scalar_target_matrix=prediction_dict[prediction_io.SCALAR_TARGETS_KEY],
        scalar_prediction_matrix=
        prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY],
        vector_target_matrix=prediction_dict[prediction_io.VECTOR_TARGETS_KEY],
        vector_prediction_matrix=
        prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY],
        mean_training_example_dict=mean_training_example_dict
    )

    evaluation_dict[evaluation.MODEL_FILE_KEY] = model_file_name
    print(SEPARATOR_STRING)

    scalar_target_names = (
        generator_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY]
    )

    for k in range(len(scalar_target_names)):
        print((
            'Variable = "{0:s}" ... stdev of target and predicted values = '
            '{1:f}, {2:f} ... MSE and skill score = {3:f}, {4:f} ... '
            'MAE and skill score = {5:f}, {6:f} ... bias = {7:f} ... '
            'correlation = {8:f}'
        ).format(
            scalar_target_names[k],
            evaluation_dict[evaluation.SCALAR_TARGET_STDEV_KEY][k],
            evaluation_dict[evaluation.SCALAR_PREDICTION_STDEV_KEY][k],
            evaluation_dict[evaluation.SCALAR_MSE_KEY][k],
            evaluation_dict[evaluation.SCALAR_MSE_SKILL_KEY][k],
            evaluation_dict[evaluation.SCALAR_MAE_KEY][k],
            evaluation_dict[evaluation.SCALAR_MAE_SKILL_KEY][k],
            evaluation_dict[evaluation.SCALAR_BIAS_KEY][k],
            evaluation_dict[evaluation.SCALAR_CORRELATION_KEY][k]
        ))

    print(SEPARATOR_STRING)

    vector_target_names = (
        generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY]
    )
    heights_m_agl = generator_option_dict[neural_net.HEIGHTS_KEY]

    for k in range(len(vector_target_names)):
        print('Variable = "{0:s}" ... PRMSE = {1:f}'.format(
            vector_target_names[k],
            evaluation_dict[evaluation.VECTOR_PRMSE_KEY][k]
        ))

    print(SEPARATOR_STRING)

    for k in range(len(vector_target_names)):
        for j in range(len(heights_m_agl)):
            print((
                'Variable = "{0:s}" at {1:d} m AGL ... '
                'stdev of target and predicted values = {2:f}, {3:f} ... '
                'MSE and skill score = {4:f}, {5:f} ... '
                'MAE and skill score = {6:f}, {7:f} ... bias = {8:f} ... '
                'correlation = {9:f}'
            ).format(
                vector_target_names[k], int(numpy.round(heights_m_agl[j])),
                evaluation_dict[evaluation.VECTOR_TARGET_STDEV_KEY][j, k],
                evaluation_dict[evaluation.VECTOR_PREDICTION_STDEV_KEY][j, k],
                evaluation_dict[evaluation.VECTOR_MSE_KEY][j, k],
                evaluation_dict[evaluation.VECTOR_MSE_SKILL_KEY][j, k],
                evaluation_dict[evaluation.VECTOR_MAE_KEY][j, k],
                evaluation_dict[evaluation.VECTOR_MAE_SKILL_KEY][j, k],
                evaluation_dict[evaluation.VECTOR_BIAS_KEY][j, k],
                evaluation_dict[evaluation.VECTOR_CORRELATION_KEY][j, k]
            ))

        print(SEPARATOR_STRING)

    print('Writing results to: "{0:s}"...'.format(evaluation_file_name))
    evaluation.write_file(
        evaluation_dict=evaluation_dict,
        pickle_file_name=evaluation_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        evaluation_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
