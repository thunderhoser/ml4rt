"""Finds extreme examples (those with the best and worst predictions)."""

import copy
import os.path
import argparse
import numpy
from gewittergefahr.gg_utils import error_checking
from ml4rt.io import example_io
from ml4rt.io import prediction_io
from ml4rt.utils import normalization
from ml4rt.machine_learning import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

PRMSE_NAME = 'prmse'
LOW_BIAS_NAME = 'low_bias'
HIGH_BIAS_NAME = 'high_bias'
MSE_SKILL_SCORE_NAME = 'mse_skill_score'

VALID_CRITERION_NAMES_WITH_HEIGHT = [
    LOW_BIAS_NAME, HIGH_BIAS_NAME, MSE_SKILL_SCORE_NAME
]
VALID_CRITERION_NAMES_SANS_HEIGHT = (
    [PRMSE_NAME] + VALID_CRITERION_NAMES_WITH_HEIGHT
)

INPUT_FILE_ARG_NAME = 'input_prediction_file_name'
TARGET_ARG_NAME = 'target_name'
TARGET_HEIGHT_ARG_NAME = 'target_height_m_agl'
CRITERION_ARG_NAME = 'criterion_name'
NUM_BEST_ARG_NAME = 'num_best_examples'
NUM_WORST_ARG_NAME = 'num_worst_examples'
BEST_FILE_ARG_NAME = 'best_prediction_file_name'
WORST_FILE_ARG_NAME = 'worst_prediction_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing predicted and actual values from model.  '
    'Will be read by `prediction_io.write_file`.'
)
TARGET_HELP_STRING = (
    'Name of target variable.  Will find best/worst predictions for this '
    'variable only.'
)
TARGET_HEIGHT_HELP_STRING = (
    'Height of target variable.  Will find best/worst predictions for `{0:s}` '
    'at this height only.  If you do not want to focus on a specific height, '
    'leave this alone.'
).format(TARGET_ARG_NAME)

CRITERION_HELP_STRING = (
    'Criterion used to select best and worst predictions.  If PRMSE, will be '
    'based on entire profiles (since PRMSE is defined only one per profile).  '
    'Otherwise, will be based on min/max over all heights in the profile.  '
    'Criterion must be in the following list:\n{0:s}'
).format(str(VALID_CRITERION_NAMES_SANS_HEIGHT))

NUM_BEST_HELP_STRING = 'Number of best-predicted examples to save.'
NUM_WORST_HELP_STRING = 'Number of worst-predicted examples to save.'
BEST_FILE_HELP_STRING = (
    'Path to output file, containing predicted and actual values only for '
    'best-predicted examples.  Will be written by `prediction_io.write_file`.'
)
WORST_FILE_HELP_STRING = (
    'Same as `{0:s}` but for worst-predicted examples.'
).format(BEST_FILE_ARG_NAME)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_ARG_NAME, type=str, required=True, help=TARGET_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_HEIGHT_ARG_NAME, type=int, required=False, default=-1,
    help=TARGET_HEIGHT_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CRITERION_ARG_NAME, type=str, required=True,
    help=CRITERION_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_BEST_ARG_NAME, type=int, required=False, default=100,
    help=NUM_BEST_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_WORST_ARG_NAME, type=int, required=False, default=100,
    help=NUM_WORST_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + BEST_FILE_ARG_NAME, type=str, required=True,
    help=BEST_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + WORST_FILE_ARG_NAME, type=str, required=True,
    help=WORST_FILE_HELP_STRING
)


def _run(input_prediction_file_name, target_name, target_height_m_agl,
         criterion_name, num_best_examples, num_worst_examples,
         best_prediction_file_name, worst_prediction_file_name):
    """Finds extreme examples (those with the best and worst predictions).

    This is effectively the main method.

    :param input_prediction_file_name: See documentation at top of file.
    :param target_name: Same.
    :param target_height_m_agl: Same.
    :param criterion_name: Same.
    :param num_best_examples: Same.
    :param num_worst_examples: Same.
    :param best_prediction_file_name: Same.
    :param worst_prediction_file_name: Same.
    :raises: ValueError: if criterion is invalid.
    :raises: ValueError: if `target_name` is not a target variable for the
        model.
    """

    error_checking.assert_is_geq(num_best_examples, 0)
    error_checking.assert_is_geq(num_worst_examples, 0)
    error_checking.assert_is_greater(num_best_examples + num_worst_examples, 0)

    if target_height_m_agl < 0:
        target_height_m_agl = None

    valid_criterion_names = (
        VALID_CRITERION_NAMES_SANS_HEIGHT if target_height_m_agl is None
        else VALID_CRITERION_NAMES_WITH_HEIGHT
    )

    if criterion_name not in valid_criterion_names:
        error_string = (
            '\n"{0:s}" is not a valid criterion.  Must be in the following '
            'list:\n{1:s}'
        ).format(criterion_name, str(valid_criterion_names))

        raise ValueError(error_string)

    print('Reading data from: "{0:s}"...'.format(input_prediction_file_name))
    prediction_dict = prediction_io.read_file(input_prediction_file_name)

    model_file_name = prediction_dict[prediction_io.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        os.path.split(model_file_name)[0]
    )

    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)

    generator_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    vector_target_names = (
        generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY]
    )

    if target_name not in vector_target_names:
        error_string = (
            '\n"{0:s}" is not a target variable for the model.  Must be in the '
            'following list:\n{1:s}'
        ).format(target_name, str(vector_target_names))

        raise ValueError(error_string)

    target_index = vector_target_names.index(target_name)
    vector_target_matrix = (
        prediction_dict[prediction_io.VECTOR_TARGETS_KEY][..., target_index]
    )
    vector_prediction_matrix = (
        prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY][..., target_index]
    )

    if target_height_m_agl is not None:
        height_index = example_io.match_heights(
            heights_m_agl=generator_option_dict[neural_net.HEIGHTS_KEY],
            desired_height_m_agl=target_height_m_agl
        )

        vector_target_matrix = numpy.take(
            vector_target_matrix, indices=[height_index], axis=1
        )
        vector_prediction_matrix = numpy.take(
            vector_prediction_matrix, indices=[height_index], axis=1
        )

        # vector_target_matrix = vector_target_matrix[:, [height_index], :]
        # vector_prediction_matrix = (
        #     vector_prediction_matrix[:, [height_index], :]
        # )

    if criterion_name == PRMSE_NAME:
        scores_to_find_best = numpy.sqrt(numpy.mean(
            (vector_target_matrix - vector_prediction_matrix) ** 2, axis=1
        ))
        scores_to_find_worst = scores_to_find_best + 0.
    elif criterion_name == HIGH_BIAS_NAME:
        scores_to_find_best = numpy.max(
            numpy.absolute(vector_prediction_matrix - vector_target_matrix),
            axis=1
        )
        scores_to_find_worst = numpy.max(
            vector_prediction_matrix - vector_target_matrix, axis=1
        )
    elif criterion_name == LOW_BIAS_NAME:
        scores_to_find_best = numpy.max(
            numpy.absolute(vector_prediction_matrix - vector_target_matrix),
            axis=1
        )
        scores_to_find_worst = numpy.max(
            vector_target_matrix - vector_prediction_matrix, axis=1
        )
    else:
        normalization_file_name = (
            generator_option_dict[neural_net.NORMALIZATION_FILE_KEY]
        )
        print((
            'Reading training examples (for climatology) from: "{0:s}"...'
        ).format(
            normalization_file_name
        ))
        training_example_dict = example_io.read_file(normalization_file_name)
        training_example_dict = example_io.subset_by_height(
            example_dict=training_example_dict,
            heights_m_agl=generator_option_dict[neural_net.HEIGHTS_KEY]
        )

        dummy_example_dict = {
            example_io.SCALAR_PREDICTOR_NAMES_KEY:
                generator_option_dict[neural_net.SCALAR_PREDICTOR_NAMES_KEY],
            example_io.VECTOR_PREDICTOR_NAMES_KEY:
                generator_option_dict[neural_net.VECTOR_PREDICTOR_NAMES_KEY],
            example_io.SCALAR_TARGET_NAMES_KEY:
                generator_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY],
            example_io.VECTOR_TARGET_NAMES_KEY:
                generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY],
            example_io.HEIGHTS_KEY:
                generator_option_dict[neural_net.HEIGHTS_KEY]
        }

        mean_training_example_dict = normalization.create_mean_example(
            new_example_dict=dummy_example_dict,
            training_example_dict=training_example_dict
        )

        climo_prediction_matrix = mean_training_example_dict[
            example_io.VECTOR_TARGET_VALS_KEY
        ][..., target_index]

        num_examples = vector_prediction_matrix.shape[0]
        climo_prediction_matrix = numpy.repeat(
            climo_prediction_matrix, repeats=num_examples, axis=0
        )

        mse_matrix = (vector_prediction_matrix - vector_target_matrix) ** 2
        climo_mse_matrix = (climo_prediction_matrix - vector_target_matrix) ** 2
        mse_skill_score_matrix = (
            (climo_mse_matrix - mse_matrix) / climo_mse_matrix
        )

        scores_to_find_best = -1 * numpy.min(mse_skill_score_matrix, axis=1)
        scores_to_find_worst = scores_to_find_best + 0.

    best_example_indices = (
        numpy.argsort(scores_to_find_best)[:num_best_examples]
    )
    worst_example_indices = (
        numpy.argsort(-1 * scores_to_find_worst)[:num_worst_examples]
    )

    print(SEPARATOR_STRING)

    for i in range(num_best_examples):
        print((
            'Negatively oriented score (based on {0:s}) for {1:d}th-best '
            'example: {2:.4e}'
        ).format(
            criterion_name.upper(), i + 1,
            scores_to_find_best[best_example_indices[i]]
        ))

    print(SEPARATOR_STRING)

    for i in range(num_worst_examples):
        print((
            'Negatively oriented score (based on {0:s}) for {1:d}th-worst '
            'example: {2:.4e}'
        ).format(
            criterion_name.upper(), i + 1,
            scores_to_find_worst[worst_example_indices[i]]
        ))

    print(SEPARATOR_STRING)

    if num_best_examples > 0:
        best_prediction_dict = prediction_io.subset_by_index(
            prediction_dict=copy.deepcopy(prediction_dict),
            desired_indices=best_example_indices
        )

        print('Writing best-predicted examples to: "{0:s}"...'.format(
            best_prediction_file_name
        ))

        prediction_io.write_file(
            netcdf_file_name=best_prediction_file_name,
            scalar_target_matrix=
            best_prediction_dict[prediction_io.SCALAR_TARGETS_KEY],
            vector_target_matrix=
            best_prediction_dict[prediction_io.VECTOR_TARGETS_KEY],
            scalar_prediction_matrix=
            best_prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY],
            vector_prediction_matrix=
            best_prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY],
            example_id_strings=
            best_prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
            model_file_name=best_prediction_dict[prediction_io.MODEL_FILE_KEY]
        )

    if num_worst_examples > 0:
        worst_prediction_dict = prediction_io.subset_by_index(
            prediction_dict=copy.deepcopy(prediction_dict),
            desired_indices=worst_example_indices
        )

        print('Writing worst-predicted examples to: "{0:s}"...'.format(
            worst_prediction_file_name
        ))

        prediction_io.write_file(
            netcdf_file_name=worst_prediction_file_name,
            scalar_target_matrix=
            worst_prediction_dict[prediction_io.SCALAR_TARGETS_KEY],
            vector_target_matrix=
            worst_prediction_dict[prediction_io.VECTOR_TARGETS_KEY],
            scalar_prediction_matrix=
            worst_prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY],
            vector_prediction_matrix=
            worst_prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY],
            example_id_strings=
            worst_prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
            model_file_name=worst_prediction_dict[prediction_io.MODEL_FILE_KEY]
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_prediction_file_name=getattr(
            INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME
        ),
        target_name=getattr(INPUT_ARG_OBJECT, TARGET_ARG_NAME),
        target_height_m_agl=getattr(INPUT_ARG_OBJECT, TARGET_HEIGHT_ARG_NAME),
        criterion_name=getattr(INPUT_ARG_OBJECT, CRITERION_ARG_NAME),
        num_best_examples=getattr(INPUT_ARG_OBJECT, NUM_BEST_ARG_NAME),
        num_worst_examples=getattr(INPUT_ARG_OBJECT, NUM_WORST_ARG_NAME),
        best_prediction_file_name=getattr(INPUT_ARG_OBJECT, BEST_FILE_ARG_NAME),
        worst_prediction_file_name=getattr(
            INPUT_ARG_OBJECT, WORST_FILE_ARG_NAME
        )
    )
