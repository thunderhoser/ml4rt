"""Runs permutation-based importance test."""

import os.path
import argparse
from ml4rt.utils import misc as misc_utils
from ml4rt.machine_learning import neural_net
from ml4rt.machine_learning import permutation
from ml4rt.scripts import make_saliency_maps

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MSE_NAME = 'mse'
DUAL_WEIGHTED_MSE_NAME = 'dual_weighted_mse_name'
VALID_COST_FUNCTION_NAMES = [MSE_NAME, DUAL_WEIGHTED_MSE_NAME]

MODEL_FILE_ARG_NAME = 'input_model_file_name'
EXAMPLE_FILE_ARG_NAME = make_saliency_maps.EXAMPLE_FILE_ARG_NAME
NUM_EXAMPLES_ARG_NAME = make_saliency_maps.NUM_EXAMPLES_ARG_NAME
EXAMPLE_DIR_ARG_NAME = make_saliency_maps.EXAMPLE_DIR_ARG_NAME
EXAMPLE_ID_FILE_ARG_NAME = make_saliency_maps.EXAMPLE_ID_FILE_ARG_NAME
COST_FUNCTION_ARG_NAME = 'cost_function_name'
DO_BACKWARDS_ARG_NAME = 'do_backwards_test'
SHUFFLE_TOGETHER_ARG_NAME = 'shuffle_profiles_together'
NUM_BOOTSTRAP_ARG_NAME = 'num_bootstrap_reps'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

MODEL_FILE_HELP_STRING = (
    'Path to trained model.  Will be read by `neural_net.read_model`.'
)
EXAMPLE_FILE_HELP_STRING = make_saliency_maps.EXAMPLE_FILE_HELP_STRING
NUM_EXAMPLES_HELP_STRING = make_saliency_maps.NUM_EXAMPLES_HELP_STRING
EXAMPLE_DIR_HELP_STRING = make_saliency_maps.EXAMPLE_DIR_HELP_STRING
EXAMPLE_ID_FILE_HELP_STRING = make_saliency_maps.EXAMPLE_ID_FILE_HELP_STRING
COST_FUNCTION_HELP_STRING = (
    'Cost function.  Must be in the following list:\n{0:s}'
).format(str(VALID_COST_FUNCTION_NAMES))

DO_BACKWARDS_HELP_STRING = (
    'Boolean flag.  If 1, will run backwards permutation test.  If 0, will run '
    'forward permutation test.'
)
SHUFFLE_TOGETHER_HELP_STRING = (
    'Boolean flag.  If 1, vertical profiles will be shuffled together.  If 0, '
    'all scalar variables will be shuffled independently (i.e., shuffling will '
    'be done along both the example and height axes), so vertical profiles will'
    ' be destroyed by shuffling.'
)
NUM_BOOTSTRAP_HELP_STRING = (
    'Number of bootstrap replicates used to estimate cost function.'
)
OUTPUT_FILE_HELP_STRING = (
    'Name of output file.  Results will be saved here by '
    '`permutation.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_FILE_ARG_NAME, type=str, required=False, default='',
    help=EXAMPLE_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_EXAMPLES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=False, default='',
    help=EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_ID_FILE_ARG_NAME, type=str, required=False, default='',
    help=EXAMPLE_ID_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + COST_FUNCTION_ARG_NAME, type=str, required=False, default=MSE_NAME,
    help=COST_FUNCTION_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DO_BACKWARDS_ARG_NAME, type=int, required=False, default=0,
    help=DO_BACKWARDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SHUFFLE_TOGETHER_ARG_NAME, type=int, required=False, default=1,
    help=SHUFFLE_TOGETHER_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_BOOTSTRAP_ARG_NAME, type=int, required=False, default=1000,
    help=NUM_BOOTSTRAP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(model_file_name, example_file_name, num_examples, example_dir_name,
         example_id_file_name, cost_function_name, do_backwards_test,
         shuffle_profiles_together, num_bootstrap_reps, output_file_name):
    """Runs permutation-based importance test.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param example_file_name: Same.
    :param num_examples: Same.
    :param example_dir_name: Same.
    :param example_id_file_name: Same.
    :param cost_function_name: Same.
    :param do_backwards_test: Same.
    :param shuffle_profiles_together: Same.
    :param num_bootstrap_reps: Same.
    :param output_file_name: Same.
    :raises: ValueError: if
        `cost_function_name not in VALID_COST_FUNCTION_NAMES`.
    """

    if cost_function_name not in VALID_COST_FUNCTION_NAMES:
        error_string = (
            '\nCost function ("{0:s}") should be in the following list:\n{1:s}'
        ).format(cost_function_name, str(VALID_COST_FUNCTION_NAMES))

        raise ValueError(error_string)

    if cost_function_name == MSE_NAME:
        cost_function = permutation.mse_cost_function
    else:
        cost_function = permutation.dual_weighted_mse_cost_function

    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = neural_net.read_model(model_file_name)

    metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(metafile_name))
    metadata_dict = neural_net.read_metafile(metafile_name)

    predictor_matrix, target_matrices = (
        misc_utils.get_examples_for_inference(
            model_metadata_dict=metadata_dict,
            example_file_name=example_file_name,
            num_examples=num_examples, example_dir_name=example_dir_name,
            example_id_file_name=example_id_file_name
        )[:2]
    )
    print(SEPARATOR_STRING)

    if not isinstance(target_matrices, list):
        target_matrices = [target_matrices]

    if do_backwards_test:
        result_dict = permutation.run_backwards_test(
            predictor_matrix=predictor_matrix, target_matrices=target_matrices,
            model_object=model_object, model_metadata_dict=metadata_dict,
            cost_function=cost_function,
            shuffle_profiles_together=shuffle_profiles_together,
            num_bootstrap_reps=num_bootstrap_reps
        )
    else:
        result_dict = permutation.run_forward_test(
            predictor_matrix=predictor_matrix, target_matrices=target_matrices,
            model_object=model_object, model_metadata_dict=metadata_dict,
            cost_function=cost_function,
            shuffle_profiles_together=shuffle_profiles_together,
            num_bootstrap_reps=num_bootstrap_reps
        )

    print(SEPARATOR_STRING)

    print('Writing results of permutation test to: "{0:s}"...'.format(
        output_file_name
    ))

    permutation.write_file(
        result_dict=result_dict, netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        example_file_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_FILE_ARG_NAME),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        example_id_file_name=getattr(
            INPUT_ARG_OBJECT, EXAMPLE_ID_FILE_ARG_NAME
        ),
        cost_function_name=getattr(INPUT_ARG_OBJECT, COST_FUNCTION_ARG_NAME),
        do_backwards_test=bool(getattr(
            INPUT_ARG_OBJECT, DO_BACKWARDS_ARG_NAME
        )),
        shuffle_profiles_together=bool(getattr(
            INPUT_ARG_OBJECT, SHUFFLE_TOGETHER_ARG_NAME
        )),
        num_bootstrap_reps=getattr(INPUT_ARG_OBJECT, NUM_BOOTSTRAP_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
