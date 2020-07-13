"""Runs permutation-based importance test."""

import os.path
import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from ml4rt.io import example_io
from ml4rt.utils import misc as misc_utils
from ml4rt.machine_learning import neural_net
from ml4rt.machine_learning import permutation

# TODO(thunderhoser): Make cost function an input arg.

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MODEL_FILE_ARG_NAME = 'input_model_file_name'
EXAMPLE_FILE_ARG_NAME = 'input_example_file_name'
EXAMPLE_INDICES_ARG_NAME = 'example_indices'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
DO_BACKWARDS_ARG_NAME = 'do_backwards_test'
SHUFFLE_TOGETHER_ARG_NAME = 'shuffle_profiles_together'
NUM_BOOTSTRAP_ARG_NAME = 'num_bootstrap_reps'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

MODEL_FILE_HELP_STRING = (
    'Path to trained model.  Will be read by `neural_net.read_model`.'
)
EXAMPLE_FILE_HELP_STRING = (
    'Path to file with data examples.  Will be read by `example_io.read_file`.'
)
EXAMPLE_INDICES_HELP_STRING = (
    'Indices of examples to use.  If you do not want to use specific examples, '
    'leave this alone.'
)
NUM_EXAMPLES_HELP_STRING = (
    '[used only if `{0:s}` is not specified] Number of examples to use (these '
    'will be selected randomly).  If you want to use all examples, leave this '
    'alone.'
).format(EXAMPLE_INDICES_ARG_NAME)

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
    '--' + EXAMPLE_FILE_ARG_NAME, type=str, required=True,
    help=EXAMPLE_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_INDICES_ARG_NAME, type=int, nargs='+', required=False,
    default=[-1], help=EXAMPLE_INDICES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_EXAMPLES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DO_BACKWARDS_ARG_NAME, type=int, required=False, default=0,
    help=DO_BACKWARDS_HELP_STRING)

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


def _run(model_file_name, example_file_name, example_indices, num_examples,
         do_backwards_test, shuffle_profiles_together, num_bootstrap_reps,
         output_file_name):
    """Runs permutation-based importance test.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param example_file_name: Same.
    :param example_indices: Same.
    :param num_examples: Same.
    :param do_backwards_test: Same.
    :param shuffle_profiles_together: Same.
    :param num_bootstrap_reps: Same.
    :param output_file_name: Same.
    """

    example_dict = example_io.read_file(example_file_name)
    num_examples_total = len(example_dict[example_io.VALID_TIMES_KEY])

    example_indices = misc_utils.subset_examples(
        indices_to_keep=example_indices, num_examples_to_keep=num_examples,
        num_examples_total=num_examples_total
    )
    print(example_indices)

    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = neural_net.read_model(model_file_name)

    metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(metafile_name))
    metadata_dict = neural_net.read_metafile(metafile_name)

    year = example_io.file_name_to_year(example_file_name)
    first_time_unix_sec, last_time_unix_sec = (
        time_conversion.first_and_last_times_in_year(year)
    )

    generator_option_dict = metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    generator_option_dict[neural_net.EXAMPLE_DIRECTORY_KEY] = (
        os.path.split(example_file_name)[0]
    )
    generator_option_dict[neural_net.BATCH_SIZE_KEY] = num_examples_total
    generator_option_dict[neural_net.FIRST_TIME_KEY] = first_time_unix_sec
    generator_option_dict[neural_net.LAST_TIME_KEY] = last_time_unix_sec

    generator = neural_net.data_generator(
        option_dict=generator_option_dict, for_inference=True,
        net_type_string=metadata_dict[neural_net.NET_TYPE_KEY],
        is_loss_constrained_mse=False
    )

    print(SEPARATOR_STRING)
    predictor_matrix, target_matrices = next(generator)[:2]
    print(SEPARATOR_STRING)

    if not isinstance(target_matrices, list):
        target_matrices = [target_matrices]

    predictor_matrix = predictor_matrix[example_indices, ...]
    target_matrices = [t[example_indices, ...] for t in target_matrices]

    if do_backwards_test:
        result_dict = permutation.run_backwards_test(
            predictor_matrix=predictor_matrix, target_matrices=target_matrices,
            model_object=model_object, model_metadata_dict=metadata_dict,
            cost_function=permutation.mse_cost_function,
            shuffle_profiles_together=shuffle_profiles_together,
            num_bootstrap_reps=num_bootstrap_reps
        )
    else:
        result_dict = permutation.run_forward_test(
            predictor_matrix=predictor_matrix, target_matrices=target_matrices,
            model_object=model_object, model_metadata_dict=metadata_dict,
            cost_function=permutation.mse_cost_function,
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
        example_indices=numpy.array(
            getattr(INPUT_ARG_OBJECT, EXAMPLE_INDICES_ARG_NAME), dtype=int
        ),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        do_backwards_test=bool(getattr(
            INPUT_ARG_OBJECT, DO_BACKWARDS_ARG_NAME
        )),
        shuffle_profiles_together=bool(getattr(
            INPUT_ARG_OBJECT, SHUFFLE_TOGETHER_ARG_NAME
        )),
        num_bootstrap_reps=getattr(INPUT_ARG_OBJECT, NUM_BOOTSTRAP_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
