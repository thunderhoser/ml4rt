"""Runs permutation-based importance test."""

import os
import sys
import argparse

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import misc as misc_utils
import neural_net
import permutation

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MODEL_FILE_ARG_NAME = 'input_model_file_name'
EXAMPLE_FILE_ARG_NAME = 'input_example_file_name'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
EXAMPLE_ID_FILE_ARG_NAME = 'input_example_id_file_name'
HEATING_RATE_WEIGHT_ARG_NAME = 'heating_rate_weight'
FLUX_WEIGHT_ARG_NAME = 'flux_weight'
INCLUDE_NET_FLUX_ARG_NAME = 'include_net_flux'
DO_BACKWARDS_ARG_NAME = 'do_backwards_test'
SHUFFLE_TOGETHER_ARG_NAME = 'shuffle_profiles_together'
NUM_BOOTSTRAP_ARG_NAME = 'num_bootstrap_reps'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

MODEL_FILE_HELP_STRING = (
    'Path to trained model.  Will be read by `neural_net.read_model`.'
)
EXAMPLE_FILE_HELP_STRING = (
    '[use only if you want random examples] Path to file with data examples.  '
    'Will be read by `example_io.read_file`.'
)
NUM_EXAMPLES_HELP_STRING = (
    '[use only if you want random examples] Number of examples to use.  If you '
    'want to use all examples in `{0:s}`, leave this alone.'
).format(EXAMPLE_FILE_ARG_NAME)

EXAMPLE_DIR_HELP_STRING = (
    '[use only if you want specific examples] Name of directory with data '
    'examples.  Files therein will be found by `example_io.find_file` and read '
    'by `example_io.read_file`.'
)
EXAMPLE_ID_FILE_HELP_STRING = (
    '[use only if you want specific examples] Path to file with desired IDs.  '
    'Will be read by `misc.read_example_ids_from_netcdf`.'
)

HEATING_RATE_WEIGHT_HELP_STRING = 'Weight for heating rates in loss function.'
FLUX_WEIGHT_HELP_STRING = 'Weight for fluxes in loss function.'
INCLUDE_NET_FLUX_HELP_STRING = (
    'Boolean flag.  If 1 (0), will (not) penalize net flux in loss function.'
)
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
    '--' + HEATING_RATE_WEIGHT_ARG_NAME, type=float, required=True,
    help=HEATING_RATE_WEIGHT_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FLUX_WEIGHT_ARG_NAME, type=float, required=True,
    help=FLUX_WEIGHT_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + INCLUDE_NET_FLUX_ARG_NAME, type=int, required=True,
    help=INCLUDE_NET_FLUX_HELP_STRING
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
         example_id_file_name, heating_rate_weight, flux_weight,
         include_net_flux, do_backwards_test, shuffle_profiles_together,
         num_bootstrap_reps, output_file_name):
    """Runs permutation-based importance test.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param example_file_name: Same.
    :param num_examples: Same.
    :param example_dir_name: Same.
    :param example_id_file_name: Same.
    :param heating_rate_weight: Same.
    :param flux_weight: Same.
    :param include_net_flux: Same.
    :param do_backwards_test: Same.
    :param shuffle_profiles_together: Same.
    :param num_bootstrap_reps: Same.
    :param output_file_name: Same.
    """

    cost_function = permutation.make_cost_function(
        heating_rate_weight=heating_rate_weight, flux_weight=flux_weight,
        include_net_flux=include_net_flux
    )

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
        heating_rate_weight=getattr(
            INPUT_ARG_OBJECT, HEATING_RATE_WEIGHT_ARG_NAME
        ),
        flux_weight=getattr(INPUT_ARG_OBJECT, FLUX_WEIGHT_ARG_NAME),
        include_net_flux=bool(getattr(
            INPUT_ARG_OBJECT, INCLUDE_NET_FLUX_ARG_NAME
        )),
        do_backwards_test=bool(getattr(
            INPUT_ARG_OBJECT, DO_BACKWARDS_ARG_NAME
        )),
        shuffle_profiles_together=bool(getattr(
            INPUT_ARG_OBJECT, SHUFFLE_TOGETHER_ARG_NAME
        )),
        num_bootstrap_reps=getattr(INPUT_ARG_OBJECT, NUM_BOOTSTRAP_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
