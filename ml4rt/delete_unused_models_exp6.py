"""Deletes unused models (those from non-terminal epoch) for Experiment 6."""

import os
import glob
import argparse
import numpy

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

DENSE_LAYER_COUNTS = numpy.array([2, 3, 4, 5], dtype=int)
DENSE_LAYER_DROPOUT_RATES = numpy.array([0, 0.1, 0.2, 0.3, 0.4, 0.5])
SCALAR_LOSS_FUNCTION_WEIGHTS = numpy.array([1, 2.5, 5, 10, 25, 50])

EXPERIMENT_DIR_ARG_NAME = 'experiment_dir_name'
EXPERIMENT_DIR_HELP_STRING = 'Name of top-level directory with models.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EXPERIMENT_DIR_ARG_NAME, type=str, required=True,
    help=EXPERIMENT_DIR_HELP_STRING
)


def _delete_unused_models_one_hp_set(model_dir_name):
    """Deletes unused models for one set of hyperparameters.

    :param model_dir_name: Name of directory with trained model and evaluation
        data.
    """

    model_file_pattern = '{0:s}/model*.h5'.format(model_dir_name)
    model_file_names = glob.glob(model_file_pattern)

    if len(model_file_names) == 0:
        return

    model_file_names.sort()

    for i in range(len(model_file_names)):
        if i == len(model_file_names) - 1:
            print('KEEPING file: "{0:s}"...'.format(model_file_names[i]))
            continue

        print('DELETING file: "{0:s}"...'.format(model_file_names[i]))
        # os.remove(model_file_names[i])


def _run(experiment_dir_name):
    """Deletes unused models (those from non-terminal epoch) for Experiment 6.

    This is effectively the main method.

    :param experiment_dir_name: See documentation at top of file.
    """

    num_dense_layer_counts = len(DENSE_LAYER_COUNTS)
    num_dropout_rates = len(DENSE_LAYER_DROPOUT_RATES)
    num_loss_function_weights = len(SCALAR_LOSS_FUNCTION_WEIGHTS)

    for i in range(num_dense_layer_counts):
        for j in range(num_dropout_rates):
            for k in range(num_loss_function_weights):
                this_model_dir_name = (
                    '{0:s}/num-dense-layers={1:d}_dense-dropout={2:.3f}_'
                    'scalar-lf-weight={3:05.1f}'
                ).format(
                    experiment_dir_name, DENSE_LAYER_COUNTS[i],
                    DENSE_LAYER_DROPOUT_RATES[j],
                    SCALAR_LOSS_FUNCTION_WEIGHTS[k]
                )

                _delete_unused_models_one_hp_set(
                    model_dir_name=this_model_dir_name
                )
                print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        experiment_dir_name=getattr(INPUT_ARG_OBJECT, EXPERIMENT_DIR_ARG_NAME)
    )
