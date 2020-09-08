"""Deletes unused models (those from non-terminal epoch) for Experiment 1."""

import os
import glob
import argparse
import numpy

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

PLATEAU_LR_MULTIPLIERS = numpy.array([0.5, 0.6, 0.7, 0.8, 0.9])
BATCH_SIZES = numpy.array(
    [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192], dtype=int
)

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
        os.remove(model_file_names[i])


def _run(experiment_dir_name):
    """Deletes unused models (those from non-terminal epoch) for Experiment 1.

    This is effectively the main method.

    :param experiment_dir_name: See documentation at top of file.
    """

    num_multipliers = len(PLATEAU_LR_MULTIPLIERS)
    num_batch_sizes = len(BATCH_SIZES)

    for i in range(num_multipliers):
        for j in range(num_batch_sizes):
            this_model_dir_name = (
                '{0:s}/plateau-lr-multiplier={1:.1f}_batch-size={2:04d}'
            ).format(
                experiment_dir_name, PLATEAU_LR_MULTIPLIERS[i], BATCH_SIZES[j]
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
