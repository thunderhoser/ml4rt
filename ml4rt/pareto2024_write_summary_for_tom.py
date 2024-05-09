"""Writes accuracy/complexity summary of each model for Tom Beucler."""

import os
import sys
import argparse
import numpy
from tensorflow.keras import backend as K

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import prediction_io
import neural_net

CNN_DEPTHS = numpy.array([1, 2, 3, 4, 5, 6], dtype=int)
CNN_CHANNEL_COUNTS = numpy.array([2, 4, 8, 16, 32, 64], dtype=int)

DENSE_NET_MODEL_DEPTHS = numpy.array([1, 2, 3, 4, 5, 6], dtype=int)
DENSE_NET_NEURON_COUNTS = numpy.array(
    [64, 128, 256, 512, 1024, 2048], dtype=int
)

UNET_TYPE_MODEL_DEPTHS = numpy.array([3, 4, 5], dtype=int)
UNET_TYPE_CHANNEL_COUNTS = numpy.array([2, 4, 8, 16, 32, 64], dtype=int)

INPUT_DIR_ARG_NAME = 'input_top_experiment_dir_name'
OUTPUT_FILE_ARG_NAME = 'output_csv_file_name'

INPUT_DIR_HELP_STRING = (
    'Name of top-level directory, containing a subdirectory for each model '
    'type (U-net++, U-net, CNN, dense net, linear regression).'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Results will be saved here in CSV format.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _read_results_one_model(model_dir_name):
    """Reads results for one model.

    :param model_dir_name: Path to model directory.
    :return: num_trainable_params: Number of trainable parameters.
    :return: overall_mse_value_k2_day02: MSE on heating rates for full testing
        set.
    :return: simple_mse_value_k2_day02: MSE on heating rates for simple testing
        set.
    :return: complex_mse_value_k2_day02: MSE on heating rates for complex
        testing set.
    """

    model_file_name = '{0:s}/model.keras'.format(model_dir_name)
    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = neural_net.read_model(model_file_name)

    num_trainable_params = numpy.sum(numpy.array(
        [K.count_params(w) for w in model_object.trainable_weights], dtype=int
    ))

    overall_prediction_file_name = '{0:s}/testing/predictions.nc'.format(
        model_dir_name
    )
    print('Reading predictions on full testing set from: "{0:s}"...'.format(
        overall_prediction_file_name
    ))
    overall_prediction_dict = prediction_io.read_file(
        overall_prediction_file_name
    )
    overall_mse_value_k2_day02 = numpy.mean(
        (overall_prediction_dict[prediction_io.VECTOR_TARGETS_KEY] -
         overall_prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY][..., 0]) ** 2
    )

    simple_prediction_file_name = (
        '{0:s}/testing_simple_data/predictions.nc'.format(model_dir_name)
    )
    print('Reading predictions on simple testing set from: "{0:s}"...'.format(
        simple_prediction_file_name
    ))
    simple_prediction_dict = prediction_io.read_file(
        simple_prediction_file_name
    )
    simple_mse_value_k2_day02 = numpy.mean(
        (simple_prediction_dict[prediction_io.VECTOR_TARGETS_KEY] -
         simple_prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY][..., 0]) ** 2
    )

    complex_prediction_file_name = (
        '{0:s}/testing_complex_data/predictions.nc'.format(model_dir_name)
    )
    print('Reading predictions on complex testing set from: "{0:s}"...'.format(
        complex_prediction_file_name
    ))
    complex_prediction_dict = prediction_io.read_file(
        complex_prediction_file_name
    )
    complex_mse_value_k2_day02 = numpy.mean(
        (complex_prediction_dict[prediction_io.VECTOR_TARGETS_KEY] -
         complex_prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY][..., 0]) ** 2
    )

    return (
        num_trainable_params, overall_mse_value_k2_day02,
        simple_mse_value_k2_day02, complex_mse_value_k2_day02
    )


def _run(top_experiment_dir_name, output_file_name):
    """Writes accuracy/complexity summary of each model for Tom Beucler.

    This is effectively the main method.

    :param top_experiment_dir_name: See documentation at top of file.
    :param output_file_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    hyperparam_strings = []
    trainable_param_counts = []
    overall_hr_mse_values_k2_day02 = []
    simple_hr_mse_values_k2_day02 = []
    complex_hr_mse_values_k2_day02 = []

    this_model_dir_name = '{0:s}/linear_regression'.format(top_experiment_dir_name)
    hyperparam_strings.append('linear_regression')

    this_num_params, this_overall_mse, this_simple_mse, this_complex_mse = (
        _read_results_one_model(this_model_dir_name)
    )

    trainable_param_counts.append(this_num_params)
    overall_hr_mse_values_k2_day02.append(this_overall_mse)
    simple_hr_mse_values_k2_day02.append(this_simple_mse)
    complex_hr_mse_values_k2_day02.append(this_complex_mse)

    for i in range(len(DENSE_NET_MODEL_DEPTHS)):
        for j in range(len(DENSE_NET_NEURON_COUNTS)):
            this_hyperparam_string = (
                'dense_net/num-levels={2:d}_num-neurons-per-layer={1:04d}'
            ).format(
                DENSE_NET_MODEL_DEPTHS[i],
                DENSE_NET_NEURON_COUNTS[j]
            )
            hyperparam_strings.append(this_hyperparam_string)

            this_model_dir_name = '{0:s}/{1:s}'.format(
                top_experiment_dir_name, this_hyperparam_string
            )
            (
                this_num_params,
                this_overall_mse,
                this_simple_mse,
                this_complex_mse
            ) = _read_results_one_model(this_model_dir_name)

            trainable_param_counts.append(this_num_params)
            overall_hr_mse_values_k2_day02.append(this_overall_mse)
            simple_hr_mse_values_k2_day02.append(this_simple_mse)
            complex_hr_mse_values_k2_day02.append(this_complex_mse)

    for i in range(len(CNN_DEPTHS)):
        for j in range(len(CNN_CHANNEL_COUNTS)):
            this_hyperparam_string = (
                'cnn/num-levels={0:d}_num-first-layer-channels={1:02d}'
            ).format(
                CNN_DEPTHS[i],
                CNN_CHANNEL_COUNTS[j]
            )
            hyperparam_strings.append(this_hyperparam_string)

            this_model_dir_name = '{0:s}/{1:s}'.format(
                top_experiment_dir_name, this_hyperparam_string
            )
            (
                this_num_params,
                this_overall_mse,
                this_simple_mse,
                this_complex_mse
            ) = _read_results_one_model(this_model_dir_name)

            trainable_param_counts.append(this_num_params)
            overall_hr_mse_values_k2_day02.append(this_overall_mse)
            simple_hr_mse_values_k2_day02.append(this_simple_mse)
            complex_hr_mse_values_k2_day02.append(this_complex_mse)

    for i in range(len(UNET_TYPE_MODEL_DEPTHS)):
        for j in range(len(UNET_TYPE_CHANNEL_COUNTS)):
            this_hyperparam_string = (
                'u_net/num-levels={0:d}_num-first-layer-channels={1:02d}'
            ).format(
                UNET_TYPE_MODEL_DEPTHS[i],
                UNET_TYPE_CHANNEL_COUNTS[j]
            )
            hyperparam_strings.append(this_hyperparam_string)

            this_model_dir_name = '{0:s}/{1:s}'.format(
                top_experiment_dir_name, this_hyperparam_string
            )
            (
                this_num_params,
                this_overall_mse,
                this_simple_mse,
                this_complex_mse
            ) = _read_results_one_model(this_model_dir_name)

            trainable_param_counts.append(this_num_params)
            overall_hr_mse_values_k2_day02.append(this_overall_mse)
            simple_hr_mse_values_k2_day02.append(this_simple_mse)
            complex_hr_mse_values_k2_day02.append(this_complex_mse)

    for i in range(len(UNET_TYPE_MODEL_DEPTHS)):
        for j in range(len(UNET_TYPE_CHANNEL_COUNTS)):
            this_hyperparam_string = (
                'u_net_plusplus/'
                'num-levels={0:d}_num-first-layer-channels={1:02d}'
            ).format(
                UNET_TYPE_MODEL_DEPTHS[i],
                UNET_TYPE_CHANNEL_COUNTS[j]
            )
            hyperparam_strings.append(this_hyperparam_string)

            this_model_dir_name = '{0:s}/{1:s}'.format(
                top_experiment_dir_name, this_hyperparam_string
            )
            (
                this_num_params,
                this_overall_mse,
                this_simple_mse,
                this_complex_mse
            ) = _read_results_one_model(this_model_dir_name)

            trainable_param_counts.append(this_num_params)
            overall_hr_mse_values_k2_day02.append(this_overall_mse)
            simple_hr_mse_values_k2_day02.append(this_simple_mse)
            complex_hr_mse_values_k2_day02.append(this_complex_mse)

    print('Writing summary to: "{0:s}"...'.format(output_file_name))
    with open(output_file_name, 'w') as output_file_handle:
        output_file_handle.write(
            'model_hyperparams, num_trainable_params, '
            'overall_heating_rate_mse_k2_day02, '
            'simple_heating_rate_mse_k2_day02, '
            'complex_heating_rate_mse_k2_day02\n'
        )

        for i in range(len(hyperparam_strings)):
            new_line = '{0:s}, {1:d}, {2:.10f}, {3:.10f}, {4:.10f}\n'.format(
                hyperparam_strings[i],
                trainable_param_counts[i],
                overall_hr_mse_values_k2_day02[i],
                simple_hr_mse_values_k2_day02[i],
                complex_hr_mse_values_k2_day02[i]
            )

            output_file_handle.write(new_line)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_experiment_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
