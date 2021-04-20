"""Makes templates for Experiment 8-dense."""

import os
import sys
import copy
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import dense_net_architecture
import architecture_utils
import custom_losses
import normalization
import file_system_utils
import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

HOME_DIR_NAME = '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist'
OUTPUT_DIR_NAME = '{0:s}/ml4rt_models/experiment08_dense/templates'.format(
    HOME_DIR_NAME
)

DENSE_LAYER_COUNTS = numpy.array([2, 3, 4, 5], dtype=int)
DENSE_LAYER_DROPOUT_RATES = numpy.array([0, 0.1, 0.2, 0.3, 0.4, 0.5])
L2_WEIGHTS = numpy.logspace(-7, -3, num=9)

HEATING_RATE_LOSS_FUNCTION = custom_losses.dual_weighted_mse()

FLUX_LOSS_FUNCTION = custom_losses.scaled_mse_for_net_flux(1.)
FLUX_LOSS_STRING = (
    'custom_losses.scaled_mse_for_net_flux({0:.10f})'.format(1.)
)
LOSS_DICT = {
    'conv_output': 'custom_losses.dual_weighted_mse()',
    'dense_output': FLUX_LOSS_STRING
}

NUM_INPUTS = 73 * 12 + 2
NUM_HEIGHTS = 73
NUM_FLUX_COMPONENTS = 2

DEFAULT_OPTION_DICT = {
    dense_net_architecture.NUM_INPUTS_KEY: NUM_INPUTS,
    dense_net_architecture.NUM_HEIGHTS_KEY: NUM_HEIGHTS,
    dense_net_architecture.NUM_FLUX_COMPONENTS_KEY: NUM_FLUX_COMPONENTS,
    dense_net_architecture.INNER_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    dense_net_architecture.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    dense_net_architecture.OUTPUT_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    dense_net_architecture.OUTPUT_ACTIV_FUNCTION_ALPHA_KEY: 0.,
    dense_net_architecture.L1_WEIGHT_KEY: 0.,
    dense_net_architecture.USE_BATCH_NORM_KEY: True
}

DUMMY_GENERATOR_OPTION_DICT = {
    neural_net.VECTOR_TARGET_NORM_TYPE_KEY: None,
    neural_net.SCALAR_TARGET_NORM_TYPE_KEY: normalization.Z_SCORE_NORM_STRING
}


def _run():
    """Makes templates for Experiment 8-dense.

    This is effectively the main method.
    """

    num_dense_layer_counts = len(DENSE_LAYER_COUNTS)
    num_dropout_rates = len(DENSE_LAYER_DROPOUT_RATES)
    num_l2_weights = len(L2_WEIGHTS)

    for i in range(num_dense_layer_counts):
        these_neuron_counts = architecture_utils.get_dense_layer_dimensions(
            num_input_units=NUM_INPUTS,
            num_classes=NUM_HEIGHTS + NUM_FLUX_COMPONENTS,
            num_dense_layers=DENSE_LAYER_COUNTS[i]
        )[1]

        for j in range(num_dropout_rates):
            these_dropout_rates = numpy.full(
                DENSE_LAYER_COUNTS[i], DENSE_LAYER_DROPOUT_RATES[j]
            )
            these_dropout_rates[-1] = numpy.nan

            for k in range(num_l2_weights):
                this_option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)

                this_option_dict[
                    dense_net_architecture.HIDDEN_LAYER_NEURON_NUMS_KEY
                ] = these_neuron_counts

                this_option_dict[
                    dense_net_architecture.HIDDEN_LAYER_DROPOUT_RATES_KEY
                ] = these_dropout_rates

                this_option_dict[dense_net_architecture.L2_WEIGHT_KEY] = (
                    L2_WEIGHTS[k]
                )

                print(this_option_dict)
                print(SEPARATOR_STRING)

                this_model_object = dense_net_architecture.create_model(
                    option_dict=this_option_dict,
                    heating_rate_loss_function=HEATING_RATE_LOSS_FUNCTION,
                    flux_loss_function=FLUX_LOSS_FUNCTION
                )

                this_model_file_name = (
                    '{0:s}/num-dense-layers={1:d}_dense-dropout={2:.3f}_'
                    'l2-weight={3:.10f}/model.h5'
                ).format(
                    OUTPUT_DIR_NAME, DENSE_LAYER_COUNTS[i],
                    DENSE_LAYER_DROPOUT_RATES[j], L2_WEIGHTS[k]
                )

                file_system_utils.mkdir_recursive_if_necessary(
                    file_name=this_model_file_name
                )

                print('Writing model to: "{0:s}"...'.format(
                    this_model_file_name
                ))
                this_model_object.save(
                    filepath=this_model_file_name, overwrite=True,
                    include_optimizer=True
                )

                this_metafile_name = neural_net.find_metafile(
                    model_dir_name=os.path.split(this_model_file_name)[0],
                    raise_error_if_missing=False
                )

                print('Writing metadata to: "{0:s}"...'.format(
                    this_metafile_name
                ))
                neural_net._write_metafile(
                    dill_file_name=this_metafile_name, num_epochs=100,
                    num_training_batches_per_epoch=100,
                    training_option_dict=DUMMY_GENERATOR_OPTION_DICT,
                    num_validation_batches_per_epoch=100,
                    validation_option_dict=DUMMY_GENERATOR_OPTION_DICT,
                    net_type_string=neural_net.DENSE_NET_TYPE_STRING,
                    loss_function_or_dict=LOSS_DICT,
                    do_early_stopping=True, plateau_lr_multiplier=0.6
                )


if __name__ == '__main__':
    _run()
