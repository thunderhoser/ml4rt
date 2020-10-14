"""Makes U-net templates for Experiment 6."""

import sys
import copy
import os.path
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import u_net_architecture
import architecture_utils
import custom_losses
import normalization
import file_system_utils
import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

# if 'hfe' in socket.gethostname():
#     HOME_DIR_NAME = '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist'
# else:
#     HOME_DIR_NAME = os.path.expanduser('~')

HOME_DIR_NAME = '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist'
OUTPUT_DIR_NAME = '{0:s}/ml4rt_models/experiment06/templates'.format(
    HOME_DIR_NAME
)

DENSE_LAYER_COUNTS = numpy.array([2, 3, 4, 5], dtype=int)
DENSE_LAYER_DROPOUT_RATES = numpy.array([0.125, 0.25, 0.375, 0.5, 0.625, 0.75])
SCALAR_LOSS_FUNCTION_WEIGHTS = numpy.array([1, 2.5, 5, 10, 25, 50])

NUM_LEVELS = 4
VECTOR_LOSS_FUNCTION = custom_losses.dual_weighted_mse()

DEFAULT_OPTION_DICT = {
    u_net_architecture.NUM_LEVELS_KEY: NUM_LEVELS,
    u_net_architecture.NUM_HEIGHTS_KEY: 73,
    u_net_architecture.NUM_HEIGHTS_FOR_LOSS_KEY: 73,
    u_net_architecture.NUM_INPUT_CHANNELS_KEY: 14,
    u_net_architecture.CONV_LAYER_DROPOUT_RATES_KEY:
        numpy.full(NUM_LEVELS + 1, -1.),
    u_net_architecture.UPCONV_LAYER_DROPOUT_RATES_KEY:
        numpy.full(NUM_LEVELS, -1.),
    u_net_architecture.SKIP_LAYER_DROPOUT_RATES_KEY:
        numpy.full(NUM_LEVELS, -1.),
    u_net_architecture.OUTPUT_LAYER_DROPOUT_RATES_KEY: numpy.full(2, numpy.nan),
    u_net_architecture.INNER_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    u_net_architecture.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    u_net_architecture.OUTPUT_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    u_net_architecture.OUTPUT_ACTIV_FUNCTION_ALPHA_KEY: 0.,
    u_net_architecture.L1_WEIGHT_KEY: 0.,
    u_net_architecture.L2_WEIGHT_KEY: 0.001,
    u_net_architecture.USE_BATCH_NORM_KEY: True,
    u_net_architecture.ZERO_OUT_TOP_HR_KEY: False,
    u_net_architecture.HEATING_RATE_INDEX_KEY: None
}

DUMMY_GENERATOR_OPTION_DICT = {
    neural_net.VECTOR_TARGET_NORM_TYPE_KEY: None,
    neural_net.SCALAR_TARGET_NORM_TYPE_KEY: normalization.Z_SCORE_NORM_STRING
}


def _run():
    """Makes U-net templates for Experiment 6.

    This is effectively the main method.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=OUTPUT_DIR_NAME
    )

    num_dense_layer_counts = len(DENSE_LAYER_COUNTS)
    num_dropout_rates = len(DENSE_LAYER_DROPOUT_RATES)
    num_loss_function_weights = len(SCALAR_LOSS_FUNCTION_WEIGHTS)

    for i in range(num_dense_layer_counts):
        these_neuron_counts = architecture_utils.get_dense_layer_dimensions(
            num_input_units=4096, num_classes=2,
            num_dense_layers=DENSE_LAYER_COUNTS[i], for_classification=False
        )[1]

        for j in range(num_dropout_rates):
            these_dropout_rates = numpy.full(
                DENSE_LAYER_COUNTS[i], DENSE_LAYER_DROPOUT_RATES[j]
            )
            these_dropout_rates[-1] = numpy.nan

            for k in range(num_loss_function_weights):
                this_option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)

                this_option_dict[
                    u_net_architecture.DENSE_LAYER_NEURON_NUMS_KEY
                ] = these_neuron_counts

                this_option_dict[
                    u_net_architecture.DENSE_LAYER_DROPOUT_RATES_KEY
                ] = these_dropout_rates

                print(this_option_dict)
                print(SEPARATOR_STRING)

                this_scalar_loss_function = (
                    custom_losses.scaled_mse(SCALAR_LOSS_FUNCTION_WEIGHTS[k]) +
                    custom_losses.scaled_mse_for_net_flux(
                        SCALAR_LOSS_FUNCTION_WEIGHTS[k]
                    )
                )
                this_loss_dict = {
                    'conv_output': VECTOR_LOSS_FUNCTION,
                    'dense_output': this_scalar_loss_function
                }

                this_model_object = u_net_architecture.create_model(
                    option_dict=this_option_dict,
                    vector_loss_function=VECTOR_LOSS_FUNCTION,
                    num_output_channels=1,
                    scalar_loss_function=this_scalar_loss_function
                )

                this_model_file_name = (
                    '{0:s}/model_num-dense-layers={1:d}_dense-dropout={2:.3f}_'
                    'scalar-lf-weight={3:05.1f}.h5'
                ).format(
                    OUTPUT_DIR_NAME, DENSE_LAYER_COUNTS[i],
                    DENSE_LAYER_DROPOUT_RATES[j],
                    SCALAR_LOSS_FUNCTION_WEIGHTS[k]
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
                    net_type_string=neural_net.U_NET_TYPE_STRING,
                    loss_function_or_dict=this_loss_dict,
                    do_early_stopping=True, plateau_lr_multiplier=0.6
                )


if __name__ == '__main__':
    _run()
