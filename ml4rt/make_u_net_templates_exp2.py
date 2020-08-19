"""Makes U-net templates for Experiment 2."""

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
import file_system_utils
import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

# if 'hfe' in socket.gethostname():
#     HOME_DIR_NAME = '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist'
# else:
#     HOME_DIR_NAME = os.path.expanduser('~')

HOME_DIR_NAME = '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist'
OUTPUT_DIR_NAME = '{0:s}/ml4rt_models/experiment02/templates'.format(
    HOME_DIR_NAME
)

CONV_LAYER_DROPOUT_RATES = numpy.array([0.3, 0.4, 0.5, 0.6, 0.7])
UPCONV_LAYER_DROPOUT_RATES = numpy.array([-1, 0.3, 0.4, 0.5, 0.6, 0.7])
SKIP_LAYER_DROPOUT_RATES = numpy.array([-1, 0.3, 0.4, 0.5, 0.6, 0.7])

NUM_LEVELS = 4
LOSS_FUNCTION = custom_losses.dual_weighted_mse()

DEFAULT_OPTION_DICT = {
    u_net_architecture.NUM_LEVELS_KEY: NUM_LEVELS,
    u_net_architecture.NUM_HEIGHTS_KEY: 73,
    u_net_architecture.NUM_HEIGHTS_FOR_LOSS_KEY: 73,
    u_net_architecture.NUM_INPUT_CHANNELS_KEY: 16,
    u_net_architecture.OUTPUT_LAYER_DROPOUT_RATES_KEY: numpy.full(2, numpy.nan),
    u_net_architecture.DENSE_LAYER_NEURON_NUMS_KEY: None,
    u_net_architecture.DENSE_LAYER_DROPOUT_RATES_KEY: None,
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

DUMMY_GENERATOR_OPTION_DICT = {neural_net.VECTOR_TARGET_NORM_TYPE_KEY: None}


def _run():
    """Makes U-net templates for Experiment 2.

    This is effectively the main method.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=OUTPUT_DIR_NAME
    )

    num_conv_dropout_rates = len(CONV_LAYER_DROPOUT_RATES)
    num_upconv_dropout_rates = len(UPCONV_LAYER_DROPOUT_RATES)
    num_skip_dropout_rates = len(SKIP_LAYER_DROPOUT_RATES)

    for i in range(num_conv_dropout_rates):
        for j in range(num_upconv_dropout_rates):
            for k in range(num_skip_dropout_rates):
                this_option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)

                this_option_dict[
                    u_net_architecture.CONV_LAYER_DROPOUT_RATES_KEY
                ] = numpy.full(NUM_LEVELS + 1, CONV_LAYER_DROPOUT_RATES[i])

                this_option_dict[
                    u_net_architecture.UPCONV_LAYER_DROPOUT_RATES_KEY
                ] = numpy.full(NUM_LEVELS, UPCONV_LAYER_DROPOUT_RATES[j])

                this_option_dict[
                    u_net_architecture.SKIP_LAYER_DROPOUT_RATES_KEY
                ] = numpy.full(NUM_LEVELS, SKIP_LAYER_DROPOUT_RATES[k])

                print(this_option_dict)
                print(SEPARATOR_STRING)

                this_model_object = u_net_architecture.create_model(
                    option_dict=this_option_dict,
                    vector_loss_function=LOSS_FUNCTION, num_output_channels=1
                )

                this_model_file_name = (
                    '{0:s}/model_conv-dropout={1:.1f}_upconv-dropout={2:.1f}_'
                    'skip-dropout={3:.1f}.h5'
                ).format(
                    OUTPUT_DIR_NAME, CONV_LAYER_DROPOUT_RATES[i],
                    UPCONV_LAYER_DROPOUT_RATES[j], SKIP_LAYER_DROPOUT_RATES[k]
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
                    loss_function_or_dict=LOSS_FUNCTION,
                    do_early_stopping=True, plateau_lr_multiplier=0.6
                )


if __name__ == '__main__':
    _run()
