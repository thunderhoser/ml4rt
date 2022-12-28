"""Makes templates for second shortwave experiment with UQ."""

import os
import sys
import copy
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import u_net_pp_architecture
import architecture_utils
import custom_losses
import file_system_utils
import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

HOME_DIR_NAME = '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist'
OUTPUT_DIR_NAME = (
    '{0:s}/ml4rt_models/shortwave_uq_experiment02/templates'
).format(HOME_DIR_NAME)

ENSEMBLE_SIZE = 25
MODEL_DEPTH = 5

DENSE_LAYER_DROPOUT_RATES = numpy.array([0.1, 0.2, 0.3, 0.4, 0.5])
CONV_LAYER_DROPOUT_RATES = numpy.array([0.1, 0.2, 0.3, 0.4, 0.5])
CONV_LAYERS_WITH_DROPOUT_COUNTS = numpy.array([1, 2, 3, 4, 5], dtype=int)

DEFAULT_OPTION_DICT = {
    u_net_pp_architecture.INPUT_DIMENSIONS_KEY:
        numpy.array([127, 26], dtype=int),
    u_net_pp_architecture.NUM_LEVELS_KEY: MODEL_DEPTH,
    u_net_pp_architecture.CONV_LAYER_COUNTS_KEY:
        numpy.full(MODEL_DEPTH + 1, 1, dtype=int),
    u_net_pp_architecture.CHANNEL_COUNTS_KEY: numpy.round(
        numpy.logspace(5, 5 + MODEL_DEPTH, num=MODEL_DEPTH + 1, base=2.)
    ).astype(int),

    u_net_pp_architecture.ENCODER_DROPOUT_RATES_KEY:
        numpy.full(MODEL_DEPTH + 1, 0.),
    u_net_pp_architecture.ENCODER_MC_DROPOUT_FLAGS_KEY:
        numpy.full(MODEL_DEPTH + 1, 0, dtype=bool),
    u_net_pp_architecture.UPCONV_DROPOUT_RATES_KEY:
        numpy.full(MODEL_DEPTH, 0.),
    u_net_pp_architecture.UPCONV_MC_DROPOUT_FLAGS_KEY:
        numpy.full(MODEL_DEPTH, 0, dtype=bool),
    u_net_pp_architecture.SKIP_MC_DROPOUT_FLAGS_KEY:
        numpy.full(MODEL_DEPTH, 1, dtype=bool),
    u_net_pp_architecture.INCLUDE_PENULTIMATE_KEY: True,
    u_net_pp_architecture.PENULTIMATE_MC_DROPOUT_FLAG_KEY: True,
    u_net_pp_architecture.INNER_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    u_net_pp_architecture.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    u_net_pp_architecture.CONV_OUTPUT_ACTIV_FUNC_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    u_net_pp_architecture.CONV_OUTPUT_ACTIV_FUNC_ALPHA_KEY: 0.,
    u_net_pp_architecture.DENSE_OUTPUT_ACTIV_FUNC_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    u_net_pp_architecture.DENSE_OUTPUT_ACTIV_FUNC_ALPHA_KEY: 0.,
    u_net_pp_architecture.L1_WEIGHT_KEY: 0.,
    u_net_pp_architecture.L2_WEIGHT_KEY: 1e-7,
    u_net_pp_architecture.USE_BATCH_NORM_KEY: True
}

DUMMY_GENERATOR_OPTION_DICT = {
    neural_net.VECTOR_TARGET_NORM_TYPE_KEY: None,
    neural_net.SCALAR_TARGET_NORM_TYPE_KEY: None
}


def _run():
    """Makes templates for second shortwave experiment with UQ.

    This is effectively the main method.
    """

    num_dense_dropout_rates = len(DENSE_LAYER_DROPOUT_RATES)
    num_conv_dropout_rates = len(CONV_LAYER_DROPOUT_RATES)
    num_layer_counts = len(CONV_LAYERS_WITH_DROPOUT_COUNTS)

    for i in range(num_dense_dropout_rates):
        for j in range(num_conv_dropout_rates):
            for k in range(num_layer_counts):
                this_option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)

                these_skip_layer_dropout_rates = numpy.full(MODEL_DEPTH, 0.)
                these_skip_layer_dropout_rates[
                    :(CONV_LAYERS_WITH_DROPOUT_COUNTS[k] - 1)
                ] = CONV_LAYER_DROPOUT_RATES[j]

                this_option_dict.update({
                    u_net_pp_architecture.PENULTIMATE_DROPOUT_RATE_KEY:
                        CONV_LAYER_DROPOUT_RATES[j],
                    u_net_pp_architecture.SKIP_DROPOUT_RATES_KEY:
                        these_skip_layer_dropout_rates
                })

                max_num_channels = this_option_dict[
                    u_net_pp_architecture.CHANNEL_COUNTS_KEY
                ][-1]

                these_neuron_counts = (
                    architecture_utils.get_dense_layer_dimensions(
                        num_input_units=int(numpy.round(
                            3 * max_num_channels
                        )),
                        num_classes=2,
                        num_dense_layers=4,
                        for_classification=False
                    )[1]
                )

                these_neuron_counts[-1] = 2 * ENSEMBLE_SIZE
                these_neuron_counts[-2] = max([
                    these_neuron_counts[-2], these_neuron_counts[-1]
                ])

                these_dropout_rates = numpy.array([
                    DENSE_LAYER_DROPOUT_RATES[k], DENSE_LAYER_DROPOUT_RATES[k],
                    DENSE_LAYER_DROPOUT_RATES[k], 0.
                ])
                these_mc_dropout_flags = numpy.array([1, 1, 1, 0], dtype=bool)

                this_option_dict.update({
                    u_net_pp_architecture.DENSE_LAYER_NEURON_NUMS_KEY:
                        these_neuron_counts,
                    u_net_pp_architecture.DENSE_LAYER_DROPOUT_RATES_KEY:
                        these_dropout_rates,
                    u_net_pp_architecture.DENSE_LAYER_MC_DROPOUT_FLAGS_KEY:
                        these_mc_dropout_flags
                })

                vector_loss_function = custom_losses.dual_weighted_crps()
                scalar_loss_function = (
                    custom_losses.unscaled_crps_for_net_flux()
                )
                loss_dict = {
                    'conv_output': 'custom_losses.dual_weighted_crps()',
                    'dense_output':
                        'custom_losses.unscaled_crps_for_net_flux()'
                }

                this_model_object = u_net_pp_architecture.create_model(
                    option_dict=this_option_dict,
                    vector_loss_function=vector_loss_function,
                    use_deep_supervision=False, num_output_channels=1,
                    scalar_loss_function=scalar_loss_function,
                    ensemble_size=ENSEMBLE_SIZE
                )

                this_model_file_name = (
                    '{0:s}/dense-layer-dropout-rate={1:.2f}_'
                    'conv-layer-dropout-rate={2:.2f}_'
                    'num-conv-layers-with-dropout={3:d}/model.h5'
                ).format(
                    OUTPUT_DIR_NAME,
                    DENSE_LAYER_DROPOUT_RATES[i],
                    CONV_LAYER_DROPOUT_RATES[j],
                    CONV_LAYERS_WITH_DROPOUT_COUNTS[k]
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
                    net_type_string=neural_net.U_NET_TYPE_STRING,
                    loss_function_or_dict=loss_dict,
                    do_early_stopping=True, plateau_lr_multiplier=0.6
                )


if __name__ == '__main__':
    _run()
