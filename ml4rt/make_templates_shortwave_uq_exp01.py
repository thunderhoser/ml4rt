"""Makes templates for shortwave experiment with uncertainty quantification."""

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
    '{0:s}/ml4rt_models/shortwave_uq_experiment01/templates'
).format(HOME_DIR_NAME)

ENSEMBLE_SIZE = 100

MODEL_DEPTH = 5
FIRST_LAYER_CHANNEL_COUNTS = numpy.array([32, 64, 128], dtype=int)
UQ_METHOD_STRINGS = ['mc-dropout', 'crps', 'mc-crps']
DENSE_LAYER_DROPOUT_RATES = numpy.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])

DEFAULT_OPTION_DICT = {
    u_net_pp_architecture.INPUT_DIMENSIONS_KEY:
        numpy.array([127, 26], dtype=int),
    # u_net_pp_architecture.NUM_LEVELS_KEY: NUM_LEVELS,
    # u_net_pp_architecture.CONV_LAYER_COUNTS_KEY:
    #     numpy.full(NUM_LEVELS + 1, 2, dtype=int),
    # u_net_pp_architecture.CHANNEL_COUNTS_KEY: numpy.round(
    #     numpy.logspace(6, 10, num=NUM_LEVELS + 1, base=2.)
    # ).astype(int),
    # u_net_pp_architecture.ENCODER_DROPOUT_RATES_KEY:
    #     numpy.full(NUM_LEVELS + 1, 0.),
    # u_net_pp_architecture.UPCONV_DROPOUT_RATES_KEY: numpy.full(NUM_LEVELS, 0.),
    # u_net_pp_architecture.SKIP_DROPOUT_RATES_KEY: numpy.full(NUM_LEVELS, 0.),
    u_net_pp_architecture.INCLUDE_PENULTIMATE_KEY: False,
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
    u_net_pp_architecture.USE_BATCH_NORM_KEY: True,
    # u_net_pp_architecture.DENSE_LAYER_NEURON_NUMS_KEY: DENSE_LAYER_NEURON_COUNTS,
    # u_net_pp_architecture.DENSE_LAYER_DROPOUT_RATES_KEY: numpy.full(4, 0.)
}

DUMMY_GENERATOR_OPTION_DICT = {
    neural_net.VECTOR_TARGET_NORM_TYPE_KEY: None,
    neural_net.SCALAR_TARGET_NORM_TYPE_KEY: None
}


def _run():
    """Makes templates for shortwave experiment with uncertainty quantification.

    This is effectively the main method.
    """

    num_channel_counts = len(FIRST_LAYER_CHANNEL_COUNTS)
    num_uq_methods = len(UQ_METHOD_STRINGS)
    num_dropout_rates = len(DENSE_LAYER_DROPOUT_RATES)

    for i in range(num_channel_counts):
        for j in range(num_uq_methods):
            for k in range(num_dropout_rates):
                this_option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)

                these_channel_counts_all = (
                    FIRST_LAYER_CHANNEL_COUNTS[i] *
                    numpy.logspace(0, MODEL_DEPTH, num=MODEL_DEPTH + 1, base=2.)
                )

                these_channel_counts_all = (
                    numpy.round(these_channel_counts_all).astype(int)
                )

                this_option_dict.update({
                    u_net_pp_architecture.NUM_LEVELS_KEY: MODEL_DEPTH,
                    u_net_pp_architecture.CONV_LAYER_COUNTS_KEY:
                        numpy.full(MODEL_DEPTH + 1, 1, dtype=int),
                    u_net_pp_architecture.CHANNEL_COUNTS_KEY:
                        these_channel_counts_all,
                    u_net_pp_architecture.ENCODER_DROPOUT_RATES_KEY:
                        numpy.full(MODEL_DEPTH + 1, 0.),
                    u_net_pp_architecture.ENCODER_MC_DROPOUT_FLAGS_KEY:
                        numpy.full(MODEL_DEPTH + 1, 0, dtype=bool),
                    u_net_pp_architecture.UPCONV_DROPOUT_RATES_KEY:
                        numpy.full(MODEL_DEPTH, 0.),
                    u_net_pp_architecture.UPCONV_MC_DROPOUT_FLAGS_KEY:
                        numpy.full(MODEL_DEPTH, 0, dtype=bool),
                    u_net_pp_architecture.SKIP_DROPOUT_RATES_KEY:
                        numpy.full(MODEL_DEPTH, 0.),
                    u_net_pp_architecture.SKIP_MC_DROPOUT_FLAGS_KEY:
                        numpy.full(MODEL_DEPTH, 0, dtype=bool)
                })

                these_neuron_counts = (
                    architecture_utils.get_dense_layer_dimensions(
                        num_input_units=int(numpy.round(
                            3 * these_channel_counts_all[-1]
                        )),
                        num_classes=2,
                        num_dense_layers=4,
                        for_classification=False
                    )[1]
                )

                if 'crps' in UQ_METHOD_STRINGS[j]:
                    these_neuron_counts[-1] = ENSEMBLE_SIZE + 0
                else:
                    these_neuron_counts[-1] = 1

                these_dropout_rates = numpy.array([
                    DENSE_LAYER_DROPOUT_RATES[k], DENSE_LAYER_DROPOUT_RATES[k],
                    DENSE_LAYER_DROPOUT_RATES[k], 0.
                ])
                these_mc_dropout_flags = numpy.array([
                    'mc-' in UQ_METHOD_STRINGS[j],
                    'mc-' in UQ_METHOD_STRINGS[j],
                    'mc-' in UQ_METHOD_STRINGS[j],
                    0
                ], dtype=bool)

                this_option_dict.update({
                    u_net_pp_architecture.DENSE_LAYER_NEURON_NUMS_KEY:
                        these_neuron_counts,
                    u_net_pp_architecture.DENSE_LAYER_DROPOUT_RATES_KEY:
                        these_dropout_rates,
                    u_net_pp_architecture.DENSE_LAYER_MC_DROPOUT_FLAGS_KEY:
                        these_mc_dropout_flags
                })

                if 'crps' in UQ_METHOD_STRINGS[j]:
                    vector_loss_function = custom_losses.dual_weighted_crps()
                    scalar_loss_function = (
                        custom_losses.unscaled_crps_for_net_flux()
                    )
                    loss_dict = {
                        'conv_output': 'custom_losses.dual_weighted_crps()',
                        'dense_output':
                            'custom_losses.unscaled_crps_for_net_flux()'
                    }
                else:
                    vector_loss_function = custom_losses.dual_weighted_mse()
                    scalar_loss_function = (
                        custom_losses.scaled_mse_for_net_flux(1.)
                    )
                    loss_dict = {
                        'conv_output': 'custom_losses.dual_weighted_mse()',
                        'dense_output':
                            'custom_losses.scaled_mse_for_net_flux(1.)'
                    }

                this_model_object = u_net_pp_architecture.create_model(
                    option_dict=this_option_dict,
                    vector_loss_function=vector_loss_function,
                    use_deep_supervision=False, num_output_channels=1,
                    scalar_loss_function=scalar_loss_function,
                    ensemble_size=these_neuron_counts[-1]
                )

                this_model_file_name = (
                    '{0:s}/{1:s}_num-first-layer-channels={2:03d}_'
                    'dense-layer-dropout-rate={3:.1f}/model.h5'
                ).format(
                    OUTPUT_DIR_NAME, UQ_METHOD_STRINGS[j],
                    FIRST_LAYER_CHANNEL_COUNTS[i], DENSE_LAYER_DROPOUT_RATES[k]
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
