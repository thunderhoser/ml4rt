"""Makes U-net++ templates for Exp 1 with spectrally resolved shortwave RT.

Same as actual Experiment 1 but with simpler loss function (equal band weights).
"""

import os
import sys
import copy
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import u_net_pp_architecture as u_net_pp_arch
import architecture_utils
import custom_losses
import file_system_utils
import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

HOME_DIR_NAME = '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist'
OUTPUT_DIR_NAME = (
    '{0:s}/ml4rt_models/spectral_experiment01_simple/shortwave_plusplus/templates'
).format(HOME_DIR_NAME)

VECTOR_LOSS_FUNCTION = custom_losses.dual_weighted_mse_constrained_bb(
    band_weights=None
)
SCALAR_LOSS_FUNCTION = custom_losses.scaled_mse_for_net_flux_constrained_bb(
    scaling_factor=1., band_weights=None
)
VECTOR_LOSS_FUNCTION_STRING = (
    'custom_losses.dual_weighted_mse_constrained_bb('
    'band_weights=None'
    ')'
)
SCALAR_LOSS_FUNCTION_STRING = (
    'custom_losses.scaled_mse_for_net_flux_constrained_bb('
    'scaling_factor=1., '
    'band_weights=None'
    ')'
)
LOSS_DICT = {
    'conv_output': VECTOR_LOSS_FUNCTION_STRING,
    'dense_output': SCALAR_LOSS_FUNCTION_STRING
}

MODEL_DEPTHS = numpy.array([3, 4, 5], dtype=int)
CONV_LAYER_COUNTS = numpy.array([1, 2, 3, 4], dtype=int)
FIRST_LAYER_CHANNEL_COUNTS = numpy.array([4, 8, 16, 32, 64, 128], dtype=int)

DEFAULT_OPTION_DICT = {
    u_net_pp_arch.INPUT_DIMENSIONS_KEY:
        numpy.array([127, 26], dtype=int),
    # u_net_pp_arch.NUM_LEVELS_KEY: NUM_LEVELS,
    # u_net_pp_arch.CONV_LAYER_COUNTS_KEY:
    #     numpy.full(NUM_LEVELS + 1, 2, dtype=int),
    # u_net_pp_arch.CHANNEL_COUNTS_KEY: numpy.round(
    #     numpy.logspace(6, 10, num=NUM_LEVELS + 1, base=2.)
    # ).astype(int),
    # u_net_pp_arch.ENCODER_DROPOUT_RATES_KEY:
    #     numpy.full(NUM_LEVELS + 1, 0.),
    # u_net_pp_arch.UPCONV_DROPOUT_RATES_KEY: numpy.full(NUM_LEVELS, 0.),
    # u_net_pp_arch.SKIP_DROPOUT_RATES_KEY: numpy.full(NUM_LEVELS, 0.),
    u_net_pp_arch.INCLUDE_PENULTIMATE_KEY: False,
    u_net_pp_arch.INNER_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    u_net_pp_arch.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    u_net_pp_arch.CONV_OUTPUT_ACTIV_FUNC_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    u_net_pp_arch.CONV_OUTPUT_ACTIV_FUNC_ALPHA_KEY: 0.,
    u_net_pp_arch.DENSE_OUTPUT_ACTIV_FUNC_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    u_net_pp_arch.DENSE_OUTPUT_ACTIV_FUNC_ALPHA_KEY: 0.,
    u_net_pp_arch.L1_WEIGHT_KEY: 0.,
    u_net_pp_arch.L2_WEIGHT_KEY: 1e-7,
    u_net_pp_arch.USE_BATCH_NORM_KEY: True,
    u_net_pp_arch.USE_RESIDUAL_BLOCKS_KEY: False,
    # u_net_pp_arch.DENSE_LAYER_NEURON_NUMS_KEY: DENSE_LAYER_NEURON_COUNTS,
    u_net_pp_arch.DENSE_LAYER_DROPOUT_RATES_KEY: numpy.full(4, 0.),
    u_net_pp_arch.DENSE_LAYER_MC_DROPOUT_FLAGS_KEY: numpy.full(
        4, False, dtype=bool
    ),
    u_net_pp_arch.NUM_OUTPUT_WAVELENGTHS_KEY: 14,
    u_net_pp_arch.VECTOR_LOSS_FUNCTION_KEY: VECTOR_LOSS_FUNCTION,
    u_net_pp_arch.SCALAR_LOSS_FUNCTION_KEY: SCALAR_LOSS_FUNCTION,
    u_net_pp_arch.USE_DEEP_SUPERVISION_KEY: False,
    u_net_pp_arch.ENSEMBLE_SIZE_KEY: 1,
    u_net_pp_arch.DO_INLINE_NORMALIZATION_KEY: False
}

DUMMY_GENERATOR_OPTION_DICT = {
    neural_net.VECTOR_TARGET_NORM_TYPE_KEY: None,
    neural_net.SCALAR_TARGET_NORM_TYPE_KEY: None
}


def _run():
    """Makes U-net++ templates for Exp 1 with spectrally resolved shortwave RT.

    This is effectively the main method.
    """

    num_depths = len(MODEL_DEPTHS)
    num_conv_layer_counts = len(CONV_LAYER_COUNTS)
    num_channel_counts = len(FIRST_LAYER_CHANNEL_COUNTS)

    for i in range(num_depths):
        for j in range(num_conv_layer_counts):
            for k in range(num_channel_counts):
                this_option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)

                these_channel_counts_all = (
                    FIRST_LAYER_CHANNEL_COUNTS[k] *
                    numpy.logspace(
                        0, MODEL_DEPTHS[i], num=MODEL_DEPTHS[i] + 1, base=2.
                    )
                )

                these_channel_counts_all = (
                    numpy.round(these_channel_counts_all).astype(int)
                )

                this_option_dict.update({
                    u_net_pp_arch.NUM_LEVELS_KEY: MODEL_DEPTHS[i],
                    u_net_pp_arch.CONV_LAYER_COUNTS_KEY: numpy.full(
                        MODEL_DEPTHS[i] + 1, CONV_LAYER_COUNTS[j], dtype=int
                    ),
                    u_net_pp_arch.CHANNEL_COUNTS_KEY: these_channel_counts_all,
                    u_net_pp_arch.ENCODER_DROPOUT_RATES_KEY: numpy.full(
                        MODEL_DEPTHS[i] + 1, 0.
                    ),
                    u_net_pp_arch.ENCODER_MC_DROPOUT_FLAGS_KEY: numpy.full(
                        MODEL_DEPTHS[i] + 1, False, dtype=bool
                    ),
                    u_net_pp_arch.UPCONV_DROPOUT_RATES_KEY: numpy.full(
                        MODEL_DEPTHS[i], 0.
                    ),
                    u_net_pp_arch.UPCONV_MC_DROPOUT_FLAGS_KEY: numpy.full(
                        MODEL_DEPTHS[i], False, dtype=bool
                    ),
                    u_net_pp_arch.SKIP_DROPOUT_RATES_KEY: numpy.full(
                        MODEL_DEPTHS[i], 0.
                    ),
                    u_net_pp_arch.SKIP_MC_DROPOUT_FLAGS_KEY: numpy.full(
                        MODEL_DEPTHS[i], False, dtype=bool
                    )
                })

                if MODEL_DEPTHS[i] == 3:
                    multiplier = 15
                elif MODEL_DEPTHS[i] == 4:
                    multiplier = 7
                else:
                    multiplier = 3

                these_neuron_counts = (
                    architecture_utils.get_dense_layer_dimensions(
                        num_input_units=int(numpy.round(
                            multiplier * these_channel_counts_all[-1]
                        )),
                        num_classes=2,
                        num_dense_layers=4,
                        for_classification=False
                    )[1]
                )

                these_neuron_counts[-1] = 2 * 14
                these_neuron_counts[-2] = max([
                    these_neuron_counts[-2], these_neuron_counts[-1]
                ])

                this_option_dict[u_net_pp_arch.DENSE_LAYER_NEURON_NUMS_KEY] = (
                    these_neuron_counts
                )
                this_model_object = u_net_pp_arch.create_model(this_option_dict)

                this_model_file_name = (
                    '{0:s}/depth={1:d}_num-conv-layers-per-block={2:d}_'
                    'num-first-layer-channels={3:03d}/model.keras'
                ).format(
                    OUTPUT_DIR_NAME,
                    MODEL_DEPTHS[i],
                    CONV_LAYER_COUNTS[j],
                    FIRST_LAYER_CHANNEL_COUNTS[k]
                )

                file_system_utils.mkdir_recursive_if_necessary(
                    file_name=this_model_file_name
                )

                print('Writing model to: "{0:s}"...'.format(
                    this_model_file_name
                ))
                this_model_object.save(
                    filepath=this_model_file_name,
                    overwrite=True,
                    include_optimizer=True
                )

                this_metafile_name = neural_net.find_metafile(
                    model_dir_name=os.path.split(this_model_file_name)[0],
                    raise_error_if_missing=False
                )

                this_option_dict[u_net_pp_arch.VECTOR_LOSS_FUNCTION_KEY] = (
                    VECTOR_LOSS_FUNCTION_STRING
                )
                this_option_dict[u_net_pp_arch.SCALAR_LOSS_FUNCTION_KEY] = (
                    SCALAR_LOSS_FUNCTION_STRING
                )

                print('Writing metadata to: "{0:s}"...'.format(
                    this_metafile_name
                ))
                neural_net._write_metafile(
                    dill_file_name=this_metafile_name,
                    num_epochs=100,
                    num_training_batches_per_epoch=100,
                    training_option_dict=DUMMY_GENERATOR_OPTION_DICT,
                    num_validation_batches_per_epoch=100,
                    validation_option_dict=DUMMY_GENERATOR_OPTION_DICT,
                    loss_function_or_dict=LOSS_DICT,
                    do_early_stopping=True,
                    plateau_lr_multiplier=0.6,
                    u_net_3plus_architecture_dict=None,
                    u_net_plusplus_architecture_dict=this_option_dict,
                    bnn_architecture_dict=None
                )


if __name__ == '__main__':
    _run()
