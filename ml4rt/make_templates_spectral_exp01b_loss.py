"""Makes U-net++ templates for Exp 1b with spectrally resolved shortwave RT.

Same as Experiment 1, except that I'm optimizing loss function instead of
architecture.
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
    '{0:s}/ml4rt_models/spectral_experiment01b_loss/shortwave_plusplus/'
    'templates'
).format(HOME_DIR_NAME)

# The fourteen band centers (in microns) are:
# 5.85, 3.42, 2.76, 2.31, 2.04, 1.77, 1.44, 1.27, 0.96, 0.69, 0.52, 0.39, 0.30,
# 0.23

# BAND_WEIGHTS = numpy.array([
#     1, 4, 500, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5
# ], dtype=float)
#
# VECTOR_LOSS_FUNCTION = custom_losses.dual_weighted_mse_constrained_bb(
#     band_weights=BAND_WEIGHTS
# )
# SCALAR_LOSS_FUNCTION = custom_losses.scaled_mse_for_net_flux_constrained_bb(
#     scaling_factor=1., band_weights=BAND_WEIGHTS
# )
# VECTOR_LOSS_FUNCTION_STRING = (
#     'custom_losses.dual_weighted_mse_constrained_bb('
#     'band_weights='
#     'numpy.array([1, 4, 500, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5], dtype=float)'
#     ')'
# )
# SCALAR_LOSS_FUNCTION_STRING = (
#     'custom_losses.scaled_mse_for_net_flux_constrained_bb('
#     'scaling_factor=1., '
#     'band_weights='
#     'numpy.array([1, 4, 500, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5], dtype=float)'
#     ')'
# )
# LOSS_DICT = {
#     'conv_output': VECTOR_LOSS_FUNCTION_STRING,
#     'dense_output': SCALAR_LOSS_FUNCTION_STRING
# }

# MODEL_DEPTHS = numpy.array([3, 4, 5], dtype=int)
# CONV_LAYER_COUNTS = numpy.array([1, 2, 3, 4], dtype=int)
# FIRST_LAYER_CHANNEL_COUNTS = numpy.array([4, 8, 16, 32, 64, 128], dtype=int)

WEAK_BAND_WEIGHTS = numpy.array([1, 3, 5, 7, 9], dtype=float)
VERY_WEAK_BAND_WEIGHTS = numpy.array([1, 5, 10, 15, 20], dtype=float)

MODEL_DEPTH = 3
CONV_LAYER_COUNT = 1
FIRST_LAYER_CHANNEL_COUNT = 128
NUM_BANDS = 14

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
    # u_net_pp_arch.DENSE_LAYER_NEURON_NUMS_KEY: DENSE_LAYER_NEURON_COUNTS,
    u_net_pp_arch.DENSE_LAYER_DROPOUT_RATES_KEY: numpy.full(4, 0.),
    u_net_pp_arch.DENSE_LAYER_MC_DROPOUT_FLAGS_KEY: numpy.full(
        4, False, dtype=bool
    ),
    u_net_pp_arch.NUM_OUTPUT_WAVELENGTHS_KEY: NUM_BANDS,
    # u_net_pp_arch.VECTOR_LOSS_FUNCTION_KEY: VECTOR_LOSS_FUNCTION,
    # u_net_pp_arch.SCALAR_LOSS_FUNCTION_KEY: SCALAR_LOSS_FUNCTION,
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

    all_channel_counts = FIRST_LAYER_CHANNEL_COUNT * numpy.logspace(
        0, MODEL_DEPTH, num=MODEL_DEPTH + 1, base=2.
    )
    all_channel_counts = numpy.round(all_channel_counts).astype(int)

    default_option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)

    default_option_dict.update({
        u_net_pp_arch.NUM_LEVELS_KEY: MODEL_DEPTH,
        u_net_pp_arch.CONV_LAYER_COUNTS_KEY: numpy.full(
            MODEL_DEPTH + 1, CONV_LAYER_COUNT, dtype=int
        ),
        u_net_pp_arch.CHANNEL_COUNTS_KEY: all_channel_counts,
        u_net_pp_arch.ENCODER_DROPOUT_RATES_KEY: numpy.full(
            MODEL_DEPTH + 1, 0.
        ),
        u_net_pp_arch.ENCODER_MC_DROPOUT_FLAGS_KEY: numpy.full(
            MODEL_DEPTH + 1, False, dtype=bool
        ),
        u_net_pp_arch.UPCONV_DROPOUT_RATES_KEY: numpy.full(
            MODEL_DEPTH, 0.
        ),
        u_net_pp_arch.UPCONV_MC_DROPOUT_FLAGS_KEY: numpy.full(
            MODEL_DEPTH, False, dtype=bool
        ),
        u_net_pp_arch.SKIP_DROPOUT_RATES_KEY: numpy.full(
            MODEL_DEPTH, 0.
        ),
        u_net_pp_arch.SKIP_MC_DROPOUT_FLAGS_KEY: numpy.full(
            MODEL_DEPTH, False, dtype=bool
        )
    })

    if MODEL_DEPTH == 3:
        multiplier = 15
    elif MODEL_DEPTH == 4:
        multiplier = 7
    else:
        multiplier = 3

    dense_neuron_counts = architecture_utils.get_dense_layer_dimensions(
        num_input_units=int(numpy.round(multiplier * all_channel_counts[-1])),
        num_classes=2,
        num_dense_layers=4,
        for_classification=False
    )[1]

    dense_neuron_counts[-1] = 2 * NUM_BANDS
    dense_neuron_counts[-2] = max([
        dense_neuron_counts[-2], dense_neuron_counts[-1]
    ])
    default_option_dict[u_net_pp_arch.DENSE_LAYER_NEURON_NUMS_KEY] = (
        dense_neuron_counts
    )

    for i in range(len(WEAK_BAND_WEIGHTS)):
        for j in range(len(VERY_WEAK_BAND_WEIGHTS)):
            this_option_dict = copy.deepcopy(default_option_dict)

            these_band_weights = numpy.array([
                1, WEAK_BAND_WEIGHTS[i], VERY_WEAK_BAND_WEIGHTS[j],
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                WEAK_BAND_WEIGHTS[i]
            ], dtype=float)

            this_vector_loss_function = (
                custom_losses.dual_weighted_mse_constrained_bb(
                    band_weights=these_band_weights
                )
            )
            this_scalar_loss_function = (
                custom_losses.scaled_mse_for_net_flux_constrained_bb(
                    scaling_factor=1., band_weights=these_band_weights
                )
            )

            this_vector_loss_string = (
                'custom_losses.dual_weighted_mse_constrained_bb('
                'band_weights='
                'numpy.array([1, {0:.0f}, {1:.0f}, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, {0:.0f}], dtype=float)'
                ')'
            ).format(
                WEAK_BAND_WEIGHTS[i], VERY_WEAK_BAND_WEIGHTS[j]
            )

            this_scalar_loss_string = (
                'custom_losses.scaled_mse_for_net_flux_constrained_bb('
                'scaling_factor=1., '
                'band_weights='
                'numpy.array([1, {0:.0f}, {1:.0f}, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, {0:.0f}], dtype=float)'
                ')'
            ).format(
                WEAK_BAND_WEIGHTS[i], VERY_WEAK_BAND_WEIGHTS[j]
            )




    for i in range(num_depths):
        for j in range(num_conv_layer_counts):
            for k in range(num_channel_counts):
                this_option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)


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
