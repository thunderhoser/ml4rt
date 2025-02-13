"""Makes final templates for Spectral Experiment 6.

Trying to make fucking ConvNeXt blocks work.
"""

import os
import sys
import copy
import numpy
import keras

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
    '{0:s}/ml4rt_models/spectral_experiment05_final_simple_norm/templates'
).format(HOME_DIR_NAME)

ENSEMBLE_SIZE = 1
NUM_WAVELENGTHS = 14

MODEL_DEPTH = 3
NUM_CONV_LAYERS_PER_BLOCK = 1
NUM_FIRST_LAYER_CHANNELS = 128

ALL_CHANNEL_COUNTS = (
    NUM_FIRST_LAYER_CHANNELS *
    numpy.logspace(0, MODEL_DEPTH, num=MODEL_DEPTH + 1, base=2.)
)
ALL_CHANNEL_COUNTS = numpy.round(ALL_CHANNEL_COUNTS).astype(int)

DENSE_LAYER_NEURON_COUNTS = architecture_utils.get_dense_layer_dimensions(
    num_input_units=int(numpy.round(15 * ALL_CHANNEL_COUNTS[-1])),
    num_classes=2 * ENSEMBLE_SIZE * NUM_WAVELENGTHS,
    num_dense_layers=4,
    for_classification=False
)[1]

DENSE_LAYER_NEURON_COUNTS[-2] = max([
    DENSE_LAYER_NEURON_COUNTS[-2],
    2 * DENSE_LAYER_NEURON_COUNTS[-1]
])

DEFAULT_OPTION_DICT = {
    u_net_pp_arch.INPUT_DIMENSIONS_KEY: numpy.array([127, 26], dtype=int),
    u_net_pp_arch.NUM_LEVELS_KEY: MODEL_DEPTH,
    u_net_pp_arch.CONV_LAYER_COUNTS_KEY: numpy.full(
        MODEL_DEPTH + 1, NUM_CONV_LAYERS_PER_BLOCK, dtype=int
    ),
    u_net_pp_arch.CHANNEL_COUNTS_KEY: ALL_CHANNEL_COUNTS,
    u_net_pp_arch.ENCODER_DROPOUT_RATES_KEY: numpy.full(MODEL_DEPTH + 1, 0.),
    u_net_pp_arch.ENCODER_MC_DROPOUT_FLAGS_KEY: numpy.full(
        MODEL_DEPTH + 1, False, dtype=bool
    ),
    u_net_pp_arch.UPCONV_DROPOUT_RATES_KEY: numpy.full(MODEL_DEPTH, 0.),
    u_net_pp_arch.UPCONV_MC_DROPOUT_FLAGS_KEY: numpy.full(
        MODEL_DEPTH, False, dtype=bool
    ),
    u_net_pp_arch.SKIP_DROPOUT_RATES_KEY: numpy.full(MODEL_DEPTH, 0.),
    u_net_pp_arch.SKIP_MC_DROPOUT_FLAGS_KEY: numpy.full(
        MODEL_DEPTH, False, dtype=bool
    ),
    u_net_pp_arch.INCLUDE_PENULTIMATE_KEY: False,
    u_net_pp_arch.DENSE_LAYER_NEURON_NUMS_KEY: DENSE_LAYER_NEURON_COUNTS,
    u_net_pp_arch.DENSE_LAYER_DROPOUT_RATES_KEY: numpy.full(4, 0.),
    u_net_pp_arch.DENSE_LAYER_MC_DROPOUT_FLAGS_KEY: numpy.full(
        4, False, dtype=bool
    ),
    u_net_pp_arch.INNER_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    u_net_pp_arch.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    # u_net_pp_arch.CONV_OUTPUT_ACTIV_FUNC_KEY:
    #     architecture_utils.RELU_FUNCTION_STRING,
    u_net_pp_arch.CONV_OUTPUT_ACTIV_FUNC_ALPHA_KEY: 0.,
    u_net_pp_arch.DENSE_OUTPUT_ACTIV_FUNC_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    u_net_pp_arch.DENSE_OUTPUT_ACTIV_FUNC_ALPHA_KEY: 0.,
    u_net_pp_arch.L1_WEIGHT_KEY: 0.,
    u_net_pp_arch.L2_WEIGHT_KEY: 1e-7,
    u_net_pp_arch.USE_BATCH_NORM_KEY: True,
    # u_net_pp_arch.USE_RESIDUAL_BLOCKS_KEY: True,
    u_net_pp_arch.NUM_OUTPUT_WAVELENGTHS_KEY: NUM_WAVELENGTHS,
    # u_net_pp_arch.VECTOR_LOSS_FUNCTION_KEY: VECTOR_LOSS_FUNCTION,
    # u_net_pp_arch.SCALAR_LOSS_FUNCTION_KEY: SCALAR_LOSS_FUNCTION,
    u_net_pp_arch.USE_DEEP_SUPERVISION_KEY: False,
    u_net_pp_arch.ENSEMBLE_SIZE_KEY: ENSEMBLE_SIZE,
    u_net_pp_arch.INCLUDE_MASK_KEY: True
}

DUMMY_GENERATOR_OPTION_DICT = {
    neural_net.NORMALIZE_PREDICTORS_KEY: True,
    neural_net.NORMALIZE_VECTOR_TARGETS_KEY: False,
    neural_net.NORMALIZE_SCALAR_TARGETS_KEY: False
}

MIN_DUAL_WEIGHT = 0.75
BROADBAND_WEIGHT = 0.01

USE_CONVNEXT_V1_FLAGS_AXIS1 = numpy.array([1, 0, 1, 0], dtype=bool)
USE_CONVNEXT_V2_FLAGS_AXIS1 = numpy.array([0, 1, 0, 1], dtype=bool)
SIMPLIFY_CONVNEXT_FLAGS_AXIS1 = numpy.array([0, 0, 1, 1], dtype=bool)
SIMPLIFY_OUTPUT_FLAGS_AXIS2 = numpy.array([0, 1, 1], dtype=bool)
OUTPUT_ACTIV_FUNCTIONS_AXIS2 = [
    None, architecture_utils.RELU_FUNCTION_STRING, None
]
OPTIMIZER_FUNCTIONS_AXIS3 = [
    keras.optimizers.AdamW(),
    keras.optimizers.AdamW(clipnorm=1.),
    keras.optimizers.AdamW(weight_decay=0.0004)
]
OPTIMIZER_FUNCTION_STRINGS_AXIS3 = [
    'keras.optimizers.AdamW()',
    'keras.optimizers.AdamW(clipnorm=1.)',
    'keras.optimizers.AdamW(weight_decay=0.0004)'
]
OPTIMIZER_FUNCTION_DESCRIPTIONS_AXIS3 = [
    'basic', 'clipnorm', 'smaller-weight-decay'
]


def _run():
    """Makes final templates for Spectral Experiment 6.

    This is effectively the main method.
    """

    for i in range(len(SIMPLIFY_CONVNEXT_FLAGS_AXIS1)):
        for j in range(len(OUTPUT_ACTIV_FUNCTIONS_AXIS2)):
            for k in range(len(OPTIMIZER_FUNCTIONS_AXIS3)):
                this_option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)

                this_vector_loss_function = (
                    custom_losses.dual_weighted_mse_constrained_bb(
                        min_dual_weight=MIN_DUAL_WEIGHT,
                        band_weights=numpy.array([
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            BROADBAND_WEIGHT
                        ])
                    )
                )

                this_vector_loss_string = (
                    'custom_losses.dual_weighted_mse_constrained_bb('
                    'min_dual_weight={0:.2f},'
                    'band_weights=numpy.array(['
                    '1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, {1:.2f}'
                    '])'
                    ')'
                ).format(MIN_DUAL_WEIGHT, BROADBAND_WEIGHT)

                this_scalar_loss_function = (
                    custom_losses.scaled_mse_for_net_flux_constrained_bb(
                        scaling_factor=1.,
                        band_weights=numpy.array([
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            BROADBAND_WEIGHT
                        ])
                    )
                )

                this_scalar_loss_string = (
                    'custom_losses.scaled_mse_for_net_flux_constrained_bb('
                    'scaling_factor=1.,'
                    'band_weights=numpy.array(['
                    '1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, {0:.2f}'
                    '])'
                    ')'
                ).format(BROADBAND_WEIGHT)

                this_option_dict.update({
                    u_net_pp_arch.VECTOR_LOSS_FUNCTION_KEY:
                        this_vector_loss_function,
                    u_net_pp_arch.SCALAR_LOSS_FUNCTION_KEY:
                        this_scalar_loss_function,
                    u_net_pp_arch.USE_RESIDUAL_BLOCKS_KEY: False,
                    u_net_pp_arch.USE_CONVNEXT_V1_BLOCKS_KEY:
                        USE_CONVNEXT_V1_FLAGS_AXIS1[i],
                    u_net_pp_arch.USE_CONVNEXT_V2_BLOCKS_KEY:
                        USE_CONVNEXT_V2_FLAGS_AXIS1[i],
                    u_net_pp_arch.SIMPLIFY_CONVNEXT_KEY:
                        SIMPLIFY_CONVNEXT_FLAGS_AXIS1[i],
                    u_net_pp_arch.SIMPLIFY_OUTPUT_LAYER_KEY:
                        SIMPLIFY_OUTPUT_FLAGS_AXIS2[j],
                    u_net_pp_arch.CONV_OUTPUT_ACTIV_FUNC_KEY:
                        OUTPUT_ACTIV_FUNCTIONS_AXIS2[j],
                    u_net_pp_arch.CONV_OUTPUT_ACTIV_FUNC_ALPHA_KEY: 0.,
                    u_net_pp_arch.OPTIMIZER_FUNCTION_KEY:
                        OPTIMIZER_FUNCTIONS_AXIS3[k]
                })

                this_model_object = u_net_pp_arch.create_model(this_option_dict)

                this_model_file_name = (
                    '{0:s}/use-convnext-v1={1:d}_use-convnext-v2={2:d}_'
                    'simplify-convnext={3:d}_simplify-output={4:d}_'
                    'output-activation={5:s}_optimizer={6:s}/model.keras'
                ).format(
                    OUTPUT_DIR_NAME,
                    int(USE_CONVNEXT_V1_FLAGS_AXIS1[i]),
                    int(USE_CONVNEXT_V2_FLAGS_AXIS1[i]),
                    int(SIMPLIFY_CONVNEXT_FLAGS_AXIS1[i]),
                    int(SIMPLIFY_OUTPUT_FLAGS_AXIS2[j]),
                    str(OUTPUT_ACTIV_FUNCTIONS_AXIS2[j]),
                    OPTIMIZER_FUNCTION_DESCRIPTIONS_AXIS3[k]
                )

                file_system_utils.mkdir_recursive_if_necessary(
                    file_name=this_model_file_name
                )

                print('Writing model to: "{0:s}"...'.format(this_model_file_name))
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
                    this_vector_loss_string
                )
                this_option_dict[u_net_pp_arch.SCALAR_LOSS_FUNCTION_KEY] = (
                    this_scalar_loss_string
                )
                this_option_dict[u_net_pp_arch.OPTIMIZER_FUNCTION_KEY] = (
                    OPTIMIZER_FUNCTION_STRINGS_AXIS3[k]
                )

                print('Writing metadata to: "{0:s}"...'.format(this_metafile_name))
                neural_net._write_metafile(
                    dill_file_name=this_metafile_name,
                    num_epochs=100,
                    num_training_batches_per_epoch=100,
                    training_option_dict=DUMMY_GENERATOR_OPTION_DICT,
                    num_validation_batches_per_epoch=100,
                    validation_option_dict=DUMMY_GENERATOR_OPTION_DICT,
                    loss_function_or_dict={
                        neural_net.HEATING_RATE_TARGETS_KEY: this_vector_loss_string,
                        neural_net.FLUX_TARGETS_KEY: this_scalar_loss_string
                    },
                    plateau_lr_multiplier=0.9,
                    early_stopping_patience_epochs=200,
                    u_net_3plus_architecture_dict=None,
                    u_net_plusplus_architecture_dict=this_option_dict,
                    bnn_architecture_dict=None,
                    cnn_architecture_dict=None,
                    dense_architecture_dict=None,
                    u_net_architecture_dict=None,
                    use_ryan_architecture=False
                )


if __name__ == '__main__':
    _run()
