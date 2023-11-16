"""Makes templates for shortwave U-net++ models sans deep sup in 2022 paper.

One difference from 2022 paper: these models use inline normalization.
"""

import os
import sys
import copy
import numpy
import keras.layers

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import example_utils
import u_net_pp_architecture
import inline_normalization
import architecture_utils
import custom_losses
import file_system_utils
import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

HOME_DIR_NAME = '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist'
OUTPUT_DIR_NAME = (
    '{0:s}/ml4rt_models/2022paper_experiment_sw_plusplus_inline_norm/templates'
).format(HOME_DIR_NAME)

VECTOR_LOSS_FUNCTION = custom_losses.dual_weighted_mse()
SCALAR_LOSS_FUNCTION = custom_losses.scaled_mse_for_net_flux(1.)
LOSS_DICT = {
    'conv_output': 'custom_losses.dual_weighted_mse()',
    'dense_output': 'custom_losses.scaled_mse_for_net_flux(1.)'
}

MODEL_DEPTHS = numpy.array([3, 4, 5], dtype=int)
CONV_LAYER_COUNTS = numpy.array([1, 2, 3, 4], dtype=int)
FIRST_LAYER_CHANNEL_COUNTS = numpy.array([4, 8, 16, 32, 64, 128], dtype=int)

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
    u_net_pp_architecture.DENSE_LAYER_DROPOUT_RATES_KEY: numpy.full(4, 0.)
}

DUMMY_GENERATOR_OPTION_DICT = {
    neural_net.VECTOR_TARGET_NORM_TYPE_KEY: None,
    neural_net.SCALAR_TARGET_NORM_TYPE_KEY: None
}



SCALAR_PREDICTOR_NAMES = [
    example_utils.ZENITH_ANGLE_NAME, example_utils.ALBEDO_NAME,
    example_utils.AEROSOL_ALBEDO_NAME,
    example_utils.AEROSOL_ASYMMETRY_PARAM_NAME
]

VECTOR_PREDICTOR_NAMES = [
    example_utils.PRESSURE_NAME, example_utils.TEMPERATURE_NAME,
    example_utils.SPECIFIC_HUMIDITY_NAME, example_utils.RELATIVE_HUMIDITY_NAME,
    example_utils.LIQUID_WATER_CONTENT_NAME,
    example_utils.ICE_WATER_CONTENT_NAME,
    example_utils.LIQUID_WATER_PATH_NAME, example_utils.ICE_WATER_PATH_NAME,
    example_utils.WATER_VAPOUR_PATH_NAME,
    example_utils.UPWARD_LIQUID_WATER_PATH_NAME,
    example_utils.UPWARD_ICE_WATER_PATH_NAME,
    example_utils.UPWARD_WATER_VAPOUR_PATH_NAME,
    example_utils.LIQUID_EFF_RADIUS_NAME, example_utils.ICE_EFF_RADIUS_NAME,
    example_utils.O3_MIXING_RATIO_NAME, example_utils.CO2_CONCENTRATION_NAME,
    example_utils.CH4_CONCENTRATION_NAME, example_utils.N2O_CONCENTRATION_NAME,
    example_utils.AEROSOL_EXTINCTION_NAME, example_utils.HEIGHT_NAME,
    example_utils.HEIGHT_THICKNESS_NAME, example_utils.PRESSURE_THICKNESS_NAME
]

GRID_HEIGHTS_M_AGL = numpy.array([
    21, 44, 68, 93, 120, 149, 179, 212, 246, 282, 321, 361, 405, 450, 499, 550,
    604, 661, 722, 785, 853, 924, 999, 1078, 1161, 1249, 1342, 1439, 1542, 1649,
    1762, 1881, 2005, 2136, 2272, 2415, 2564, 2720, 2882, 3051, 3228, 3411,
    3601, 3798, 4002, 4214, 4433, 4659, 4892, 5132, 5379, 5633, 5894, 6162,
    6436, 6716, 7003, 7296, 7594, 7899, 8208, 8523, 8842, 9166, 9494, 9827,
    10164, 10505, 10849, 11198, 11550, 11906, 12266, 12630, 12997, 13368, 13744,
    14123, 14506, 14895, 15287, 15686, 16090, 16501, 16920, 17350, 17791, 18246,
    18717, 19205, 19715, 20249, 20809, 21400, 22022, 22681, 23379, 24119, 24903,
    25736, 26619, 27558, 28556, 29616, 30743, 31940, 33211, 34566, 36012, 37560,
    39218, 40990, 42882, 44899, 47042, 49299, 51644, 54067, 56552, 59089, 61677,
    64314, 67001, 69747, 72521, 75256, 77803
], dtype=float)

PW_LINEAR_UNIF_MODEL_FILE_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4rt_project/gfs_data/'
    'shortwave_examples_600days/orig_heights/training/'
    'piecewise_linear_models_for_uniformization.nc'
)


def _run():
    """Makes templates for shortwave U-net++ models sans deep sup in 2022 paper.

    One difference from 2022 paper: these models use inline normalization.

    This is effectively the main method.
    """

    input_layer_object = keras.layers.Input(shape=(127, 26))
    normalized_input_layer_object = (
        inline_normalization.create_normalization_layers(
            input_layer_object=input_layer_object,
            pw_linear_unif_model_file_name=PW_LINEAR_UNIF_MODEL_FILE_NAME,
            vector_predictor_names=VECTOR_PREDICTOR_NAMES,
            scalar_predictor_names=SCALAR_PREDICTOR_NAMES,
            heights_m_agl=GRID_HEIGHTS_M_AGL
        )
    )

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
                    u_net_pp_architecture.NUM_LEVELS_KEY:
                        MODEL_DEPTHS[i],
                    u_net_pp_architecture.CONV_LAYER_COUNTS_KEY:
                        numpy.full(
                            MODEL_DEPTHS[i] + 1, CONV_LAYER_COUNTS[j], dtype=int
                        ),
                    u_net_pp_architecture.CHANNEL_COUNTS_KEY:
                        these_channel_counts_all,
                    u_net_pp_architecture.ENCODER_DROPOUT_RATES_KEY:
                        numpy.full(MODEL_DEPTHS[i] + 1, 0.),
                    u_net_pp_architecture.UPCONV_DROPOUT_RATES_KEY:
                        numpy.full(MODEL_DEPTHS[i], 0.),
                    u_net_pp_architecture.SKIP_DROPOUT_RATES_KEY:
                        numpy.full(MODEL_DEPTHS[i], 0.)
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

                this_option_dict[
                    u_net_pp_architecture.DENSE_LAYER_NEURON_NUMS_KEY
                ] = these_neuron_counts

                this_model_object = u_net_pp_architecture.create_model(
                    option_dict=this_option_dict,
                    vector_loss_function=VECTOR_LOSS_FUNCTION,
                    use_deep_supervision=False, num_output_channels=1,
                    scalar_loss_function=SCALAR_LOSS_FUNCTION,
                    input_layer_object=input_layer_object,
                    normalized_input_layer_object=normalized_input_layer_object
                )

                this_model_file_name = (
                    '{0:s}/depth={1:d}_num-conv-layers-per-block={2:d}_'
                    'num-first-layer-channels={3:03d}/model.h5'
                ).format(
                    OUTPUT_DIR_NAME, MODEL_DEPTHS[i], CONV_LAYER_COUNTS[j],
                    FIRST_LAYER_CHANNEL_COUNTS[k]
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
                    loss_function_or_dict=LOSS_DICT,
                    do_early_stopping=True, plateau_lr_multiplier=0.6,
                    bnn_architecture_dict=None
                )


if __name__ == '__main__':
    _run()
