"""Makes templates for Spectral Experiment 7.

This experiment has z-score normalization and Boolean masks built in as layers.
"""

import os
import sys
import copy
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import u_net_architecture as u_net_arch
import u_net_pp_architecture as u_net_pp_arch
import architecture_utils
import custom_losses
import file_system_utils
import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

HOME_DIR_NAME = '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist'
OUTPUT_DIR_NAME = (
    '{0:s}/ml4rt_models/spectral_experiment07_shortwave/templates'
).format(HOME_DIR_NAME)

ENSEMBLE_SIZE = 1
NUM_WAVELENGTHS = 14

MODEL_DEPTH = 3
NUM_CONV_LAYERS_PER_BLOCK = 1
NUM_FIRST_LAYER_CHANNELS = 128

NORMALIZATION_FILE_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4rt_project/gfs_data/'
    'examples_with_correct_vertical_coords/shortwave_spectrally_resolved/'
    'training/normalization_params_20180901-20191221.nc'
)
VECTOR_PREDICTOR_NAMES = [
    'pressure_pascals', 'temperature_kelvins', 'specific_humidity_kg_kg01',
    'relative_humidity_unitless', 'liquid_water_content_kg_m03',
    'ice_water_content_kg_m03', 'liquid_water_path_kg_m02',
    'ice_water_path_kg_m02', 'vapour_path_kg_m02',
    'upward_liquid_water_path_kg_m02', 'upward_ice_water_path_kg_m02',
    'upward_vapour_path_kg_m02', 'liquid_effective_radius_metres',
    'ice_effective_radius_metres', 'o3_mixing_ratio_kg_kg01',
    'co2_concentration_ppmv', 'ch4_concentration_ppmv',
    'n2o_concentration_ppmv', 'aerosol_extinction_metres01', 'height_m_agl',
    'height_thickness_metres', 'pressure_thickness_pascals'
]
SCALAR_PREDICTOR_NAMES = [
    'zenith_angle_radians', 'albedo', 'aerosol_single_scattering_albedo',
    'aerosol_asymmetry_param'
]
HEIGHTS_M_AGL = neural_net.HEIGHTS_FOR_PETER_M_AGL

VECTOR_TARGET_NAME = 'shortwave_heating_rate_k_day01'
SCALAR_TARGET_NAMES = [
    'shortwave_surface_down_flux_w_m02', 'shortwave_toa_up_flux_w_m02'
]
WAVELENGTHS_METRES = numpy.array([
    0.000005847953216, 0.000003418803419, 0.000002758620690, 0.000002312138728,
    0.000002040816327, 0.000001769911504, 0.000001444043321, 0.000001269841270,
    0.000000956937799, 0.000000693240901, 0.000000517464424, 0.000000387221684,
    0.000000298507463, 0.000000227272727
])

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
    u_net_pp_arch.CONV_OUTPUT_ACTIV_FUNC_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    u_net_pp_arch.CONV_OUTPUT_ACTIV_FUNC_ALPHA_KEY: 0.,
    u_net_pp_arch.DENSE_OUTPUT_ACTIV_FUNC_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    u_net_pp_arch.DENSE_OUTPUT_ACTIV_FUNC_ALPHA_KEY: 0.,
    u_net_pp_arch.L1_WEIGHT_KEY: 0.,
    u_net_pp_arch.L2_WEIGHT_KEY: 1e-7,
    u_net_pp_arch.USE_BATCH_NORM_KEY: True,
    # u_net_pp_arch.USE_RESIDUAL_BLOCKS_KEY: True,
    u_net_pp_arch.USE_CONVNEXT_V1_BLOCKS_KEY: False,
    u_net_pp_arch.USE_CONVNEXT_V2_BLOCKS_KEY: False,
    u_net_pp_arch.SIMPLIFY_CONVNEXT_KEY: False,
    u_net_pp_arch.SIMPLIFY_OUTPUT_LAYER_KEY: False,
    # u_net_pp_arch.MEAN_VALUE_MATRIX_KEY: None,
    # u_net_pp_arch.STDEV_MATRIX_KEY: None,
    # u_net_pp_arch.HEATING_RATE_MASK_KEY: None,
    # u_net_pp_arch.FLUX_MASK_KEY: None,
    u_net_pp_arch.NUM_OUTPUT_WAVELENGTHS_KEY: NUM_WAVELENGTHS,
    # u_net_pp_arch.VECTOR_LOSS_FUNCTION_KEY: VECTOR_LOSS_FUNCTION,
    # u_net_pp_arch.SCALAR_LOSS_FUNCTION_KEY: SCALAR_LOSS_FUNCTION,
    u_net_pp_arch.USE_DEEP_SUPERVISION_KEY: False,
    u_net_pp_arch.ENSEMBLE_SIZE_KEY: ENSEMBLE_SIZE
}

DUMMY_GENERATOR_OPTION_DICT = {
    neural_net.NORMALIZE_PREDICTORS_KEY: False,
    neural_net.NORMALIZE_VECTOR_TARGETS_KEY: False,
    neural_net.NORMALIZE_SCALAR_TARGETS_KEY: False
}

MIN_DUAL_WEIGHT = 0.75
BROADBAND_WEIGHT = 0.01
USE_RESIDUAL_FLAGS = numpy.array([0, 1], dtype=bool)


def _run():
    """Makes templates for Spectral Experiment 7.

    This is effectively the main method.
    """

    for j in range(len(USE_RESIDUAL_FLAGS)):
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

        mean_value_matrix, stdev_matrix = (
            u_net_arch.get_normalization_params(
                vector_predictor_names=VECTOR_PREDICTOR_NAMES,
                scalar_predictor_names=SCALAR_PREDICTOR_NAMES,
                heights_m_agl=HEIGHTS_M_AGL,
                normalization_file_name=NORMALIZATION_FILE_NAME
            )
        )

        heating_rate_mask_matrix, flux_mask_matrix = neural_net.create_mask(
            normalization_file_name=NORMALIZATION_FILE_NAME,
            min_heating_rate_k_day01=0.001,
            min_flux_w_m02=0.01,
            heights_m_agl=HEIGHTS_M_AGL,
            target_wavelengths_metres=WAVELENGTHS_METRES,
            vector_target_name=VECTOR_TARGET_NAME,
            scalar_target_names=SCALAR_TARGET_NAMES,
            num_examples=1
        )
        heating_rate_mask_matrix = heating_rate_mask_matrix[0, ...]
        flux_mask_matrix = flux_mask_matrix[0, ...]

        this_option_dict.update({
            u_net_pp_arch.VECTOR_LOSS_FUNCTION_KEY: this_vector_loss_function,
            u_net_pp_arch.SCALAR_LOSS_FUNCTION_KEY: this_scalar_loss_function,
            u_net_pp_arch.USE_RESIDUAL_BLOCKS_KEY: USE_RESIDUAL_FLAGS[j],
            u_net_pp_arch.MEAN_VALUE_MATRIX_KEY: mean_value_matrix,
            u_net_pp_arch.STDEV_MATRIX_KEY: stdev_matrix,
            u_net_pp_arch.HEATING_RATE_MASK_KEY: heating_rate_mask_matrix,
            u_net_pp_arch.FLUX_MASK_KEY: flux_mask_matrix
        })

        this_model_object = u_net_pp_arch.create_model(this_option_dict)

        this_model_file_name = '{0:s}/use-residual={1:d}/model.keras'.format(
            OUTPUT_DIR_NAME, int(USE_RESIDUAL_FLAGS[j])
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
