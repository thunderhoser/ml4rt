"""Methods for building U-net 3+.

Based on:
https://github.com/ZJUGiveLab/UNet-Version/blob/master/models/UNet_3Plus.py
"""

import numpy
import keras
import keras.layers
from gewittergefahr.deep_learning import architecture_utils
from ml4rt.machine_learning import neural_net
from ml4rt.machine_learning import u_net_architecture as u_net_arch

INPUT_DIMENSIONS_KEY = u_net_arch.INPUT_DIMENSIONS_KEY
NUM_LEVELS_KEY = u_net_arch.NUM_LEVELS_KEY
CONV_LAYER_COUNTS_KEY = u_net_arch.CONV_LAYER_COUNTS_KEY
CHANNEL_COUNTS_KEY = u_net_arch.CHANNEL_COUNTS_KEY
ENCODER_DROPOUT_RATES_KEY = u_net_arch.ENCODER_DROPOUT_RATES_KEY
ENCODER_MC_DROPOUT_FLAGS_KEY = u_net_arch.ENCODER_MC_DROPOUT_FLAGS_KEY
UPCONV_DROPOUT_RATES_KEY = u_net_arch.UPCONV_DROPOUT_RATES_KEY
UPCONV_MC_DROPOUT_FLAGS_KEY = u_net_arch.UPCONV_MC_DROPOUT_FLAGS_KEY
SKIP_DROPOUT_RATES_KEY = u_net_arch.SKIP_DROPOUT_RATES_KEY
SKIP_MC_DROPOUT_FLAGS_KEY = u_net_arch.SKIP_MC_DROPOUT_FLAGS_KEY
INCLUDE_PENULTIMATE_KEY = u_net_arch.INCLUDE_PENULTIMATE_KEY
PENULTIMATE_DROPOUT_RATE_KEY = u_net_arch.PENULTIMATE_DROPOUT_RATE_KEY
PENULTIMATE_MC_DROPOUT_FLAG_KEY = u_net_arch.PENULTIMATE_MC_DROPOUT_FLAG_KEY
DENSE_LAYER_NEURON_NUMS_KEY = u_net_arch.DENSE_LAYER_NEURON_NUMS_KEY
DENSE_LAYER_DROPOUT_RATES_KEY = u_net_arch.DENSE_LAYER_DROPOUT_RATES_KEY
DENSE_LAYER_MC_DROPOUT_FLAGS_KEY = u_net_arch.DENSE_LAYER_MC_DROPOUT_FLAGS_KEY
INNER_ACTIV_FUNCTION_KEY = u_net_arch.INNER_ACTIV_FUNCTION_KEY
INNER_ACTIV_FUNCTION_ALPHA_KEY = u_net_arch.INNER_ACTIV_FUNCTION_ALPHA_KEY
CONV_OUTPUT_ACTIV_FUNC_KEY = u_net_arch.CONV_OUTPUT_ACTIV_FUNC_KEY
CONV_OUTPUT_ACTIV_FUNC_ALPHA_KEY = u_net_arch.CONV_OUTPUT_ACTIV_FUNC_ALPHA_KEY
DENSE_OUTPUT_ACTIV_FUNC_KEY = u_net_arch.DENSE_OUTPUT_ACTIV_FUNC_KEY
DENSE_OUTPUT_ACTIV_FUNC_ALPHA_KEY = (
    u_net_arch.DENSE_OUTPUT_ACTIV_FUNC_ALPHA_KEY
)
L1_WEIGHT_KEY = u_net_arch.L1_WEIGHT_KEY
L2_WEIGHT_KEY = u_net_arch.L2_WEIGHT_KEY
USE_BATCH_NORM_KEY = u_net_arch.USE_BATCH_NORM_KEY

NUM_OUTPUT_WAVELENGTHS_KEY = u_net_arch.NUM_OUTPUT_WAVELENGTHS_KEY
VECTOR_LOSS_FUNCTION_KEY = u_net_arch.VECTOR_LOSS_FUNCTION_KEY
SCALAR_LOSS_FUNCTION_KEY = u_net_arch.SCALAR_LOSS_FUNCTION_KEY
JOINED_LOSS_FUNCTION_KEY = u_net_arch.JOINED_LOSS_FUNCTION_KEY
USE_DEEP_SUPERVISION_KEY = u_net_arch.USE_DEEP_SUPERVISION_KEY
ENSEMBLE_SIZE_KEY = u_net_arch.ENSEMBLE_SIZE_KEY

DO_INLINE_NORMALIZATION_KEY = u_net_arch.DO_INLINE_NORMALIZATION_KEY
PW_LINEAR_UNIF_MODEL_KEY = u_net_arch.PW_LINEAR_UNIF_MODEL_KEY
SCALAR_PREDICTORS_KEY = u_net_arch.SCALAR_PREDICTORS_KEY
VECTOR_PREDICTORS_KEY = u_net_arch.VECTOR_PREDICTORS_KEY
HEIGHTS_KEY = u_net_arch.HEIGHTS_KEY

DEFAULT_ARCHITECTURE_OPTION_DICT = {
    NUM_LEVELS_KEY: 4,
    CONV_LAYER_COUNTS_KEY: numpy.full(5, 2, dtype=int),
    CHANNEL_COUNTS_KEY: numpy.array([64, 128, 256, 512, 1024], dtype=int),
    ENCODER_DROPOUT_RATES_KEY: numpy.full(5, 0.),
    ENCODER_MC_DROPOUT_FLAGS_KEY: numpy.full(5, 0, dtype=bool),
    UPCONV_DROPOUT_RATES_KEY: numpy.full(4, 0.),
    UPCONV_MC_DROPOUT_FLAGS_KEY: numpy.full(4, 0, dtype=bool),
    SKIP_DROPOUT_RATES_KEY: numpy.full(4, 0.),
    SKIP_MC_DROPOUT_FLAGS_KEY: numpy.full(4, 0, dtype=bool),
    INCLUDE_PENULTIMATE_KEY: True,
    PENULTIMATE_DROPOUT_RATE_KEY: 0.,
    PENULTIMATE_MC_DROPOUT_FLAG_KEY: False,
    DENSE_LAYER_NEURON_NUMS_KEY: None,
    DENSE_LAYER_DROPOUT_RATES_KEY: None,
    DENSE_LAYER_MC_DROPOUT_FLAGS_KEY: None,
    INNER_ACTIV_FUNCTION_KEY: architecture_utils.RELU_FUNCTION_STRING,
    INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    CONV_OUTPUT_ACTIV_FUNC_KEY: None,
    CONV_OUTPUT_ACTIV_FUNC_ALPHA_KEY: 0.,
    DENSE_OUTPUT_ACTIV_FUNC_KEY: architecture_utils.RELU_FUNCTION_STRING,
    DENSE_OUTPUT_ACTIV_FUNC_ALPHA_KEY: 0.,
    L1_WEIGHT_KEY: 0.,
    L2_WEIGHT_KEY: 0.001,
    USE_BATCH_NORM_KEY: True
}


def create_model(option_dict):
    """Creates U-net 3+.

    Architecture based on:
    https://github.com/ZJUGiveLab/UNet-Version/blob/master/models/UNet_3Plus.py

    :param option_dict: See doc for `u_net_architecture.check_args`.
    :return: model_object: Instance of `keras.models.Model`, with the
        aforementioned architecture.
    """

    option_dict[u_net_arch.DO_INLINE_NORMALIZATION_KEY] = False
    option_dict = u_net_arch.check_args(option_dict)

    input_dimensions = option_dict[INPUT_DIMENSIONS_KEY]
    num_levels = option_dict[NUM_LEVELS_KEY]
    num_conv_layers_by_level = option_dict[CONV_LAYER_COUNTS_KEY]
    num_channels_by_level = option_dict[CHANNEL_COUNTS_KEY]
    encoder_dropout_rate_by_level = option_dict[ENCODER_DROPOUT_RATES_KEY]
    encoder_mc_dropout_flag_by_level = option_dict[ENCODER_MC_DROPOUT_FLAGS_KEY]
    upconv_dropout_rate_by_level = option_dict[UPCONV_DROPOUT_RATES_KEY]
    upconv_mc_dropout_flag_by_level = option_dict[UPCONV_MC_DROPOUT_FLAGS_KEY]
    skip_dropout_rate_by_level = option_dict[SKIP_DROPOUT_RATES_KEY]
    skip_mc_dropout_flag_by_level = option_dict[SKIP_MC_DROPOUT_FLAGS_KEY]
    include_penultimate_conv = option_dict[INCLUDE_PENULTIMATE_KEY]
    penultimate_conv_dropout_rate = option_dict[PENULTIMATE_DROPOUT_RATE_KEY]
    penultimate_conv_mc_dropout_flag = (
        option_dict[PENULTIMATE_MC_DROPOUT_FLAG_KEY]
    )
    dense_layer_neuron_nums = option_dict[DENSE_LAYER_NEURON_NUMS_KEY]
    dense_layer_dropout_rates = option_dict[DENSE_LAYER_DROPOUT_RATES_KEY]
    dense_layer_mc_dropout_flags = option_dict[DENSE_LAYER_MC_DROPOUT_FLAGS_KEY]
    inner_activ_function_name = option_dict[INNER_ACTIV_FUNCTION_KEY]
    inner_activ_function_alpha = option_dict[INNER_ACTIV_FUNCTION_ALPHA_KEY]
    conv_output_activ_func_name = option_dict[CONV_OUTPUT_ACTIV_FUNC_KEY]
    conv_output_activ_func_alpha = option_dict[CONV_OUTPUT_ACTIV_FUNC_ALPHA_KEY]
    dense_output_activ_func_name = option_dict[DENSE_OUTPUT_ACTIV_FUNC_KEY]
    dense_output_activ_func_alpha = (
        option_dict[DENSE_OUTPUT_ACTIV_FUNC_ALPHA_KEY]
    )
    l1_weight = option_dict[L1_WEIGHT_KEY]
    l2_weight = option_dict[L2_WEIGHT_KEY]
    use_batch_normalization = option_dict[USE_BATCH_NORM_KEY]

    num_output_wavelengths = option_dict[NUM_OUTPUT_WAVELENGTHS_KEY]
    vector_loss_function = option_dict[VECTOR_LOSS_FUNCTION_KEY]
    scalar_loss_function = option_dict[SCALAR_LOSS_FUNCTION_KEY]
    use_deep_supervision = option_dict[USE_DEEP_SUPERVISION_KEY]
    ensemble_size = option_dict[ENSEMBLE_SIZE_KEY]

    if ensemble_size > 1:
        include_penultimate_conv = False
        use_deep_supervision = False

    has_dense_layers = dense_layer_neuron_nums is not None

    input_layer_object = keras.layers.Input(
        shape=tuple(input_dimensions.tolist())
    )
    regularizer_object = architecture_utils.get_weight_regularizer(
        l1_weight=l1_weight, l2_weight=l2_weight
    )

    encoder_conv_layer_objects = [None] * (num_levels + 1)
    encoder_pooling_layer_objects = [None] * num_levels

    for i in range(num_levels + 1):
        for k in range(num_conv_layers_by_level[i]):
            if k == 0:
                if i == 0:
                    this_input_layer_object = input_layer_object
                else:
                    this_input_layer_object = (
                        encoder_pooling_layer_objects[i - 1]
                    )
            else:
                this_input_layer_object = encoder_conv_layer_objects[i]

            this_name = 'encoder{0:d}_conv{1:d}'.format(i, k)

            encoder_conv_layer_objects[i] = architecture_utils.get_1d_conv_layer(
                num_kernel_rows=3, num_rows_per_stride=1,
                num_filters=num_channels_by_level[i],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object, layer_name=this_name
            )(this_input_layer_object)

            this_name = 'encoder{0:d}_conv{1:d}_activation'.format(i, k)

            encoder_conv_layer_objects[i] = (
                architecture_utils.get_activation_layer(
                    activation_function_string=inner_activ_function_name,
                    alpha_for_relu=inner_activ_function_alpha,
                    alpha_for_elu=inner_activ_function_alpha,
                    layer_name=this_name
                )(encoder_conv_layer_objects[i])
            )

            if encoder_dropout_rate_by_level[i] > 0:
                this_name = 'encoder{0:d}_conv{1:d}_dropout'.format(i, k)
                this_mc_flag = bool(encoder_mc_dropout_flag_by_level[i])

                encoder_conv_layer_objects[i] = (
                    architecture_utils.get_dropout_layer(
                        dropout_fraction=encoder_dropout_rate_by_level[i],
                        layer_name=this_name
                    )(encoder_conv_layer_objects[i], training=this_mc_flag)
                )

            if use_batch_normalization:
                this_name = 'encoder{0:d}_conv{1:d}_batchnorm'.format(i, k)

                encoder_conv_layer_objects[i] = (
                    architecture_utils.get_batch_norm_layer(
                        layer_name=this_name
                    )(encoder_conv_layer_objects[i])
                )

        if i != num_levels:
            this_name = 'encoder{0:d}_pooling'.format(i)

            encoder_pooling_layer_objects[i] = (
                architecture_utils.get_1d_pooling_layer(
                    num_rows_in_window=2, num_rows_per_stride=2,
                    pooling_type_string=architecture_utils.MAX_POOLING_STRING,
                    layer_name=this_name
                )(encoder_conv_layer_objects[i])
            )

    decoder_conv_layer_objects = [None] * num_levels

    for i in range(num_levels)[::-1]:
        layer_objects_to_concat = []

        for j in range(i + 1, num_levels + 1):
            if j == num_levels:
                layer_name_prefix = (
                    'decoder{0:d}_upsample_encoder{1:d}'.format(i, j)
                )

                this_layer_object = keras.layers.UpSampling1D(
                    size=2 ** (j - i), name=layer_name_prefix
                )(encoder_conv_layer_objects[j])
            else:
                layer_name_prefix = (
                    'decoder{0:d}_upsample_decoder{1:d}'.format(i, j)
                )

                this_layer_object = keras.layers.UpSampling1D(
                    size=2 ** (j - i), name=layer_name_prefix
                )(decoder_conv_layer_objects[j])

            num_upsampled_heights = this_layer_object.shape[1]
            num_desired_heights = encoder_conv_layer_objects[i].shape[1]

            if num_desired_heights != num_upsampled_heights:
                this_name = '{0:s}_padding'.format(layer_name_prefix)

                num_extra_heights = num_desired_heights - num_upsampled_heights
                num_extra_heights_bottom = int(numpy.floor(
                    float(num_extra_heights) / 2
                ))
                num_extra_heights_top = int(numpy.ceil(
                    float(num_extra_heights) / 2
                ))

                this_layer_object = keras.layers.ZeroPadding1D(
                    padding=(num_extra_heights_bottom, num_extra_heights_top),
                    name=this_name
                )(this_layer_object)

            this_name = '{0:s}_conv'.format(layer_name_prefix)

            this_layer_object = architecture_utils.get_1d_conv_layer(
                num_kernel_rows=3, num_rows_per_stride=1,
                num_filters=num_channels_by_level[0],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object, layer_name=this_name
            )(this_layer_object)

            this_name = '{0:s}_activation'.format(layer_name_prefix)

            this_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha,
                layer_name=this_name
            )(this_layer_object)

            if upconv_dropout_rate_by_level[i] > 0:
                this_name = '{0:s}_dropout'.format(layer_name_prefix)
                this_mc_flag = bool(upconv_mc_dropout_flag_by_level[i])

                this_layer_object = architecture_utils.get_dropout_layer(
                    dropout_fraction=upconv_dropout_rate_by_level[
                        upconv_dropout_rate_by_level[i]
                    ],
                    layer_name=this_name
                )(this_layer_object, training=this_mc_flag)

            if use_batch_normalization:
                this_name = '{0:s}_batchnorm'.format(layer_name_prefix)

                this_layer_object = architecture_utils.get_batch_norm_layer(
                    layer_name=this_name
                )(this_layer_object)

            layer_objects_to_concat.append(this_layer_object)

        layer_name_prefix = 'decoder{0:d}_carry_encoder{0:d}'.format(i)
        this_name = '{0:s}_conv'.format(layer_name_prefix)

        this_layer_object = architecture_utils.get_1d_conv_layer(
            num_kernel_rows=3, num_rows_per_stride=1,
            num_filters=num_channels_by_level[0],
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object, layer_name=this_name
        )(encoder_conv_layer_objects[i])

        this_name = '{0:s}_activation'.format(layer_name_prefix)

        this_layer_object = architecture_utils.get_activation_layer(
            activation_function_string=inner_activ_function_name,
            alpha_for_relu=inner_activ_function_alpha,
            alpha_for_elu=inner_activ_function_alpha,
            layer_name=this_name
        )(this_layer_object)

        if upconv_dropout_rate_by_level[i] > 0:
            this_name = '{0:s}_dropout'.format(layer_name_prefix)
            this_mc_flag = bool(upconv_mc_dropout_flag_by_level[i])

            this_layer_object = architecture_utils.get_dropout_layer(
                dropout_fraction=upconv_dropout_rate_by_level[
                    upconv_dropout_rate_by_level[i]
                ],
                layer_name=this_name
            )(this_layer_object, training=this_mc_flag)

        if use_batch_normalization:
            this_name = '{0:s}_batchnorm'.format(layer_name_prefix)

            this_layer_object = architecture_utils.get_batch_norm_layer(
                layer_name=this_name
            )(this_layer_object)

        layer_objects_to_concat.append(this_layer_object)

        for j in range(i):
            layer_name_prefix = (
                'decoder{0:d}_downsample_encoder{1:d}'.format(i, j)
            )
            this_name = '{0:s}_pooling'.format(layer_name_prefix)

            this_layer_object = architecture_utils.get_1d_pooling_layer(
                num_rows_in_window=2 ** (i - j),
                num_rows_per_stride=2 ** (i - j),
                pooling_type_string=architecture_utils.MAX_POOLING_STRING,
                layer_name=this_name
            )(encoder_conv_layer_objects[j])

            this_name = '{0:s}_conv'.format(layer_name_prefix)

            this_layer_object = architecture_utils.get_1d_conv_layer(
                num_kernel_rows=3, num_rows_per_stride=1,
                num_filters=num_channels_by_level[0],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object, layer_name=this_name
            )(this_layer_object)

            this_name = '{0:s}_activation'.format(layer_name_prefix)

            this_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha,
                layer_name=this_name
            )(this_layer_object)

            if upconv_dropout_rate_by_level[i] > 0:
                this_name = '{0:s}_dropout'.format(layer_name_prefix)
                this_mc_flag = bool(upconv_mc_dropout_flag_by_level[i])

                this_layer_object = architecture_utils.get_dropout_layer(
                    dropout_fraction=upconv_dropout_rate_by_level[
                        upconv_dropout_rate_by_level[i]
                    ],
                    layer_name=this_name
                )(this_layer_object, training=this_mc_flag)

            if use_batch_normalization:
                this_name = '{0:s}_batchnorm'.format(layer_name_prefix)

                this_layer_object = architecture_utils.get_batch_norm_layer(
                    layer_name=this_name
                )(this_layer_object)

            layer_objects_to_concat.append(this_layer_object)

        this_name = 'decoder{0:d}_concat'.format(i)
        decoder_conv_layer_objects[i] = keras.layers.Concatenate(
            axis=-1, name=this_name
        )(layer_objects_to_concat)

        this_num_channels = (
            num_channels_by_level[0] * len(layer_objects_to_concat)
        )

        for k in range(num_conv_layers_by_level[i]):
            this_name = 'decoder{0:d}_conv{1:d}'.format(i, k)

            decoder_conv_layer_objects[i] = (
                architecture_utils.get_1d_conv_layer(
                    num_kernel_rows=3, num_rows_per_stride=1,
                    num_filters=this_num_channels,
                    padding_type_string=architecture_utils.YES_PADDING_STRING,
                    weight_regularizer=regularizer_object,
                    layer_name=this_name
                )(decoder_conv_layer_objects[i])
            )

            this_name = 'decoder{0:d}_conv{1:d}_activation'.format(i, k)

            decoder_conv_layer_objects[i] = (
                architecture_utils.get_activation_layer(
                    activation_function_string=inner_activ_function_name,
                    alpha_for_relu=inner_activ_function_alpha,
                    alpha_for_elu=inner_activ_function_alpha,
                    layer_name=this_name
                )(decoder_conv_layer_objects[i])
            )

            if skip_dropout_rate_by_level[i] > 0:
                this_name = 'decoder{0:d}_conv{1:d}_dropout'.format(i, k)
                this_mc_flag = bool(skip_mc_dropout_flag_by_level[i])

                decoder_conv_layer_objects[i] = (
                    architecture_utils.get_dropout_layer(
                        dropout_fraction=skip_dropout_rate_by_level[i],
                        layer_name=this_name
                    )(decoder_conv_layer_objects[i], training=this_mc_flag)
                )

            if use_batch_normalization:
                this_name = 'decoder{0:d}_conv{1:d}_batchnorm'.format(i, k)

                decoder_conv_layer_objects[i] = (
                    architecture_utils.get_batch_norm_layer(
                        layer_name=this_name
                    )(decoder_conv_layer_objects[i])
                )

    if include_penultimate_conv:
        conv_output_layer_object = architecture_utils.get_1d_conv_layer(
            num_kernel_rows=3, num_rows_per_stride=1,
            num_filters=2 * num_output_wavelengths * ensemble_size,
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object, layer_name='penultimate_conv'
        )(decoder_conv_layer_objects[0])

        conv_output_layer_object = architecture_utils.get_activation_layer(
            activation_function_string=inner_activ_function_name,
            alpha_for_relu=inner_activ_function_alpha,
            alpha_for_elu=inner_activ_function_alpha,
            layer_name='penultimate_conv_activation'
        )(conv_output_layer_object)

        if penultimate_conv_dropout_rate > 0:
            this_mc_flag = bool(penultimate_conv_mc_dropout_flag)

            conv_output_layer_object = (
                architecture_utils.get_dropout_layer(
                    dropout_fraction=penultimate_conv_dropout_rate,
                    layer_name='penultimate_conv_dropout'
                )(conv_output_layer_object, training=this_mc_flag)
            )

        if use_batch_normalization:
            conv_output_layer_object = (
                architecture_utils.get_batch_norm_layer(
                    layer_name='penultimate_conv_bn'
                )(conv_output_layer_object)
            )
    else:
        conv_output_layer_object = decoder_conv_layer_objects[0]

    conv_output_layer_object = architecture_utils.get_1d_conv_layer(
        num_kernel_rows=1, num_rows_per_stride=1,
        num_filters=num_output_wavelengths * ensemble_size,
        padding_type_string=architecture_utils.YES_PADDING_STRING,
        weight_regularizer=regularizer_object, layer_name='last_conv'
    )(conv_output_layer_object)

    if ensemble_size > 1:
        conv_output_layer_object = keras.layers.Reshape(
            target_shape=
            (input_dimensions[0], num_output_wavelengths, 1, ensemble_size)
        )(conv_output_layer_object)
    else:
        conv_output_layer_object = keras.layers.Reshape(
            target_shape=
            (input_dimensions[0], num_output_wavelengths, 1)
        )(conv_output_layer_object)

    if conv_output_activ_func_name is not None:
        conv_output_layer_object = architecture_utils.get_activation_layer(
            activation_function_string=conv_output_activ_func_name,
            alpha_for_relu=conv_output_activ_func_alpha,
            alpha_for_elu=conv_output_activ_func_alpha,
            layer_name='last_conv_activation'
        )(conv_output_layer_object)

    conv_output_layer_object = u_net_arch.zero_top_heating_rate(
        input_layer_object=conv_output_layer_object,
        ensemble_size=ensemble_size,
        output_layer_name='conv_output'
    )

    output_layer_objects = [conv_output_layer_object]
    loss_dict = {'conv_output': vector_loss_function}

    deep_supervision_layer_objects = [None] * (num_levels + 1)

    if use_deep_supervision:
        for i in range(1, num_levels + 1):
            this_name = 'deepsup{0:d}_upsampling'.format(i)

            if i == num_levels:
                deep_supervision_layer_objects[i] = keras.layers.UpSampling1D(
                    size=2 ** i, name=this_name
                )(encoder_conv_layer_objects[i])
            else:
                deep_supervision_layer_objects[i] = keras.layers.UpSampling1D(
                    size=2 ** i, name=this_name
                )(decoder_conv_layer_objects[i])

            num_upsampled_heights = (
                deep_supervision_layer_objects[i].shape[1]
            )
            num_desired_heights = input_dimensions[0]

            if num_desired_heights != num_upsampled_heights:
                this_name = 'deepsup{0:d}_padding'.format(i)

                num_extra_heights = num_desired_heights - num_upsampled_heights
                num_extra_heights_bottom = int(numpy.floor(
                    float(num_extra_heights) / 2
                ))
                num_extra_heights_top = int(numpy.ceil(
                    float(num_extra_heights) / 2
                ))

                deep_supervision_layer_objects[i] = keras.layers.ZeroPadding1D(
                    padding=(num_extra_heights_bottom, num_extra_heights_top),
                    name=this_name
                )(deep_supervision_layer_objects[i])

            this_name = 'deepsup{0:d}_conv'.format(i)

            deep_supervision_layer_objects[i] = (
                architecture_utils.get_1d_conv_layer(
                    num_kernel_rows=1, num_rows_per_stride=1,
                    num_filters=num_output_wavelengths,
                    padding_type_string=architecture_utils.YES_PADDING_STRING,
                    weight_regularizer=regularizer_object, layer_name=this_name
                )(deep_supervision_layer_objects[i])
            )

            if conv_output_activ_func_name is not None:
                this_name = 'deepsup{0:d}_activation'.format(i)

                deep_supervision_layer_objects[i] = (
                    architecture_utils.get_activation_layer(
                        activation_function_string=conv_output_activ_func_name,
                        alpha_for_relu=conv_output_activ_func_alpha,
                        alpha_for_elu=conv_output_activ_func_alpha,
                        layer_name=this_name
                    )(deep_supervision_layer_objects[i])
                )

            deep_supervision_layer_objects[i] = keras.layers.Reshape(
                target_shape=(input_dimensions[0], num_output_wavelengths, 1)
            )(deep_supervision_layer_objects[i])

            deep_supervision_layer_objects[i] = (
                u_net_arch.zero_top_heating_rate(
                    input_layer_object=deep_supervision_layer_objects[i],
                    ensemble_size=1,
                    output_layer_name='deepsup{0:d}_output'.format(i)
                )
            )

            output_layer_objects.append(deep_supervision_layer_objects[i])
            loss_dict[this_name] = vector_loss_function

    if has_dense_layers:
        num_dense_layers = len(dense_layer_neuron_nums)
        dense_output_layer_object = architecture_utils.get_flattening_layer()(
            encoder_conv_layer_objects[-1]
        )
    else:
        num_dense_layers = 0
        dense_output_layer_object = None

    for j in range(num_dense_layers):
        dense_output_layer_object = architecture_utils.get_dense_layer(
            num_output_units=dense_layer_neuron_nums[j]
        )(dense_output_layer_object)

        if j == num_dense_layers - 1:
            if (
                    dense_layer_dropout_rates[j] <= 0 and
                    dense_output_activ_func_name is None
            ):
                this_name = 'dense_output'
            else:
                this_name = None

            num_dense_output_vars = (
                float(dense_layer_neuron_nums[j]) /
                (ensemble_size * num_output_wavelengths)
            )
            assert numpy.isclose(
                num_dense_output_vars, numpy.round(num_dense_output_vars),
                atol=1e-6
            )
            num_dense_output_vars = int(numpy.round(num_dense_output_vars))

            if ensemble_size > 1:
                these_dim = (
                    num_output_wavelengths, num_dense_output_vars, ensemble_size
                )
            else:
                these_dim = (num_output_wavelengths, num_dense_output_vars)

            dense_output_layer_object = keras.layers.Reshape(
                target_shape=these_dim, name=this_name
            )(dense_output_layer_object)

            if dense_output_activ_func_name is not None:
                this_name = (
                    None if dense_layer_dropout_rates[j] > 0 else 'dense_output'
                )

                dense_output_layer_object = (
                    architecture_utils.get_activation_layer(
                        activation_function_string=dense_output_activ_func_name,
                        alpha_for_relu=dense_output_activ_func_alpha,
                        alpha_for_elu=dense_output_activ_func_alpha,
                        layer_name=this_name
                    )(dense_output_layer_object)
                )
        else:
            dense_output_layer_object = (
                architecture_utils.get_activation_layer(
                    activation_function_string=inner_activ_function_name,
                    alpha_for_relu=inner_activ_function_alpha,
                    alpha_for_elu=inner_activ_function_alpha
                )(dense_output_layer_object)
            )

        if dense_layer_dropout_rates[j] > 0:
            this_name = (
                'dense_output' if j == num_dense_layers - 1 else None
            )
            this_mc_flag = bool(dense_layer_mc_dropout_flags[j])

            dense_output_layer_object = (
                architecture_utils.get_dropout_layer(
                    dropout_fraction=dense_layer_dropout_rates[j],
                    layer_name=this_name
                )(dense_output_layer_object, training=this_mc_flag)
            )

        if use_batch_normalization and j != num_dense_layers - 1:
            dense_output_layer_object = (
                architecture_utils.get_batch_norm_layer()(
                    dense_output_layer_object
                )
            )

    if ensemble_size > 1:
        metric_function_list = []
    else:
        metric_function_list = neural_net.METRIC_FUNCTION_LIST

    if has_dense_layers:
        output_layer_objects.insert(1, dense_output_layer_object)
        loss_dict['dense_output'] = scalar_loss_function

    model_object = keras.models.Model(
        inputs=input_layer_object, outputs=output_layer_objects
    )
    model_object.compile(
        loss=loss_dict,
        optimizer=keras.optimizers.Nadam(),
        metrics=metric_function_list
    )

    model_object.summary()
    return model_object


def create_model_1output_layer(option_dict):
    """Same as `create_model`, except that output layers are joined into one.

    :param option_dict: See doc for `u_net_architecture.check_args`.
    :return: model_object: Instance of `keras.models.Model`.
    """

    option_dict[u_net_arch.USE_DEEP_SUPERVISION_KEY] = False
    option_dict[u_net_arch.ENSEMBLE_SIZE_KEY] = 1
    option_dict[u_net_arch.DO_INLINE_NORMALIZATION_KEY] = False
    option_dict = u_net_arch.check_args(option_dict)

    input_dimensions = option_dict[INPUT_DIMENSIONS_KEY]
    num_levels = option_dict[NUM_LEVELS_KEY]
    num_conv_layers_by_level = option_dict[CONV_LAYER_COUNTS_KEY]
    num_channels_by_level = option_dict[CHANNEL_COUNTS_KEY]
    encoder_dropout_rate_by_level = option_dict[ENCODER_DROPOUT_RATES_KEY]
    encoder_mc_dropout_flag_by_level = option_dict[ENCODER_MC_DROPOUT_FLAGS_KEY]
    upconv_dropout_rate_by_level = option_dict[UPCONV_DROPOUT_RATES_KEY]
    upconv_mc_dropout_flag_by_level = option_dict[UPCONV_MC_DROPOUT_FLAGS_KEY]
    skip_dropout_rate_by_level = option_dict[SKIP_DROPOUT_RATES_KEY]
    skip_mc_dropout_flag_by_level = option_dict[SKIP_MC_DROPOUT_FLAGS_KEY]
    include_penultimate_conv = option_dict[INCLUDE_PENULTIMATE_KEY]
    penultimate_conv_dropout_rate = option_dict[PENULTIMATE_DROPOUT_RATE_KEY]
    penultimate_conv_mc_dropout_flag = (
        option_dict[PENULTIMATE_MC_DROPOUT_FLAG_KEY]
    )
    dense_layer_neuron_nums = option_dict[DENSE_LAYER_NEURON_NUMS_KEY]
    dense_layer_dropout_rates = option_dict[DENSE_LAYER_DROPOUT_RATES_KEY]
    dense_layer_mc_dropout_flags = option_dict[DENSE_LAYER_MC_DROPOUT_FLAGS_KEY]
    inner_activ_function_name = option_dict[INNER_ACTIV_FUNCTION_KEY]
    inner_activ_function_alpha = option_dict[INNER_ACTIV_FUNCTION_ALPHA_KEY]
    conv_output_activ_func_name = option_dict[CONV_OUTPUT_ACTIV_FUNC_KEY]
    conv_output_activ_func_alpha = option_dict[CONV_OUTPUT_ACTIV_FUNC_ALPHA_KEY]
    dense_output_activ_func_name = option_dict[DENSE_OUTPUT_ACTIV_FUNC_KEY]
    dense_output_activ_func_alpha = (
        option_dict[DENSE_OUTPUT_ACTIV_FUNC_ALPHA_KEY]
    )
    l1_weight = option_dict[L1_WEIGHT_KEY]
    l2_weight = option_dict[L2_WEIGHT_KEY]
    use_batch_normalization = option_dict[USE_BATCH_NORM_KEY]

    num_output_wavelengths = option_dict[NUM_OUTPUT_WAVELENGTHS_KEY]
    loss_function = option_dict[JOINED_LOSS_FUNCTION_KEY]

    has_dense_layers = dense_layer_neuron_nums is not None

    input_layer_object = keras.layers.Input(
        shape=tuple(input_dimensions.tolist())
    )
    regularizer_object = architecture_utils.get_weight_regularizer(
        l1_weight=l1_weight, l2_weight=l2_weight
    )

    encoder_conv_layer_objects = [None] * (num_levels + 1)
    encoder_pooling_layer_objects = [None] * num_levels

    for i in range(num_levels + 1):
        for k in range(num_conv_layers_by_level[i]):
            if k == 0:
                if i == 0:
                    this_input_layer_object = input_layer_object
                else:
                    this_input_layer_object = (
                        encoder_pooling_layer_objects[i - 1]
                    )
            else:
                this_input_layer_object = encoder_conv_layer_objects[i]

            this_name = 'encoder{0:d}_conv{1:d}'.format(i, k)

            encoder_conv_layer_objects[i] = architecture_utils.get_1d_conv_layer(
                num_kernel_rows=3, num_rows_per_stride=1,
                num_filters=num_channels_by_level[i],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object, layer_name=this_name
            )(this_input_layer_object)

            this_name = 'encoder{0:d}_conv{1:d}_activation'.format(i, k)

            encoder_conv_layer_objects[i] = (
                architecture_utils.get_activation_layer(
                    activation_function_string=inner_activ_function_name,
                    alpha_for_relu=inner_activ_function_alpha,
                    alpha_for_elu=inner_activ_function_alpha,
                    layer_name=this_name
                )(encoder_conv_layer_objects[i])
            )

            if encoder_dropout_rate_by_level[i] > 0:
                this_name = 'encoder{0:d}_conv{1:d}_dropout'.format(i, k)
                this_mc_flag = bool(encoder_mc_dropout_flag_by_level[i])

                encoder_conv_layer_objects[i] = (
                    architecture_utils.get_dropout_layer(
                        dropout_fraction=encoder_dropout_rate_by_level[i],
                        layer_name=this_name
                    )(encoder_conv_layer_objects[i], training=this_mc_flag)
                )

            if use_batch_normalization:
                this_name = 'encoder{0:d}_conv{1:d}_batchnorm'.format(i, k)

                encoder_conv_layer_objects[i] = (
                    architecture_utils.get_batch_norm_layer(
                        layer_name=this_name
                    )(encoder_conv_layer_objects[i])
                )

        if i != num_levels:
            this_name = 'encoder{0:d}_pooling'.format(i)

            encoder_pooling_layer_objects[i] = (
                architecture_utils.get_1d_pooling_layer(
                    num_rows_in_window=2, num_rows_per_stride=2,
                    pooling_type_string=architecture_utils.MAX_POOLING_STRING,
                    layer_name=this_name
                )(encoder_conv_layer_objects[i])
            )

    decoder_conv_layer_objects = [None] * num_levels

    for i in range(num_levels)[::-1]:
        layer_objects_to_concat = []

        for j in range(i + 1, num_levels + 1):
            if j == num_levels:
                layer_name_prefix = (
                    'decoder{0:d}_upsample_encoder{1:d}'.format(i, j)
                )

                this_layer_object = keras.layers.UpSampling1D(
                    size=2 ** (j - i), name=layer_name_prefix
                )(encoder_conv_layer_objects[j])
            else:
                layer_name_prefix = (
                    'decoder{0:d}_upsample_decoder{1:d}'.format(i, j)
                )

                this_layer_object = keras.layers.UpSampling1D(
                    size=2 ** (j - i), name=layer_name_prefix
                )(decoder_conv_layer_objects[j])

            num_upsampled_heights = this_layer_object.shape[1]
            num_desired_heights = encoder_conv_layer_objects[i].shape[1]

            if num_desired_heights != num_upsampled_heights:
                this_name = '{0:s}_padding'.format(layer_name_prefix)

                num_extra_heights = num_desired_heights - num_upsampled_heights
                num_extra_heights_bottom = int(numpy.floor(
                    float(num_extra_heights) / 2
                ))
                num_extra_heights_top = int(numpy.ceil(
                    float(num_extra_heights) / 2
                ))

                this_layer_object = keras.layers.ZeroPadding1D(
                    padding=(num_extra_heights_bottom, num_extra_heights_top),
                    name=this_name
                )(this_layer_object)

            this_name = '{0:s}_conv'.format(layer_name_prefix)

            this_layer_object = architecture_utils.get_1d_conv_layer(
                num_kernel_rows=3, num_rows_per_stride=1,
                num_filters=num_channels_by_level[0],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object, layer_name=this_name
            )(this_layer_object)

            this_name = '{0:s}_activation'.format(layer_name_prefix)

            this_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha,
                layer_name=this_name
            )(this_layer_object)

            if upconv_dropout_rate_by_level[i] > 0:
                this_name = '{0:s}_dropout'.format(layer_name_prefix)
                this_mc_flag = bool(upconv_mc_dropout_flag_by_level[i])

                this_layer_object = architecture_utils.get_dropout_layer(
                    dropout_fraction=upconv_dropout_rate_by_level[
                        upconv_dropout_rate_by_level[i]
                    ],
                    layer_name=this_name
                )(this_layer_object, training=this_mc_flag)

            if use_batch_normalization:
                this_name = '{0:s}_batchnorm'.format(layer_name_prefix)

                this_layer_object = architecture_utils.get_batch_norm_layer(
                    layer_name=this_name
                )(this_layer_object)

            layer_objects_to_concat.append(this_layer_object)

        layer_name_prefix = 'decoder{0:d}_carry_encoder{0:d}'.format(i)
        this_name = '{0:s}_conv'.format(layer_name_prefix)

        this_layer_object = architecture_utils.get_1d_conv_layer(
            num_kernel_rows=3, num_rows_per_stride=1,
            num_filters=num_channels_by_level[0],
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object, layer_name=this_name
        )(encoder_conv_layer_objects[i])

        this_name = '{0:s}_activation'.format(layer_name_prefix)

        this_layer_object = architecture_utils.get_activation_layer(
            activation_function_string=inner_activ_function_name,
            alpha_for_relu=inner_activ_function_alpha,
            alpha_for_elu=inner_activ_function_alpha,
            layer_name=this_name
        )(this_layer_object)

        if upconv_dropout_rate_by_level[i] > 0:
            this_name = '{0:s}_dropout'.format(layer_name_prefix)
            this_mc_flag = bool(upconv_mc_dropout_flag_by_level[i])

            this_layer_object = architecture_utils.get_dropout_layer(
                dropout_fraction=upconv_dropout_rate_by_level[
                    upconv_dropout_rate_by_level[i]
                ],
                layer_name=this_name
            )(this_layer_object, training=this_mc_flag)

        if use_batch_normalization:
            this_name = '{0:s}_batchnorm'.format(layer_name_prefix)

            this_layer_object = architecture_utils.get_batch_norm_layer(
                layer_name=this_name
            )(this_layer_object)

        layer_objects_to_concat.append(this_layer_object)

        for j in range(i):
            layer_name_prefix = (
                'decoder{0:d}_downsample_encoder{1:d}'.format(i, j)
            )
            this_name = '{0:s}_pooling'.format(layer_name_prefix)

            this_layer_object = architecture_utils.get_1d_pooling_layer(
                num_rows_in_window=2 ** (i - j),
                num_rows_per_stride=2 ** (i - j),
                pooling_type_string=architecture_utils.MAX_POOLING_STRING,
                layer_name=this_name
            )(encoder_conv_layer_objects[j])

            this_name = '{0:s}_conv'.format(layer_name_prefix)

            this_layer_object = architecture_utils.get_1d_conv_layer(
                num_kernel_rows=3, num_rows_per_stride=1,
                num_filters=num_channels_by_level[0],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object, layer_name=this_name
            )(this_layer_object)

            this_name = '{0:s}_activation'.format(layer_name_prefix)

            this_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha,
                layer_name=this_name
            )(this_layer_object)

            if upconv_dropout_rate_by_level[i] > 0:
                this_name = '{0:s}_dropout'.format(layer_name_prefix)
                this_mc_flag = bool(upconv_mc_dropout_flag_by_level[i])

                this_layer_object = architecture_utils.get_dropout_layer(
                    dropout_fraction=upconv_dropout_rate_by_level[
                        upconv_dropout_rate_by_level[i]
                    ],
                    layer_name=this_name
                )(this_layer_object, training=this_mc_flag)

            if use_batch_normalization:
                this_name = '{0:s}_batchnorm'.format(layer_name_prefix)

                this_layer_object = architecture_utils.get_batch_norm_layer(
                    layer_name=this_name
                )(this_layer_object)

            layer_objects_to_concat.append(this_layer_object)

        this_name = 'decoder{0:d}_concat'.format(i)
        decoder_conv_layer_objects[i] = keras.layers.Concatenate(
            axis=-1, name=this_name
        )(layer_objects_to_concat)

        this_num_channels = (
            num_channels_by_level[0] * len(layer_objects_to_concat)
        )

        for k in range(num_conv_layers_by_level[i]):
            this_name = 'decoder{0:d}_conv{1:d}'.format(i, k)

            decoder_conv_layer_objects[i] = (
                architecture_utils.get_1d_conv_layer(
                    num_kernel_rows=3, num_rows_per_stride=1,
                    num_filters=this_num_channels,
                    padding_type_string=architecture_utils.YES_PADDING_STRING,
                    weight_regularizer=regularizer_object,
                    layer_name=this_name
                )(decoder_conv_layer_objects[i])
            )

            this_name = 'decoder{0:d}_conv{1:d}_activation'.format(i, k)

            decoder_conv_layer_objects[i] = (
                architecture_utils.get_activation_layer(
                    activation_function_string=inner_activ_function_name,
                    alpha_for_relu=inner_activ_function_alpha,
                    alpha_for_elu=inner_activ_function_alpha,
                    layer_name=this_name
                )(decoder_conv_layer_objects[i])
            )

            if skip_dropout_rate_by_level[i] > 0:
                this_name = 'decoder{0:d}_conv{1:d}_dropout'.format(i, k)
                this_mc_flag = bool(skip_mc_dropout_flag_by_level[i])

                decoder_conv_layer_objects[i] = (
                    architecture_utils.get_dropout_layer(
                        dropout_fraction=skip_dropout_rate_by_level[i],
                        layer_name=this_name
                    )(decoder_conv_layer_objects[i], training=this_mc_flag)
                )

            if use_batch_normalization:
                this_name = 'decoder{0:d}_conv{1:d}_batchnorm'.format(i, k)

                decoder_conv_layer_objects[i] = (
                    architecture_utils.get_batch_norm_layer(
                        layer_name=this_name
                    )(decoder_conv_layer_objects[i])
                )

    if include_penultimate_conv:
        conv_output_layer_object = architecture_utils.get_1d_conv_layer(
            num_kernel_rows=3, num_rows_per_stride=1,
            num_filters=2 * num_output_wavelengths,
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object, layer_name='penultimate_conv'
        )(decoder_conv_layer_objects[0])

        conv_output_layer_object = architecture_utils.get_activation_layer(
            activation_function_string=inner_activ_function_name,
            alpha_for_relu=inner_activ_function_alpha,
            alpha_for_elu=inner_activ_function_alpha,
            layer_name='penultimate_conv_activation'
        )(conv_output_layer_object)

        if penultimate_conv_dropout_rate > 0:
            this_mc_flag = bool(penultimate_conv_mc_dropout_flag)

            conv_output_layer_object = (
                architecture_utils.get_dropout_layer(
                    dropout_fraction=penultimate_conv_dropout_rate,
                    layer_name='penultimate_conv_dropout'
                )(conv_output_layer_object, training=this_mc_flag)
            )

        if use_batch_normalization:
            conv_output_layer_object = (
                architecture_utils.get_batch_norm_layer(
                    layer_name='penultimate_conv_bn'
                )(conv_output_layer_object)
            )
    else:
        conv_output_layer_object = decoder_conv_layer_objects[0]

    conv_output_layer_object = architecture_utils.get_1d_conv_layer(
        num_kernel_rows=1, num_rows_per_stride=1,
        num_filters=num_output_wavelengths,
        padding_type_string=architecture_utils.YES_PADDING_STRING,
        weight_regularizer=regularizer_object, layer_name='last_conv'
    )(conv_output_layer_object)

    if conv_output_activ_func_name is not None:
        conv_output_layer_object = architecture_utils.get_activation_layer(
            activation_function_string=conv_output_activ_func_name,
            alpha_for_relu=conv_output_activ_func_alpha,
            alpha_for_elu=conv_output_activ_func_alpha,
            layer_name='last_conv_activation'
        )(conv_output_layer_object)

    conv_output_layer_object = u_net_arch.zero_top_heating_rate(
        input_layer_object=conv_output_layer_object,
        ensemble_size=1,
        output_layer_name='conv_output'
    )

    if has_dense_layers:
        num_dense_layers = len(dense_layer_neuron_nums)
        dense_output_layer_object = architecture_utils.get_flattening_layer()(
            encoder_conv_layer_objects[-1]
        )
    else:
        num_dense_layers = 0
        dense_output_layer_object = None

    for j in range(num_dense_layers):
        dense_output_layer_object = architecture_utils.get_dense_layer(
            num_output_units=dense_layer_neuron_nums[j]
        )(dense_output_layer_object)

        if j == num_dense_layers - 1:
            if (
                    dense_layer_dropout_rates[j] <= 0 and
                    dense_output_activ_func_name is None
            ):
                this_name = 'dense_output'
            else:
                this_name = None

            num_dense_output_vars = (
                float(dense_layer_neuron_nums[j]) / num_output_wavelengths
            )
            assert numpy.isclose(
                num_dense_output_vars, numpy.round(num_dense_output_vars),
                atol=1e-6
            )
            num_dense_output_vars = int(numpy.round(num_dense_output_vars))

            dense_output_layer_object = keras.layers.Reshape(
                target_shape=(num_output_wavelengths, num_dense_output_vars),
                name=this_name
            )(dense_output_layer_object)

            if dense_output_activ_func_name is not None:
                this_name = (
                    None if dense_layer_dropout_rates[j] > 0 else 'dense_output'
                )

                dense_output_layer_object = (
                    architecture_utils.get_activation_layer(
                        activation_function_string=dense_output_activ_func_name,
                        alpha_for_relu=dense_output_activ_func_alpha,
                        alpha_for_elu=dense_output_activ_func_alpha,
                        layer_name=this_name
                    )(dense_output_layer_object)
                )
        else:
            dense_output_layer_object = (
                architecture_utils.get_activation_layer(
                    activation_function_string=inner_activ_function_name,
                    alpha_for_relu=inner_activ_function_alpha,
                    alpha_for_elu=inner_activ_function_alpha
                )(dense_output_layer_object)
            )

        if dense_layer_dropout_rates[j] > 0:
            this_name = (
                'dense_output' if j == num_dense_layers - 1 else None
            )
            this_mc_flag = bool(dense_layer_mc_dropout_flags[j])

            dense_output_layer_object = (
                architecture_utils.get_dropout_layer(
                    dropout_fraction=dense_layer_dropout_rates[j],
                    layer_name=this_name
                )(dense_output_layer_object, training=this_mc_flag)
            )

        if use_batch_normalization and j != num_dense_layers - 1:
            dense_output_layer_object = (
                architecture_utils.get_batch_norm_layer()(
                    dense_output_layer_object
                )
            )

    if has_dense_layers:
        dense_output_layer_object = keras.layers.Permute((2, 1))(
            dense_output_layer_object
        )
        output_layer_object = keras.layers.Concatenate(axis=1)(
            [conv_output_layer_object, dense_output_layer_object]
        )
        model_object = keras.models.Model(
            inputs=input_layer_object, outputs=output_layer_object
        )
    else:
        model_object = keras.models.Model(
            inputs=input_layer_object, outputs=conv_output_layer_object
        )

    model_object.compile(
        loss=loss_function,
        optimizer=keras.optimizers.Nadam(),
        metrics=neural_net.METRIC_FUNCTION_LIST
    )

    model_object.summary()
    return model_object
