"""Methods for building U-net++.

Based on: https://github.com/longuyen97/UnetPlusPlus/blob/master/unetpp.py
"""

import numpy
import keras
import keras.layers
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import architecture_utils
from ml4rt.machine_learning import neural_net
from ml4rt.machine_learning import u_net_architecture as u_net_arch

POINT_ESTIMATE_TYPE_STRING = 'point_estimate'
FLIPOUT_TYPE_STRING = 'flipout'
REPARAMETERIZATION_TYPE_STRING = 'reparameterization'
VALID_CONV_LAYER_TYPE_STRINGS = [
    POINT_ESTIMATE_TYPE_STRING, FLIPOUT_TYPE_STRING,
    REPARAMETERIZATION_TYPE_STRING
]

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
DENSE_OUTPUT_ACTIV_FUNC_ALPHA_KEY = u_net_arch.DENSE_OUTPUT_ACTIV_FUNC_ALPHA_KEY
L1_WEIGHT_KEY = u_net_arch.L1_WEIGHT_KEY
L2_WEIGHT_KEY = u_net_arch.L2_WEIGHT_KEY
USE_BATCH_NORM_KEY = u_net_arch.USE_BATCH_NORM_KEY
USE_RESIDUAL_BLOCKS_KEY = u_net_arch.USE_RESIDUAL_BLOCKS_KEY

NUM_OUTPUT_WAVELENGTHS_KEY = u_net_arch.NUM_OUTPUT_WAVELENGTHS_KEY
VECTOR_LOSS_FUNCTION_KEY = u_net_arch.VECTOR_LOSS_FUNCTION_KEY
SCALAR_LOSS_FUNCTION_KEY = u_net_arch.SCALAR_LOSS_FUNCTION_KEY
JOINED_LOSS_FUNCTION_KEY = u_net_arch.JOINED_LOSS_FUNCTION_KEY
OPTIMIZER_FUNCTION_KEY = u_net_arch.OPTIMIZER_FUNCTION_KEY
USE_DEEP_SUPERVISION_KEY = u_net_arch.USE_DEEP_SUPERVISION_KEY
ENSEMBLE_SIZE_KEY = u_net_arch.ENSEMBLE_SIZE_KEY
INCLUDE_MASK_KEY = u_net_arch.INCLUDE_MASK_KEY

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
    USE_BATCH_NORM_KEY: True,
    USE_DEEP_SUPERVISION_KEY: False
}

KL_SCALING_FACTOR_KEY = 'kl_divergence_scaling_factor'
UPCONV_BNN_LAYER_TYPES_KEY = 'upconv_layer_type_string_by_level'
SKIP_BNN_LAYER_TYPES_KEY = 'skip_layer_type_string_by_level'
PENULTIMATE_BNN_LAYER_TYPE_KEY = 'penultimate_conv_layer_type_string'
DENSE_BNN_LAYER_TYPES_KEY = 'dense_layer_type_strings'
CONV_OUTPUT_BNN_LAYER_TYPE_KEY = 'conv_output_layer_type_string'


def _check_layer_type(layer_type_string):
    """Ensures that convolutional-layer type is valid.

    :param layer_type_string: Layer type (must be in list
        VALID_CONV_LAYER_TYPE_STRINGS).
    :raises ValueError: if
        `layer_type_string not in VALID_CONV_LAYER_TYPE_STRINGS`.
    """

    error_checking.assert_is_string(layer_type_string)

    if layer_type_string not in VALID_CONV_LAYER_TYPE_STRINGS:
        error_string = (
            'Valid conv-layer types (listed below) do not include "{0:s}":'
            '\n{1:s}'
        ).format(layer_type_string, str(VALID_CONV_LAYER_TYPE_STRINGS))

        raise ValueError(error_string)


def create_model(option_dict):
    """Creates U-net++.

    This method sets up the architecture, loss function, and optimizer -- and
    compiles the model -- but does not train it.

    Architecture based on:
    https://github.com/longuyen97/UnetPlusPlus/blob/master/unetpp.py

    M = number of rows in grid
    N = number of columns in grid

    :param option_dict: See doc for `u_net_architecture.check_args`.
    :return: model_object: Instance of `keras.models.Model`, with the
        aforementioned architecture.
    """

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
    use_residual_blocks = option_dict[USE_RESIDUAL_BLOCKS_KEY]

    num_output_wavelengths = option_dict[NUM_OUTPUT_WAVELENGTHS_KEY]
    vector_loss_function = option_dict[VECTOR_LOSS_FUNCTION_KEY]
    scalar_loss_function = option_dict[SCALAR_LOSS_FUNCTION_KEY]
    optimizer_function = option_dict[OPTIMIZER_FUNCTION_KEY]
    use_deep_supervision = option_dict[USE_DEEP_SUPERVISION_KEY]
    ensemble_size = option_dict[ENSEMBLE_SIZE_KEY]
    include_mask = option_dict[INCLUDE_MASK_KEY]

    # TODO(thunderhoser): Make deep supervision work again.
    assert not use_deep_supervision

    if ensemble_size > 1:
        use_deep_supervision = False
        metric_function_list = []
    else:
        metric_function_list = neural_net.METRIC_FUNCTION_LIST

    has_dense_layers = dense_layer_neuron_nums is not None
    if has_dense_layers:
        num_dense_output_vars = (
            float(dense_layer_neuron_nums[-1]) /
            (ensemble_size * num_output_wavelengths)
        )
        assert numpy.isclose(
            num_dense_output_vars, numpy.round(num_dense_output_vars),
            atol=1e-6
        )
        num_dense_output_vars = int(numpy.round(num_dense_output_vars))
    else:
        num_dense_output_vars = 0

    main_input_layer_object = keras.layers.Input(
        shape=tuple(input_dimensions.tolist()),
        name=neural_net.MAIN_PREDICTORS_KEY
    )
    hr_mask_input_layer_object = None
    flux_mask_input_layer_object = None

    if include_mask:
        hr_mask_input_layer_object = keras.layers.Input(
            shape=(input_dimensions[0], num_output_wavelengths),
            name=neural_net.HEATING_RATE_MASK_KEY
        )

        if has_dense_layers:
            flux_mask_input_layer_object = keras.layers.Input(
                shape=(num_output_wavelengths, num_dense_output_vars),
                name=neural_net.FLUX_MASK_KEY
            )

    regularizer_object = architecture_utils.get_weight_regularizer(
        l1_weight=l1_weight, l2_weight=l2_weight
    )

    last_conv_layer_matrix = numpy.full(
        (num_levels + 1, num_levels + 1), '', dtype=object
    )
    pooling_layer_by_level = [None] * num_levels

    for i in range(num_levels + 1):
        if i == 0:
            this_input_layer_object = main_input_layer_object
        else:
            this_input_layer_object = pooling_layer_by_level[i - 1]

        last_conv_layer_matrix[i, 0] = u_net_arch.get_conv_block(
            input_layer_object=this_input_layer_object,
            do_residual=use_residual_blocks,
            num_conv_layers=num_conv_layers_by_level[i],
            filter_size_px=3,
            num_filters=num_channels_by_level[i],
            regularizer_object=regularizer_object,
            activation_function_name=inner_activ_function_name,
            activation_function_alpha=inner_activ_function_alpha,
            dropout_rates=encoder_dropout_rate_by_level[i],
            monte_carlo_dropout_flags=encoder_mc_dropout_flag_by_level[i],
            use_batch_norm=use_batch_normalization,
            basic_layer_name='encoder{0:d}-0'.format(i)
        )

        if i != num_levels:
            this_name = 'encoder{0:d}-0_pooling'.format(i)

            pooling_layer_by_level[i] = architecture_utils.get_1d_pooling_layer(
                num_rows_in_window=2,
                num_rows_per_stride=2,
                pooling_type_string=architecture_utils.MAX_POOLING_STRING,
                layer_name=this_name
            )(last_conv_layer_matrix[i, 0])

        i_new = i + 0
        j = 0

        while i_new > 0:
            i_new -= 1
            j += 1

            this_name = 'decoder{0:d}-{1:d}_upsampling'.format(i_new, j)
            this_layer_object = keras.layers.UpSampling1D(
                size=2, name=this_name
            )(last_conv_layer_matrix[i_new + 1, j - 1])

            num_upconv_heights = this_layer_object.shape[1]
            num_desired_heights = (
                last_conv_layer_matrix[i_new, 0].shape[1]
            )

            if num_desired_heights == num_upconv_heights + 1:
                this_name = 'decoder{0:d}-{1:d}_padding'.format(i_new, j)
                this_layer_object = keras.layers.ZeroPadding1D(
                    padding=(0, 1), name=this_name
                )(this_layer_object)

            last_conv_layer_matrix[i_new, j] = u_net_arch.get_conv_block(
                input_layer_object=this_layer_object,
                do_residual=use_residual_blocks,
                num_conv_layers=1,
                filter_size_px=3,
                num_filters=num_channels_by_level[i_new],
                regularizer_object=regularizer_object,
                activation_function_name=inner_activ_function_name,
                activation_function_alpha=inner_activ_function_alpha,
                dropout_rates=upconv_dropout_rate_by_level[i_new],
                monte_carlo_dropout_flags=
                upconv_mc_dropout_flag_by_level[i_new],
                use_batch_norm=use_batch_normalization,
                basic_layer_name='decoder{0:d}-{1:d}'.format(i_new, j)
            )

            this_name = 'decoder{0:d}-{1:d}_skip'.format(i_new, j)
            last_conv_layer_matrix[i_new, j] = keras.layers.Concatenate(
                axis=-1, name=this_name
            )(last_conv_layer_matrix[i_new, :(j + 1)].tolist())

            # The weird if-statement for using dropout ensures that dropout is
            # used only on the outside edge of the U-shape, i.e., on the right
            # side of the U-shape.
            last_conv_layer_matrix[i_new, j] = u_net_arch.get_conv_block(
                input_layer_object=last_conv_layer_matrix[i_new, j],
                do_residual=use_residual_blocks,
                num_conv_layers=num_conv_layers_by_level[i_new],
                filter_size_px=3,
                num_filters=num_channels_by_level[i_new],
                regularizer_object=regularizer_object,
                activation_function_name=inner_activ_function_name,
                activation_function_alpha=inner_activ_function_alpha,
                dropout_rates=(
                    skip_dropout_rate_by_level[i_new] if i_new + j == num_levels
                    else 0.
                ),
                monte_carlo_dropout_flags=skip_mc_dropout_flag_by_level[i_new],
                use_batch_norm=use_batch_normalization,
                basic_layer_name='decoder{0:d}-{1:d}_skip'.format(i_new, j)
            )

    if include_penultimate_conv:
        last_conv_layer_matrix[0, -1] = u_net_arch.get_conv_block(
            input_layer_object=last_conv_layer_matrix[0, -1],
            do_residual=use_residual_blocks,
            num_conv_layers=1,
            filter_size_px=3,
            num_filters=2 * num_output_wavelengths * ensemble_size,
            regularizer_object=regularizer_object,
            activation_function_name=inner_activ_function_name,
            activation_function_alpha=inner_activ_function_alpha,
            dropout_rates=penultimate_conv_dropout_rate,
            monte_carlo_dropout_flags=penultimate_conv_mc_dropout_flag,
            use_batch_norm=use_batch_normalization,
            basic_layer_name='penultimate_conv'
        )

    conv_output_layer_object = u_net_arch.get_conv_block(
        input_layer_object=last_conv_layer_matrix[0, -1],
        do_residual=use_residual_blocks,
        num_conv_layers=1,
        filter_size_px=1,
        num_filters=num_output_wavelengths * ensemble_size,
        regularizer_object=regularizer_object,
        activation_function_name=conv_output_activ_func_name,
        activation_function_alpha=conv_output_activ_func_alpha,
        dropout_rates=-1.,
        monte_carlo_dropout_flags=False,
        use_batch_norm=False,
        basic_layer_name='last_conv'
    )

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

    if hr_mask_input_layer_object is None:
        conv_output_layer_object = u_net_arch.zero_top_heating_rate(
            input_layer_object=conv_output_layer_object,
            ensemble_size=ensemble_size,
            output_layer_name=neural_net.HEATING_RATE_TARGETS_KEY
        )
    else:
        if ensemble_size > 1:
            hr_mask_layer_object = keras.layers.Reshape(
                target_shape=
                (input_dimensions[0], num_output_wavelengths, 1, ensemble_size)
            )(hr_mask_input_layer_object)
        else:
            hr_mask_layer_object = keras.layers.Reshape(
                target_shape=
                (input_dimensions[0], num_output_wavelengths, 1)
            )(hr_mask_input_layer_object)

        conv_output_layer_object = keras.layers.Multiply(
            name=neural_net.HEATING_RATE_TARGETS_KEY
        )([conv_output_layer_object, hr_mask_layer_object])

    output_layer_objects = [conv_output_layer_object]
    loss_dict = {neural_net.HEATING_RATE_TARGETS_KEY: vector_loss_function}
    metric_dict = {neural_net.HEATING_RATE_TARGETS_KEY: metric_function_list}

    deep_supervision_layer_objects = [None] * num_levels

    if use_deep_supervision:
        for i in range(1, num_levels):
            deep_supervision_layer_objects[i] = u_net_arch.get_conv_block(
                input_layer_object=last_conv_layer_matrix[0, i],
                do_residual=use_residual_blocks,
                num_conv_layers=1,
                filter_size_px=1,
                num_filters=num_output_wavelengths,
                regularizer_object=regularizer_object,
                activation_function_name=conv_output_activ_func_name,
                activation_function_alpha=conv_output_activ_func_alpha,
                dropout_rates=-1.,
                monte_carlo_dropout_flags=False,
                use_batch_norm=False,
                basic_layer_name='deepsup{0:d}'.format(i)
            )

            deep_supervision_layer_objects[i] = keras.layers.Reshape(
                target_shape=
                (input_dimensions[0], num_output_wavelengths, 1)
            )(deep_supervision_layer_objects[i])

            this_name = 'deepsup{0:d}_output'.format(i)
            deep_supervision_layer_objects[i] = (
                u_net_arch.zero_top_heating_rate(
                    input_layer_object=deep_supervision_layer_objects[i],
                    ensemble_size=1,
                    output_layer_name=this_name
                )
            )

            output_layer_objects.append(deep_supervision_layer_objects[i])
            loss_dict[this_name] = vector_loss_function
            metric_dict[this_name] = metric_function_list

    if has_dense_layers:
        num_dense_layers = len(dense_layer_neuron_nums)
        dense_output_layer_object = architecture_utils.get_flattening_layer()(
            last_conv_layer_matrix[-1, 0]
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
                    dense_layer_dropout_rates[j] <= 0
                    and dense_output_activ_func_name is None
                    and flux_mask_input_layer_object is None
            ):
                this_name = neural_net.FLUX_TARGETS_KEY
            else:
                this_name = None

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
                if (
                        dense_layer_dropout_rates[j] <= 0
                        and flux_mask_input_layer_object is None
                ):
                    this_name = neural_net.FLUX_TARGETS_KEY
                else:
                    this_name = None

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
            if (
                    j == num_dense_layers - 1
                    and flux_mask_input_layer_object is None
            ):
                this_name = neural_net.FLUX_TARGETS_KEY
            else:
                this_name = None

            this_mc_flag = bool(dense_layer_mc_dropout_flags[j])

            dense_output_layer_object = (
                architecture_utils.get_dropout_layer(
                    dropout_fraction=dense_layer_dropout_rates[j],
                    layer_name=this_name
                )(dense_output_layer_object, training=this_mc_flag)
            )

        if (
                j == num_dense_layers - 1
                and flux_mask_input_layer_object is not None
        ):
            if ensemble_size > 1:
                these_dim = (
                    num_output_wavelengths, num_dense_output_vars, ensemble_size
                )
                flux_mask_layer_object = keras.layers.Reshape(
                    target_shape=these_dim
                )(flux_mask_input_layer_object)
            else:
                flux_mask_layer_object = flux_mask_input_layer_object

            dense_output_layer_object = keras.layers.Multiply(
                name=neural_net.FLUX_TARGETS_KEY
            )([dense_output_layer_object, flux_mask_layer_object])

        if use_batch_normalization and j != num_dense_layers - 1:
            dense_output_layer_object = (
                architecture_utils.get_batch_norm_layer()(
                    dense_output_layer_object
                )
            )

    if has_dense_layers:
        output_layer_objects.insert(1, dense_output_layer_object)
        loss_dict[neural_net.FLUX_TARGETS_KEY] = scalar_loss_function
        metric_dict[neural_net.FLUX_TARGETS_KEY] = metric_function_list

    input_layer_objects = [
        main_input_layer_object,
        hr_mask_input_layer_object,
        flux_mask_input_layer_object
    ]
    input_layer_objects = [l for l in input_layer_objects if l is not None]

    model_object = keras.models.Model(
        inputs=input_layer_objects, outputs=output_layer_objects
    )
    model_object.compile(
        loss=loss_dict,
        optimizer=optimizer_function,
        metrics=metric_dict
    )

    model_object.summary()
    return model_object
