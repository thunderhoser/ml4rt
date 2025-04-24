"""Methods for implementing ConvNeXt and ConvNeXt-v2 blocks."""

import os
import sys
import keras
import tensorflow

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import architecture_utils

EPSILON_FOR_LAYER_NORM = 1e-6
EXPANSION_FACTOR_FOR_CONVNEXT = 4
INIT_VALUE_FOR_LAYER_SCALE = 1e-6


# @keras.saving.register_keras_serializable()
class LayerScale(keras.layers.Layer):
    """Layer-scale module.

    Scavenged from: https://github.com/danielabdi-noaa/HRRRemulator/blob/
                    master/tfmodel/convnext.py
    """

    def __init__(self, init_values, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim

    def build(self, _):
        self.gamma = self.add_weight(
            shape=(self.projection_dim,),
            initializer=keras.initializers.Constant(self.init_values),
            trainable=True,
            name="gamma",
        )

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "init_values": self.init_values,
                "projection_dim": self.projection_dim,
            }
        )
        return config


# @keras.saving.register_keras_serializable()
class GRN(keras.layers.Layer):
    """Global response normalization.

    Scavenged from: https://github.com/facebookresearch/ConvNeXt-V2/blob/
                    2553895753323c6fe0b2bf390683f5ea358a42b9/models/
                    utils.py#L105-L116
    """

    def __init__(self, init_values, projection_dim, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim
        self.epsilon = epsilon

    def build(self, _):

        # TODO(thunderhoser): Not sure what the initial values should be.
        # The linked webpage uses zeros; the ChatGPT suggestion
        # (https://chatgpt.com/c/6740daf4-b010-8013-86ee-b6329e8ea6e9)
        # uses gamma = 1 and beta = 0; meanwhile, LayerScale from ConvNext 1
        # uses gamma = 1e-6.
        self.gamma = self.add_weight(
            shape=(self.projection_dim,),
            initializer=keras.initializers.Constant(self.init_values),
            trainable=True,
            name="gamma",
        )
        self.beta = self.add_weight(
            shape=(self.projection_dim,),
            initializer=keras.initializers.Constant(self.init_values),
            trainable=True,
            name="beta",
        )

    def call(self, inputs):
        # gx = tensorflow.norm(
        #     inputs, ord=2, axis=(1, 2), keepdims=True
        # )
        gx = tensorflow.sqrt(tensorflow.reduce_sum(
            tensorflow.square(inputs), axis=(1, 2), keepdims=True
        ))
        denominator = self.epsilon + tensorflow.math.reduce_mean(
            gx, axis=-1, keepdims=True
        )
        nx = gx / denominator

        return (self.gamma * nx * inputs) + self.beta + inputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "init_values": self.init_values,
                "projection_dim": self.projection_dim,
                "epsilon": self.epsilon
            }
        )
        return config


# @keras.saving.register_keras_serializable()
class StochasticDepth(keras.layers.Layer):
    def __init__(self, survival_prob=0.9, **kwargs):
        super().__init__(**kwargs)
        self.survival_prob = survival_prob

    def call(self, inputs, training=None):
        if not training:
            return inputs[0] + inputs[1]

        batch_size = tensorflow.shape(inputs[0])[0]
        random_tensor = self.survival_prob + tensorflow.random.uniform(
            [batch_size, 1, 1, 1]
        )
        binary_tensor = tensorflow.floor(random_tensor)
        output = inputs[0] + binary_tensor * inputs[1] / self.survival_prob
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "survival_prob": self.survival_prob
            }
        )
        return config


def get_convnext_v1_block(
        input_layer_object, num_conv_layers, filter_size_px, num_filters,
        regularizer_object, do_activation, dropout_rate, use_simple_version,
        basic_layer_name):
    """Creates basic ConvNeXt block (version 1).

    :param input_layer_object: Input layer to the ConvNeXt block.
    :param num_conv_layers: Number of conv layers in block.
    :param filter_size_px: Filter size (the same for every conv layer).
    :param num_filters: Number of filters (the same for every conv layer).
    :param regularizer_object: Regularizer for conv layers (instance o
        `keras.regularizers.l1_l2` or similar).
    :param do_activation: Boolean flag.
    :param dropout_rate: Dropout rate for the whole conv block.  If
        dropout_rate <= 0, there will be no dropout.  If dropout_rate > 0, will
        use stochastic dropout or "stochastic depth".
    :param use_simple_version: Boolean flag.  If True, will use the simple
        version of ConvNeXt, without layer normalization.
    :param basic_layer_name: Basic layer name.  Each layer name will be made
        unique by adding a suffix.
    :return: output_layer_object: Output layer from the ConvNeXt block.
    """

    # TODO(thunderhoser): HACK.
    if filter_size_px == 3:
        actual_filter_size_px = 7
    else:
        actual_filter_size_px = filter_size_px + 0

    current_layer_object = None

    for i in range(num_conv_layers):
        if i == 0:
            this_input_layer_object = input_layer_object
        else:
            this_input_layer_object = current_layer_object

        this_name = '{0:s}_conv{1:d}'.format(basic_layer_name, i)

        current_layer_object = architecture_utils.get_1d_depthwise_conv_layer(
            num_kernel_rows=actual_filter_size_px,
            num_rows_per_stride=1,
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object,
            layer_name=this_name
        )(this_input_layer_object)

        if not use_simple_version:
            this_name = '{0:s}_lyrnorm{1:d}'.format(basic_layer_name, i)
            current_layer_object = keras.layers.LayerNormalization(
                epsilon=EPSILON_FOR_LAYER_NORM, name=this_name
            )(
                current_layer_object
            )

        this_name = '{0:s}_dense{1:d}a'.format(basic_layer_name, i)
        dense_layer_object = architecture_utils.get_dense_layer(
            num_output_units=EXPANSION_FACTOR_FOR_CONVNEXT * num_filters,
            weight_regularizer=regularizer_object,
            layer_name=this_name
        )
        current_layer_object = dense_layer_object(current_layer_object)

        if do_activation:
            this_name = '{0:s}_gelu{1:d}'.format(basic_layer_name, i)
            current_layer_object = keras.layers.Activation(
                'gelu', name=this_name
            )(current_layer_object)

        this_name = '{0:s}_dense{1:d}b'.format(basic_layer_name, i)
        current_layer_object = architecture_utils.get_dense_layer(
            num_output_units=num_filters,
            weight_regularizer=regularizer_object,
            layer_name=this_name
        )(current_layer_object)

        this_name = '{0:s}_lyrscale{1:d}'.format(basic_layer_name, i)
        current_layer_object = LayerScale(
            INIT_VALUE_FOR_LAYER_SCALE, num_filters, name=this_name
        )(current_layer_object)

        if i != num_conv_layers - 1:
            continue

        if input_layer_object.shape[-1] == num_filters:
            new_layer_object = input_layer_object
        else:
            this_name = '{0:s}_preresidual_conv'.format(basic_layer_name)
            new_layer_object = architecture_utils.get_1d_conv_layer(
                num_kernel_rows=1,
                num_rows_per_stride=1,
                num_filters=num_filters,
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object,
                layer_name=this_name
            )(input_layer_object)

        this_name = '{0:s}_residual'.format(basic_layer_name)

        if dropout_rate > 0:
            current_layer_object = StochasticDepth(
                survival_prob=1. - dropout_rate, name=this_name
            )([new_layer_object, current_layer_object])
        else:
            current_layer_object = keras.layers.Add(name=this_name)([
                new_layer_object, current_layer_object
            ])

    return current_layer_object


def get_convnext_v2_block(
        input_layer_object, num_conv_layers, filter_size_px, num_filters,
        regularizer_object, do_activation, dropout_rate, use_simple_version,
        basic_layer_name):
    """Creates ConvNeXt-v2 block (version 2).

    :param input_layer_object: See documentation for `get_convnext_v1_block`.
    :param num_conv_layers: Same.
    :param filter_size_px: Same.
    :param num_filters: Same.
    :param regularizer_object: Same.
    :param do_activation: Same.
    :param dropout_rate: Same.
    :param use_simple_version: Boolean flag.  If True, will use the simple
        version of ConvNeXt, without layer normalization or global residual
        normalization.
    :param basic_layer_name: Same.
    :return: output_layer_object: Same.
    """

    # TODO(thunderhoser): HACK.
    if filter_size_px == 3:
        actual_filter_size_px = 7
    else:
        actual_filter_size_px = filter_size_px + 0

    current_layer_object = None

    for i in range(num_conv_layers):
        if i == 0:
            this_input_layer_object = input_layer_object
        else:
            this_input_layer_object = current_layer_object

        this_name = '{0:s}_conv{1:d}'.format(basic_layer_name, i)

        current_layer_object = architecture_utils.get_1d_depthwise_conv_layer(
            num_kernel_rows=actual_filter_size_px,
            num_rows_per_stride=1,
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object,
            layer_name=this_name
        )(this_input_layer_object)

        if not use_simple_version:
            this_name = '{0:s}_lyrnorm{1:d}'.format(basic_layer_name, i)
            current_layer_object = keras.layers.LayerNormalization(
                epsilon=EPSILON_FOR_LAYER_NORM, name=this_name
            )(
                current_layer_object
            )

        this_name = '{0:s}_dense{1:d}a'.format(basic_layer_name, i)
        current_layer_object = architecture_utils.get_dense_layer(
            num_output_units=EXPANSION_FACTOR_FOR_CONVNEXT * num_filters,
            weight_regularizer=regularizer_object,
            layer_name=this_name
        )(current_layer_object)

        if do_activation:
            this_name = '{0:s}_gelu{1:d}'.format(basic_layer_name, i)
            current_layer_object = keras.layers.Activation(
                'gelu', name=this_name
            )(current_layer_object)

        if not use_simple_version:
            this_name = '{0:s}_grn{1:d}'.format(basic_layer_name, i)
            current_layer_object = GRN(
                init_values=INIT_VALUE_FOR_LAYER_SCALE,
                projection_dim=EXPANSION_FACTOR_FOR_CONVNEXT * num_filters,
                name=this_name
            )(current_layer_object)

        this_name = '{0:s}_dense{1:d}b'.format(basic_layer_name, i)
        current_layer_object = architecture_utils.get_dense_layer(
            num_output_units=num_filters,
            weight_regularizer=regularizer_object,
            layer_name=this_name
        )(current_layer_object)

        if i != num_conv_layers - 1:
            continue

        if input_layer_object.shape[-1] == num_filters:
            new_layer_object = input_layer_object
        else:
            this_name = '{0:s}_preresidual_conv'.format(basic_layer_name)
            new_layer_object = architecture_utils.get_1d_conv_layer(
                num_kernel_rows=1,
                num_rows_per_stride=1,
                num_filters=num_filters,
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object,
                layer_name=this_name
            )(input_layer_object)

        this_name = '{0:s}_residual'.format(basic_layer_name)

        if dropout_rate > 0:
            current_layer_object = StochasticDepth(
                survival_prob=1. - dropout_rate, name=this_name
            )([new_layer_object, current_layer_object])
        else:
            current_layer_object = keras.layers.Add(name=this_name)([
                new_layer_object, current_layer_object
            ])

    return current_layer_object
