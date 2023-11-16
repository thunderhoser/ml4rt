"""Inline normalization of predictor data (with NN layers)."""

import numpy
import keras
import keras.layers
from keras import backend as K
import tensorflow
import tensorflow.math
from gewittergefahr.gg_utils import error_checking
from ml4rt.utils import normalization
from ml4rt.utils import example_utils

tensorflow.compat.v1.disable_eager_execution()


def _get_var_slicing_function(height_index, channel_index):
    """Creates function that grabs one variable from input tensor.

    :param height_index: Will grab the [j]th height, where j = `height_index`.
    :param channel_index: Will grab the [k]th channel, where
        k = `channel_index`.
    :return: var_slicing_function: Function handle (see below).
    """

    def var_slicing_function(input_tensor_3d):
        """Grabs one variable from input tensor.

        :param input_tensor_3d: Input tensor with three axes, where the second
            axis is height and the third is channel.
        :return: output_tensor_2d: Tensor with two axes, containing only one
            variable.  The second axis has a length of 1.  In other words, the
            second axis could be squeezed out, but it is not.
        """

        return K.expand_dims(
            input_tensor_3d[:, height_index, channel_index], axis=-1
        )

    return var_slicing_function


def _get_norm_ppf_function():
    """Creates function that computes point-perfect function (PPF) for SND.

    SND = standard normal distribution

    :return: norm_ppf_function: Function handle (see below).
    """

    def norm_ppf_function(input_tensor):
        """Computes PPF for standard normal distribution.

        :param input_tensor: Input tensor, containing quantiles ranging from
            0...1.
        :return: output_tensor: Output tensor, containing z-scores.
        """

        return (
            tensorflow.math.sqrt(2.) *
            tensorflow.math.erfinv(2 * input_tensor - 1)
        )

    return norm_ppf_function


def _get_zeroing_function():
    """Creates function that replaces all inputs with zero.

    :return: zeroing_function: Function handle (see below).
    """

    def zeroing_function(input_tensor_2d):
        """Replaces all inputs with zero.

        :param input_tensor_2d: Input tensor (shape E x 1).
        :return: output_tensor_3d: Zero tensor (shape E x 1 x 1).
        """

        output_tensor_2d = K.minimum(K.maximum(input_tensor_2d, 0.), 0.)
        return K.expand_dims(output_tensor_2d, axis=-2)

    return zeroing_function


def _get_pw_linear_regression_function(slopes, intercepts):
    """Creates function that computes piecewise-linear regression.

    B = number of bins (pieces)

    :param slopes: length-B numpy array of linear-regression slopes.
    :param intercepts: length-B numpy array of linear-regression intercepts.
    :return: pw_linear_regression_function: Function handle (see below).
    """

    def pw_linear_regression_function(input_tensor_2d,
                                      discretized_input_tensor_2d):
        """Computes piecewise-linear regression.

        E = number of examples
        B = number of bins (pieces)

        :param input_tensor_2d: Actual input tensor (shape E x 1).
        :param discretized_input_tensor_2d: Discretized input tensor (one-hot
            encoding of bin membership with shape E x B).
        :return: output_tensor_3d: Output tensor (shape E x 1 x 1), containing
            results of piecewise-linear regression.
        """

        # slope_tensor = K.sum(
        #     numpy.expand_dims(slopes, axis=0) * discretized_input_tensor_2d,
        #     axis=1, keepdims=True
        # )
        # intercept_tensor = K.sum(
        #     numpy.expand_dims(intercepts, axis=0) * discretized_input_tensor_2d,
        #     axis=1, keepdims=True
        # )

        slope_tensor = K.expand_dims(
            K.constant(slopes, dtype='float32'),
            axis=0
        )
        intercept_tensor = K.expand_dims(
            K.constant(intercepts, dtype='float32'),
            axis=0
        )

        # Output is E-by-1 tensor with the same slope repeated.
        slope_tensor = K.sum(
            slope_tensor * discretized_input_tensor_2d,
            axis=1, keepdims=True
        )

        # Output is E-by-1 tensor with the same intercept repeated.
        intercept_tensor = K.sum(
            intercept_tensor * discretized_input_tensor_2d,
            axis=1, keepdims=True
        )

        output_tensor_2d = intercept_tensor + slope_tensor * input_tensor_2d
        return K.expand_dims(output_tensor_2d, axis=-2)

    return pw_linear_regression_function


def create_normalization_layers(
        input_layer_object, pw_linear_unif_model_file_name,
        vector_predictor_names, scalar_predictor_names, heights_m_agl):
    """Creates normalization layers.

    This method implements two-step normalization: uniformization followed by
    z-score transformation.

    :param input_layer_object: Input layer (instance of `keras.layers.Input`),
        containing unnormalized predictors.
    :param pw_linear_unif_model_file_name: Path to file with piecewise-linear
        model approximating the full uniformization.  This will be read by
        `normalization.read_piecewise_linear_models_for_unif`.
    :param vector_predictor_names: 1-D list with names of vector predictor
        variables (those defined at every height).
    :param scalar_predictor_names: 1-D list with names of scalar predictor
        variables.
    :param heights_m_agl: 1-D numpy array of heights (metres above ground level)
        for vector predictors.
    :return: output_layer_object: Layer containing normalized predictors (in
        z-score units), with same shape as input layer.
    """

    # Check input args.
    error_checking.assert_is_string_list(vector_predictor_names)
    error_checking.assert_is_string_list(scalar_predictor_names)
    predictor_names = vector_predictor_names + scalar_predictor_names

    num_channels = len(predictor_names)
    error_checking.assert_equals(
        num_channels, input_layer_object.get_shape()[2]
    )

    error_checking.assert_is_numpy_array(heights_m_agl, num_dimensions=1)
    error_checking.assert_is_numpy_array_without_nan(heights_m_agl)

    num_heights = len(heights_m_agl)
    error_checking.assert_equals(
        num_heights, input_layer_object.get_shape()[1]
    )

    heights_m_agl = numpy.round(heights_m_agl).astype(int)
    error_checking.assert_is_greater_numpy_array(heights_m_agl, 0)

    # Do actual stuff.
    print('Reading models from: "{0:s}"...'.format(
        pw_linear_unif_model_file_name
    ))
    pw_linear_model_table_xarray = (
        normalization.read_piecewise_linear_models_for_unif(
            pw_linear_unif_model_file_name
        )
    )

    pwlmt = pw_linear_model_table_xarray
    output_layer_object_by_height = [None] * num_heights

    for j in range(num_heights):
        output_layer_object_by_channel = [None] * num_channels

        for k in range(num_channels):
            print((
                'Creating normalization layers for {0:s} at {1:d} m AGL...'
            ).format(
                predictor_names[k], heights_m_agl[j]
            ))

            this_function = _get_var_slicing_function(
                height_index=j, channel_index=k
            )
            this_name = 'slice_height{0:d}_channel{1:d}'.format(j, k)
            layer_object_jk = keras.layers.Lambda(
                this_function, name=this_name
            )(input_layer_object)

            if predictor_names[k] in vector_predictor_names:
                j_other = example_utils.match_heights(
                    heights_m_agl=pwlmt.coords[normalization.HEIGHT_DIM].values,
                    desired_height_m_agl=heights_m_agl[j]
                )
                k_other = numpy.where(
                    pwlmt.coords[normalization.VECTOR_PREDICTOR_DIM].values ==
                    predictor_names[k]
                )[0][0]

                num_pieces = numpy.sum(numpy.invert(numpy.isnan(
                    pwlmt[normalization.VECTOR_SLOPE_KEY].values[
                        k_other, j_other, :
                    ]
                )))

                bin_edges = pwlmt[normalization.VECTOR_BREAK_POINT_KEY].values[
                    k_other, j_other, :(num_pieces + 1)
                ]
                slopes = pwlmt[normalization.VECTOR_SLOPE_KEY].values[
                    k_other, j_other, :num_pieces
                ]
                intercepts = pwlmt[normalization.VECTOR_INTERCEPT_KEY].values[
                    k_other, j_other, :num_pieces
                ]
            else:
                k_other = numpy.where(
                    pwlmt.coords[normalization.SCALAR_PREDICTOR_DIM].values ==
                    predictor_names[k]
                )[0][0]

                num_pieces = numpy.sum(numpy.invert(numpy.isnan(
                    pwlmt[normalization.SCALAR_SLOPE_KEY].values[k_other, :]
                )))

                bin_edges = pwlmt[normalization.SCALAR_BREAK_POINT_KEY].values[
                    k_other, :(num_pieces + 1)
                ]
                slopes = pwlmt[normalization.SCALAR_SLOPE_KEY].values[
                    k_other, :num_pieces
                ]
                intercepts = pwlmt[normalization.SCALAR_INTERCEPT_KEY].values[
                    k_other, :num_pieces
                ]

            if num_pieces == 0:
                this_name = 'zeroing_height{0:d}_channel{1:d}'.format(j, k)
                output_layer_object_by_channel[k] = keras.layers.Lambda(
                    _get_zeroing_function(), name=this_name
                )(layer_object_jk)

                continue

            this_name = 'discretize_height{0:d}_channel{1:d}'.format(j, k)
            discretized_layer_object_jk = keras.layers.Discretization(
                bin_boundaries=bin_edges[1:-1], output_mode='one_hot',
                name=this_name
            )(layer_object_jk)

            this_function = _get_pw_linear_regression_function(
                slopes=slopes, intercepts=intercepts
            )
            this_name = 'uniformize_height{0:d}_channel{1:d}'.format(j, k)
            layer_object_jk = keras.layers.Lambda(
                this_function, name=this_name,
                arguments={
                    'discretized_input_tensor_2d': discretized_layer_object_jk
                }
            )(layer_object_jk)

            this_name = 'z_score_height{0:d}_channel{1:d}'.format(j, k)
            layer_object_jk = keras.layers.Lambda(
                _get_norm_ppf_function(), name=this_name
            )(layer_object_jk)

            output_layer_object_by_channel[k] = layer_object_jk

        this_name = 'concat_height{0:d}'.format(j)
        output_layer_object_by_height[j] = keras.layers.Concatenate(
            axis=-1, name=this_name
        )(output_layer_object_by_channel)

    output_layer_object = keras.layers.Concatenate(
        axis=-2, name='concat_all_heights'
    )(output_layer_object_by_height)

    return output_layer_object
