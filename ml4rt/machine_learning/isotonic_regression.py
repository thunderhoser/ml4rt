"""Methods for building, training, and applying isotonic-regression models."""

import os.path
import dill
import numpy
import keras
import tensorflow
from sklearn.isotonic import IsotonicRegression
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4rt.machine_learning import neural_net


# TODO(thunderhoser): Using keras, instead of tensorflow.keras, might fuck me
# over here.
class IsotonicRegressionLayer(keras.layers.Layer):
    """For every target variable, applies trained isotonic-regression model.

    E = number of examples (batch size)
    H = number of heights
    W = number of wavelengths
    F = number of flux variables
    T = number of atomic target variables = W*H + W*F
    """

    def __init__(self, x_threshold_array_list, y_threshold_array_list,
                 **kwargs):
        """Initializer.

        For each isotonic-regression model, the "x-thresholds" are the original
        (uncorrected) values, and the "y-thresholds" are the new
        (bias-corrected) values.

        :param x_threshold_array_list: List containing x-thresholds for
            isotonic-regression models.  This list has length T, and
            x_threshold_array_list[j] is a 1-D numpy array of x-thresholds in
            the isotonic-regression model for the [j]th atomic target variable.
        :param y_threshold_array_list: Same as above but for y-thresholds.
        :param kwargs: Keyword arguments.
        """

        super(IsotonicRegressionLayer, self).__init__(**kwargs)

        num_atomic_target_vars = len(x_threshold_array_list)
        max_num_thresholds = max([
            len(this_array) for this_array in x_threshold_array_list
        ])

        # Pad all threshold arrays to the same length.
        x_threshold_matrix = numpy.full(
            (num_atomic_target_vars, max_num_thresholds),
            numpy.nan, dtype=numpy.float32
        )
        y_threshold_matrix = numpy.full(
            (num_atomic_target_vars, max_num_thresholds),
            numpy.nan, dtype=numpy.float32
        )
        num_thresholds_by_atomic_target = numpy.full(
            num_atomic_target_vars, 0, dtype=numpy.int32
        )

        for j in range(num_atomic_target_vars):
            this_length = len(x_threshold_array_list[j])
            num_thresholds_by_atomic_target[j] = this_length

            x_threshold_matrix[j, :this_length] = x_threshold_array_list[j]
            y_threshold_matrix[j, :this_length] = y_threshold_array_list[j]

            if this_length == max_num_thresholds:
                continue

            x_threshold_matrix[j, this_length:] = x_threshold_array_list[j][-1]
            y_threshold_matrix[j, this_length:] = y_threshold_array_list[j][-1]

        # Store lookup tables as non-trainable variables.
        self.num_atomic_target_vars = num_atomic_target_vars
        self.max_num_thresholds = max_num_thresholds
        self.x_threshold_tensor = tensorflow.Variable(
            x_threshold_matrix, trainable=False, dtype=tensorflow.float32
        )
        self.y_threshold_tensor = tensorflow.Variable(
            y_threshold_matrix, trainable=False, dtype=tensorflow.float32
        )
        self.num_thresholds_by_atomic_target = tensorflow.Variable(
            num_thresholds_by_atomic_target,
            trainable=False, dtype=tensorflow.int32
        )

    def call(self, uncorrected_output_tensors):
        """Implements layer.

        :param uncorrected_output_tensors: length-2 list, where
            uncorrected_output_tensors[0] has dimensions E x H x W x 1 and
            contains predicted heating rates, while
            uncorrected_output_tensors[1] has dimensions E x W x F and
            contains predicted fluxes.
        :return: corrected_heating_rate_tensor_k_day01: Bias-corrected version
            of uncorrected_output_tensors[0].
        :return: corrected_flux_tensor_w_m02: Bias-corrected version
            of uncorrected_output_tensors[1].
        """

        heating_rate_tensor_k_day01, flux_tensor_w_m02 = (
            uncorrected_output_tensors
        )

        # Flatten uncorrected outputs to shape E x T.
        num_examples = tensorflow.shape(heating_rate_tensor_k_day01)[0]
        uncorrected_output_tensor_flat = tensorflow.concat([
            tensorflow.reshape(heating_rate_tensor_k_day01, (num_examples, -1)),
            tensorflow.reshape(flux_tensor_w_m02, (num_examples, -1))
        ], axis=-1)

        num_atomic_target_vars = self.num_atomic_target_vars
        max_num_thresholds = self.max_num_thresholds
        y_interp_list = []

        for j in range(num_atomic_target_vars):
            x_thresholds = self.x_threshold_tensor[j, :]
            y_thresholds = self.y_threshold_tensor[j, :]
            x_values = uncorrected_output_tensor_flat[:, j]

            indices = tensorflow.searchsorted(
                x_thresholds, x_values, side='left'
            )
            indices = tensorflow.clip_by_value(
                indices, 1, max_num_thresholds - 1
            )

            x0_tensor = tensorflow.gather(x_thresholds, indices - 1)
            x1_tensor = tensorflow.gather(x_thresholds, indices)
            y0_tensor = tensorflow.gather(y_thresholds, indices - 1)
            y1_tensor = tensorflow.gather(y_thresholds, indices)

            slope_tensor = (
                (y1_tensor - y0_tensor) / (x1_tensor - x0_tensor + 1e-10)
            )
            these_y_interp = y0_tensor + slope_tensor * (x_values - x0_tensor)

            these_y_interp = tensorflow.maximum(these_y_interp, y_thresholds[0])
            these_y_interp = tensorflow.minimum(
                these_y_interp, y_thresholds[-1]
            )

            y_interp_list.append(
                tensorflow.expand_dims(these_y_interp, axis=-1)
            )

        # Concatenate the results across the 1806 target variables
        y_interp_tensor = tensorflow.concat(y_interp_list, axis=-1)

        # Reshape bias-corrected predictions.
        num_heights = tensorflow.shape(heating_rate_tensor_k_day01)[1]
        num_wavelengths = tensorflow.shape(heating_rate_tensor_k_day01)[2]
        num_flux_vars = tensorflow.shape(flux_tensor_w_m02)[-1]
        num_atomic_heating_rates = num_heights * num_wavelengths

        corrected_heating_rate_tensor_k_day01 = tensorflow.reshape(
            y_interp_tensor[:, :num_atomic_heating_rates],
            (num_examples, num_heights, num_wavelengths, 1)
        )
        corrected_flux_tensor_w_m02 = tensorflow.reshape(
            y_interp_tensor[:, num_atomic_heating_rates:],
            (num_examples, num_wavelengths, num_flux_vars)
        )

        return (
            corrected_heating_rate_tensor_k_day01, corrected_flux_tensor_w_m02
        )

    def get_config(self):
        """Returns layer configuration."""

        config = super().get_config()
        config.update({
            'x_threshold_tensor': self.x_threshold_tensor.numpy().tolist(),
            'y_threshold_tensor': self.y_threshold_tensor.numpy().tolist(),
            'num_thresholds_by_atomic_target':
                self.num_thresholds_by_atomic_target.numpy().tolist(),
            'num_atomic_target_vars': self.num_atomic_target_vars,
            'max_num_thresholds': self.max_num_thresholds
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Instantiates layer from configuration.

        I don't know if I actually need this method.
        """

        # TODO(thunderhoser): Shouldn't `num_thresholds_by_atomic_target` be
        # involved?  ChatGPT says yes, and it gave me a new version of both
        # `get_config` and `from_config`, but I'm hesitant to fuck with things
        # right now.

        return cls(
            numpy.array(config['x_threshold_tensor']),
            numpy.array(config['y_threshold_tensor'])
        )


class IsotonicRegressionLayerNoPadding(keras.layers.Layer):
    """For every target variable, applies trained isotonic-regression model.

    E = number of examples (batch size)
    H = number of heights
    W = number of wavelengths
    F = number of flux variables
    T = number of atomic target variables = W*H + W*F
    """

    def __init__(self, x_threshold_array_list, y_threshold_array_list,
                 **kwargs):
        """Initializer.

        For each isotonic-regression model, the "x-thresholds" are the original
        (uncorrected) values, and the "y-thresholds" are the new
        (bias-corrected) values.

        :param x_threshold_array_list: List containing x-thresholds for
            isotonic-regression models.  This list has length T, and
            x_threshold_array_list[j] is a 1-D numpy array of x-thresholds in
            the isotonic-regression model for the [j]th atomic target variable.
        :param y_threshold_array_list: Same as above but for y-thresholds.
        :param kwargs: Keyword arguments.
        """

        super(IsotonicRegressionLayerNoPadding, self).__init__(**kwargs)

        num_atomic_target_vars = len(x_threshold_array_list)
        max_num_thresholds = max([
            len(this_array) for this_array in x_threshold_array_list
        ])

        # Pad all threshold arrays to the same length.
        num_thresholds_by_atomic_target = numpy.full(
            num_atomic_target_vars, 0, dtype=numpy.int32
        )
        for j in range(num_atomic_target_vars):
            num_thresholds_by_atomic_target[j] = len(x_threshold_array_list[j])

        # Store lookup tables as non-trainable variables.
        self.num_atomic_target_vars = num_atomic_target_vars
        self.max_num_thresholds = max_num_thresholds
        self.num_thresholds_by_atomic_target = tensorflow.Variable(
            num_thresholds_by_atomic_target,
            trainable=False, dtype=tensorflow.int32
        )

        self.x_thresholds_by_atomic_target = [
            tensorflow.Variable(
                these_x, trainable=False, dtype=tensorflow.float32
            ) for these_x in x_threshold_array_list
        ]
        self.y_thresholds_by_atomic_target = [
            tensorflow.Variable(
                these_y, trainable=False, dtype=tensorflow.float32
            ) for these_y in y_threshold_array_list
        ]

    def call(self, uncorrected_output_tensors):
        """Implements layer.

        :param uncorrected_output_tensors: length-2 list, where
            uncorrected_output_tensors[0] has dimensions E x H x W x 1 and
            contains predicted heating rates, while
            uncorrected_output_tensors[1] has dimensions E x W x F and
            contains predicted fluxes.
        :return: corrected_heating_rate_tensor_k_day01: Bias-corrected version
            of uncorrected_output_tensors[0].
        :return: corrected_flux_tensor_w_m02: Bias-corrected version
            of uncorrected_output_tensors[1].
        """

        heating_rate_tensor_k_day01, flux_tensor_w_m02 = (
            uncorrected_output_tensors
        )

        # Flatten uncorrected outputs to shape E x T.
        num_examples = tensorflow.shape(heating_rate_tensor_k_day01)[0]
        uncorrected_output_tensor_flat = tensorflow.concat([
            tensorflow.reshape(heating_rate_tensor_k_day01, (num_examples, -1)),
            tensorflow.reshape(flux_tensor_w_m02, (num_examples, -1))
        ], axis=-1)

        num_atomic_target_vars = self.num_atomic_target_vars
        max_num_thresholds = self.max_num_thresholds
        y_interp_list = []

        for j in range(num_atomic_target_vars):
            x_thresholds = self.x_thresholds_by_atomic_target[j]
            y_thresholds = self.y_thresholds_by_atomic_target[j]
            x_values = uncorrected_output_tensor_flat[:, j]

            indices = tensorflow.searchsorted(
                x_thresholds, x_values, side='left'
            )
            indices = tensorflow.clip_by_value(
                indices, 1, max_num_thresholds - 1
            )

            x0_tensor = tensorflow.gather(x_thresholds, indices - 1)
            x1_tensor = tensorflow.gather(x_thresholds, indices)
            y0_tensor = tensorflow.gather(y_thresholds, indices - 1)
            y1_tensor = tensorflow.gather(y_thresholds, indices)

            slope_tensor = (
                (y1_tensor - y0_tensor) / (x1_tensor - x0_tensor + 1e-10)
            )
            these_y_interp = y0_tensor + slope_tensor * (x_values - x0_tensor)

            these_y_interp = tensorflow.maximum(these_y_interp, y_thresholds[0])
            these_y_interp = tensorflow.minimum(
                these_y_interp, y_thresholds[-1]
            )

            y_interp_list.append(
                tensorflow.expand_dims(these_y_interp, axis=-1)
            )

        # Concatenate the results across the 1806 target variables
        y_interp_tensor = tensorflow.concat(y_interp_list, axis=-1)

        # Reshape bias-corrected predictions.
        num_heights = tensorflow.shape(heating_rate_tensor_k_day01)[1]
        num_wavelengths = tensorflow.shape(heating_rate_tensor_k_day01)[2]
        num_flux_vars = tensorflow.shape(flux_tensor_w_m02)[-1]
        num_atomic_heating_rates = num_heights * num_wavelengths

        corrected_heating_rate_tensor_k_day01 = tensorflow.reshape(
            y_interp_tensor[:, :num_atomic_heating_rates],
            (num_examples, num_heights, num_wavelengths, 1)
        )
        corrected_flux_tensor_w_m02 = tensorflow.reshape(
            y_interp_tensor[:, num_atomic_heating_rates:],
            (num_examples, num_wavelengths, num_flux_vars)
        )

        return (
            corrected_heating_rate_tensor_k_day01, corrected_flux_tensor_w_m02
        )

    def get_config(self):
        """Returns layer configuration."""

        config = super().get_config()
        config.update({
            'x_thresholds_by_atomic_target': [
                xs.numpy().tolist() for xs in self.x_thresholds_by_atomic_target
            ],
            'y_thresholds_by_atomic_target': [
                ys.numpy().tolist() for ys in self.y_thresholds_by_atomic_target
            ],
            'num_thresholds_by_atomic_target':
                self.num_thresholds_by_atomic_target.numpy().tolist(),
            'num_atomic_target_vars': self.num_atomic_target_vars,
            'max_num_thresholds': self.max_num_thresholds
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Instantiates layer from configuration.

        I don't know if I actually need this method.
        """

        # TODO(thunderhoser): Shouldn't `num_thresholds_by_atomic_target` be
        # involved?  ChatGPT says yes, and it gave me a new version of both
        # `get_config` and `from_config`, but I'm hesitant to fuck with things
        # right now.

        return cls(
            [numpy.array(xs) for xs in config['x_thresholds_by_atomic_target']],
            [numpy.array(ys) for ys in config['y_thresholds_by_atomic_target']]
        )


class IsotonicRegressionLayerMemoryHeavy(keras.layers.Layer):
    """For every target variable, applies trained isotonic-regression model.

    E = number of examples (batch size)
    H = number of heights
    W = number of wavelengths
    F = number of flux variables
    T = number of atomic target variables = W*H + W*F
    """

    def __init__(self, x_threshold_array_list, y_threshold_array_list,
                 **kwargs):
        """Initializer.

        For each isotonic-regression model, the "x-thresholds" are the original
        (uncorrected) values, and the "y-thresholds" are the new
        (bias-corrected) values.

        :param x_threshold_array_list: List containing x-thresholds for
            isotonic-regression models.  This list has length T, and
            x_threshold_array_list[j] is a 1-D numpy array of x-thresholds in
            the isotonic-regression model for the [j]th atomic target variable.
        :param y_threshold_array_list: Same as above but for y-thresholds.
        :param kwargs: Keyword arguments.
        """

        super(IsotonicRegressionLayerMemoryHeavy, self).__init__(**kwargs)

        num_atomic_target_vars = len(x_threshold_array_list)
        max_num_thresholds = max([
            len(this_array) for this_array in x_threshold_array_list
        ])

        # Pad all threshold arrays to the same length.
        x_threshold_matrix = numpy.full(
            (num_atomic_target_vars, max_num_thresholds),
            numpy.nan, dtype=numpy.float32
        )
        y_threshold_matrix = numpy.full(
            (num_atomic_target_vars, max_num_thresholds),
            numpy.nan, dtype=numpy.float32
        )
        num_thresholds_by_atomic_target = numpy.full(
            num_atomic_target_vars, 0, dtype=numpy.int32
        )

        for j in range(num_atomic_target_vars):
            this_length = len(x_threshold_array_list[j])
            num_thresholds_by_atomic_target[j] = this_length

            x_threshold_matrix[j, :this_length] = x_threshold_array_list[j]
            y_threshold_matrix[j, :this_length] = y_threshold_array_list[j]

            if this_length == max_num_thresholds:
                continue

            x_threshold_matrix[j, this_length:] = x_threshold_array_list[j][-1]
            y_threshold_matrix[j, this_length:] = y_threshold_array_list[j][-1]

        # Store lookup tables as non-trainable variables.
        self.num_atomic_target_vars = num_atomic_target_vars
        self.max_num_thresholds = max_num_thresholds
        self.x_threshold_tensor = tensorflow.Variable(
            x_threshold_matrix, trainable=False, dtype=tensorflow.float32
        )
        self.y_threshold_tensor = tensorflow.Variable(
            y_threshold_matrix, trainable=False, dtype=tensorflow.float32
        )
        self.num_thresholds_by_atomic_target = tensorflow.Variable(
            num_thresholds_by_atomic_target,
            trainable=False, dtype=tensorflow.int32
        )

    def call(self, uncorrected_output_tensors):
        """Implements layer.

        :param uncorrected_output_tensors: length-2 list, where
            uncorrected_output_tensors[0] has dimensions E x H x W x 1 and
            contains predicted heating rates, while
            uncorrected_output_tensors[1] has dimensions E x W x F and
            contains predicted fluxes.
        :return: corrected_heating_rate_tensor_k_day01: Bias-corrected version
            of uncorrected_output_tensors[0].
        :return: corrected_flux_tensor_w_m02: Bias-corrected version
            of uncorrected_output_tensors[1].
        """

        heating_rate_tensor_k_day01, flux_tensor_w_m02 = (
            uncorrected_output_tensors
        )

        # Flatten uncorrected outputs to shape E x T.
        num_examples = tensorflow.shape(heating_rate_tensor_k_day01)[0]
        uncorrected_output_tensor_flat = tensorflow.concat([
            tensorflow.reshape(heating_rate_tensor_k_day01, (num_examples, -1)),
            tensorflow.reshape(flux_tensor_w_m02, (num_examples, -1))
        ], axis=-1)

        # Expand threshold matrices to match batch size.
        x_threshold_tensor_expanded = tensorflow.tile(
            self.x_threshold_tensor[None, :, :], [num_examples, 1, 1]
        )
        y_threshold_tensor_expanded = tensorflow.tile(
            self.y_threshold_tensor[None, :, :], [num_examples, 1, 1]
        )

        # Find relevant thresholds.
        index_tensor = tensorflow.searchsorted(
            x_threshold_tensor_expanded,
            uncorrected_output_tensor_flat[..., None], side='left'
        )

        max_num_thresholds = self.max_num_thresholds
        index_tensor = tensorflow.clip_by_value(
            index_tensor, 1, max_num_thresholds - 1
        )

        # Gather relevant x (uncorrected) and y (bias-corrected) values.
        x0_tensor = tensorflow.gather(
            x_threshold_tensor_expanded, index_tensor - 1,
            axis=2, batch_dims=2
        )
        x1_tensor = tensorflow.gather(
            x_threshold_tensor_expanded, index_tensor,
            axis=2, batch_dims=2
        )
        y0_tensor = tensorflow.gather(
            y_threshold_tensor_expanded, index_tensor - 1,
            axis=2, batch_dims=2
        )
        y1_tensor = tensorflow.gather(
            y_threshold_tensor_expanded, index_tensor,
            axis=2, batch_dims=2
        )
        y_min_tensor = tensorflow.gather(
            y_threshold_tensor_expanded, tensorflow.minimum(index_tensor, 0),
            axis=2, batch_dims=2
        )
        y_max_tensor = tensorflow.gather(
            y_threshold_tensor_expanded,
            tensorflow.maximum(index_tensor, max_num_thresholds - 1),
            axis=2, batch_dims=2
        )

        # Do the linear interpolation.
        slope_tensor = (y1_tensor - y0_tensor) / (x1_tensor - x0_tensor + 1e-10)
        y_interp_tensor = y0_tensor + slope_tensor * (
            uncorrected_output_tensor_flat[..., None] - x0_tensor
        )
        y_interp_tensor = tensorflow.maximum(y_interp_tensor, y_min_tensor)
        y_interp_tensor = tensorflow.minimum(y_interp_tensor, y_max_tensor)

        # Reshape bias-corrected predictions.
        num_heights = tensorflow.shape(heating_rate_tensor_k_day01)[1]
        num_wavelengths = tensorflow.shape(heating_rate_tensor_k_day01)[2]
        num_flux_vars = tensorflow.shape(flux_tensor_w_m02)[-1]
        num_atomic_heating_rates = num_heights * num_wavelengths

        corrected_heating_rate_tensor_k_day01 = tensorflow.reshape(
            y_interp_tensor[:, :num_atomic_heating_rates],
            (num_examples, num_heights, num_wavelengths, 1)
        )
        corrected_flux_tensor_w_m02 = tensorflow.reshape(
            y_interp_tensor[:, num_atomic_heating_rates:],
            (num_examples, num_wavelengths, num_flux_vars)
        )

        return (
            corrected_heating_rate_tensor_k_day01, corrected_flux_tensor_w_m02
        )

    def get_config(self):
        """Returns layer configuration."""

        config = super().get_config()
        config.update({
            'x_threshold_tensor': self.x_threshold_tensor.numpy().tolist(),
            'y_threshold_tensor': self.y_threshold_tensor.numpy().tolist(),
            'num_thresholds_by_atomic_target':
                self.num_thresholds_by_atomic_target.numpy().tolist(),
            'num_atomic_target_vars': self.num_atomic_target_vars,
            'max_num_thresholds': self.max_num_thresholds
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Instantiates layer from configuration.

        I don't know if I actually need this method.
        """

        return cls(
            numpy.array(config['x_threshold_tensor']),
            numpy.array(config['y_threshold_tensor'])
        )


def train_models(
        orig_vector_prediction_matrix, orig_scalar_prediction_matrix,
        vector_target_matrix, scalar_target_matrix):
    """Trains isotonic-regression models.

    E = number of examples
    H = number of heights
    T_v = number of vector target variables
    T_s = number of scalar target variables
    W = number of wavelengths
    S = number of ensemble members

    :param orig_vector_prediction_matrix: numpy array (E x H x W x T_v x S) of
        predicted values for vector target variables.
    :param orig_scalar_prediction_matrix: numpy array (E x W x T_s x S) of
        predicted values for scalar target variables.
    :param vector_target_matrix: numpy array (E x H x W x T_v) of actual values
        for vector target variables.
    :param scalar_target_matrix: numpy array (E x W x T_s) of actual values for
        scalar target variables.
    :return: scalar_model_object_matrix: numpy array of models
        (instances of `sklearn.isotonic.IsotonicRegression`) for scalar target
        variables.  Dimensions are W x T_s.
    :return: vector_model_object_matrix: numpy array of models
        (instances of `sklearn.isotonic.IsotonicRegression`) for vector target
        variables.  Dimensions are H x W x T_v.
    """

    # Check input args.
    num_examples = None
    num_heights = 0
    num_wavelengths = None
    num_vector_targets = 0
    num_scalar_targets = 0
    ensemble_size = None

    have_vectors = (
        orig_vector_prediction_matrix is not None
        or vector_target_matrix is not None
    )

    if have_vectors:
        orig_vector_prediction_matrix = orig_vector_prediction_matrix.astype(
            numpy.float32
        )
        vector_target_matrix = vector_target_matrix.astype(numpy.float32)

        error_checking.assert_is_numpy_array(
            orig_vector_prediction_matrix, num_dimensions=5
        )
        error_checking.assert_is_numpy_array_without_nan(
            orig_vector_prediction_matrix
        )

        error_checking.assert_is_numpy_array(
            vector_target_matrix,
            exact_dimensions=numpy.array(
                orig_vector_prediction_matrix.shape[:-1], dtype=int
            )
        )
        error_checking.assert_is_numpy_array_without_nan(vector_target_matrix)

        num_examples = vector_target_matrix.shape[0]
        num_heights = vector_target_matrix.shape[1]
        num_wavelengths = vector_target_matrix.shape[2]
        num_vector_targets = vector_target_matrix.shape[3]
        ensemble_size = orig_vector_prediction_matrix.shape[4]

    have_scalars = (
        orig_scalar_prediction_matrix is not None
        or scalar_target_matrix is not None
    )

    if have_scalars:
        error_checking.assert_is_numpy_array(
            orig_scalar_prediction_matrix, num_dimensions=4
        )

        if num_examples is None:
            num_examples = orig_scalar_prediction_matrix.shape[0]
        if num_wavelengths is None:
            num_wavelengths = orig_scalar_prediction_matrix.shape[1]
        if ensemble_size is None:
            ensemble_size = orig_scalar_prediction_matrix.shape[3]

        expected_dim = numpy.array([
            num_examples, num_wavelengths,
            orig_scalar_prediction_matrix.shape[2], ensemble_size
        ], dtype=int)

        error_checking.assert_is_numpy_array(
            orig_scalar_prediction_matrix, exact_dimensions=expected_dim
        )
        error_checking.assert_is_numpy_array_without_nan(
            orig_scalar_prediction_matrix
        )

        error_checking.assert_is_numpy_array(
            scalar_target_matrix,
            exact_dimensions=numpy.array(
                orig_scalar_prediction_matrix.shape[:-1], dtype=int
            )
        )
        error_checking.assert_is_numpy_array_without_nan(scalar_target_matrix)

        num_scalar_targets = scalar_target_matrix.shape[2]

    # Do actual stuff.
    scalar_model_object_matrix = numpy.full(
        (num_wavelengths, num_scalar_targets), '', dtype=object
    )
    vector_model_object_matrix = numpy.full(
        (num_heights, num_wavelengths, num_vector_targets), '', dtype=object
    )

    for t in range(num_scalar_targets):
        for w in range(num_wavelengths):
            print((
                'Training isotonic-regression model for {0:d}th of {1:d} '
                'scalar target variables at {2:d}th of {3:d} wavelengths...'
            ).format(
                t + 1, num_scalar_targets,
                w + 1, num_wavelengths
            ))

            this_prediction_matrix = (
                orig_scalar_prediction_matrix[:, w, t, :] + 0.
            )
            this_target_matrix = numpy.expand_dims(
                scalar_target_matrix[:, w, t], axis=-1
            )
            this_target_matrix = numpy.repeat(
                this_target_matrix, axis=-1, repeats=ensemble_size
            )

            scalar_model_object_matrix[w, t] = IsotonicRegression(
                increasing=True, out_of_bounds='clip'
            )
            scalar_model_object_matrix[w, t].fit(
                X=numpy.ravel(this_prediction_matrix),
                y=numpy.ravel(this_target_matrix)
            )

        print('\n')

    for t in range(num_vector_targets):
        for w in range(num_wavelengths):
            for h in range(num_heights):
                print((
                    'Training isotonic-regression model for {0:d}th of {1:d} '
                    'vector target variables at {2:d}th of {3:d} wavelengths '
                    'and {4:d}th of {5:d} heights...'
                ).format(
                    t + 1, num_vector_targets,
                    w + 1, num_wavelengths,
                    h + 1, num_heights
                ))

                vector_model_object_matrix[h, w, t] = IsotonicRegression(
                    increasing=True, out_of_bounds='clip'
                )

                this_prediction_matrix = (
                    orig_vector_prediction_matrix[:, h, w, t, :] + 0.
                )
                this_target_matrix = numpy.expand_dims(
                    vector_target_matrix[:, h, w, t], axis=-1
                )
                this_target_matrix = numpy.repeat(
                    this_target_matrix, axis=-1, repeats=ensemble_size
                )

                vector_model_object_matrix[h, w, t].fit(
                    X=numpy.ravel(this_prediction_matrix),
                    y=numpy.ravel(this_target_matrix)
                )

            print('\n')

    return scalar_model_object_matrix, vector_model_object_matrix


def apply_models(
        orig_vector_prediction_matrix, orig_scalar_prediction_matrix,
        scalar_model_object_matrix, vector_model_object_matrix):
    """Applies isotonic-regression models.

    :param orig_vector_prediction_matrix: See doc for `train_models`.
    :param orig_scalar_prediction_matrix: Same.
    :param scalar_model_object_matrix: Same.
    :param vector_model_object_matrix: Same.
    :return: new_vector_prediction_matrix: Same as
        `orig_vector_prediction_matrix` but with transformed values.
    :return: new_scalar_prediction_matrix: Same as
        `orig_scalar_prediction_matrix` but with transformed values.
    """

    # Check input args.
    num_examples = None
    num_heights = 0
    num_wavelengths = None
    num_vector_targets = 0
    num_scalar_targets = 0
    ensemble_size = None

    have_vectors = (
        orig_vector_prediction_matrix is not None
        or vector_model_object_matrix.size > 0
    )

    if have_vectors:
        orig_vector_prediction_matrix = orig_vector_prediction_matrix.astype(
            numpy.float16
        )

        error_checking.assert_is_numpy_array(
            orig_vector_prediction_matrix, num_dimensions=5
        )
        error_checking.assert_is_numpy_array_without_nan(
            orig_vector_prediction_matrix
        )

        error_checking.assert_is_numpy_array(
            vector_model_object_matrix, num_dimensions=3
        )

        num_heights = orig_vector_prediction_matrix.shape[1]
        num_wavelengths = orig_vector_prediction_matrix.shape[2]
        num_vector_targets = orig_vector_prediction_matrix.shape[3]
        expected_dim = numpy.array(
            [num_heights, num_wavelengths, num_vector_targets], dtype=int
        )
        error_checking.assert_is_numpy_array(
            vector_model_object_matrix, exact_dimensions=expected_dim
        )

        num_examples = orig_vector_prediction_matrix.shape[0]
        ensemble_size = orig_vector_prediction_matrix.shape[4]

    have_scalars = (
        orig_scalar_prediction_matrix is not None
        or scalar_model_object_matrix.size > 0
    )

    if have_scalars:
        error_checking.assert_is_numpy_array(
            orig_scalar_prediction_matrix, num_dimensions=4
        )

        if num_examples is None:
            num_examples = orig_scalar_prediction_matrix.shape[0]
        if ensemble_size is None:
            ensemble_size = orig_scalar_prediction_matrix.shape[3]

        num_scalar_targets = orig_scalar_prediction_matrix.shape[2]
        expected_dim = numpy.array(
            [num_examples, num_wavelengths, num_scalar_targets, ensemble_size],
            dtype=int
        )

        error_checking.assert_is_numpy_array(
            orig_scalar_prediction_matrix, exact_dimensions=expected_dim
        )
        error_checking.assert_is_numpy_array_without_nan(
            orig_scalar_prediction_matrix
        )

        expected_dim = numpy.array(
            [num_wavelengths, num_scalar_targets], dtype=int
        )
        error_checking.assert_is_numpy_array(
            numpy.array(scalar_model_object_matrix),
            exact_dimensions=expected_dim
        )

    if have_vectors:
        new_vector_prediction_matrix = numpy.full(
            orig_vector_prediction_matrix.shape, numpy.nan, dtype=numpy.float32
        )
    else:
        new_vector_prediction_matrix = numpy.full(
            (num_examples, 0, num_wavelengths, 0, ensemble_size),
            numpy.nan,
            dtype=numpy.float32
        )

    if have_scalars:
        new_scalar_prediction_matrix = numpy.full(
            orig_scalar_prediction_matrix.shape, numpy.nan
        )
    else:
        new_scalar_prediction_matrix = numpy.full(
            (num_examples, num_wavelengths, 0, ensemble_size), numpy.nan
        )

    for t in range(num_scalar_targets):
        for w in range(num_wavelengths):
            print((
                'Applying isotonic-regression model to {0:d}th of {1:d} scalar '
                'target variables at {2:d}th of {3:d} wavelengths...'
            ).format(
                t + 1, num_scalar_targets,
                w + 1, num_wavelengths
            ))

            for s in range(ensemble_size):
                new_scalar_prediction_matrix[:, w, t, s] = (
                    scalar_model_object_matrix[w, t].predict(
                        orig_scalar_prediction_matrix[:, w, t, s]
                    )
                )

        print('\n')

    for t in range(num_vector_targets):
        for w in range(num_wavelengths):
            for h in range(num_heights):
                print((
                    'Applying isotonic-regression model to {0:d}th of {1:d} '
                    'vector target variables at {2:d}th of {3:d} heights '
                    'and {4:d}th of {5:d} wavelengths...'
                ).format(
                    t + 1, num_vector_targets,
                    h + 1, num_heights,
                    w + 1, num_wavelengths
                ))

                for s in range(ensemble_size):
                    new_vector_prediction_matrix[:, h, w, t, s] = (
                        vector_model_object_matrix[h, w, t].predict(
                            orig_vector_prediction_matrix[:, h, w, t, s]
                        )
                    )

            print('\n')

    return new_vector_prediction_matrix, new_scalar_prediction_matrix


def find_file(model_dir_name, raise_error_if_missing=True):
    """Finds Dill file with set of isotonic-regression models.

    :param model_dir_name: Name of directory.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: dill_file_name: Path to Dill file with models.
    """

    error_checking.assert_is_string(model_dir_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    dill_file_name = '{0:s}/isotonic_regression.dill'.format(model_dir_name)

    if raise_error_if_missing and not os.path.isfile(dill_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            dill_file_name
        )
        raise ValueError(error_string)

    return dill_file_name


def write_file(dill_file_name, scalar_model_object_matrix,
               vector_model_object_matrix):
    """Writes set of isotonic-regression models to Dill file.

    :param dill_file_name: Path to output file.
    :param scalar_model_object_matrix: See doc for `train_models`.
    :param vector_model_object_matrix: Same.
    """

    error_checking.assert_is_numpy_array(
        numpy.array(scalar_model_object_matrix), num_dimensions=2
    )
    error_checking.assert_is_numpy_array(
        vector_model_object_matrix, num_dimensions=3
    )

    file_system_utils.mkdir_recursive_if_necessary(file_name=dill_file_name)

    dill_file_handle = open(dill_file_name, 'wb')
    dill.dump(scalar_model_object_matrix, dill_file_handle)
    dill.dump(vector_model_object_matrix, dill_file_handle)
    dill_file_handle.close()


def read_file(dill_file_name):
    """Reads set of isotonic-regression models from Dill file.

    :param dill_file_name: Path to input file.
    :return: scalar_model_object_matrix: See doc for `train_models`.
    :return: vector_model_object_matrix: Same.
    """

    error_checking.assert_file_exists(dill_file_name)

    dill_file_handle = open(dill_file_name, 'rb')
    scalar_model_object_matrix = dill.load(dill_file_handle)
    vector_model_object_matrix = dill.load(dill_file_handle)
    dill_file_handle.close()

    scalar_model_object_matrix = numpy.array(scalar_model_object_matrix)
    if len(scalar_model_object_matrix.shape) == 1:
        scalar_model_object_matrix = numpy.expand_dims(
            scalar_model_object_matrix, axis=0
        )

    if len(vector_model_object_matrix.shape) == 1:
        vector_model_object_matrix = numpy.expand_dims(
            vector_model_object_matrix, axis=-2
        )

    return scalar_model_object_matrix, vector_model_object_matrix


def add_ir_to_neural_net(
        nn_model_object, nn_metafile_name, scalar_model_object_matrix,
        vector_model_object_matrix, layer_type_string='default'):
    """Adds suite of trained isotonic-regression models to trained neural net.

    :param nn_model_object: Trained instance of `keras.models.Model` or
        `keras.models.Sequential`.
    :param nn_metafile_name: Path to metafile for neural network
        (will be read by `neural_net.read_metafile`).
    :param scalar_model_object_matrix: See doc for `train_models`.
    :param vector_model_object_matrix: Same.
    :param layer_type_string: Type of isotonic-regression layer ("default",
        "no_padding", or "memory_heavy").
    :return: nn_model_object: Same as input, except with isotonic regression
        built in.
    """

    error_checking.assert_is_numpy_array(
        numpy.array(scalar_model_object_matrix), num_dimensions=2
    )
    error_checking.assert_is_numpy_array(
        vector_model_object_matrix, num_dimensions=3
    )

    print('Reading metadata from: "{0:s}"...'.format(nn_metafile_name))
    nn_metadata_dict = neural_net.read_metafile(nn_metafile_name)
    training_option_dict = nn_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    tod = training_option_dict

    target_wavelengths_metres = tod[neural_net.TARGET_WAVELENGTHS_KEY]
    num_wavelengths = len(target_wavelengths_metres)

    assert (
        vector_model_object_matrix.shape[1] ==
        scalar_model_object_matrix.shape[0]
    )
    num_model_wavelengths = scalar_model_object_matrix.shape[0]

    if num_model_wavelengths == num_wavelengths + 1:
        vector_model_object_matrix = vector_model_object_matrix[:, :-1, :]
        scalar_model_object_matrix = scalar_model_object_matrix[:-1, :]
        num_model_wavelengths = scalar_model_object_matrix.shape[0]

    assert num_model_wavelengths == num_wavelengths

    model_objects = numpy.concatenate([
        numpy.ravel(vector_model_object_matrix),
        numpy.ravel(scalar_model_object_matrix)
    ])
    num_models = len(model_objects)
    
    x_threshold_array_list = numpy.array(
        [mdl.X_thresholds_ for mdl in model_objects], dtype=object
    )
    y_threshold_array_list = numpy.array(
        [mdl.y_thresholds_ for mdl in model_objects], dtype=object
    )

    for j in range(num_models):
        _, unique_indices = numpy.unique(
            number_rounding.round_to_nearest(x_threshold_array_list[j], 1e-6),
            return_index=True
        )
        x_threshold_array_list[j] = x_threshold_array_list[j][unique_indices]
        y_threshold_array_list[j] = y_threshold_array_list[j][unique_indices]

    if layer_type_string == 'default':
        ir_layer_object = IsotonicRegressionLayer(
            x_threshold_array_list=x_threshold_array_list,
            y_threshold_array_list=y_threshold_array_list
        )(nn_model_object.output)
    elif layer_type_string == 'no_padding':
        ir_layer_object = IsotonicRegressionLayerNoPadding(
            x_threshold_array_list=x_threshold_array_list,
            y_threshold_array_list=y_threshold_array_list
        )(nn_model_object.output)
    elif layer_type_string == 'memory_heavy':
        ir_layer_object = IsotonicRegressionLayerMemoryHeavy(
            x_threshold_array_list=x_threshold_array_list,
            y_threshold_array_list=y_threshold_array_list
        )(nn_model_object.output)

    nn_model_object = keras.models.Model(
        inputs=nn_model_object.input, outputs=ir_layer_object
    )
    nn_model_object.summary()

    return nn_model_object
