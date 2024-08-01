"""Methods for building Peter Ukkonen's BRNN architecture.

BRNN = bidirectional recurrent neural network

https://doi.org/10.1029/2021MS002875
"""

import numpy
import keras
import keras.layers
from keras import Model
from tensorflow.keras import backend as K

# Assumes scalar predictors in the following order:
# [1] Solar zenith angle (radians)
# [2] Albedo (unitless)
# [3] Latitude (deg north)
# [4] Longitude (deg east)
# [5] Column-integrated liquid-water path (LWP; kg m^-2)
# [6] Column-integrated ice-water path (IWP; kg m^-2)
# [7] Aerosol single-scattering albedo (SSA; unitless)
# [8] Aerosol asymmetry parameter (unitless)
SCALAR_PREDICTOR_MAX_VALUES = numpy.array([
    1.4835263, 0.8722802, 89.910324, 359.8828,
    2.730448, 8.228799, 1., 0.96304035
], dtype=numpy.float32)

# Assumes vector predictors in the following order:
# [1] Pressure (Pa)
# [2] Temperature (K)
# [3] Specific humidity (kg kg^-1)
# [4] Liquid-water content (LWC; kg m^-3)
# [5] Ice-water content (IWC; kg m^-3)
# [6] Ozone mixing ratio (kg kg^-1)
# [7] CO2 concentration (ppmv)
# [8] CH4 concentration (ppmv)
# [9] N2O concentration (ppmv)
# [10] Aerosol extinction (m^-1)
# [11] Liquid effective radius (m)
# [12] Ice effective radius (m)
# [13] Height thickness of layer (m)
# [14] Pressure thickness of layer (Pa)
# [15] Height at layer center (m above ground level)
# [16] Downward-integrated liquid-water path (LWP; kg m^-2)
# [17] Downward-integrated ice-water path (IWP; kg m^-2)
# [18] Downward-integrated water-vapour path (WVP; kg m^-2)
# [19] Upward-integrated LWP (kg m^-2)
# [20] Upward-integrated IWP (kg m^-2)
# [21] Upward-integrated WVP (kg m^-2)
# [22] Relative humidity (unitless)
VECTOR_PREDICTOR_MAX_VALUES = numpy.array([
    1.04399836e+05, 3.18863373e+02, 2.22374070e-02, 1.33532193e-03,
    1.47309888e-03, 1.73750886e-05, 1.15184314e+03, 6.53699684e+00,
    1.38715994e+00, 9.72749013e-03, 1.70985604e-05, 6.53300012e-05,
    7.95751719e+04, 3.15813086e+03, 1.78493945e+03, 2.73044801e+00,
    8.22879887e+00, 7.84459991e+01, 2.73044801e+00, 8.22879887e+00,
    7.84459991e+01, 1.23876810e+00
], dtype=numpy.float32)

NUM_HEIGHTS = 127

GRAVITATIONAL_CONSTANT_M_S02 = 9.8066
SPECIFIC_HEAT_J_KG01_K01 = 1004


class HeatingRateLayer(keras.layers.Layer):
    """Layer to convert fluxes to heating rates."""

    def __init__(self, name=None, unit_string='K d-1', **kwargs):
        """Constructor.

        :param name: Layer name (string).
        :param unit_string: Heating-rate units (must be either "K s-1" for Kelvins
            per second or "K d-1" for Kelvins per day).
        :param kwargs: Keyword arguments inherited from class
            `keras.layers.Layer`.
        """

        super(HeatingRateLayer, self).__init__(name=name, **kwargs)

        units_to_time_scale_seconds = {
            'K s-1': 1,
            'K d-1': 3600 * 24
        }
        time_scale_seconds = units_to_time_scale_seconds[unit_string]

        self.g_cp = numpy.float32(
            GRAVITATIONAL_CONSTANT_M_S02 / SPECIFIC_HEAT_J_KG01_K01
            * time_scale_seconds
        )

    def get_config(self):
        cfg = super().get_config()
        return cfg

    def build(self, input_shape):
        pass

    def call(self, input_tensor_list):
        """Does the dirty work.

        E = number of data examples
        H = number of heights

        :param input_tensor_list: length-2 list of Keras tensors.

        input_tensor_list[0]: E-by-H-by-2 tensor, where
        input_tensor_list[0][..., 0] contains predicted downwelling fluxes in
        W m^-2 and input_tensor_list[0][..., 1] contains predicted upwelling
        fluxes in W m^-2.

        input_tensor_list[1]: E-by-H-by-1 tensor, containing pressures in Pa.

        :return: heating_rate_tensor: E-by-(H minus 1) tensor of heating rates,
            in whatever units were specified when calling the constructor.
        """

        flux_tensor_w_m02, pressure_tensor_pa = input_tensor_list
        net_flux_tensor_w_m02 = (
            flux_tensor_w_m02[..., 0] - flux_tensor_w_m02[..., 1]
        )
        net_flux_diff_tensor_w_m02 = (
            net_flux_tensor_w_m02[..., :-1] - net_flux_tensor_w_m02[..., 1:]
        )
        pressure_diff_tensor_pa = (
            pressure_tensor_pa[..., :-1, 0] - pressure_tensor_pa[..., 1:, 0]
        )
        return -self.g_cp * net_flux_diff_tensor_w_m02 / pressure_diff_tensor_pa


def create_model(
        num_neurons=64, use_lstm=True,
        output_activ_func_name='sigmoid', init_state_activ_func_name='linear',
        add_extra_dense_layer=False, repeat_scalars_over_height=True,
        simplify_inputs=True):
    """Creates BRNN model.

    :param num_neurons: Number of neurons (same for every hidden layer).
    :param use_lstm: Boolean flag.  If True, will use an LSTM (long-short-term
        memory) layer for the bidirectional part.  If False, will use a GRU
        (gated recurrent unit) for this.
    :param output_activ_func_name: Name of activation function for quasi-output
        layer.  The "quasi-output" layer is the one that produces normalized
        flux predictions, before they have been multiplied with
        top-of-atmosphere incoming flux.
    :param init_state_activ_func_name: Name of activation function for initial
        state before recurrent layer.
    :param add_extra_dense_layer: Boolean flag.  If True, will throw in an extra
        dense layer before the recurrent (LSTM or GRU) layer.
    :param repeat_scalars_over_height: Boolean flag.  If True, will repeat
        scalars over height, so that scalar and vector predictors can be passed
        in a single tensor.
    :param simplify_inputs: Boolean flag.  If True, will remove some predictor
        variables.
    :return: model_object: Untrained instance of `keras.models.Model`, ready for
        training.
    """

    # Set up input layers.
    scalar_input_dims = (len(SCALAR_PREDICTOR_MAX_VALUES),)
    scalar_input_layer_object = keras.layers.Input(
        shape=scalar_input_dims, name='scalar_predictor_matrix'
    )

    vector_input_dims = (NUM_HEIGHTS, len(VECTOR_PREDICTOR_MAX_VALUES))
    vector_input_layer_object = keras.layers.Input(
        shape=vector_input_dims, name='vector_predictor_matrix'
    )

    toa_down_flux_input_dims = (1, 1)
    toa_down_flux_input_layer_object = keras.layers.Input(
        shape=toa_down_flux_input_dims, name='toa_down_flux_input_matrix'
    )

    input_layer_objects = [
        scalar_input_layer_object,
        vector_input_layer_object,
        toa_down_flux_input_layer_object
    ]

    # Pre-process inputs.
    scalar_layer_object = input_layer_objects[0]
    vector_layer_object = input_layer_objects[1]
    toa_down_flux_layer_object_w_m02 = input_layer_objects[2]

    pressure_layer_object_pa = vector_layer_object[..., :1]
    albedo_layer_object = scalar_layer_object[:, 1:2]

    vector_layer_object = vector_layer_object / VECTOR_PREDICTOR_MAX_VALUES
    scalar_layer_object = scalar_layer_object / SCALAR_PREDICTOR_MAX_VALUES

    if simplify_inputs:
        vector_layer_object = vector_layer_object[:, :, :14]

        scalar_layer_object = keras.layers.Concatenate(axis=-1)([
            scalar_layer_object[:, 0:1],
            scalar_layer_object[:, 1:2],
            scalar_layer_object[:, 6:7],
            scalar_layer_object[:, 7:8]
        ])

    if repeat_scalars_over_height:
        vector_layer_object = keras.layers.Concatenate(axis=-1)([
            vector_layer_object,
            keras.layers.RepeatVector(NUM_HEIGHTS)(scalar_layer_object)
        ])

    # Do the rest.
    first_init_state_layer_object = keras.layers.Dense(
        num_neurons, activation=init_state_activ_func_name, name='init_state1'
    )(albedo_layer_object)

    if use_lstm:
        second_init_state_layer_object = keras.layers.Dense(
            num_neurons, activation=init_state_activ_func_name,
            name='init_state2'
        )(albedo_layer_object)

        init_state = [
            first_init_state_layer_object, second_init_state_layer_object
        ]
        recurrent_layer_object = keras.layers.LSTM
    else:
        init_state = first_init_state_layer_object
        recurrent_layer_object = keras.layers.GRU

    if add_extra_dense_layer:
        vector_layer_object = keras.layers.Dense(
            num_neurons, activation=init_state_activ_func_name,
            name='extra_dense_layer'
        )(vector_layer_object)

    fwd_hidden_state_object = recurrent_layer_object(
        num_neurons, return_sequences=True, go_backwards=False
    )(vector_layer_object, initial_state=init_state)

    bwd_hidden_state_object = recurrent_layer_object(
        num_neurons, return_sequences=True, go_backwards=True
    )(fwd_hidden_state_object)

    bwd_hidden_state_object = keras.layers.Lambda(
        lambda x: K.reverse(x, axes=1),
        output_shape=bwd_hidden_state_object.shape[1:]
    )(bwd_hidden_state_object)

    hidden_state_object = keras.layers.Concatenate(axis=2)(
        [fwd_hidden_state_object, bwd_hidden_state_object]
    )

    norm_flux_layer_object = keras.layers.Dense(
        2, activation=output_activ_func_name, name='normalized_sw_flux'
    )(hidden_state_object)

    flux_layer_object_w_m02 = keras.layers.Multiply(
        name='physical_sw_flux'
    )([norm_flux_layer_object, toa_down_flux_layer_object_w_m02])

    heating_rate_layer_object_k_per_time = HeatingRateLayer(
        name='sw_heating_rate'
    )([flux_layer_object_w_m02, pressure_layer_object_pa])

    new_dimensions = heating_rate_layer_object_k_per_time.shape[1:] + (1,)
    heating_rate_layer_object_k_per_time = keras.layers.Reshape(
        target_shape=new_dimensions
    )(heating_rate_layer_object_k_per_time)

    heating_rate_layer_object_k_per_time = keras.layers.ZeroPadding1D(
        padding=(0, 1)
    )(heating_rate_layer_object_k_per_time)

    output_layer_objects = keras.layers.Concatenate(axis=-1)(
        [flux_layer_object_w_m02, heating_rate_layer_object_k_per_time]
    )

    model = Model(inputs=input_layer_objects, outputs=output_layer_objects)
    model.summary()

    model.compile(
        loss='mse',
        optimizer=keras.optimizers.Nadam(),
        metrics=['mean_squared_error', 'mean_absolute_error']
    )

    return model
