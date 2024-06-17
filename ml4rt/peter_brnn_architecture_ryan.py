"""Methods for building Ryan's version of Peter Ukkonen's BRNN architecture.

Ryan's version has two different output layers -- one for heating rates and one
for fluxes -- with extra dimensions.

https://doi.org/10.1029/2021MS002875
"""

import os
import sys
import numpy
import keras
import keras.layers
from keras import Model
from tensorflow.keras import backend as K

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import neural_net
import custom_losses

# Assumes scalar predictors in the following order:
# solar zenith angle (rad), albedo (1), latitude (deg N), longitude (deg E),
# column LWP (kg m^-2), column IWP (kg m^-2), aerosol SSA (1), aerosol asymm (1)
SCALAR_PREDICTOR_MAX_VALUES = numpy.array([
    1.4835263, 0.8722802, 89.910324, 359.8828,
    2.730448, 8.228799, 1., 0.96304035
], dtype=numpy.float32)

# Assumes vector predictors in the following order:
# pressure (Pa), temperature (K), spec humidity (kg kg^-1), LWC (kg m^-3),
# IWC (kg m^-3), O3 mixing ratio (kg kg^-1), CO2 (ppmv), CH4 (ppmv),
# N2O (ppmv), aerosol extinction (m^-1), liquid eff rad (m), ice eff rad (m),
# height thickness (m), pressure thickness (Pa), height (m AGL), dLWP (kg m^-2),
# dIWP (kg m^-2), dWVP (kg m^-2), uLWP (kg m^-2), uIWP (kg m^-2),
# uWVP (kg m^-2), relative humidity (unitless)
VECTOR_PREDICTOR_MAX_VALUES = numpy.array([
    1.04399836e+05, 3.18863373e+02, 2.22374070e-02, 1.33532193e-03,
    1.47309888e-03, 1.73750886e-05, 1.15184314e+03, 6.53699684e+00,
    1.38715994e+00, 9.72749013e-03, 1.70985604e-05, 6.53300012e-05,
    7.95751719e+04, 3.15813086e+03, 1.78493945e+03, 2.73044801e+00,
    8.22879887e+00, 7.84459991e+01, 2.73044801e+00, 8.22879887e+00,
    7.84459991e+01, 1.23876810e+00
], dtype=numpy.float32)

NUM_HEIGHTS = 127

VECTOR_LOSS_FUNCTION = custom_losses.dual_weighted_mse()
SCALAR_LOSS_FUNCTION = custom_losses.scaled_mse_for_net_flux(0.64)


class HeatingRateLayer(keras.layers.Layer):
    def __init__(self, name=None, hr_units='K d-1', **kwargs):
        super(HeatingRateLayer, self).__init__(name=name, **kwargs)
        time_scale = {'K s-1': 1, 'K d-1': 3600 * 24}[hr_units]
        self.g_cp = numpy.float32(9.8066 / 1004 * time_scale)

    def get_config(self):
        cfg = super().get_config()
        return cfg

    def build(self, input_shape):
        pass

    def call(self, inputs):
        fluxes, hlpress = inputs
        netflux = fluxes[..., 0] - fluxes[..., 1]
        flux_diff = netflux[..., :-1] - netflux[..., 1:]
        net_press = hlpress[..., :-1, 0] - hlpress[..., 1:, 0]
        return -self.g_cp * flux_diff / net_press


def rnn_sw(
        nneur=64, lstm=True,
        activ_last='sigmoid', activ_surface='linear',
        add_dense=False, add_scalars_to_levels=True, simpler_inputs=True):

    scalar_input_dims = (len(SCALAR_PREDICTOR_MAX_VALUES),)
    scalar_input_layer_object = keras.layers.Input(
        shape=scalar_input_dims, name='scalar_predictor_matrix'
    )

    vector_input_dims = (NUM_HEIGHTS, len(VECTOR_PREDICTOR_MAX_VALUES))
    vector_input_layer_object = keras.layers.Input(
        shape=vector_input_dims, name='vector_predictor_matrix'
    )

    toa_flux_input_dims = (1, 1)
    toa_flux_input_layer_object = keras.layers.Input(
        shape=toa_flux_input_dims, name='toa_flux_input_matrix'
    )

    all_inp = [
        scalar_input_layer_object,
        vector_input_layer_object,
        toa_flux_input_layer_object
    ]
    scalar_inp = all_inp[0]
    lay_inp = all_inp[1]
    incflux = all_inp[2]

    hl_p = lay_inp[..., :1]
    albedos = scalar_inp[:, 1:2]

    lay_inp = lay_inp / VECTOR_PREDICTOR_MAX_VALUES
    scalar_inp = scalar_inp / SCALAR_PREDICTOR_MAX_VALUES

    if simpler_inputs:
        lay_inp = lay_inp[:, :, :14]

        scalar_inp = keras.layers.Concatenate(axis=-1)([
            scalar_inp[:, 0:1],
            scalar_inp[:, 1:2],
            scalar_inp[:, 6:7],
            scalar_inp[:, 7:8]
        ])

    if add_scalars_to_levels:
        # new_dimensions = (1, scalar_inp.shape[1])
        # lay_inp2 = keras.layers.Reshape(target_shape=new_dimensions)(scalar_inp)
        lay_inp2 = keras.layers.RepeatVector(127)(scalar_inp)
        lay_inp = keras.layers.Concatenate(axis=-1)([lay_inp, lay_inp2])

    mlp_surface_outp = keras.layers.Dense(
        nneur, activation=activ_surface, name='dense_surface'
    )(albedos)

    if lstm:
        mlp_surface_outp2 = keras.layers.Dense(
            nneur, activation=activ_surface, name='dense_surface2'
        )(albedos)

        init_state = [mlp_surface_outp, mlp_surface_outp2]
        rnnlayer = keras.layers.LSTM
    else:
        init_state = mlp_surface_outp
        rnnlayer = keras.layers.GRU

    if add_dense:
        lay_inp = keras.layers.Dense(
            nneur, activation=activ_surface, name='dense_lay'
        )(lay_inp)

    hidden1 = rnnlayer(nneur, return_sequences=True, go_backwards=False)(
        lay_inp, initial_state=init_state
    )
    hidden2 = rnnlayer(nneur, return_sequences=True, go_backwards=True)(
        hidden1
    )

    reverse_layer_object = keras.layers.Lambda(
        lambda x: K.reverse(x, axes=1),
        output_shape=hidden2.shape[1:]
    )
    hidden2 = reverse_layer_object(hidden2)
    # hidden2 = K.reverse(hidden2, axes=1)

    hidden2 = keras.layers.Concatenate(axis=2)([hidden1, hidden2])

    flux_sw = keras.layers.Dense(2, activation=activ_last, name='sw_denorm')(
        hidden2
    )
    flux_sw = keras.layers.Multiply(name='sw')([flux_sw, incflux])
    hr_sw = HeatingRateLayer(name='hr_sw')([flux_sw, hl_p])
    hr_sw = keras.layers.ReLU()(hr_sw)

    new_dimensions = hr_sw.shape[1:] + (1,)
    hr_sw = keras.layers.Reshape(target_shape=new_dimensions)(hr_sw)
    # hr_sw = K.expand_dims(hr_sw, axis=-1)

    # Layer now has dimensions E x H x 1.
    hr_sw = keras.layers.ZeroPadding1D(padding=(0, 1))(hr_sw)

    # Layer now has dimensions E x H x W x 1 = E x H x 1 x 1.
    new_dimensions = (hr_sw.shape[1], 1, hr_sw.shape[2])
    hr_sw = keras.layers.Reshape(
        target_shape=new_dimensions, name=neural_net.HEATING_RATE_TARGETS_KEY
    )(hr_sw)

    # Layer now has dimensions E x W x 2 = E x 1 x 2.
    flux_sw = keras.layers.Concatenate(
        axis=-1, name=neural_net.FLUX_TARGETS_KEY
    )([
        flux_sw[:, :1, :1],
        flux_sw[:, -1:, 1:]
    ])

    loss_dict = {
        neural_net.HEATING_RATE_TARGETS_KEY: VECTOR_LOSS_FUNCTION,
        neural_net.FLUX_TARGETS_KEY: SCALAR_LOSS_FUNCTION
    }
    metric_dict = {
        neural_net.HEATING_RATE_TARGETS_KEY: neural_net.METRIC_FUNCTION_LIST,
        neural_net.FLUX_TARGETS_KEY: neural_net.METRIC_FUNCTION_LIST
    }

    model = Model(inputs=all_inp, outputs=[hr_sw, flux_sw])
    model.summary()

    # TODO(thunderhoser): Try different initial learning rate, early stopping,
    # ReduceLROnPlateau, etc.
    model.compile(
        loss=loss_dict,
        optimizer=keras.optimizers.Nadam(),
        metrics=metric_dict
    )

    return model
