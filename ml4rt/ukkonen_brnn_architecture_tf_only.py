import xarray
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Layer, AveragePooling1D, Conv1D, ZeroPadding1D, GRU, LSTM
from tensorflow.keras.layers import Multiply, Concatenate


def get_inpout(example_file_name):
    example_table_xarray = xarray.open_dataset(example_file_name)

    inp_spec = {}
    for key in ["scalar_predictor_matrix","vector_predictor_matrix"]:
        inp_spec[key] = example_table_xarray[key]

    inp_spec["toa_down"] = example_table_xarray["vector_target_matrix"][:,-1:,:1]
    out_spec = example_table_xarray['vector_target_matrix']

    return inp_spec,out_spec


class HRLayer(tf.keras.layers.Layer):
    def __init__(self, name=None, hr_units='K d-1', **kwargs):
        super(HRLayer, self).__init__(name=name, **kwargs)
        time_scale = {'K s-1': 1, 'K d-1': 3600 * 24}[hr_units]
        self.g_cp = tf.constant(9.8066 / 1004 * time_scale, dtype=tf.float32)

    def get_config(self):
        cfg = super().get_config()
        return cfg

    def build(self, input_shape):
        pass

    def call(self, inputs):
        fluxes = inputs[0]
        hlpress = inputs[1]
        netflux = fluxes[..., 0] - fluxes[..., 1]
        flux_diff = netflux[..., :-1] - netflux[..., 1:]
        net_press = hlpress[..., :-1, 0] - hlpress[..., 1:, 0]
        return -self.g_cp * tf.math.divide(flux_diff, net_press)


def rnn_sw(inp_spec, outp_spec, nneur=64, lstm=True, activ_last='sigmoid', activ_surface='linear', activ_dense='relu', add_dense=False, add_scalars_to_levels=True, simpler_inputs=True):
    # Assume inputs have the order
    # scalar, column, hl, inter, pressure_hl
    kw = 5
    all_inp = [Input(shape=inp_spec[k].shape[1:], name=k) for k in inp_spec.keys()]

    scalar_inp = all_inp[0]
    lay_inp = all_inp[1]
    incflux = all_inp[2]

    hl_p = lay_inp[..., :1]

    sca_norm = tf.constant([1.4835263, 0.8722802, 89.910324, 359.8828, 2.730448, 8.228799, 1., 0.96304035])
    vec_norm = tf.constant([1.04399836e+05, 3.18863373e+02, 2.22374070e-02, 1.33532193e-03, 1.47309888e-03, 1.73750886e-05, 1.15184314e+03, 6.53699684e+00, 1.38715994e+00, 9.72749013e-03, 1.70985604e-05, 6.53300012e-05, 7.95751719e+04, 3.15813086e+03, 1.78493945e+03, 2.73044801e+00, 8.22879887e+00, 7.84459991e+01, 2.73044801e+00, 8.22879887e+00, 7.84459991e+01, 1.23876810e+00])

    # Extract scalar variables we need
    albedos = scalar_inp[:, 1:2]

    lay_inp = lay_inp / vec_norm
    scalar_inp = scalar_inp / sca_norm

    if simpler_inputs:
        lay_inp = lay_inp[:, :, 0:14]

        scalar_inp = Concatenate(axis=-1)([
            scalar_inp[:, 0:1],
            scalar_inp[:, 1:2],
            scalar_inp[:, 6:7],
            scalar_inp[:, 7:8]
        ])

    if add_scalars_to_levels:
        lay_inp2 = tf.repeat(tf.expand_dims(scalar_inp, axis=1), repeats=127, axis=1)
        lay_inp = tf.concat([lay_inp, lay_inp2], axis=-1)

    # Outputs are the raw fluxes scaled by incoming flux
    ny = 2

    # MLP for initial state of RNN
    mlp_surface_outp = Dense(nneur, activation=activ_surface, name='dense_surface')(albedos)

    if lstm:
        mlp_surface_outp2 = Dense(nneur, activation=activ_surface, name='dense_surface2')(albedos)
        init_state = [mlp_surface_outp, mlp_surface_outp2]
        rnnlayer = LSTM
    else:
        init_state = mlp_surface_outp
        rnnlayer = GRU

    if add_dense:
        lay_inp = Dense(nneur, activation=activ_surface, name='dense_lay')(lay_inp)

    hidden1 = rnnlayer(nneur, return_sequences=True, go_backwards=False)(lay_inp, initial_state=init_state)

    # Second RNN layer
    hidden2 = rnnlayer(nneur, return_sequences=True, go_backwards=True)(hidden1)
    hidden2 = tf.reverse(hidden2, axis=[1])
    hidden2 = tf.concat([hidden1, hidden2], axis=2)

    flux_sw = Conv1D(ny, kernel_size=1, activation=activ_last, name='sw_denorm')(hidden2)
    flux_sw = Multiply(name='sw')([flux_sw, incflux])

    hr_sw = HRLayer(name='hr_sw')([flux_sw, hl_p])
    hr_sw = tf.expand_dims(hr_sw, axis=-1)
    hr_sw = ZeroPadding1D(padding=(0, 1))(hr_sw)

    outputs = tf.concat([flux_sw, hr_sw], axis=-1)

    model = Model(inputs=all_inp, outputs=outputs)

    return model


if __name__ == '__main__':
    inp_spec, outp_spec = get_inpout(
        '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4rt_project/gfs_data/'
        'examples_with_correct_vertical_coords/shortwave/training/'
        'for_pareto_paper_2024/simple/learning_examples.nc'
    )

    model_object = rnn_sw(
        inp_spec=inp_spec,
        outp_spec=outp_spec,
        nneur=64,
        lstm=True,
        activ_last='sigmoid',
        activ_surface='linear',
        activ_dense='relu',
        add_dense=False,
        add_scalars_to_levels=True,
        simpler_inputs=True
    )