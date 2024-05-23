import xarray
import tensorflow as tf
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Layer, Conv1D, GRU, LSTM
from tensorflow.keras.layers import Multiply, ZeroPadding1D
# from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
# from tensorflow.keras.optimizers import RMSprop,Adam,Nadam,SGD
# from tensorflow.keras import backend as K
# from tensorflow.python.ops.nn import swish as myswish
# from tensorflow import nn

class HRLayer(tf.keras.layers.Layer):
# Computes heating rate
    def __init__(self,name=None,
                 hr_units = 'K d-1',
                 **kwargs,
             ):
        super(HRLayer, self).__init__(name=name,**kwargs)
        time_scale  = {'K s-1':1,
                       'K d-1':3600*24}[hr_units]
        self.g_cp = tf.constant(9.8066 / 1004 * time_scale,dtype=tf.float32)

    def get_config(self):
        cfg = super().get_config()
        return cfg

    def build(self, input_shape):
        pass

    def call(self, inputs):
        fluxes = inputs[0]
        hlpress = inputs[1]
        netflux = fluxes[...,0] - fluxes[...,1]
        flux_diff = netflux[...,:-1] - netflux[...,1:]
        net_press = hlpress[...,:-1,0]-hlpress[...,1:,0]
        return -self.g_cp * tf.math.divide(flux_diff,net_press)


def get_inpout(example_file_name):
    example_table_xarray = xarray.open_dataset(example_file_name)

    inp_spec = {}
    for key in ["scalar_predictor_matrix","vector_predictor_matrix"]:
        inp_spec[key] = example_table_xarray[key]

    inp_spec["toa_down"] = example_table_xarray["vector_target_matrix"][:,-1:,:1]
    out_spec = example_table_xarray['vector_target_matrix']

    return inp_spec,out_spec
    
    
def rnn_sw(inp_spec,outp_spec, nneur=64, 
          lstm = True,
          activ_last = 'sigmoid',
          activ_surface = 'linear',
          activ_dense ='relu', # For initial layer-wise dense if add_dense=True
          add_dense=False,
          add_scalars_to_levels=True, 
          # ^True needed because aerosol ssa and asymmetry are for some reason scalars in this dataset,
          # but these are layer-wise properties
          simpler_inputs=True
):
    #Assume inputs have the order
    #scalar, column, hl, inter, pressure_hl
    kw = 5
    all_inp = []
    for k in inp_spec.keys():
        all_inp.append(Input(inp_spec[k].shape[1:],name=k))

    scalar_inp = all_inp[0]
    lay_inp = all_inp[1]
    incflux = all_inp[2]

    hl_p = lay_inp[...,:1]

    sca_norm = tf.constant([  1.4835263 ,   0.8722802 ,  89.910324  , 359.8828    ,
         2.730448  ,   8.228799  ,   1.        ,   0.96304035])
    # "zenith_angle_radians",  "albedo",  "latitude_deg_n", "longitude_deg_e", 
    #  "column_liquid_water_path_kg_m02",  "column_ice_water_path_kg_m02", 
    #  "aerosol_single_scattering_albedo", "aerosol_asymmetry_param" ;

    vec_norm = tf.constant([1.04399836e+05, 3.18863373e+02, 2.22374070e-02, 1.33532193e-03,
       1.47309888e-03, 1.73750886e-05, 1.15184314e+03, 6.53699684e+00,
       1.38715994e+00, 9.72749013e-03, 1.70985604e-05, 6.53300012e-05,
       7.95751719e+04, 3.15813086e+03, 1.78493945e+03, 2.73044801e+00,
       8.22879887e+00, 7.84459991e+01, 2.73044801e+00, 8.22879887e+00,
       7.84459991e+01, 1.23876810e+00])
    # "pressure_pascals",  "temperature_kelvins", "specific_humidity_kg_kg01", "liquid_water_content_kg_m03", 
    # "ice_water_content_kg_m03", "o3_mixing_ratio_kg_kg01", "co2_concentration_ppmv", "ch4_concentration_ppmv", 
    # "n2o_concentration_ppmv", "aerosol_extinction_metres01", "liquid_effective_radius_metres", "ice_effective_radius_metres", 
    # "height_m_agl", "height_thickness_metres", "pressure_thickness_pascals", "liquid_water_path_kg_m02", 
    # "ice_water_path_kg_m02", "vapour_path_kg_m02", "upward_liquid_water_path_kg_m02", "upward_ice_water_path_kg_m02", 
    # "upward_vapour_path_kg_m02", "relative_humidity_unitless" ;

    # extract scalar variables we need
    # cos_sza = tf.math.cos(scalar_inp[:,0])
    albedos = scalar_inp[:,1:2] # in ecRad/IFS this would be a vector (provided for several bands)

    lay_inp = lay_inp / vec_norm
    scalar_inp = scalar_inp / sca_norm

    if simpler_inputs:
        # Skip water paths (vertically integrated quantities) - probably not needed for RNN
        lay_inp = lay_inp[:,:,0:14]
        # We also don't want to use non physical inputs such as lat lon
        scalar_inp = scalar_inp[:,[0,1,6,7]] 
        # last two are ssa and asymmetry. odd that these are scalars here? optical properties are layer-wise quantities
        # assumed vertically constant in an "aerosol layer" spanning many vertical layers?
        # in any case we need to add these to layer wise inputs:

    if add_scalars_to_levels:
      lay_inp2 = tf.repeat(tf.expand_dims(scalar_inp,axis=1),repeats=127, axis=1)
      lay_inp = tf.concat([lay_inp,lay_inp2],axis=-1)
    print(lay_inp.shape)

    # Outputs are the raw fluxes scaled by incoming flux
    ny = 2
    # incoming flux from inputs
    # incflux = Multiply()([cos_sza, solar_irrad])
    # incflux = tf.expand_dims(tf.expand_dims(incflux,axis=1),axis=2)

    # We used a dense NN to predict the initial state of the RNN iterating from the surface upwards
    mlp_surface_outp = Dense(nneur, activation=activ_surface,name='dense_surface')(albedos)

    if lstm:
        mlp_surface_outp2 = Dense(nneur, activation=activ_surface,name='dense_surface2')(albedos)
        init_state = [mlp_surface_outp,mlp_surface_outp2]
        rnnlayer = LSTM
    else:
        init_state = mlp_surface_outp
        rnnlayer = GRU

    if add_dense:
        lay_inp = Dense(nneur, activation=activ_surface,name='dense_lay')(lay_inp)

    hidden1 = rnnlayer(nneur,return_sequences=True,
                       go_backwards=False)(lay_inp, initial_state=init_state)

    # Second RNN layer
    hidden2 = rnnlayer(nneur,return_sequences=True,
                       go_backwards = True)(hidden1)#,#
    # When go_backwards=True, TF/Keras does not automatically then reverse the RNN output 
    # so that it's in the original direction, meaning we need to do it manually!
    # This fix was missing from my 2022 paper, which is probably why I needed 3 RNNs..
    hidden2 = tf.reverse(hidden2,axis=[1])
    hidden2  = tf.concat([hidden1,hidden2],axis=2)

    # Conv1D with kernel_size 1 is the same as Dense applied to vertical layers
    flux_sw = Conv1D(ny, kernel_size = 1, activation=activ_last,
                     name='sw_denorm'
    )(hidden2)

    flux_sw = Multiply(name='sw')([flux_sw, incflux]) # Normalize outputs by incoming flux at TOA
    
    hr_sw = HRLayer(name='hr_sw')([flux_sw, hl_p])
    hr_sw = tf.expand_dims(hr_sw,axis=-1)
    hr_sw = ZeroPadding1D(padding=(0,1))(hr_sw)

    outputs = tf.concat([flux_sw,hr_sw],axis=-1)
    print(outputs.shape)

    model = Model(inputs=all_inp, outputs=outputs)
    model.summary()

    # return all_inp, outputs
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
