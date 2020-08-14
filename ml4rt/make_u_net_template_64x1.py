"""Makes U-net template where output is 64 heights x 1 channel."""

import sys
import socket
import os.path

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import u_net_architecture
import custom_losses
import file_system_utils
import neural_net

if 'hfe' in socket.gethostname():
    HOME_DIR_NAME = '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist'
else:
    HOME_DIR_NAME = os.path.expanduser('~')

MODEL_FILE_NAME = '{0:s}/ml4rt_models/templates/u_net_64x1.h5'.format(
    HOME_DIR_NAME
)

ARCHITECTURE_OPTION_DICT = {
    u_net_architecture.NUM_HEIGHTS_KEY: 64,
    u_net_architecture.NUM_HEIGHTS_FOR_LOSS_KEY: 64,
    u_net_architecture.NUM_INPUT_CHANNELS_KEY: 16,
    u_net_architecture.OUTPUT_ACTIV_FUNCTION_KEY:
        u_net_architecture.RELU_FUNCTION_STRING
}

LOSS_FUNCTION = custom_losses.dual_weighted_mse()


def _run():
    """Main method."""

    file_system_utils.mkdir_recursive_if_necessary(file_name=MODEL_FILE_NAME)

    model_object = u_net_architecture.create_model(
        option_dict=ARCHITECTURE_OPTION_DICT,
        vector_loss_function=LOSS_FUNCTION, num_output_channels=1
    )

    print('Writing model to: "{0:s}"...'.format(MODEL_FILE_NAME))
    model_object.save(
        filepath=MODEL_FILE_NAME, overwrite=True, include_optimizer=True
    )

    metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(MODEL_FILE_NAME)[0],
        raise_error_if_missing=False
    )

    dummy_option_dict = {neural_net.VECTOR_TARGET_NORM_TYPE_KEY: None}

    print('Writing metadata to: "{0:s}"...'.format(metafile_name))
    neural_net._write_metafile(
        dill_file_name=metafile_name, num_epochs=100,
        num_training_batches_per_epoch=100,
        training_option_dict=dummy_option_dict,
        num_validation_batches_per_epoch=100,
        validation_option_dict=dummy_option_dict,
        net_type_string=neural_net.U_NET_TYPE_STRING,
        loss_function_or_dict=LOSS_FUNCTION
    )


if __name__ == '__main__':
    _run()
