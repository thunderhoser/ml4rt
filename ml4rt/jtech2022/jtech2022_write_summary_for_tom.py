"""Writes accuracy/complexity summary of each model for Tom Beucler."""

import os
import warnings
import argparse
import numpy
from gewittergefahr.gg_utils import file_system_utils
from ml4rt.utils import evaluation
from ml4rt.utils import example_utils

FIRST_LAYER_CHANNEL_COUNTS = numpy.array([4, 8, 16, 32, 64, 128], dtype=int)
MODEL_DEPTHS = numpy.array([3, 4, 5], dtype=int)
MODEL_WIDTHS = numpy.array([1, 2, 3, 4], dtype=int)

NUM_WEIGHTS_ARRAY_PLUSPLUS = numpy.array([
    71300, 238499, 809015, 2769136, 9562039, 33240174, 76988, 260627, 896279,
    3115696, 10943287, 38755182, 82676, 282755, 983543, 3462256, 12324535,
    44270190, 88364, 304883, 1070807, 3808816, 13705783, 49785198, 88569,
    311439, 1114490, 4036476, 14783144, 54689538, 110997, 399783, 1465130,
    5433564, 20360552, 76977282, 133425, 488127, 1815770, 6830652, 25937960,
    99265026, 155853, 576471, 2166410, 8227740, 31515368, 121552770, 171767,
    651494, 2501732, 9675391, 37636990, 147105982, 260471, 1003430, 3903716,
    15271807, 59999614, 236510398, 349175, 1355366, 5305700, 20868223,
    82362238, 325914814, 437879, 1707302, 6707684, 26464639, 104724862,
    415319230
], dtype=int)

NUM_WEIGHTS_ARRAY_PLUSPLUS_DEEP = numpy.array([
    71310, 238517, 809049, 2769202, 9562169, 33240432, 76998, 260645, 896313,
    3115762, 10943417, 38755440, 82686, 282773, 983577, 3462322, 12324665,
    44270448, 88374, 304901, 1070841, 3808882, 13705913, 49785456, 88584,
    311466, 1114541, 4036575, 14783339, 54689925, 111012, 399810, 1465181,
    5433663, 20360747, 76977669, 133440, 488154, 1815821, 6830751, 25938155,
    99265413, 155868, 576498, 2166461, 8227839, 31515563, 121553157, 171787,
    651530, 2501800, 9675523, 37637250, 147106498, 260491, 1003466, 3903784,
    15271939, 59999874, 236510914, 349195, 1355402, 5305768, 20868355,
    82362498, 325915330, 437899, 1707338, 6707752, 26464771, 104725122,
    415319746
], dtype=int)

NUM_WEIGHTS_ARRAY_PLUSPLUSPLUS = numpy.array([
    70464, 234907, 794151, 2708688, 9318263, 32261102, 77172, 261091, 897591,
    3119856, 10957751, 38808686, 83880, 287275, 1001031, 3531024, 12597239,
    45356270, 90588, 313459, 1104471, 3942192, 14236727, 51903854, 76073,
    261295, 913594, 3232252, 11564968, 41814274, 97853, 347191, 1254730,
    4591900, 16993768, 63509890, 119633, 433087, 1595866, 5951548, 22422568,
    85205506, 141413, 518983, 1937002, 7311196, 27851368, 106901122, 99547,
    362958, 1348276, 5062943, 19189950, 73323326, 174823, 661830, 2539300,
    9818111, 38192766, 149298878, 250099, 960702, 3730324, 14573279, 57195582,
    225274430, 325375, 1259574, 4921348, 19328447, 76198398, 301249982
], dtype=int)

NUM_WEIGHTS_ARRAY_PLUSPLUSPLUS_DEEP = numpy.array([
    70531, 235038, 794410, 2709203, 9319290, 32263153, 77239, 261222, 897850,
    3120371, 10958778, 38810737, 83947, 287406, 1001290, 3531539, 12598266,
    45358321, 90655, 313590, 1104730, 3942707, 14237754, 51905905, 76201,
    261547, 914094, 3233248, 11566956, 41818246, 97981, 347443, 1255230,
    4592896, 16995756, 63513862, 119761, 433339, 1596366, 5952544, 22424556,
    85209478, 141541, 519235, 1937502, 7312192, 27853356, 106905094, 99776,
    363411, 1349177, 5064740, 19193539, 73330499, 175052, 662283, 2540201,
    9819908, 38196355, 149306051, 250328, 961155, 3731225, 14575076, 57199171,
    225281603, 325604, 1260027, 4922249, 19330244, 76201987, 301257155
], dtype=int)

INPUT_DIR_ARG_NAME = 'input_top_experiment_dir_name'
PLUSPLUSPLUS_ARG_NAME = 'plusplusplus_flag'
DEEP_SUPERVISION_ARG_NAME = 'deep_supervision_flag'
OUTPUT_FILE_ARG_NAME = 'output_csv_file_name'

INPUT_DIR_HELP_STRING = (
    'Name of top-level directory, containing a subdirectory for each of the 4 '
    'experiments (U-net++ with and sans deep supervision, U-net3+ with and '
    'sans deep supervision).'
)
PLUSPLUSPLUS_HELP_STRING = (
    'Boolean flag.  If 1 (0), will write summary for U-net3+ (U-net++) models.'
)
DEEP_SUPERVISION_HELP_STRING = (
    'Boolean flag.  If 1 (0), will write summary for (non) deeply supervised '
    'models.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Results will be saved here in CSV format.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PLUSPLUSPLUS_ARG_NAME, type=int, required=True,
    help=PLUSPLUSPLUS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DEEP_SUPERVISION_ARG_NAME, type=int, required=True,
    help=DEEP_SUPERVISION_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(top_experiment_dir_name, plusplusplus_flag, deep_supervision_flag,
         output_file_name):
    """Writes accuracy/complexity summary of each model for Tom Beucler.

    This is effectively the main method.

    :param top_experiment_dir_name: See documentation at top of file.
    :param plusplusplus_flag: Same.
    :param deep_supervision_flag: Same.
    :param output_file_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    num_depths = len(MODEL_DEPTHS)
    num_widths = len(MODEL_WIDTHS)
    num_channel_counts = len(FIRST_LAYER_CHANNEL_COUNTS)

    if plusplusplus_flag:
        if deep_supervision_flag:
            num_weights_array = NUM_WEIGHTS_ARRAY_PLUSPLUSPLUS_DEEP
            subdir_name = '2022paper_experiment_sw_plusplusplus_deep'
            model_type_string = 'u_net_plusplusplus_with_deep_supervision'
        else:
            num_weights_array = NUM_WEIGHTS_ARRAY_PLUSPLUSPLUS
            subdir_name = '2022paper_experiment_sw_plusplusplus'
            model_type_string = 'u_net_plusplusplus_sans_deep_supervision'
    else:
        if deep_supervision_flag:
            num_weights_array = NUM_WEIGHTS_ARRAY_PLUSPLUS_DEEP
            subdir_name = '2022paper_experiment_sw_plusplus_deep'
            model_type_string = 'u_net_plusplus_with_deep_supervision'
        else:
            num_weights_array = NUM_WEIGHTS_ARRAY_PLUSPLUS
            subdir_name = '2022paper_experiment_sw_plusplus'
            model_type_string = 'u_net_plusplus_sans_deep_supervision'

    these_dim = (num_widths, num_depths, num_channel_counts)
    num_weights_matrix = numpy.reshape(num_weights_array, these_dim)
    num_weights_matrix = numpy.swapaxes(num_weights_matrix, 0, 1)

    hyperparam_string_matrix = numpy.full(
        num_weights_matrix.shape, '', dtype='object'
    )
    mae_matrix_w_m02 = numpy.full(num_weights_matrix.shape, numpy.nan)
    mse_matrix_w_m04 = numpy.full(num_weights_matrix.shape, numpy.nan)

    for i in range(num_depths):
        for j in range(num_widths):
            for k in range(num_channel_counts):
                hyperparam_string_matrix[i, j, k] = (
                    'num-levels={0:d}_num-conv-layers-per-block={1:d}_'
                    'num-first-layer-channels={2:03d}'
                ).format(
                    MODEL_DEPTHS[i], MODEL_WIDTHS[j],
                    FIRST_LAYER_CHANNEL_COUNTS[k]
                )

                this_eval_file_name = (
                    '{0:s}/{1:s}/depth={2:d}_num-conv-layers-per-block={3:d}_'
                    'num-first-layer-channels={4:03d}/'
                    'model/validation/evaluation.nc'
                ).format(
                    top_experiment_dir_name, subdir_name,
                    MODEL_DEPTHS[i], MODEL_WIDTHS[j],
                    FIRST_LAYER_CHANNEL_COUNTS[k]
                )

                if not os.path.isfile(this_eval_file_name):
                    warning_string = (
                        'POTENTIAL ERROR: Cannot find evaluation file expected '
                        'at: "{0:s}"'
                    ).format(this_eval_file_name)

                    warnings.warn(warning_string)
                    continue

                print('Reading data from: "{0:s}"...'.format(
                    this_eval_file_name
                ))
                this_eval_table_xarray = evaluation.read_file(
                    this_eval_file_name
                )
                et = this_eval_table_xarray

                target_var_index = numpy.where(
                    et.coords[evaluation.SCALAR_FIELD_DIM].values ==
                    example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME
                )[0][0]
                mae_matrix_w_m02[i, j, k] = numpy.nanmean(
                    et[evaluation.SCALAR_MAE_KEY].values[target_var_index, :]
                )
                mse_matrix_w_m04[i, j, k] = numpy.nanmean(
                    et[evaluation.SCALAR_MSE_KEY].values[target_var_index, :]
                )

    print('Writing summary to: "{0:s}"...'.format(output_file_name))
    with open(output_file_name, 'w') as output_file_handle:
        output_file_handle.write(
            'model_type, model_hyperparams, num_model_params, '
            'shortwave_surface_down_flux_mae_w_m02, '
            'shortwave_surface_down_flux_mse_w2_m04\n'
        )

        for i in range(num_depths):
            for j in range(num_widths):
                for k in range(num_channel_counts):
                    new_line = '{0:s}, {1:s}, {2:d}, {3:.10f}, {4:.10f}\n'.format(
                        model_type_string,
                        hyperparam_string_matrix[i, j, k],
                        num_weights_matrix[i, j, k],
                        mae_matrix_w_m02[i, j, k],
                        mse_matrix_w_m04[i, j, k]
                    )

                    output_file_handle.write(new_line)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_experiment_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        plusplusplus_flag=bool(
            getattr(INPUT_ARG_OBJECT, PLUSPLUSPLUS_ARG_NAME)
        ),
        deep_supervision_flag=bool(
            getattr(INPUT_ARG_OBJECT, DEEP_SUPERVISION_ARG_NAME)
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
