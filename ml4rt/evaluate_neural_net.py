"""Evaluates trained neural net."""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import prediction_io
import evaluation

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_FILE_ARG_NAME = 'input_prediction_file_name'
NUM_BOOTSTRAP_REPS_ARG_NAME = 'num_bootstrap_reps'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing predicted and actual target values.  Will '
    'be read by `prediction_io.read_file`.'
)
NUM_BOOTSTRAP_REPS_HELP_STRING = 'Number of bootstrap replicates.'
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Evaluation scores will be written here by '
    '`evaluation.write_file`, to a file name determined by '
    '`evaluation.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_BOOTSTRAP_REPS_ARG_NAME, type=int, required=True,
    help=NUM_BOOTSTRAP_REPS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(prediction_file_name, num_bootstrap_reps, output_dir_name):
    """Evaluates trained neural net.

    This is effectively the main method.

    :param prediction_file_name: See documentation at top of file.
    :param num_bootstrap_reps: Same.
    :param output_dir_name: Same.
    """

    file_metadata_dict = prediction_io.file_name_to_metadata(
        prediction_file_name
    )
    output_file_name = evaluation.find_file(
        directory_name=output_dir_name,
        zenith_angle_bin=file_metadata_dict[prediction_io.ZENITH_ANGLE_BIN_KEY],
        albedo_bin=file_metadata_dict[prediction_io.ALBEDO_BIN_KEY],
        month=file_metadata_dict[prediction_io.MONTH_KEY],
        surface_down_flux_bin=
        file_metadata_dict[prediction_io.SURFACE_DOWN_FLUX_BIN_KEY],
        aerosol_optical_depth_bin=
        file_metadata_dict[prediction_io.AEROSOL_OPTICAL_DEPTH_BIN_KEY],
        grid_row=file_metadata_dict[prediction_io.GRID_ROW_KEY],
        grid_column=file_metadata_dict[prediction_io.GRID_COLUMN_KEY],
        raise_error_if_missing=False
    )

    result_table_xarray = evaluation.get_scores_all_variables(
        prediction_file_name=prediction_file_name,
        num_bootstrap_reps=num_bootstrap_reps
    )
    print(SEPARATOR_STRING)

    t = result_table_xarray
    scalar_target_names = t.coords[evaluation.SCALAR_FIELD_DIM].values

    for k in range(len(scalar_target_names)):
        print((
            'Variable = "{0:s}" ... stdev of target and predicted values = '
            '{1:f}, {2:f} ... MSE and skill score = {3:f}, {4:f} ... '
            'MAE and skill score = {5:f}, {6:f} ... bias = {7:f} ... '
            'correlation = {8:f} ... KGE = {9:f}'
        ).format(
            scalar_target_names[k],
            numpy.nanmean(t[evaluation.SCALAR_TARGET_STDEV_KEY].values[k, :]),
            numpy.nanmean(
                t[evaluation.SCALAR_PREDICTION_STDEV_KEY].values[k, :]
            ),
            numpy.nanmean(t[evaluation.SCALAR_MSE_KEY].values[k, :]),
            numpy.nanmean(t[evaluation.SCALAR_MSE_SKILL_KEY].values[k, :]),
            numpy.nanmean(t[evaluation.SCALAR_MAE_KEY].values[k, :]),
            numpy.nanmean(t[evaluation.SCALAR_MAE_SKILL_KEY].values[k, :]),
            numpy.nanmean(t[evaluation.SCALAR_BIAS_KEY].values[k, :]),
            numpy.nanmean(t[evaluation.SCALAR_CORRELATION_KEY].values[k, :]),
            numpy.nanmean(t[evaluation.SCALAR_KGE_KEY].values[k, :])
        ))

    print(SEPARATOR_STRING)

    vector_target_names = t.coords[evaluation.VECTOR_FIELD_DIM].values
    heights_m_agl = t.coords[evaluation.HEIGHT_DIM].values

    for k in range(len(vector_target_names)):
        print('Variable = "{0:s}" ... PRMSE = {1:f}'.format(
            vector_target_names[k],
            numpy.nanmean(t[evaluation.VECTOR_PRMSE_KEY].values[k, :])
        ))

    print(SEPARATOR_STRING)

    for k in range(len(vector_target_names)):
        for j in range(len(heights_m_agl)):
            print((
                'Variable = "{0:s}" at {1:d} m AGL ... '
                'stdev of target and predicted values = {2:f}, {3:f} ... '
                'MSE and skill score = {4:f}, {5:f} ... '
                'MAE and skill score = {6:f}, {7:f} ... bias = {8:f} ... '
                'correlation = {9:f} ... KGE = {10:f}'
            ).format(
                vector_target_names[k], int(numpy.round(heights_m_agl[j])),
                numpy.nanmean(
                    t[evaluation.VECTOR_TARGET_STDEV_KEY].values[j, k, :]
                ),
                numpy.nanmean(
                    t[evaluation.VECTOR_PREDICTION_STDEV_KEY].values[j, k, :]
                ),
                numpy.nanmean(t[evaluation.VECTOR_MSE_KEY].values[j, k, :]),
                numpy.nanmean(
                    t[evaluation.VECTOR_MSE_SKILL_KEY].values[j, k, :]
                ),
                numpy.nanmean(t[evaluation.VECTOR_MAE_KEY].values[j, k, :]),
                numpy.nanmean(
                    t[evaluation.VECTOR_MAE_SKILL_KEY].values[j, k, :]
                ),
                numpy.nanmean(t[evaluation.VECTOR_BIAS_KEY].values[j, k, :]),
                numpy.nanmean(
                    t[evaluation.VECTOR_CORRELATION_KEY].values[j, k, :]
                ),
                numpy.nanmean(t[evaluation.VECTOR_KGE_KEY].values[j, k, :])
            ))

        print(SEPARATOR_STRING)

    try:
        aux_target_field_names = (
            t.coords[evaluation.AUX_TARGET_FIELD_DIM].values
        )
        aux_predicted_field_names = (
            t.coords[evaluation.AUX_PREDICTED_FIELD_DIM].values
        )
    except:
        aux_target_field_names = []
        aux_predicted_field_names = []

    for k in range(len(aux_target_field_names)):
        print((
            'Target variable = "{0:s}" ... predicted variable = "{1:s}" ... '
            'stdev of target and predicted values = {2:f}, {3:f} ... '
            'MSE and skill score = {4:f}, {5:f} ... '
            'MAE and skill score = {6:f}, {7:f} ... bias = {8:f} ... '
            'correlation = {9:f} ... KGE = {10:f}'
        ).format(
            aux_target_field_names[k], aux_predicted_field_names[k],
            numpy.nanmean(t[evaluation.AUX_TARGET_STDEV_KEY].values[k, :]),
            numpy.nanmean(t[evaluation.AUX_PREDICTION_STDEV_KEY].values[k, :]),
            numpy.nanmean(t[evaluation.AUX_MSE_KEY].values[k, :]),
            numpy.nanmean(t[evaluation.AUX_MSE_SKILL_KEY].values[k, :]),
            numpy.nanmean(t[evaluation.AUX_MAE_KEY].values[k, :]),
            numpy.nanmean(t[evaluation.AUX_MAE_SKILL_KEY].values[k, :]),
            numpy.nanmean(t[evaluation.AUX_BIAS_KEY].values[k, :]),
            numpy.nanmean(t[evaluation.AUX_CORRELATION_KEY].values[k, :]),
            numpy.nanmean(t[evaluation.AUX_KGE_KEY].values[k, :])
        ))

    print(SEPARATOR_STRING)

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    evaluation.write_file(
        result_table_xarray=result_table_xarray,
        netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        num_bootstrap_reps=getattr(
            INPUT_ARG_OBJECT, NUM_BOOTSTRAP_REPS_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
