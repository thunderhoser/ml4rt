"""Writes summary evaluation table for one model.

The summary table is written in LaTeX and printed to an ASCII file.
"""

import os
import argparse
import numpy
from gewittergefahr.gg_utils import file_system_utils
from ml4rt.io import prediction_io
from ml4rt.utils import evaluation
from ml4rt.utils import example_utils
from ml4rt.machine_learning import neural_net

METRES_TO_MICRONS = 1e6

INPUT_FILE_ARG_NAME = 'input_prediction_file_name'
OUTPUT_FILE_ARG_NAME = 'output_tex_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing predictions from one model.  Will be read '
    'by `prediction_io.read_file`.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  The summary table will be written here in LaTeX '
    'code.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _compute_scores(prediction_dict, wavelength_metres):
    """Computes scores from prediction file.

    :param prediction_dict: Dictionary returned by `prediction_io.read_file`.
    :param wavelength_metres: Will compute scores for this wavelength.
    :return: rmse_k_day01: RMSE for heating rate (Kelvins per day).
    :return: near_sfc_rmse_k_day01: RMSE for near-surface heating rate.
    :return: profile_rmse_k_day01: PRMSE for heating rate.
    :return: bias_k_day01: Bias for heating rate.
    :return: near_sfc_bias_k_day01: Bias for near-surface heating rate.
    :return: down_flux_rmse_w_m02: RMSE for surface downwelling flux (Watts per
        square meter).
    :return: up_flux_rmse_w_m02: RMSE for TOA upwelling flux.
    """

    w = example_utils.match_wavelengths(
        wavelengths_metres=
        prediction_dict[prediction_io.TARGET_WAVELENGTHS_KEY],
        desired_wavelength_metres=wavelength_metres
    )
    vector_target_matrix = (
        prediction_dict[prediction_io.VECTOR_TARGETS_KEY][:, :, w, ...]
    )
    vector_prediction_matrix = numpy.mean(
        prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY][:, :, w, ...],
        axis=-1
    )

    rmse_k_day01 = numpy.sqrt(numpy.mean(
        (vector_prediction_matrix - vector_target_matrix) ** 2
    ))
    near_sfc_rmse_k_day01 = numpy.sqrt(numpy.mean(
        (vector_prediction_matrix[:, 0, :] - vector_target_matrix[:, 0, :]) ** 2
    ))
    profile_rmse_k_day01 = evaluation._get_prmse_one_variable(
        target_matrix=vector_target_matrix,
        prediction_matrix=vector_prediction_matrix
    )
    bias_k_day01 = numpy.mean(
        vector_prediction_matrix - vector_target_matrix
    )
    near_sfc_bias_k_day01 = numpy.mean(
        vector_prediction_matrix[:, 0, :] - vector_target_matrix[:, 0, :]
    )

    model_file_name = prediction_dict[prediction_io.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading data from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    training_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]

    scalar_target_names = (
        training_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY]
    )
    if not isinstance(scalar_target_names, list):
        scalar_target_names = scalar_target_names.tolist()

    is_model_shortwave = (
        example_utils.SHORTWAVE_HEATING_RATE_NAME in
        training_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY]
    )

    if is_model_shortwave:
        d = scalar_target_names.index(
            example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME
        )
    else:
        d = scalar_target_names.index(
            example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME
        )

    target_down_flux_matrix_w_m02 = (
        prediction_dict[prediction_io.SCALAR_TARGETS_KEY][:, w, d]
    )
    predicted_down_flux_matrix_w_m02 = numpy.mean(
        prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY][:, w, d, ...],
        axis=-1
    )
    down_flux_rmse_w_m02 = numpy.sqrt(numpy.mean(
        (predicted_down_flux_matrix_w_m02 - target_down_flux_matrix_w_m02) ** 2
    ))

    if is_model_shortwave:
        u = scalar_target_names.index(
            example_utils.SHORTWAVE_TOA_UP_FLUX_NAME
        )
    else:
        u = scalar_target_names.index(
            example_utils.LONGWAVE_TOA_UP_FLUX_NAME
        )

    target_up_flux_matrix_w_m02 = (
        prediction_dict[prediction_io.SCALAR_TARGETS_KEY][:, w, u]
    )
    predicted_up_flux_matrix_w_m02 = numpy.mean(
        prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY][:, w, u, ...],
        axis=-1
    )
    up_flux_rmse_w_m02 = numpy.sqrt(numpy.mean(
        (predicted_up_flux_matrix_w_m02 - target_up_flux_matrix_w_m02) ** 2
    ))

    return (
        rmse_k_day01, near_sfc_rmse_k_day01, profile_rmse_k_day01,
        bias_k_day01, near_sfc_bias_k_day01,
        down_flux_rmse_w_m02, up_flux_rmse_w_m02
    )


def _run(prediction_file_name, output_file_name):
    """Writes summary evaluation table for one model.

    This is effectively the main method.

    :param prediction_file_name: See documentation at top of this script.
    :param output_file_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(prediction_file_name)
    wavelengths_metres = prediction_dict[prediction_io.TARGET_WAVELENGTHS_KEY]

    with open(output_file_name, 'w') as output_file_handle:
        output_file_handle.write('\\begin{table}[t]\r\n')
        output_file_handle.write('\t\\captionsetup{justification = centering}\r\n')
        output_file_handle.write('\t\\caption{RT performance.}\r\n')
        output_file_handle.write('\t\\label{table:rt_performance}\r\n')
        output_file_handle.write('\t\\begin{center}\r\n')
        output_file_handle.write('\t\t\\begin{tabular}{l | l | l | l | l | l | l | l}\r\n')
        output_file_handle.write('\t\t\t\\hline \\hline\r\n')
        output_file_handle.write('\t\t\t\\multicolumn{1}{c|}{\\textbf{Wavelength}} & \\multicolumn{1}{|c|}{\\textbf{HR}} & \\multicolumn{1}{|c|}{\\textbf{Near-sfc}} & \\multicolumn{1}{|c|}{\\textbf{HR}} & \\multicolumn{1}{|c|}{\\textbf{HR}} & \\multicolumn{1}{|c|}{\\textbf{Near-sfc}} & \\multicolumn{1}{|c|}{\\textbf{$F_{\\textrm{down}}^{\\textrm{sfc}}$}} & \\multicolumn{1}{|c}{\\textbf{$F_{\\textrm{up}}^{\\textrm{TOA}}$}} \\\\\r\n')
        output_file_handle.write('\t\t\t\\multicolumn{1}{c|}{\\textbf{}} & \\multicolumn{1}{|c|}{\\textbf{RMSE}} & \\multicolumn{1}{|c|}{\\textbf{HR RMSE}} & \\multicolumn{1}{|c|}{\\textbf{PRMSE}} & \\multicolumn{1}{|c|}{\\textbf{bias}} & \\multicolumn{1}{|c|}{\\textbf{HR bias}} & \\multicolumn{1}{|c|}{\\textbf{RMSE}} & \\multicolumn{1}{|c}{\\textbf{RMSE}} \\\\\r\n')
        output_file_handle.write('\t\t\t\\multicolumn{1}{c|}{\\textbf{($\\mu$m)}} & \\multicolumn{1}{|c|}{\\textbf{(K day\\textsuperscript{-1})}} & \\multicolumn{1}{|c|}{\\textbf{(K day\\textsuperscript{-1})}} & \\multicolumn{1}{|c|}{\\textbf{(K day\\textsuperscript{-1})}} & \\multicolumn{1}{|c|}{\\textbf{(K day\\textsuperscript{-1})}} & \\multicolumn{1}{|c|}{\\textbf{(K day\\textsuperscript{-1})}} & \\multicolumn{1}{|c|}{\\textbf{(W m\\textsuperscript{-2})}} & \\multicolumn{1}{|c}{\\textbf{(W m\\textsuperscript{-2})}} \\\\\r\n')
        output_file_handle.write('\t\t\t\\hline\r\n')

        for this_wavelength_metres in wavelengths_metres:
            (
                rmse_k_day01, near_sfc_rmse_k_day01, profile_rmse_k_day01,
                bias_k_day01, near_sfc_bias_k_day01,
                down_flux_rmse_w_m02, up_flux_rmse_w_m02
            ) = _compute_scores(
                prediction_dict=prediction_dict,
                wavelength_metres=this_wavelength_metres
            )

            print('Writing results for {0:.2f} microns to: "{1:s}"...'.format(
                METRES_TO_MICRONS * this_wavelength_metres,
                output_file_name
            ))

            output_file_handle.write((
                '\t\t\t{0:.2f} & {1:.4f} & {2:.4f} & {3:.4f} & '
                '{4:.4f} & {5:.4f} & {6:.4f} & {7:.4f} \\\\\r\n'
            ).format(
                METRES_TO_MICRONS * this_wavelength_metres,
                rmse_k_day01,
                near_sfc_rmse_k_day01,
                profile_rmse_k_day01,
                bias_k_day01,
                near_sfc_bias_k_day01,
                down_flux_rmse_w_m02,
                up_flux_rmse_w_m02
            ))

        output_file_handle.write('\t\t\\end{tabular}\r\n')
        output_file_handle.write('\t\\end{center}\r\n')
        output_file_handle.write('\\end{table}')


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
