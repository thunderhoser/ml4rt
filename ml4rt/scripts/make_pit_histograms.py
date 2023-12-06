"""Creates PIT (prob integ transform) histogram for each target variable."""

import argparse
import numpy
from ml4rt.utils import pit_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

METRES_TO_MICRONS = 1e6

INPUT_FILE_ARG_NAME = 'input_prediction_file_name'
NUM_BINS_ARG_NAME = 'num_bins'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing predictions and target values.  Will be '
    'read by `prediction_io.read_file`.'
)
NUM_BINS_HELP_STRING = 'Number of bins in each histogram.'
OUTPUT_FILE_HELP_STRING = (
    'Path to output (NetCDF) file.  Results will be written here by '
    '`pit_utils.write_results`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_BINS_ARG_NAME, type=int, required=True,
    help=NUM_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(prediction_file_name, num_bins, output_file_name):
    """Creates PIT (prob integ transform) histogram for each target variable.

    This is effectively the main method.

    :param prediction_file_name: See documentation at top of file.
    :param num_bins: Same.
    :param output_file_name: Same.
    """

    result_table_xarray = pit_utils.get_histogram_all_vars(
        prediction_file_name=prediction_file_name, num_bins=num_bins
    )
    print(SEPARATOR_STRING)

    rtx = result_table_xarray
    scalar_target_names = rtx.coords[pit_utils.SCALAR_FIELD_DIM].values
    wavelengths_microns = (
        METRES_TO_MICRONS * rtx.coords[pit_utils.WAVELENGTH_DIM].values
    )

    for t in range(len(scalar_target_names)):
        for w in range(len(wavelengths_microns)):
            print((
                'Variable = {0:s} at {1:.2f} microns ... '
                'PITD = {2:f} ... low-PIT bias = {3:f} ... '
                'medium-PIT bias = {4:f} ... high-PIT bias = {5:f} ... '
                'extreme-PIT frequency = {6:f}'
            ).format(
                scalar_target_names[t],
                wavelengths_microns[w],
                rtx[pit_utils.SCALAR_PITD_KEY].values[t, w],
                rtx[pit_utils.SCALAR_LOW_BIN_BIAS_KEY].values[t, w],
                rtx[pit_utils.SCALAR_MIDDLE_BIN_BIAS_KEY].values[t, w],
                rtx[pit_utils.SCALAR_HIGH_BIN_BIAS_KEY].values[t, w],
                rtx[pit_utils.SCALAR_EXTREME_PIT_FREQ_KEY].values[t, w]
            ))

        print(SEPARATOR_STRING)

    vector_target_names = rtx.coords[pit_utils.VECTOR_FIELD_DIM].values
    heights_m_agl = rtx.coords[pit_utils.HEIGHT_DIM].values

    for t in range(len(vector_target_names)):
        for w in range(len(wavelengths_microns)):
            print((
                'Variable = {0:s} at {1:.2f} microns ... '
                'PITD = {2:f} ... low-PIT bias = {3:f} ... '
                'medium-PIT bias = {4:f} ... high-PIT bias = {5:f} ... '
                'extreme-PIT frequency = {6:f}'
            ).format(
                vector_target_names[t],
                wavelengths_microns[w],
                rtx[pit_utils.VECTOR_FLAT_PITD_KEY].values[t, w],
                rtx[pit_utils.VECTOR_FLAT_LOW_BIN_BIAS_KEY].values[t, w],
                rtx[pit_utils.VECTOR_FLAT_MIDDLE_BIN_BIAS_KEY].values[t, w],
                rtx[pit_utils.VECTOR_FLAT_HIGH_BIN_BIAS_KEY].values[t, w],
                rtx[pit_utils.VECTOR_FLAT_EXTREME_PIT_FREQ_KEY].values[t, w]
            ))

        print(SEPARATOR_STRING)

    for t in range(len(vector_target_names)):
        for w in range(len(wavelengths_microns)):
            for h in range(len(heights_m_agl)):
                print((
                    'Variable = {0:s} at {1:.2f} microns and {2:d} m AGL ... '
                    'PITD = {3:f} ... low-PIT bias = {4:f} ... '
                    'medium-PIT bias = {5:f} ... high-PIT bias = {6:f} ... '
                    'extreme-PIT frequency = {7:f}'
                ).format(
                    vector_target_names[t],
                    wavelengths_microns[w],
                    int(numpy.round(heights_m_agl[h])),
                    rtx[pit_utils.VECTOR_PITD_KEY].values[t, h, w],
                    rtx[pit_utils.VECTOR_LOW_BIN_BIAS_KEY].values[t, h, w],
                    rtx[pit_utils.VECTOR_MIDDLE_BIN_BIAS_KEY].values[t, h, w],
                    rtx[pit_utils.VECTOR_HIGH_BIN_BIAS_KEY].values[t, h, w],
                    rtx[pit_utils.VECTOR_EXTREME_PIT_FREQ_KEY].values[t, h, w]
                ))

            print(SEPARATOR_STRING)

    try:
        aux_target_field_names = (
            rtx.coords[pit_utils.AUX_TARGET_FIELD_DIM].values
        )
        aux_predicted_field_names = (
            rtx.coords[pit_utils.AUX_PREDICTED_FIELD_DIM].values
        )
    except:
        aux_target_field_names = []
        aux_predicted_field_names = []

    for t in range(len(aux_target_field_names)):
        for w in range(len(wavelengths_microns)):
            print((
                'Target variable = {0:s} at {1:.2f} microns ... '
                'predicted variable = {2:s} at {1:.2f} microns ... '
                'PITD = {3:f} ... low-PIT bias = {4:f} ... '
                'medium-PIT bias = {5:f} ... high-PIT bias = {6:f} ... '
                'extreme-PIT frequency = {7:f}'
            ).format(
                aux_target_field_names[t],
                wavelengths_microns[w],
                aux_predicted_field_names[t],
                rtx[pit_utils.AUX_PITD_KEY].values[t, w],
                rtx[pit_utils.AUX_LOW_BIN_BIAS_KEY].values[t, w],
                rtx[pit_utils.AUX_MIDDLE_BIN_BIAS_KEY].values[t, w],
                rtx[pit_utils.AUX_HIGH_BIN_BIAS_KEY].values[t, w],
                rtx[pit_utils.AUX_EXTREME_PIT_FREQ_KEY].values[t, w]
            ))

    print(SEPARATOR_STRING)

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    pit_utils.write_results(
        result_table_xarray=result_table_xarray,
        netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        num_bins=getattr(INPUT_ARG_OBJECT, NUM_BINS_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
