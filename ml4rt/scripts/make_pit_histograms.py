"""Creates PIT (prob integ transform) histogram for each target variable."""

import argparse
import numpy
from ml4rt.utils import uq_evaluation

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

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
    '`uq_evaluation.write_pit_histograms`.'
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

    result_table_xarray = uq_evaluation.get_pit_histogram_all_vars(
        prediction_file_name=prediction_file_name, num_bins=num_bins
    )
    print(SEPARATOR_STRING)

    t = result_table_xarray
    scalar_target_names = t.coords[uq_evaluation.SCALAR_FIELD_DIM].values

    for k in range(len(scalar_target_names)):
        this_skill_score = (
            (1. - t[uq_evaluation.SCALAR_PITD_KEY].values[k]) /
            (1. - t[uq_evaluation.SCALAR_PERFECT_PITD_KEY].values[k])
        )

        print((
            'Variable = {0:s} ... PITD = {1:f} ... perfect PITD = {2:f} ... '
            'PITD "skill" score = {3:f}'
        ).format(
            scalar_target_names[k],
            t[uq_evaluation.SCALAR_PITD_KEY].values[k],
            t[uq_evaluation.SCALAR_PERFECT_PITD_KEY].values[k],
            this_skill_score
        ))

    print(SEPARATOR_STRING)
    vector_target_names = t.coords[uq_evaluation.VECTOR_FIELD_DIM].values
    heights_m_agl = t.coords[uq_evaluation.HEIGHT_DIM].values

    for k in range(len(vector_target_names)):
        for j in range(len(heights_m_agl)):
            this_skill_score = (
                (1. - t[uq_evaluation.VECTOR_PITD_KEY].values[k, j]) /
                (1. - t[uq_evaluation.VECTOR_PERFECT_PITD_KEY].values[k, j])
            )

            print((
                'Variable = {0:s} at {1:d} m AGL ... PITD = {2:f} ... '
                'perfect PITD = {3:f} ... PITD "skill" score = {4:f}'
            ).format(
                vector_target_names[k], int(numpy.round(heights_m_agl[j])),
                t[uq_evaluation.VECTOR_PITD_KEY].values[k, j],
                t[uq_evaluation.VECTOR_PERFECT_PITD_KEY].values[k, j],
                this_skill_score
            ))

        print(SEPARATOR_STRING)

    try:
        aux_target_field_names = (
            t.coords[uq_evaluation.AUX_TARGET_FIELD_DIM].values
        )
        aux_predicted_field_names = (
            t.coords[uq_evaluation.AUX_PREDICTED_FIELD_DIM].values
        )
    except:
        aux_target_field_names = []
        aux_predicted_field_names = []

    for k in range(len(aux_target_field_names)):
        this_skill_score = (
            (1. - t[uq_evaluation.AUX_PITD_KEY].values[k]) /
            (1. - t[uq_evaluation.AUX_PERFECT_PITD_KEY].values[k])
        )

        print((
            'Target variable = {0:s} ... predicted variable = {1:s} ... '
            'PITD = {2:f} ... perfect PITD = {3:f} ... '
            'PITD "skill" score = {4:f}'
        ).format(
            aux_target_field_names[k], aux_predicted_field_names[k],
            t[uq_evaluation.AUX_PITD_KEY].values[k],
            t[uq_evaluation.AUX_PERFECT_PITD_KEY].values[k],
            this_skill_score
        ))

    print(SEPARATOR_STRING)

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    uq_evaluation.write_pit_histograms(
        pit_histogram_table_xarray=result_table_xarray,
        netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        num_bins=getattr(INPUT_ARG_OBJECT, NUM_BINS_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
