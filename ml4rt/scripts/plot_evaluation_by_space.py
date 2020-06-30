"""Plots evaluation scores by spatial region."""

import argparse
import numpy
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4rt.utils import evaluation

# TODO(thunderhoser): Just need method to plot a single score, which needs to
# handle weird grid bullshit.

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_DIR_ARG_NAME = 'input_evaluation_dir_name'
NUM_ROWS_ARG_NAME = 'num_grid_rows'
NUM_COLUMNS_ARG_NAME = 'num_grid_columns'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of directory with evaluation files (one for each grid cell).  Files '
    'will be found by `evaluation.find_file` and read by '
    '`evaluation.read_file`.'
)
NUM_ROWS_HELP_STRING = 'Number of rows in grid.'
NUM_COLUMNS_HELP_STRING = 'Number of columns in grid.'
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_ROWS_ARG_NAME, type=int, required=True,
    help=NUM_ROWS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_COLUMNS_ARG_NAME, type=int, required=True,
    help=NUM_COLUMNS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_scores_one_field(
        mae_matrix, rmse_matrix, bias_matrix, mae_skill_score_matrix,
        mse_skill_score_matrix, correlation_matrix, field_name, output_dir_name,
        height_m_agl=None):
    """Plots all evaluation scores for one field.

    M = number of rows in grid
    N = number of columns in grid

    :param mae_matrix: M-by-N numpy array of MAE (mean absolute error) values.
    :param rmse_matrix: Same but for RMSE (root mean squared error).
    :param bias_matrix: Same but for bias.
    :param mae_skill_score_matrix: Same but for MAE skill score.
    :param mse_skill_score_matrix: Same but for MSE skill score.
    :param correlation_matrix: Same but for correlation.
    :param field_name: Name of field for which scores are being plotted.
    :param output_dir_name: Name of output directory.  Figures will be saved
        here.
    :param height_m_agl: Height (metres above ground level).  If plotting for
        scalar field, leave this argument alone.
    """


def _run(evaluation_dir_name, num_grid_rows, num_grid_columns,
         output_dir_name):
    """Plots evaluation scores by spatial region.

    This is effectively the main method.

    :param evaluation_dir_name: See documentation at top of file.
    :param num_grid_rows: Same.
    :param num_grid_columns: Same.
    :param output_dir_name: Same.
    """

    error_checking.assert_is_geq(num_grid_rows, 2)
    error_checking.assert_is_geq(num_grid_columns, 2)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    eval_table_matrix_xarray = numpy.full(
        (num_grid_rows, num_grid_columns), '', dtype=object
    )

    for i in range(num_grid_rows):
        for j in range(num_grid_columns):
            this_file_name = evaluation.find_file(
                directory_name=evaluation_dir_name, grid_row=i, grid_column=j,
                raise_error_if_missing=True
            )

            print('Reading data from: "{0:s}"...'.format(this_file_name))
            eval_table_matrix_xarray[i, j] = evaluation.read_file(
                this_file_name
            )

        if i == num_grid_rows - 1:
            print(SEPARATOR_STRING)
        else:
            print('\n')

    evaluation_tables_xarray = numpy.reshape(
        eval_table_matrix_xarray, num_grid_rows * num_grid_columns
    )
    grid_dim_tuple = (num_grid_rows, num_grid_columns)

    scalar_field_names = (
        evaluation_tables_xarray[0].coords[evaluation.SCALAR_FIELD_DIM].values
    )
    scalar_mae_matrix = numpy.vstack([
        t[evaluation.SCALAR_MAE_KEY].values for t in evaluation_tables_xarray
    ])
    scalar_rmse_matrix = numpy.sqrt(numpy.vstack([
        t[evaluation.SCALAR_MSE_KEY].values for t in evaluation_tables_xarray
    ]))
    scalar_bias_matrix = numpy.vstack([
        t[evaluation.SCALAR_BIAS_KEY].values for t in evaluation_tables_xarray
    ])
    scalar_mae_skill_matrix = numpy.vstack([
        t[evaluation.SCALAR_MAE_SKILL_KEY].values
        for t in evaluation_tables_xarray
    ])
    scalar_mse_skill_matrix = numpy.vstack([
        t[evaluation.SCALAR_MSE_SKILL_KEY].values
        for t in evaluation_tables_xarray
    ])
    scalar_correlation_matrix = numpy.vstack([
        t[evaluation.SCALAR_CORRELATION_KEY].values
        for t in evaluation_tables_xarray
    ])

    for k in range(len(scalar_field_names)):
        _plot_scores_one_field(
            mae_matrix=numpy.reshape(scalar_mae_matrix[:, k], grid_dim_tuple),
            rmse_matrix=numpy.reshape(scalar_rmse_matrix[:, k], grid_dim_tuple),
            bias_matrix=numpy.reshape(scalar_bias_matrix[:, k], grid_dim_tuple),
            mae_skill_score_matrix=numpy.reshape(
                scalar_mae_skill_matrix[:, k], grid_dim_tuple
            ),
            mse_skill_score_matrix=numpy.reshape(
                scalar_mse_skill_matrix[:, k], grid_dim_tuple
            ),
            correlation_matrix=numpy.reshape(
                scalar_correlation_matrix[:, k], grid_dim_tuple
            ),
            field_name=scalar_field_names[k], output_dir_name=output_dir_name
        )

        if k == len(scalar_field_names) - 1:
            print(SEPARATOR_STRING)
        else:
            print('\n')

    try:
        aux_field_names = (
            evaluation_tables_xarray[0].coords[
                evaluation.AUX_TARGET_FIELD_DIM
            ].values
        )
        aux_mae_matrix = numpy.vstack([
            t[evaluation.AUX_MAE_KEY].values for t in evaluation_tables_xarray
        ])
        aux_rmse_matrix = numpy.sqrt(numpy.vstack([
            t[evaluation.AUX_MSE_KEY].values for t in evaluation_tables_xarray
        ]))
        aux_bias_matrix = numpy.vstack([
            t[evaluation.AUX_BIAS_KEY].values for t in evaluation_tables_xarray
        ])
        aux_mae_skill_matrix = numpy.vstack([
            t[evaluation.AUX_MAE_SKILL_KEY].values
            for t in evaluation_tables_xarray
        ])
        aux_mse_skill_matrix = numpy.vstack([
            t[evaluation.AUX_MSE_SKILL_KEY].values
            for t in evaluation_tables_xarray
        ])
        aux_correlation_matrix = numpy.vstack([
            t[evaluation.AUX_CORRELATION_KEY].values
            for t in evaluation_tables_xarray
        ])
    except KeyError:
        aux_field_names = []

    for k in range(len(aux_field_names)):
        _plot_scores_one_field(
            mae_matrix=numpy.reshape(aux_mae_matrix[:, k], grid_dim_tuple),
            rmse_matrix=numpy.reshape(aux_rmse_matrix[:, k], grid_dim_tuple),
            bias_matrix=numpy.reshape(aux_bias_matrix[:, k], grid_dim_tuple),
            mae_skill_score_matrix=numpy.reshape(
                aux_mae_skill_matrix[:, k], grid_dim_tuple
            ),
            mse_skill_score_matrix=numpy.reshape(
                aux_mse_skill_matrix[:, k], grid_dim_tuple
            ),
            correlation_matrix=numpy.reshape(
                aux_correlation_matrix[:, k], grid_dim_tuple
            ),
            field_name=aux_field_names[k], output_dir_name=output_dir_name
        )

        if k == len(aux_field_names) - 1:
            print(SEPARATOR_STRING)
        else:
            print('\n')

    vector_field_names = (
        evaluation_tables_xarray[0].coords[evaluation.VECTOR_FIELD_DIM].values
    )
    heights_m_agl = numpy.round(
        evaluation_tables_xarray[0].coords[evaluation.HEIGHT_DIM].values
    ).astype(int)

    vector_mae_matrix = numpy.stack([
        t[evaluation.VECTOR_MAE_KEY].values for t in evaluation_tables_xarray
    ], axis=0)
    vector_rmse_matrix = numpy.sqrt(numpy.stack([
        t[evaluation.VECTOR_MSE_KEY].values for t in evaluation_tables_xarray
    ], axis=0))
    vector_bias_matrix = numpy.stack([
        t[evaluation.VECTOR_BIAS_KEY].values for t in evaluation_tables_xarray
    ], axis=0)
    vector_mae_skill_matrix = numpy.stack([
        t[evaluation.VECTOR_MAE_SKILL_KEY].values
        for t in evaluation_tables_xarray
    ], axis=0)
    vector_mse_skill_matrix = numpy.stack([
        t[evaluation.VECTOR_MSE_SKILL_KEY].values
        for t in evaluation_tables_xarray
    ], axis=0)
    vector_correlation_matrix = numpy.stack([
        t[evaluation.VECTOR_CORRELATION_KEY].values
        for t in evaluation_tables_xarray
    ], axis=0)

    for k in range(len(vector_field_names)):
        for j in range(len(heights_m_agl)):
            _plot_scores_one_field(
                mae_matrix=numpy.reshape(
                    vector_mae_matrix[:, j, k], grid_dim_tuple
                ),
                rmse_matrix=numpy.reshape(
                    vector_rmse_matrix[:, j, k], grid_dim_tuple
                ),
                bias_matrix=numpy.reshape(
                    vector_bias_matrix[:, j, k], grid_dim_tuple
                ),
                mae_skill_score_matrix=numpy.reshape(
                    vector_mae_skill_matrix[:, j, k], grid_dim_tuple
                ),
                mse_skill_score_matrix=numpy.reshape(
                    vector_mse_skill_matrix[:, j, k], grid_dim_tuple
                ),
                correlation_matrix=numpy.reshape(
                    vector_correlation_matrix[:, j, k], grid_dim_tuple
                ),
                field_name=vector_field_names[k], height_m_agl=heights_m_agl[j],
                output_dir_name=output_dir_name
            )

        if k != len(vector_field_names) - 1:
            print('\n')


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        evaluation_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        num_grid_rows=getattr(INPUT_ARG_OBJECT, NUM_ROWS_ARG_NAME),
        num_grid_columns=getattr(INPUT_ARG_OBJECT, NUM_COLUMNS_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
