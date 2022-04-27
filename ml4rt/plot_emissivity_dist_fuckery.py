"""USE ONCE AND DESTROY."""

import os
import sys
import numpy
import xarray
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import histograms
import file_system_utils
import gg_plotting_utils
import imagemagick_utils

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

FACE_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
EDGE_COLOUR = numpy.full(3, 0.)
EDGE_WIDTH = 1.5

FONT_SIZE = 36
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)


def _plot_histogram_one_target(
        target_values, target_name, num_bins, letter_label, output_dir_name):
    """Plots histogram for one target variable.

    :param target_values: 1-D numpy array of values.
    :param target_name: Name of target variable.
    :param num_bins: Number of bins in histogram.
    :param letter_label: Letter label (will be used to label panel).
    :param output_dir_name: Name of output directory.  Figure will be saved
        here.
    :return: output_file_name: Path to output file.
    """

    min_value = numpy.min(target_values)
    max_value = numpy.max(target_values)

    num_examples_by_bin = histograms.create_histogram(
        input_values=target_values, num_bins=num_bins,
        min_value=min_value, max_value=max_value
    )[1]
    frequency_by_bin = (
        num_examples_by_bin.astype(float) / numpy.sum(num_examples_by_bin)
    )

    for this_freq in frequency_by_bin:
        print(this_freq)

    bin_edges = numpy.linspace(min_value, max_value, num=num_bins + 1)
    bin_centers = numpy.array([
        numpy.mean(bin_edges[[k, k + 1]])
        for k in range(num_bins)
    ])

    x_tick_coords = 0.5 + numpy.linspace(
        0, num_bins - 1, num=num_bins, dtype=float
    )

    x_tick_labels = ['{0:.2f}'.format(c) for c in bin_centers]
    x_tick_labels = [
        x_tick_labels[k] if numpy.mod(k, 3) == 0 else ' '
        for k in range(num_bins)
    ]

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    axes_object.bar(
        x=x_tick_coords, height=frequency_by_bin, width=1.,
        color=FACE_COLOUR, edgecolor=EDGE_COLOUR, linewidth=EDGE_WIDTH
    )

    axes_object.set_xlim([
        x_tick_coords[0] - 0.5, x_tick_coords[-1] + 0.5
    ])
    axes_object.set_xticks(x_tick_coords)
    axes_object.set_xticklabels(x_tick_labels, rotation=90.)

    axes_object.set_ylabel('Frequency')
    axes_object.set_xlabel(target_name)
    gg_plotting_utils.label_axes(
        axes_object=axes_object, label_string='({0:s})'.format(letter_label)
    )

    output_file_name = '{0:s}/histogram_{1:s}.jpg'.format(
        output_dir_name, target_name.replace('_', '-').replace(' ', '-').lower()
    )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name, output_file_name=output_file_name
    )
    imagemagick_utils.resize_image(
        input_file_name=output_file_name, output_file_name=output_file_name,
        output_size_pixels=int(2.5e6)
    )

    return output_file_name


EMISSIVITY_FILE_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4rt_project/gfs_data/'
    'gfs_emissivities_20200201-20200207.nc'
)
OUTPUT_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4rt_project/gfs_data/'
    'gfs_emissivities_20200201-20200207'
)

file_system_utils.mkdir_recursive_if_necessary(directory_name=OUTPUT_DIR_NAME)
emissivity_table_xarray = xarray.open_dataset(EMISSIVITY_FILE_NAME)

_plot_histogram_one_target(
    target_values=numpy.ravel(
        emissivity_table_xarray['full_emissivities'].values
    ),
    target_name='Full emissivity', num_bins=100, letter_label='a',
    output_dir_name=OUTPUT_DIR_NAME
)

_plot_histogram_one_target(
    target_values=numpy.ravel(
        emissivity_table_xarray['approx_emissivities'].values
    ),
    target_name='Approx emissivity', num_bins=100, letter_label='b',
    output_dir_name=OUTPUT_DIR_NAME
)
