"""Makes animations with case studies (for LEAP-RT seminar in Dec 2020)."""

import os
import sys
import glob
import argparse

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import imagemagick_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
CONCAT_FIGURE_SIZE_PX = int(5e6)

INPUT_DIR_ARG_NAME = 'input_dir_name'
NUM_FRAMES_ARG_NAME = 'num_frames_per_gif'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of top-level input directory, containing individual plots of extreme '
    'examples.'
)
NUM_FRAMES_HELP_STRING = 'Number of frames per GIF.'
OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory.  GIFs will be saved here, one for each'
    ' set of extreme examples.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_FRAMES_ARG_NAME, type=int, required=False, default=10,
    help=NUM_FRAMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _make_animation_one_set(input_predictor_dir_name, input_prediction_dir_name,
                            num_frames_per_gif, output_dir_name):
    """Makes animation for one set of extreme cases.

    :param input_predictor_dir_name: Name of directory with predictor plots.
    :param input_prediction_dir_name: Name of directory with prediction plots.
    :param num_frames_per_gif: Number of frames in GIF.
    :param output_dir_name: Name of output directory.  GIF will be saved here.
    """

    predictor_file_pattern = (
        '{0:s}/*predictor-set-0.jpg'.format(input_predictor_dir_name)
    )

    input_predictor_file_names = glob.glob(predictor_file_pattern)
    input_predictor_file_names.sort()
    input_predictor_file_names = input_predictor_file_names[:num_frames_per_gif]

    pathless_predictor_file_names = [
        os.path.split(f)[-1] for f in input_predictor_file_names
    ]
    output_predictor_file_names = [
        '{0:s}/{1:s}'.format(output_dir_name, f)
        for f in pathless_predictor_file_names
    ]

    pathless_prediction_file_names = [
        f.replace('_predictor-set-0.jpg', '_shortwave-heating-rate-k-day01.jpg')
        for f in pathless_predictor_file_names
    ]
    input_prediction_file_names = [
        '{0:s}/{1:s}'.format(input_prediction_dir_name, f)
        for f in pathless_prediction_file_names
    ]
    output_prediction_file_names = [
        '{0:s}/{1:s}'.format(output_dir_name, f)
        for f in pathless_prediction_file_names
    ]

    pathless_concat_file_names = [
        f.replace('_predictor-set-0.jpg', '_concat.jpg')
        for f in pathless_predictor_file_names
    ]
    concat_file_names = [
        '{0:s}/{1:s}'.format(output_dir_name, f)
        for f in pathless_concat_file_names
    ]

    for i in range(num_frames_per_gif):
        imagemagick_utils.trim_whitespace(
            input_file_name=input_predictor_file_names[i],
            output_file_name=output_predictor_file_names[i]
        )
        imagemagick_utils.resize_image(
            input_file_name=output_predictor_file_names[i],
            output_file_name=output_predictor_file_names[i],
            output_size_pixels=int(5e6)
        )

        imagemagick_utils.trim_whitespace(
            input_file_name=input_prediction_file_names[i],
            output_file_name=output_prediction_file_names[i]
        )
        imagemagick_utils.resize_image(
            input_file_name=output_prediction_file_names[i],
            output_file_name=output_prediction_file_names[i],
            output_size_pixels=int(5e6)
        )

        print('Concatenating panels to: "{0:s}"...'.format(
            concat_file_names[i]
        ))
        these_input_file_names = [
            output_predictor_file_names[i], output_prediction_file_names[i]
        ]

        imagemagick_utils.concatenate_images(
            input_file_names=these_input_file_names,
            output_file_name=concat_file_names[i],
            num_panel_rows=1, num_panel_columns=2
        )
        imagemagick_utils.trim_whitespace(
            input_file_name=concat_file_names[i],
            output_file_name=concat_file_names[i]
        )
        # imagemagick_utils.resize_image(
        #     input_file_name=concat_file_names[i],
        #     output_file_name=concat_file_names[i],
        #     output_size_pixels=CONCAT_FIGURE_SIZE_PX
        # )

    gif_file_name = '{0:s}/animation.gif'.format(output_dir_name)
    print('Making animation: "{0:s}"...'.format(gif_file_name))

    imagemagick_utils.create_gif(
        input_file_names=concat_file_names, output_file_name=gif_file_name,
        num_seconds_per_frame=1., resize_factor=0.5
    )


def _run(top_input_dir_name, num_frames_per_gif, top_output_dir_name):
    """Makes animations with case studies (for LEAP-RT seminar in Dec 2020).

    This is effectively the main method.

    :param top_input_dir_name: See documentation at top of file.
    :param num_frames_per_gif: Same.
    :param top_output_dir_name: Same.
    """

    error_checking.assert_is_geq(num_frames_per_gif, 2)

    this_predictor_dir_name = (
        '{0:s}/averaged=0_scaled=0/predictions_high-bias/predictor_plots'
    ).format(top_input_dir_name)

    this_prediction_dir_name = (
        '{0:s}/averaged=0_scaled=0/predictions_high-bias/prediction_plots'
    ).format(top_input_dir_name)

    this_output_dir_name = (
        '{0:s}/averaged=0_scaled=0/high_bias'.format(top_output_dir_name)
    )

    _make_animation_one_set(
        input_predictor_dir_name=this_predictor_dir_name,
        input_prediction_dir_name=this_prediction_dir_name,
        num_frames_per_gif=num_frames_per_gif,
        output_dir_name=this_output_dir_name
    )
    print(SEPARATOR_STRING)

    this_predictor_dir_name = (
        '{0:s}/averaged=0_scaled=0/predictions_low-absolute-error/'
        'predictor_plots'
    ).format(top_input_dir_name)

    this_prediction_dir_name = (
        '{0:s}/averaged=0_scaled=0/predictions_low-absolute-error/'
        'prediction_plots'
    ).format(top_input_dir_name)

    this_output_dir_name = (
        '{0:s}/averaged=0_scaled=0/low_error'.format(top_output_dir_name)
    )

    _make_animation_one_set(
        input_predictor_dir_name=this_predictor_dir_name,
        input_prediction_dir_name=this_prediction_dir_name,
        num_frames_per_gif=num_frames_per_gif,
        output_dir_name=this_output_dir_name
    )
    print(SEPARATOR_STRING)

    this_predictor_dir_name = (
        '{0:s}/averaged=0_scaled=0/predictions_large-heating-rate/'
        'predictor_plots'
    ).format(top_input_dir_name)

    this_prediction_dir_name = (
        '{0:s}/averaged=0_scaled=0/predictions_large-heating-rate/'
        'prediction_plots'
    ).format(top_input_dir_name)

    this_output_dir_name = (
        '{0:s}/averaged=0_scaled=0/large_heating_rate'
    ).format(top_output_dir_name)

    _make_animation_one_set(
        input_predictor_dir_name=this_predictor_dir_name,
        input_prediction_dir_name=this_prediction_dir_name,
        num_frames_per_gif=num_frames_per_gif,
        output_dir_name=this_output_dir_name
    )
    print(SEPARATOR_STRING)

    this_predictor_dir_name = (
        '{0:s}/averaged=1/predictions_large-heating-rate/predictor_plots'
    ).format(top_input_dir_name)

    this_prediction_dir_name = (
        '{0:s}/averaged=1/predictions_large-heating-rate/prediction_plots'
    ).format(top_input_dir_name)

    this_output_dir_name = (
        '{0:s}/averaged=1/large_heating_rate'.format(top_output_dir_name)
    )

    _make_animation_one_set(
        input_predictor_dir_name=this_predictor_dir_name,
        input_prediction_dir_name=this_prediction_dir_name,
        num_frames_per_gif=num_frames_per_gif,
        output_dir_name=this_output_dir_name
    )
    print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        num_frames_per_gif=getattr(INPUT_ARG_OBJECT, NUM_FRAMES_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
