"""Converts all imports of ml4rt code from absolute to relative."""

import os
import glob


def _run():
    """Converts all imports of ml4rt code from absolute to relative.

    This is effectively the main method.
    """

    script_dir_name = os.getcwd()
    package_dir_name = '/'.join(script_dir_name.split('/')[:-1])

    module_file_pattern = '{0:s}/*.py'.format(package_dir_name)
    module_file_names = glob.glob(module_file_pattern)

    for this_file_name in module_file_names:
        this_command_string = (
            "sed -i 's/\\# THIS_DIRECTORY_NAME/THIS_DIRECTORY_NAME/g' {0:s}"
        ).format(this_file_name)
        os.system(this_command_string)

        this_command_string = (
            "sed -i 's/\\#     os.path.join/    os.path.join/g' {0:s}"
        ).format(this_file_name)
        os.system(this_command_string)

        this_command_string = (
            "sed -i 's/\\# ))/))/g' {0:s}"
        ).format(this_file_name)
        os.system(this_command_string)

        this_command_string = (
            "sed -i 's/\\# sys.path.append/sys.path.append/g' {0:s}"
        ).format(this_file_name)
        os.system(this_command_string)

        this_command_string = (
            "sed -i 's/from ml4rt import/import/g' {0:s}"
        ).format(this_file_name)
        os.system(this_command_string)


if __name__ == '__main__':
    _run()
