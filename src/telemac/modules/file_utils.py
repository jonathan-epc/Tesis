import os
import shutil

from loguru import logger


def move_file(src, dst):
    """
    Moves a file from the source path to the destination path.

    Parameters
    ----------
    src : str
        The path to the source file.
    dst : str
        The path to the destination file.

    Raises
    ------
    Exception
        If an error occurs while moving the file.

    Examples
    --------
    >>> move_file('source.txt', 'destination.txt')
    """
    try:
        shutil.move(src, dst)
    except Exception as e:
        logger.error(f"Error moving file {src} to {dst}: {e}")
        raise


def setup_output_dir(output_dir):
    """
    Sets up the output directory by creating it if it does not exist.

    Parameters
    ----------
    output_dir : str
        The path to the output directory.

    Examples
    --------
    >>> setup_output_dir('output_directory')
    """
    os.makedirs(output_dir, exist_ok=True)
