import os
from modules.file_utils import move_file
from loguru import logger

def add_continuation_lines(steering_file, i):
    """
    Adds continuation lines to the steering file.

    Parameters
    ----------
    steering_file : str
        The path to the steering file.
    i : int
        The index of the previous computation file.

    Examples
    --------
    >>> add_continuation_lines('steering.txt', 1)
    """
    with open(steering_file, 'r') as f:
        content = f.readlines()

    continuation_lines = [
        "\nCOMPUTATION CONTINUED = YES",
        f"\nPREVIOUS COMPUTATION FILE = 'results/results_{i}.slf'\n"
    ]

    # Add lines after the last non-empty line
    for idx in range(len(content) - 1, -1, -1):
        if content[idx].strip():
            content.insert(idx + 1, '\n'.join(continuation_lines))
            break

    with open(steering_file, 'w') as f:
        f.writelines(content)
    logger.info(f"Continuation lines added to file {steering_file}")

def update_duration(steering_file):
    """
    Updates the duration in the steering file by adding 60 units.

    Parameters
    ----------
    steering_file : str
        The path to the steering file.

    Examples
    --------
    >>> update_duration('steering.txt')
    """
    with open(steering_file, 'r') as f:
        content = f.readlines()

    for idx, line in enumerate(content):
        if line.strip().startswith("DURATION"):
            current_duration = int(line.split("=")[1].strip())
            new_duration = current_duration + 60
            content[idx] = f"DURATION = {new_duration}\n"
            break

    with open(steering_file, 'w') as f:
        f.writelines(content)
    logger.info(f"Duration updated to {new_duration} s in file {steering_file}")

def prepare_steering_file(src_file, dst_file, result_file, i, continue_simulation):
    """
    Prepares the steering file for the next simulation step.

    Parameters
    ----------
    src_file : str
        The path to the source steering file.
    dst_file : str
        The path to the destination steering file.
    result_file : str
        The path to the result file.
    i : int
        The index of the previous computation file.
    continue_simulation : bool
        Whether to continue the simulation from the previous result.

    Examples
    --------
    >>> prepare_steering_file('src.txt', 'dst.txt', 'result.txt', 1, True)
    """
    if os.path.exists(result_file):
        if continue_simulation:
            add_continuation_lines(src_file, i)
        else:
            update_duration(src_file)
    move_file(src_file, dst_file)
