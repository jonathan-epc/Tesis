import os
from modules.file_utils import move_file
def add_continuation_lines(steering_file, i):
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

def update_duration(steering_file):
    with open(steering_file, 'r') as f:
        content = f.readlines()
    
    for idx, line in enumerate(content):
        if line.strip().startswith("DURATION ="):
            current_duration = int(line.split("=")[1].strip())
            new_duration = current_duration + 60
            content[idx] = f"DURATION = {new_duration}\n"
            break
    
    with open(steering_file, 'w') as f:
        f.writelines(content)

def prepare_steering_file(src_file, dst_file, result_file, i, continue_simulation):
    move_file(src_file, dst_file)
    if os.path.exists(result_file):
        if continue_simulation:
            add_continuation_lines(dst_file, i)
        else:
            update_duration(dst_file)
            