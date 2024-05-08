import glob
import os
import subprocess
from tqdm import tqdm


def run_on_all_cas_files(linux = False):
    cas_files = glob.glob("steering*.cas")

    for filename in tqdm(cas_files, total = len(cas_files), desc="Running"):
        run_telemac2d(filename, linux)

    # Shutdown the computer
    # os.system("shutdown /s /t 1")


def run_telemac2d(filename, linux):
    if linux:
        command = f"telemac2d.py --ncsize=4 {filename} >> outputs/{filename}.txt"
    else:python -m telemac2d steering_0.cas
        command = f"python -m telemac2d --ncsize=4 {filename} >> outputs/{filename}.txt"
    subprocess.run(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True
    )


# run_telemac2d("steering_0.cas", linux = False)

run_on_all_cas_files(linux = False)
