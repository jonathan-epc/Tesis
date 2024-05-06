# +
import glob
import os
import subprocess

from tqdm import tqdm


# -

def run_on_all_cas_files():
    cas_files = glob.glob("steering*.cas")
    progress_bar = tqdm(total=len(cas_files), desc="Running")

    for filename in cas_files:
        run_telemac2d(filename)
        progress_bar.update(1)  # Update progress bar

    progress_bar.close()
    # Shutdown the computer
    # os.system("shutdown /s /t 1")


def run_telemac2d(filename):
    command = f"python -m telemac2d {filename}"
    subprocess.run(command, shell=True)


def run_telemac2d(filename):
    command = f"telemac2d.py --ncsize=4 {filename} >> outputs/{filename}.txt"
    subprocess.run(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True
    )


# +
# run_telemac2d("steering_0.cas")
# -

run_on_all_cas_files()
