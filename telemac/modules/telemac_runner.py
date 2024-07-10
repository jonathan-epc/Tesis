import os
import platform
import subprocess
from loguru import logger

def run_telemac2d(filename, output_dir):
    command = ["telemac2d.py"] if platform.system() == "Linux" else ["python", "-m", "telemac2d"]
    output_file = os.path.join(output_dir, f"{filename}.txt")
    
    try:
        with open(output_file, "w") as output_fh:
            subprocess.run(
                command + ["--ncsize=8", filename],
                check=True,
                stdout=output_fh,
                stderr=subprocess.STDOUT,
            )
        logger.info(f"Completed Telemac2D simulation for {filename}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running Telemac2D simulation for {filename}: {e}")
        raise