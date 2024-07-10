import shutil
import os
from loguru import logger
def move_file(src, dst):
    try:
        shutil.move(src, dst)
    except Exception as e:
        logger.error(f"Error moving file {src} to {dst}: {e}")
        raise

def setup_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)