import argparse
import os
from datetime import datetime
from logger_config import setup_logger
from run_configurations import OUTPUT_FOLDER

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Telemac2D simulations.")
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start file index (default: 0 for all files)",
    )
    parser.add_argument(
        "--end", 
        type=int, 
        default=0, 
        help="End file index (default: 0 for all files)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_FOLDER,
        help=f"Directory for the simulations log files (default: {OUTPUT_FOLDER})",
    )

    return parser.parse_args()

def setup_logging():
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    process_id = os.getpid()

    log_name = f"{script_name}_{timestamp}_{process_id}"
    logger = setup_logger(log_name)
    
    return logger

def determine_file_range(args, steering_folder):
    if args.start == 0 and args.end == 0:
        cas_files = [f for f in os.listdir(steering_folder) if f.endswith(".cas")]
        return 0, len(cas_files)
    else:
        return args.start, args.end