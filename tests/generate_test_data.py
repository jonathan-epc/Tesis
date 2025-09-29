# tests/generate_test_data.py

from pathlib import Path

import h5py
import numpy as np


def create_fake_test_data():
    """
    Creates a small, fake HDF5 dataset for testing purposes.
    This file is small enough to be committed to Git.
    """
    output_dir = Path(__file__).resolve().parent.parent / "data"
    output_dir.mkdir(exist_ok=True)
    file_path = output_dir / "test_data.hdf5"

    print(f"Generating fake test data at: {file_path}")

    # Define dataset parameters
    num_samples = 10
    height, width = 11, 401

    # Define variable names for fields and scalars
    field_vars = ["B", "H", "U", "V"]
    scalar_vars = ["H0", "Q0", "n", "nut", "SLOPE"]

    with h5py.File(file_path, "w") as f:
        # Create a dummy statistics group (can be empty for tests)
        stats_group = f.create_group("statistics")
        for var in field_vars + scalar_vars:
            stats_group.attrs[f"{var}_mean"] = 0.5
            stats_group.attrs[f"{var}_variance"] = 1.0
            stats_group.attrs[f"{var}_min"] = 0.0

        # Create a few simulation samples
        for i in range(num_samples):
            sim_group = f.create_group(f"simulation_{i}")

            # Add scalar attributes
            for scalar in scalar_vars:
                sim_group.attrs[scalar] = np.random.rand()

            # Add field datasets
            for field in field_vars:
                sim_group.create_dataset(field, data=np.random.rand(height, width))

    print("Fake test data generated successfully.")


if __name__ == "__main__":
    create_fake_test_data()
