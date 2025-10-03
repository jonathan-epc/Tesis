import shutil
from pathlib import Path


def organize_publication_plots():
    """
    Finds specific plots from a nested directory structure, renames them,
    and copies them to a new 'publication_plots' directory.
    """
    # --- 1. DEFINE PATHS ---

    # Get the directory where the script is running.
    # Assumes the script is in '.../ML/' and 'plots' is a subfolder.
    # If not, change 'source_root' to the full path of your plots folder.
    # e.g., source_root = Path("C:/Users/Name/Documents/Tesis/ML/plots")
    base_path = Path.cwd()
    source_root = base_path
    target_root = base_path / "publication_plots"

    # Define the language folders to process
    languages = ["en", "es"]

    # Define the prefixes for the "example" plots
    example_prefixes = [f"{v}_comparison_case" for v in ["U", "B", "U_", "B_", "H"]]

    print(f"Source folder: {source_root}")
    print(f"Target folder: {target_root}")
    print("-" * 30)

    # --- 2. CREATE TARGET DIRECTORY STRUCTURE ---

    try:
        target_root.mkdir(exist_ok=True)
        for lang in languages:
            (target_root / lang).mkdir(exist_ok=True)
    except OSError as e:
        print(f"Error creating directories: {e}")
        return

    # --- 3. PROCESS FILES ---

    for lang in languages:
        source_lang_dir = source_root / lang
        target_lang_dir = target_root / lang

        if not source_lang_dir.is_dir():
            print(f"Warning: Source directory not found, skipping: {source_lang_dir}")
            continue

        print(f"\nProcessing language: '{lang}'")

        # Iterate through each experiment subfolder (e.g., 'dab_barsa')
        for exp_dir in source_lang_dir.iterdir():
            if not exp_dir.is_dir():
                continue  # Skip any non-directory items

            folder_name = exp_dir.name
            print(f"  Scanning subfolder: {folder_name}")

            # --- Rule 1: Find and copy the "overview" plot ---
            # Find files starting with "predictions" (e.g., predictions.pdf, predictions.png)
            for file_path in exp_dir.glob("predictions*"):
                if file_path.is_file():
                    # New name: folder_name + "_overview" + extension
                    new_name = f"{folder_name}_overview{file_path.suffix}"
                    target_path = target_lang_dir / new_name
                    shutil.copy2(file_path, target_path)
                    print(f"    -> Copied and renamed to: {new_name}")

            # --- Rule 2: Find and copy ONE "example" plot ---
            # We only want one set of example plots per folder.
            found_example = False
            for prefix in example_prefixes:
                # Use glob to find files starting with the current prefix
                # This will find both .pdf and .png if they exist
                matching_files = list(exp_dir.glob(f"{prefix}*"))

                if matching_files:
                    # We found a match! Copy all versions (pdf, png)
                    for file_path in matching_files:
                        if file_path.is_file():
                            # New name: folder_name + "_example" + extension
                            new_name = f"{folder_name}_example{file_path.suffix}"
                            target_path = target_lang_dir / new_name
                            shutil.copy2(file_path, target_path)
                            print(f"    -> Copied and renamed to: {new_name}")

                    found_example = True
                    break  # Stop searching for other example prefixes in this folder

            if not found_example:
                print(f"    - Warning: No 'example' plot found in {folder_name}")

    print("\n" + "-" * 30)
    print("Script finished successfully!")
    print(f"All selected plots have been copied to: {target_root}")


if __name__ == "__main__":
    organize_publication_plots()
