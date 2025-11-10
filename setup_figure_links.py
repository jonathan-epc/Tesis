# Tesis/setup_figure_links.py

import os  # Import the os module
from pathlib import Path

# --- CONFIGURATION ---

# 1. Define the root of your project
# This path points to the 'Tesis/' directory
PROJECT_ROOT = Path(__file__).resolve().parent

# 2. Define the single source folder where plots are generated
# This path is inside your code repo: 'Tesis/publication_figures'
SOURCE_FIGURES_DIR = PROJECT_ROOT / "publication_figures"

# 3. NEW: Define the path to your separate paper repository.
# This assumes 'Tesis-Paper/' and 'Tesis/' are in the same parent folder.
# e.g., YourProjects/Tesis/ and YourProjects/Tesis-Paper/
try:
    PAPER_REPO_ROOT = PROJECT_ROOT.parent / "Tesis-Paper"
    # Check if the directory actually exists
    if not PAPER_REPO_ROOT.is_dir():
        raise FileNotFoundError
except FileNotFoundError:
    print(f"[ERROR] Paper repository not found at expected location: {PAPER_REPO_ROOT}")
    print(
        "Please make sure your 'Tesis-Paper' directory is next to your 'Tesis' directory."
    )
    # You can also hardcode the path if you prefer:
    # PAPER_REPO_ROOT = Path("/path/to/your/Tesis-Paper")
    exit()


# 4. Define all the LaTeX projects where you want to use these figures.
# The paths now point to the new, separate paper repository.
LATEX_PROJECTS = {
    PAPER_REPO_ROOT / "Articulos/article": "figures",
    PAPER_REPO_ROOT / "Articulos/sochid": "figures",
    PAPER_REPO_ROOT / "Articulos/thesis": "images",  # Thesis might use 'images'
}


# --- SCRIPT LOGIC (No changes needed below this line) ---
def create_symlinks():
    """
    Creates symbolic links from the central publication_figures directory
    to each specified LaTeX project folder.
    """
    print("--- Setting up Symbolic Links for LaTeX Figures ---")

    if not SOURCE_FIGURES_DIR.is_dir():
        print(f"\n[ERROR] Source directory not found: '{SOURCE_FIGURES_DIR}'")
        print("Please run the plot generation script first to create it.")
        return

    for project_path, link_name in LATEX_PROJECTS.items():
        print(
            f"\nProcessing LaTeX project: {os.path.relpath(project_path, PAPER_REPO_ROOT)}"
        )

        if not project_path.is_dir():
            print(f"  [WARNING] Project directory not found. Skipping: {project_path}")
            continue

        target_link = project_path / link_name

        if target_link.exists() or target_link.is_symlink():
            if (
                target_link.is_symlink()
                and target_link.resolve() == SOURCE_FIGURES_DIR.resolve()
            ):
                print(f"  [INFO] Correct link '{link_name}' already exists. Skipping.")
                continue
            else:
                print(
                    f"  [WARNING] A file or directory named '{link_name}' already exists and is NOT the correct symlink."
                )
                print("  Please remove it manually and re-run this script.")
                continue

        try:
            target_link.symlink_to(SOURCE_FIGURES_DIR, target_is_directory=True)
            print(
                f"  [SUCCESS] Created link: '{target_link}' -> '{SOURCE_FIGURES_DIR}'"
            )
        except OSError as e:
            print(f"  [ERROR] Failed to create symlink: {e}")
            print(
                "  On Windows, you may need to run this script as an Administrator or enable Developer Mode."
            )
        except Exception as e:
            print(f"  [ERROR] An unexpected error occurred: {e}")


if __name__ == "__main__":
    create_symlinks()
