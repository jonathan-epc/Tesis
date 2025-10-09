# Tesis/setup_figure_links.py

from pathlib import Path

# --- CONFIGURATION ---

# 1. Define the root of your project
PROJECT_ROOT = Path(__file__).resolve().parent

# 2. Define the single source folder where plots are generated
SOURCE_FIGURES_DIR = PROJECT_ROOT / "publication_figures"

# 3. Define all the LaTeX projects where you want to use these figures
#    This is a dictionary mapping the project path to the desired name
#    of the symlink inside that project.
LATEX_PROJECTS = {
    PROJECT_ROOT / "docs/Articulos/article": "figures",
    PROJECT_ROOT / "docs/Articulos/sochid": "figures",
    PROJECT_ROOT / "docs/Articulos/thesis": "images",  # Thesis might use 'images'
}

# --- SCRIPT LOGIC ---


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
        print(f"\nProcessing LaTeX project: {project_path.name}")

        if not project_path.is_dir():
            print(f"  [WARNING] Project directory not found. Skipping: {project_path}")
            continue

        target_link = project_path / link_name

        # Check if a file/directory/link already exists at the target
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
                # To be safe, we don't automatically delete user files.
                # If you are confident, you can uncomment the next lines:
                # if target_link.is_dir():
                #     shutil.rmtree(target_link)
                # else:
                #     target_link.unlink()
                # print(f"  [ACTION] Removed existing '{link_name}'.")
                continue  # Skip to next project after warning

        try:
            # On Windows, target_is_directory=True is needed for directory links
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
