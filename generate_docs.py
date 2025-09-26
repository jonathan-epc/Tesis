import os
from pathlib import Path

# --- Configuration ---

# The root folder to analyze (current working directory)
ROOT_FOLDER: Path = Path.cwd()

# The single output file for the result
OUTPUT_FILENAME: str = "project_summary.txt"

# Extensions to include for content extraction
EXTENSIONS_TO_INCLUDE: set[str] = {".py", ".yaml", ".yml", ".config", ".env"}

# Extensions to exclude from the folder structure listing
EXTENSIONS_TO_EXCLUDE_STRUCTURE: set[str] = {
    ".pdf",
    ".png",
    ".log",
    ".csv",
    ".json",
    ".pth",
    ".gif",
}

# Relative paths to completely exclude from both structure and content.
# Using a set of Path objects for efficient lookups.
PATHS_TO_EXCLUDE: set[Path] = {
    Path(p)
    for p in [
        "ML/wandb",
        ".git",
        ".ipynb_checkpoints",
        "ML/runs",
        "ML/runs.old",
        "ML/plots",
        "neuraloperator",
        "Articulos",
        "telemac/.ipynb_checkpoints",
        "modules/__pycache__",
        "__pycache__",
        "ML/__pycache__",
        "modules/.ipynb_checkpoints",
        "ML/modules/__pycache__",
        "ML/modules/.ipynb_checkpoints",
        "telemac/logs/.ipynb_checkpoints",
        "telemac/modules/.ipynb_checkpoints",
        "ML/studies",
        "ML/.ipynb_checkpoints",
        ".ruff_cache/",
    ]
}

# --- Main Logic ---


def generate_project_summary(root_folder: Path, output_filename: str):
    """
    Walks the directory tree once to generate the folder structure and
    appends the content of specified files to a single output file.

    Args:
        root_folder: The starting directory for the analysis.
        output_filename: The name of the text file to create.
    """
    structure_lines: list[str] = []
    content_files: list[Path] = []

    def _is_path_excluded(path: Path) -> bool:
        """A nested helper to check if a path should be excluded."""
        try:
            relative_path = path.relative_to(root_folder)
        except ValueError:
            # This can happen if a symlink points outside the root. Exclude it.
            return True

        # Check against any part of the path (e.g., '.git', '__pycache__')
        if any(
            part in p.name for p in PATHS_TO_EXCLUDE for part in relative_path.parts
        ):
            return True

        # Check if the path itself or any of its parents are in the exclusion list
        if relative_path in PATHS_TO_EXCLUDE:
            return True
        return any(parent in PATHS_TO_EXCLUDE for parent in relative_path.parents)

        return False

    # --- Single pass over the directory tree ---
    for current_str_path, dirnames, filenames in os.walk(root_folder, topdown=True):
        current_path = Path(current_str_path)

        # Prune directories from traversal for efficiency
        dirnames[:] = [d for d in dirnames if not _is_path_excluded(current_path / d)]

        # Skip processing the contents of the current directory if it's excluded
        if _is_path_excluded(current_path):
            continue

        # --- Process current directory for the structure output ---
        relative_dir = current_path.relative_to(root_folder)
        level = len(relative_dir.parts)

        if str(relative_dir) == ".":
            # Root folder
            structure_lines.append(f"{root_folder.name}/\n")
            indent = "    "
        else:
            # Subfolders
            indent = "    " * level
            structure_lines.append(f"{indent}{current_path.name}/\n")

        # --- Process files in the current directory ---
        file_indent = indent + "    "
        for fname in sorted(filenames):
            file_path = current_path / fname

            # Add to structure list if its extension is not excluded
            if file_path.suffix not in EXTENSIONS_TO_EXCLUDE_STRUCTURE:
                structure_lines.append(f"{file_indent}{fname}\n")

            # Add to content list if its extension is included
            if file_path.suffix in EXTENSIONS_TO_INCLUDE:
                content_files.append(file_path)

    # --- Write all collected information to the output file ---
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            # 1. Write the complete folder structure
            f.write("### FOLDER STRUCTURE ###\n")
            f.writelines(structure_lines)

            # 2. Write the content of the collected files
            f.write("\n\n### FILE CONTENTS ###\n")
            for file_path in content_files:
                relative_path = file_path.relative_to(root_folder)
                # Use as_posix() for consistent '/' separators in the header
                f.write(f"\n--- Content of: {relative_path.as_posix()} ---\n")
                try:
                    with file_path.open(
                        "r", encoding="utf-8", errors="ignore"
                    ) as content_file:
                        f.write(content_file.read())
                    f.write("\n")
                except Exception as e:
                    f.write(f"Could not read file: {e}\n\n")

        print(f"Successfully created '{output_filename}'")
    except OSError as e:
        print(f"Error writing to file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main():
    """Main execution function."""
    generate_project_summary(ROOT_FOLDER, OUTPUT_FILENAME)


if __name__ == "__main__":
    main()
