# generate_docs.py (Updated for Refactored Project)
import os
from pathlib import Path

# --- Configuration ---
ROOT_FOLDER: Path = Path.cwd()
OUTPUT_FILENAME: str = "project_summary.txt"

# Extensions to include for content extraction
EXTENSIONS_TO_INCLUDE: set[str] = {".py", ".yaml", ".yml", ".toml", ".gitattributes"}

# Directories/files to completely exclude from both structure and content
# Updated for the refactored project structure.
PATHS_TO_EXCLUDE: set[str] = {
    ".git",
    ".github",  # Exclude CI configuration from content
    ".ipynb_checkpoints",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "blender",
    "studies",  # Exclude Optuna DBs
    "savepoints",  # Exclude saved models
    "logs",
    "plots",
    "wandb",
    # Exclude old/temporary/generated files if any
    "wandb.xlsx",
    "project_summary.txt",
    "generate_docs.py",
    "readme2.md",
    "runs",
    "metrics",
    "runs.old",
    "geometry",
    "steering",
}

# --- Main Logic ---


def generate_project_summary(root_folder: Path, output_filename: str):
    """
    Walks the directory tree, generates the folder structure, and appends the
    content of specified files to a single output file.
    """
    structure_lines: list[str] = []
    content_files: list[Path] = []

    for current_path, dirnames, filenames in os.walk(root_folder, topdown=True):
        current_path = Path(current_path)

        # Prune excluded directories from traversal
        dirnames[:] = [d for d in dirnames if d not in PATHS_TO_EXCLUDE]

        # Skip processing the current directory if it's in the exclusion list
        rel_path_str = str(current_path.relative_to(root_folder))
        if (
            any(ex in rel_path_str for ex in PATHS_TO_EXCLUDE if ex != ".")
            and rel_path_str in PATHS_TO_EXCLUDE
        ):
            continue

        # --- Process current directory for structure output ---
        level = len(current_path.relative_to(root_folder).parts)
        indent = "    " * level
        if current_path == root_folder:
            structure_lines.append(f"{root_folder.name}/\n")
        else:
            structure_lines.append(f"{indent}{current_path.name}/\n")

        # --- Process files in the current directory ---
        file_indent = "    " * (level + 1)
        for fname in sorted(filenames):
            if fname in PATHS_TO_EXCLUDE:
                continue

            file_path = current_path / fname
            structure_lines.append(f"{file_indent}{fname}\n")

            if file_path.suffix in EXTENSIONS_TO_INCLUDE:
                content_files.append(file_path)

    # --- Write all collected information to the output file ---
    try:
        with open(output_filename, "w", encoding="utf-8", errors="ignore") as f:
            f.write("### FOLDER STRUCTURE ###\n")
            f.writelines(structure_lines)
            f.write("\n\n### FILE CONTENTS ###\n")

            for file_path in sorted(content_files):
                relative_path = file_path.relative_to(root_folder).as_posix()
                f.write(f"\n--- Content of: {relative_path} ---\n")
                try:
                    f.write(file_path.read_text(encoding="utf-8", errors="ignore"))
                    f.write("\n")
                except Exception as e:
                    f.write(f"Could not read file: {e}\n\n")

        print(f"Successfully created '{output_filename}'")
    except OSError as e:
        print(f"Error writing to file: {e}")


if __name__ == "__main__":
    generate_project_summary(ROOT_FOLDER, OUTPUT_FILENAME)
