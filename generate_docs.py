#!/usr/bin/env python3
# generate_docs.py
import os
import re
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set, Tuple, Pattern
from tqdm import tqdm

# ==============================================================================
# SCRIPT CONFIGURATION
# ==============================================================================
@dataclass
class Config:
    """Configuration for the project summary generator."""

    root_folder: Path = Path.cwd()
    output_filename: str = "project_summary.txt"
    extensions_to_include: Set[str] = field(default_factory=lambda: {
        ".py", ".yaml", ".yml", ".toml", ".md", ".gitignore", ".gitattributes", ".txt", ".tex"
    })

    # --- Exclusion Rules ---
    paths_to_exclude: Set[str] = field(
        default_factory=lambda: {
            ".git", ".github", ".vscode", "__pycache__", ".ipynb_checkpoints", 
            ".pytest_cache", ".ruff_cache", ".DS_Store", "venv", "node_modules",
            "blender", "studies", "savepoints", "logs", "plots", 
            "publication_figures", "wandb", "runs", "metrics", "runs.old",
            "geometry", "steering"
        }
    )
    
    path_prefixes_to_exclude: Set[str] = field(
        default_factory=lambda: {
            "Articulos/sochid", "Articulos/article", "Presentaciones",
        }
    )
    
    patterns_to_exclude: Set[str] = field(
        default_factory=lambda: {
            "*.aux", "*.bbl", "*.blg", "*.fdb_latexmk", "*.fls", "*.log", "*.out", "*.pdf",
            "*.png", "*.jpg", "*.gif", "*.sty", "*.bst", "*.csl", "*.spl", "*.toc", "*.lof",
            "*.lot", "*.nav", "*.snm", "*.run.xml", "*.bcf", "*.gz", "*.synctex.gz",
            "wandb.xlsx", "readme2.md"
        }
    )

    # --- Redaction & Cleaning ---
    redact_comments: bool = False
    strip_non_semantic_latex: bool = False
    redaction_patterns: List[Tuple[str, str]] = field(
        default_factory=lambda: [
            (r'[\w\.-]+@[\w\.-]+\.\w+', r'[EMAIL]'),
            (r'\\author\[\d+\]\{[^}]+\}', r'\\author{[AUTHOR]}'),
            (r'\\author\{[^}]+\}', r'\\author{[AUTHOR]}'),
            (r'\bJONATHAN\s+(?:ESTEBAN\s+)?POBLETE\s+CÁCERES\b', r'[AUTHOR]'),
            (r'PROFESOR\s+GUÍA:[^\n]*', r'PROFESOR GUÍA: [NAME]'),
            (r'\b(?:YARKO\s+NIÑO|LUIS\s+ZAMORANO|PAULA\s+AGUIRRE|VALENTIN\s+BARRIERE)[^\s]*', r'[NAME]'),
            (r'Advisors:[^\n]*', r'Advisors: [NAMES]'),
            (r'\\affiliation\[\d+\]\{[^}]+\}', r'\\affiliation{[AFFILIATION]}'),
            (r'\\institute\{[^}]+\}', r'\\institute{[INSTITUTE]}'),
            (r'(?:Departamento\s+de\s+Ingeniería\s+Civil|Advanced\s+Mining\s+Technology\s+Center|Instituto\s+Nacional\s+de\s+Hidráulica)[^,\n]*', r'[AFFILIATION]'),
        ]
    )
    
    _compiled_patterns: List[Tuple[Pattern, str]] = field(default_factory=list, init=False)
    _comment_pattern: Pattern = field(default=None, init=False)
    _latex_cleanup_patterns: List[Tuple[Pattern, str]] = field(default_factory=list, init=False)

    def __post_init__(self):
        # Default to Articulos/thesis if it exists and we are in root and haven't overridden root
        # This preserves the specific behavior of the new script while being safe
        if self.root_folder == Path.cwd():
            thesis_dir = self.root_folder / "Articulos" / "thesis"
            if thesis_dir.is_dir():
                print(f"INFO: Automatically shifting scan root to: {thesis_dir}")
                self.root_folder = thesis_dir

        self._compiled_patterns = [
            (re.compile(p, re.IGNORECASE), r) for p, r in self.redaction_patterns
        ]
        # Re-using the same comment pattern: % not preceded by \
        self._comment_pattern = re.compile(r'(?<!\\)%.*')
        
        if self.strip_non_semantic_latex:
            # Common LaTeX commands that don't add semantic value for LLMs
            cmds = [r'newpage', r'clearpage', r'vspace', r'hspace', r'noindent', r'centering']
            self._latex_cleanup_patterns = [
                (re.compile(rf'\\{cmd}(?:\{{[^}}]*\}}|\[[^\]]*\])?'), '') 
                for cmd in cmds
            ]

def _is_excluded(path: Path, root: Path, config: Config) -> bool:
    if path.name in config.paths_to_exclude:
        return True
    
    # Check prefixes
    try:
        rel_path = path.relative_to(root).as_posix()
        if any(rel_path.startswith(prefix) for prefix in config.path_prefixes_to_exclude):
            return True
    except ValueError:
        pass
    
    # Check glob patterns
    return any(path.match(p) for p in config.patterns_to_exclude)

def clean_content(content: str, config: Config, is_latex: bool = False) -> str:
    """Optimized cleaning and redaction."""
    # 1. Redaction
    for pattern, replacement in config._compiled_patterns:
        content = pattern.sub(replacement, content)

    # 2. LaTeX Specifics
    if is_latex:
        if config.redact_comments:
            content = config._comment_pattern.sub('', content)
        
        if config.strip_non_semantic_latex:
            for pattern, replacement in config._latex_cleanup_patterns:
                content = pattern.sub(replacement, content)
            
            # Additional LaTeX cleaning: Strip environments but keep content
            # e.g. \begin{center} ... \end{center} -> content
            envs = [r'center', r'flushright', r'flushleft', r'small', r'textit', r'textbf']
            for env in envs:
                content = re.sub(rf'\\begin\{{{env}\}}', '', content)
                content = re.sub(rf'\\end\{{{env}\}}', '', content)
                content = re.sub(rf'\\{env}\{{([^}}]*)\}}', r'\1', content)

    # 3. Whitespace Optimization (Lighter Output)
    lines = []
    for line in content.splitlines():
        trimmed = line.strip() 
        if trimmed:
            lines.append(trimmed)
    
    return "\n".join(lines)

def generate_project_summary(config: Config):
    structure_lines: List[str] = []
    content_files: List[Path] = []
    
    # Ensure output file and script itself are excluded
    config.paths_to_exclude.add(config.output_filename)
    config.paths_to_exclude.add("generate_docs.py")
    config.paths_to_exclude.add("generate_docs_new.py")
    
    print(f"Scanning: {config.root_folder}")
    
    for root, dirs, files in os.walk(config.root_folder, topdown=True):
        current_path = Path(root)
        
        # Prune directories in-place
        dirs[:] = [d for d in dirs if not _is_excluded(current_path / d, config.root_folder, config)]
        
        try:
            rel_to_root = current_path.relative_to(config.root_folder)
        except ValueError:
            continue

        level = len(rel_to_root.parts)
        indent = "  " * level
        
        # Folder line
        display_name = current_path.name if str(rel_to_root) != "." else f"{config.root_folder.name}/"
        if not display_name.endswith("/"):
            display_name += "/"
        structure_lines.append(f"{indent}{display_name}")
        
        for fname in sorted(files):
            file_path = current_path / fname
            if _is_excluded(file_path, config.root_folder, config):
                continue

            structure_lines.append(f"{indent}  {fname}")
            if file_path.suffix.lower() in config.extensions_to_include:
                content_files.append(file_path)

    print(f"Writing to '{config.output_filename}' ({len(content_files)} files)")
    
    try:
        with open(config.output_filename, "w", encoding="utf-8", errors="ignore") as f:
            f.write("### FOLDER STRUCTURE ###\n")
            f.write("\n".join(structure_lines) + "\n\n")
            f.write("### FILE CONTENTS ###\n")

            for file_path in tqdm(sorted(content_files), desc="Processing"):
                rel_path = file_path.relative_to(config.root_folder).as_posix()
                f.write(f"\n--- Content of: {rel_path} ---\n")
                
                try:
                    raw = file_path.read_text(encoding="utf-8", errors="ignore")
                    cleaned = clean_content(raw, config, is_latex=file_path.suffix.lower() == ".tex")
                    if cleaned:
                        f.write(cleaned + "\n")
                except Exception as e:
                    f.write(f"[ERROR reading {rel_path}: {e}]\n")

        print(f"✓ Created '{config.output_filename}' ({os.path.getsize(config.output_filename) // 1024} KB)")
        
    except Exception as e:
        print(f"✗ Failure: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pack project docs for LLMs.")
    parser.add_argument("-o", "--output", help="Output filename")
    parser.add_argument("-r", "--root", help="Root directory to scan")
    parser.add_argument("--bib", action="store_true", help="Include .bib files")
    parser.add_argument("--md", action="store_true", help="Include .md files")
    parser.add_argument("--txt", action="store_true", help="Include .txt files")
    parser.add_argument("--clean", action="store_true", help="Enable LaTeX comment and command cleaning")
    parser.add_argument("--all", action="store_true", help="Scan from script root instead of thesis folder")
    args = parser.parse_args()

    cfg = Config()
    if args.output: cfg.output_filename = args.output
    if args.root: cfg.root_folder = Path(args.root).resolve()
    
    # If using --all, we force the root to be the current directory (where script is usually run)
    # and skip the smart subfolder detection in __post_init__
    if args.all:
        cfg.root_folder = Path.cwd()
        # To prevent __post_init__ from shifting it if it finds Articulos/thesis
        # We can just run __post_init__ manually after setting it if we want to bypass logic,
        # but here I'll just let it run and then override if needed.
        # Actually, let's just make __post_init__ smarter or handle it here.

    if args.bib: cfg.extensions_to_include.add(".bib")
    if args.md: cfg.extensions_to_include.add(".md")
    if args.txt: cfg.extensions_to_include.add(".txt")
    
    if args.clean:
        cfg.redact_comments = True
        cfg.strip_non_semantic_latex = True
    
    # Re-initialize patterns and smart root shifting
    # If --all is passed, we want to scan the absolute root, so we should block the shifting.
    if args.all:
        original_post_init = cfg.__post_init__
        def dummy_post_init(): pass
        cfg.__post_init__ = dummy_post_init
        # Actually just setting it manually is easier.
        cfg._compiled_patterns = [
            (re.compile(p, re.IGNORECASE), r) for p, r in cfg.redaction_patterns
        ]
        cfg._comment_pattern = re.compile(r'(?<!\\)%.*')
        if cfg.strip_non_semantic_latex:
             cmds = [r'newpage', r'clearpage', r'vspace', r'hspace', r'noindent', r'centering']
             cfg._latex_cleanup_patterns = [(re.compile(rf'\\{cmd}(?:\{{[^}}]*\}}|\[[^\]]*\])?'), '') for cmd in cmds]
    else:
        cfg.__post_init__()
    
    generate_project_summary(cfg)
