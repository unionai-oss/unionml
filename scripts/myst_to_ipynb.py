"""Convert myst markdown to ipython notebook."""

import hashlib
import os
from pathlib import Path

import jupytext
from nbformat import NotebookNode

COLAB_BADGE = "https://colab.research.google.com/assets/colab-badge.svg"
COLAB_URL = "https://colab.research.google.com"
REPO_URL = "github/unionai-oss/unionml/blob/main"


def create_or_replace_cell_badge(cell: NotebookNode, output_path: str) -> NotebookNode:
    """Prepends a badge to the beginning of the cell."""
    badge = f"[![Open In Colab]({COLAB_BADGE})]({COLAB_URL}/{REPO_URL}/{output_path})"
    cell.update({"source": badge})
    return cell


def convert_notebook_str(
    output_path: Path,
    notebook_str: str,
) -> NotebookNode:
    """Makes notebook cell ids deterministic."""
    notebook = jupytext.reads(notebook_str, fmt="myst")
    for i, cell in enumerate(notebook.cells):
        tags = cell.get("metadata", {}).get("tags", [])
        if "add-colab-badge" in tags:
            create_or_replace_cell_badge(cell, output_path)

    # get notebook hash based on updated cell content
    myst_str = jupytext.writes(notebook, fmt="myst")
    notebook_hash = hashlib.md5(myst_str.encode())

    for i, cell in enumerate(notebook.cells):
        cell_id = notebook_hash.copy()
        cell_id.update(str(i).encode())
        cell.update({"id": cell_id.hexdigest()})
    return notebook


def main(file: Path, output_path: Path):
    """Convert a myst markdown file to a jupyter notebook."""
    with open(file) as f:
        notebook_str = f.read()
    notebook = convert_notebook_str(output_path, notebook_str)
    jupytext.write(notebook, file, fmt="myst")
    jupytext.write(notebook, output_path, fmt="ipynb")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        type=Path,
        help="filepath to myst markdown notebook or directory containing myst markdown notebooks.",
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="filepath to output notebook or to directory to write notebooks to."
    )

    args = parser.parse_args()

    if args.input.is_dir():
        if not args.output.is_dir():
            raise ValueError("output must be a directory if input is a directory.")

        existing_notebooks = [*args.output.glob("*.ipynb")]
        converted_notebooks = []
        for fp in args.input.glob("*.md"):
            out_fp = (args.output / fp.stem).with_suffix(".ipynb")
            main(fp, out_fp)
            converted_notebooks.append(out_fp)

        # clean up notebooks that weren't no longer exist in the input directory
        for fp in existing_notebooks:
            if fp not in converted_notebooks:
                os.remove(fp)
    else:
        output = args.output
        if output.is_file() and output.suffix != ".ipynb":
            raise ValueError("output file extension must be .ipynb")
        elif output.is_dir():
            output = (output / args.input.stem).with_suffix(".ipynb")
        main(args.input, output)
