"""Convert myst markdown to ipython notebook and myst markdown with custom modifications.

This script takes myst markdown files and converts it to a notebook for the purpose of
interactive computing and myst markdown source files for this repo's documentation.

The custom modification is to add a "Open in Colab" badge to the top of a notebook
based on a custom "add-colab-badge" tag in a code cell.
"""

import hashlib
import os
from pathlib import Path

import jupytext
from nbformat import NotebookNode

COLAB_BADGE = "https://colab.research.google.com/assets/colab-badge.svg"
COLAB_URL = "https://colab.research.google.com"
REPO_URL = "github/unionai-oss/unionml/blob/main"


def add_cell_badge(cell: NotebookNode, output_path: str) -> NotebookNode:
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
            add_cell_badge(cell, output_path)

    # get notebook hash based on updated cell content
    myst_str = jupytext.writes(notebook, fmt="myst")
    notebook_hash = hashlib.md5(myst_str.encode())

    for i, cell in enumerate(notebook.cells):
        cell_id = notebook_hash.copy()
        cell_id.update(str(i).encode())
        cell.update({"id": cell_id.hexdigest()})
    return notebook


def main(file: Path, output_path: Path, doc_output_path: Path):
    """Convert a myst markdown file based on custom logic."""
    with open(file) as f:
        notebook_str = f.read()
    notebook = convert_notebook_str(output_path, notebook_str)
    jupytext.write(notebook, output_path, fmt="ipynb")
    jupytext.write(notebook, doc_output_path, fmt="myst")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        type=Path,
        help="filepath to myst markdown notebook or directory containing myst markdown notebooks.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="filepath to output notebook or to directory to write notebooks to.",
    )
    parser.add_argument(
        "-d",
        "--docs-output",
        type=Path,
        help="filepath to output myst file or to directory to write myst files for the docs.",
    )

    args = parser.parse_args()

    if args.input.is_dir():
        if not args.output.is_dir():
            raise ValueError("output must be a directory if input is a directory.")
        if not args.docs_output.is_dir():
            raise ValueError("output must be a directory if input is a directory.")

        existing_notebooks = [*args.output.glob("*.ipynb")]
        converted_notebooks = []
        for fp in args.input.glob("*.md"):
            out_fp = (args.output / fp.stem).with_suffix(".ipynb")
            docs_out_fp = (args.docs_output / fp.stem).with_suffix(".md")
            main(fp, out_fp, docs_out_fp)
            converted_notebooks.append(out_fp)

        # clean up notebooks that weren't no longer exist in the input directory
        for fp in existing_notebooks:
            if fp not in converted_notebooks:
                os.remove(fp)
    else:
        output = args.output
        docs_output = args.docs_output
        if output.is_file() and output.suffix != ".ipynb":
            raise ValueError("output file extension must be .ipynb")
        elif output.is_dir():
            output = (output / args.input.stem).with_suffix(".ipynb")

        if docs_output.is_file() and docs_output.suffix != ".ipynb":
            raise ValueError("output file extension must be .md")
        elif output.is_dir():
            docs_output = (docs_output / args.input.stem).with_suffix(".md")

        main(args.input, output, docs_output)
