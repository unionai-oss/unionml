"""Convert myst markdown to ipython notebook."""

import hashlib
import os
from pathlib import Path

import jupytext
from nbformat import NotebookNode

# This indicates where code-blocks should be convert to code-cells
# when converted to ipynb format. This logic is for cells that should be
# non-executable in the docs but executable in notebook form.
CODE_CELL_MARKER = "<!-- ipynb:{code-cell} -->"
CODE_CELL_DIRECTIVE = "```{code-cell} python"


def convert_notebook_str(notebook_str: str) -> NotebookNode:
    """Makes notebook cell ids deterministic."""
    notebook = jupytext.reads(notebook_str, fmt="myst")
    notebook_hash = hashlib.md5(notebook_str.encode())
    for i, cell in enumerate(notebook.cells):
        cell_id = notebook_hash.copy()
        cell_id.update(str(i).encode())
        cell.id = cell_id.hexdigest()
    return notebook


def main(file: Path, output_path: Path):
    """Convert a myst markdown file to a jupyter notebook."""
    with open(file) as f:
        lines = [*f.readlines()]

    notebook_str = []
    for curr, prev in zip(lines, [None] + lines[:-1]):
        if curr.startswith(CODE_CELL_MARKER):
            continue
        if prev is not None and prev.startswith(CODE_CELL_MARKER):
            notebook_str.append(f"""{CODE_CELL_DIRECTIVE}\n""")
        else:
            notebook_str.append(curr)
    notebook_str = "".join(notebook_str)
    jupytext.write(convert_notebook_str(notebook_str), output_path, fmt="ipynb")


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
