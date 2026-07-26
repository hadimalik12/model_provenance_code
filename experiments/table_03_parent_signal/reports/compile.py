"""Purpose-named entrypoint for the augmented parent-signal report."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("compile_augmented.py")), run_name="__main__")
