"""
Prints the exact Python call stack at the moment `torch` (or `transformers`,
or `sentence_transformers`) is first imported. This is 100% reliable because
it comes straight from Python's own import machinery - no log files, no
PowerShell encoding/wrapping issues involved at all.

Usage:
    python trace_heavy_import.py torch
    python trace_heavy_import.py transformers
"""
import sys
import traceback
import importlib.abc
import importlib.machinery

TARGET = sys.argv[1] if len(sys.argv) > 1 else "torch"

class TracingFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == TARGET or name.startswith(TARGET + "."):
            print(f"\n{'='*70}")
            print(f"'{name}' is about to be imported. Call stack (most recent call last):")
            print(f"{'='*70}")
            # Skip the last 2 frames (this finder's own machinery)
            stack = traceback.extract_stack()[:-1]
            for frame in stack:
                print(f"  File \"{frame.filename}\", line {frame.lineno}, in {frame.name}")
                if frame.line:
                    print(f"    {frame.line}")
            print(f"{'='*70}\n")
            # Remove ourselves so we only report the FIRST import site
            sys.meta_path.remove(self)
        return None  # let the normal import machinery continue handling it

sys.meta_path.insert(0, TracingFinder())

print(f"Importing app.main and watching for the first import of '{TARGET}'...\n")
import app.main  # noqa: E402

print(f"\nDone. If you saw no stack trace above, '{TARGET}' was never imported.")
