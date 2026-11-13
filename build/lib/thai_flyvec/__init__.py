"""Sparse binary word embeddings for Thai, inspired by the fruit fly brain."""

import sys


def use_utf8_output() -> None:
    """Make stdout and stderr UTF-8, so Thai text prints when the output is redirected.

    On Windows a redirected or piped stream uses the ANSI code page, which has
    no Thai letters.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
