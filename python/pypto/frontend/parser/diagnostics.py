#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 CANN community contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PTO Script Parser Source and Diagnostics.

This module provides source code management and diagnostic reporting utilities
for the PTO Script Parser. It handles:

1. Source Code Management:
   - Reading and caching source code from functions, strings, or files
   - Converting source to Python AST and PTO doc AST
   - Tracking line numbers and source locations

2. Diagnostic Reporting:
   - Error, warning, and info message formatting
   - Pretty-printing of errors with source context
   - Color-coded terminal output for better readability

Key Classes:
    - Source: Represents source code with location information
    - Diagnostics: Manages diagnostic messages and error reporting
    - DiagnosticLevel: Enumeration of diagnostic severity levels

The diagnostic system provides rich error messages with source context,
making it easier for users to identify and fix issues in their PTO scripts.
"""

import inspect
import enum
import linecache
import sys
from typing import NoReturn, Union

from . import doc
from .error import ParserError, RenderedParserError

CONTEXT_LINES_BEFORE = 2
CONTEXT_LINES_AFTER = 4


class DiagnosticLevel(enum.IntEnum):
    """The diagnostic level, see diagnostic.h for more details."""

    BUG = 10
    ERROR = 20
    WARNING = 30
    INFO = 40
    DEBUG = 50


class Source:
    """Source code class for PTO Script.

    It is constructed by source code str or doc AST tree.

    Parameters
    ----------
    source_name : str
        The filename of the file where the source code locates.

    start_line : int
        The first line number of the source code.

    start_column : int
        The first column number of the first line of the source code.

    source : str
        The source code str of source code.

    full_source : str
        The complete source code of the file where the source code locates.
    """

    source_name: str
    start_line: int
    start_column: int
    source: str
    full_source: str

    def __init__(self, program: Union[str, doc.AST]):
        if isinstance(program, str):
            self.source_name = "<str>"
            self.start_line = 1
            self.start_column = 0
            self.source = program
            self.full_source = program
            return

        self.source_name = inspect.getsourcefile(program)  # type: ignore
        lines, self.start_line = getsourcelines(program)  # type: ignore
        if lines:
            self.start_column = len(lines[0]) - len(lines[0].lstrip())
        else:
            self.start_column = 0
        if self.start_column and lines:
            self.source = "\n".join([l[self.start_column :].rstrip() for l in lines])
        else:
            self.source = "".join(lines)
        try:
            # It will cause a problem when running in Jupyter Notebook.
            # `mod` will be <module '__main__'>, which is a built-in module
            # and `getsource` will throw a TypeError
            mod = inspect.getmodule(program)
            if mod:
                self.full_source = inspect.getsource(mod)
            else:
                self.full_source = self.source
        except TypeError:
            # It's a work around for Jupyter problem.
            # Since `findsource` is an internal API of inspect, we just use it
            # as a fallback method.
            src, _ = inspect.findsource(program)  # type: ignore
            self.full_source = "".join(src)

    def as_ast(self) -> doc.AST:
        """Parse the source code into AST.

        Returns
        -------
        res : doc.AST
            The AST of source code.
        """
        return doc.parse(self.source)


_getfile = inspect.getfile  # pylint: disable=invalid-name
_findsource = inspect.findsource  # pylint: disable=invalid-name


def _patched_inspect_getfile(obj):
    """Work out which source or compiled file an object was defined in."""
    if not inspect.isclass(obj):
        return _getfile(obj)
    mod = getattr(obj, "__module__", None)
    if mod is not None:
        file = getattr(sys.modules[mod], "__file__", None)
        if file is not None:
            return file
    for _, member in inspect.getmembers(obj):
        if inspect.isfunction(member):
            if obj.__qualname__ + "." + member.__name__ == member.__qualname__:
                return inspect.getfile(member)
    raise TypeError(f"Source for {obj:!r} not found")


def findsource(obj):
    """Return the entire source file and starting line number for an object."""

    if not inspect.isclass(obj):
        return _findsource(obj)

    file = inspect.getsourcefile(obj)
    if file:
        linecache.checkcache(file)
    else:
        file = inspect.getfile(obj)
        if not (file.startswith("<") and file.endswith(">")):
            raise OSError("source code not available")

    module = inspect.getmodule(obj, file)
    if module:
        lines = linecache.getlines(file, module.__dict__)
    else:
        lines = linecache.getlines(file)
    if not lines:
        raise OSError("could not get source code")
    qual_names = obj.__qualname__.replace(".<locals>", "<locals>").split(".")
    in_comment = 0
    scope_stack = []
    indent_info = {}
    for i, line in enumerate(lines):
        n_comment = line.count('"""')
        if n_comment:
            # update multi-line comments status
            in_comment = in_comment ^ (n_comment & 1)
            continue
        if in_comment:
            # skip lines within multi-line comments
            continue
        indent = len(line) - len(line.lstrip())
        tokens = line.split()
        if len(tokens) > 1:
            name = None
            if tokens[0] == "def":
                name = tokens[1].split(":")[0].split("(")[0] + "<locals>"
            elif tokens[0] == "class":
                name = tokens[1].split(":")[0].split("(")[0]
            # pop scope if we are less indented
            while scope_stack and indent_info[scope_stack[-1]] >= indent:
                scope_stack.pop()
            if name:
                scope_stack.append(name)
                indent_info[name] = indent
                if scope_stack == qual_names:
                    return lines, i

    raise OSError("could not find class definition")


def getsourcelines(obj):
    """Extract the block of code at the top of the given list of lines."""
    obj = inspect.unwrap(obj)
    lines, l_num = findsource(obj)
    return inspect.getblock(lines[l_num:]), l_num + 1


inspect.getfile = _patched_inspect_getfile


class Span:
    """Span source name for diagnostics."""

    source_name: str
    line: int
    end_line: int
    column: int
    end_column: int

    def __init__(
        self,
        source_name: str,
        lineno: int,
        end_lineno: int,
        col_offset: int,
        end_col_offset: int,
    ):
        self.source_name = source_name
        self.line = lineno
        self.end_line = end_lineno
        self.column = col_offset
        self.end_column = end_col_offset


class DiagnosticItem:
    """Diagnostic item class for diagnostics.

    Parameters
    ----------
    level : DiagnosticLevel
        The diagnostic level.
    """

    level: DiagnosticLevel
    span: Span
    message: str

    def __init__(self, level: DiagnosticLevel, span: Span, message: str):
        self.level = level
        self.span = span
        self.message = message

    def render_to_console(self) -> None:
        """Render the diagnostic item to string."""
        # ANSI color codes for different diagnostic levels
        colors = {
            DiagnosticLevel.BUG: "\033[95m",  # Magenta
            DiagnosticLevel.ERROR: "\033[91m",  # Red
            DiagnosticLevel.WARNING: "\033[93m",  # Yellow
            DiagnosticLevel.INFO: "\033[94m",  # Blue
            DiagnosticLevel.DEBUG: "\033[92m",  # Green
        }
        names = {
            DiagnosticLevel.BUG: "INTERNAL BUG",
            DiagnosticLevel.ERROR: "ERROR",
            DiagnosticLevel.WARNING: "WARNING",
            DiagnosticLevel.INFO: "INFO",
            DiagnosticLevel.DEBUG: "DEBUG",
        }
        reset_color = "\033[0m"
        bold = "\033[1m"

        color = colors.get(self.level, "")
        level_name = names.get(self.level, "")

        # Print header with color
        print(
            f"{bold}{color}{level_name}{reset_color} {bold}{self.span.source_name}"
            f":{self.span.line}:{self.span.column}:{reset_color} {self.message}"
        )

        # Read and display source code
        with open(self.span.source_name, "r", encoding="utf-8") as f:
            lines = f.readlines()

        context_before = CONTEXT_LINES_BEFORE
        context_after = CONTEXT_LINES_AFTER
        start_line = max(0, self.span.line - 1 - context_before)
        end_line = min(len(lines), self.span.end_line + context_after)

        # Calculate line number width for alignment
        line_num_width = len(str(end_line))

        # Print context lines
        for i in range(start_line, end_line):
            line_num = i + 1
            line_content = lines[i].rstrip("\n")

            # Check if this is an error line
            is_error_line = self.span.line <= line_num <= self.span.end_line

            if is_error_line:
                print(
                    f"{color}{line_num:>{line_num_width}} |{reset_color} {line_content}"
                )
            else:
                print(f"{line_num:>{line_num_width}} | {line_content}")

            # Add caret indicator for error lines
            if is_error_line:
                # Calculate spaces before caret
                if line_num == self.span.line:
                    start_col = self.span.column - 1
                else:
                    start_col = 0

                if line_num == self.span.end_line:
                    end_col = self.span.end_column - 1
                else:
                    end_col = len(line_content)

                # Print caret line
                spaces = " " * (line_num_width + 3 + start_col)
                carets = "^" * max(1, end_col - start_col)
                print(f"{color}{spaces}{carets}{reset_color}")

        print()  # Empty line for separation


class DiagnosticContext:
    """Diagnostic context for diagnostics.

    Parameters
    ----------
    source : Source
        The source code.
    """

    source: Source
    diagnostics: list[DiagnosticItem]

    def __init__(self, source: Source):
        self.source = source
        self.diagnostics = []

    def emit(self, diagnostic: DiagnosticItem) -> None:
        """Emit a diagnostic.

        Parameters
        ----------
        diagnostic : DiagnosticItem
            The diagnostic to emit.
        """
        self.diagnostics.append(diagnostic)

    def render(self) -> None:
        """Render the diagnostics to console."""
        for diagnostic in self.diagnostics:
            diagnostic.render_to_console()
        self.diagnostics.clear()


class Diagnostics:
    """Diagnostics class for error reporting in parser.

    Parameters
    ----------
    source : Source
        The source code.

    ctx : DiagnosticContext
        The diagnostic context for diagnostics.
    """

    source: Source
    context: DiagnosticContext

    def __init__(self, source: Source):
        self.source = source
        self.context = DiagnosticContext(source)

    def __del__(self) -> None:
        """Render the diagnostics and destroy the diagnostics."""
        self._render()

    def emit(
        self, node: doc.AST, message: str, level: DiagnosticLevel = DiagnosticLevel.INFO
    ) -> None:
        """Emit a diagnostic.

        Parameters
        ----------
        node : doc.AST
            The node with diagnostic information.

        message : str
            The diagnostic message.

        level : DiagnosticLevel
            The diagnostic level.
        """
        lineno = getattr(node, "lineno", 1)
        col_offset = getattr(node, "col_offset", self.source.start_column)
        end_lineno = getattr(node, "end_lineno", lineno)
        end_col_offset = getattr(node, "end_col_offset", col_offset)
        lineno += self.source.start_line - 1
        end_lineno += self.source.start_line - 1
        col_offset += self.source.start_column + 1
        end_col_offset += self.source.start_column + 1
        self.context.emit(
            DiagnosticItem(
                level=level,
                span=Span(
                    source_name=self.source.source_name,
                    lineno=lineno,
                    end_lineno=end_lineno,
                    col_offset=col_offset,
                    end_col_offset=end_col_offset,
                ),
                message=message,
            )
        )

    def _render(self) -> None:
        """Render the diagnostics to console."""
        self.context.render()

    def bug(self, node: doc.AST, message: str) -> NoReturn:
        """Emit a diagnostic bug.

        Parameters
        ----------
        node : doc.AST
            The node with diagnostic bug.

        message : str
            The diagnostic message.

        Raises
        ------
        ParserError
            raise the ParserError.
        """
        self.emit(node, message, DiagnosticLevel.BUG)
        self._render()
        raise RenderedParserError(node, message)

    def error(self, node: doc.AST, message: str) -> NoReturn:
        """Emit a diagnostic error.

        Parameters
        ----------
        node : doc.AST
            The node with diagnostic error.

        message : str
            The diagnostic message.

        Raises
        ------
        ParserError
            raise the ParserError.
        """
        self.emit(node, message, DiagnosticLevel.ERROR)
        self._render()
        raise RenderedParserError(node, message)

    def warning(self, node: doc.AST, message: str) -> None:
        """Emit a diagnostic warning.

        Parameters
        ----------
        node : doc.AST
            The node with diagnostic warning.
        """
        self.emit(node, message, DiagnosticLevel.WARNING)

    def info(self, node: doc.AST, message: str) -> None:
        """Emit a diagnostic info.

        Parameters
        ----------
        node : doc.AST
            The node with diagnostic info.

        message : str
            The diagnostic message.
        """
        self.emit(node, message, DiagnosticLevel.INFO)

    def debug(self, node: doc.AST, message: str) -> None:
        """Emit a diagnostic debug.

        Parameters
        ----------
        node : doc.AST
            The node with diagnostic debug.

        message : str
            The diagnostic message.
        """
        self.emit(node, message, DiagnosticLevel.DEBUG)
