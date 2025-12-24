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

"""Variable table and context management for PTO Script Parser.

This module provides context management and variable scoping utilities for
parsing PTO scripts. It implements a stack-based variable table that tracks
variables across different scopes and blocks during parsing.

The main components are:
- VarTableFrame: Represents a single scope/block of variables
- VarTable: Stack of frames managing variable lifetime and shadowing
- Context managers for automatic frame cleanup
"""


from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from pypto.frontend.parser.doc_core import AST
from pypto.frontend.parser.error import ParserError


def _deferred(exit_f: Callable[[], None]) -> Iterator[None]:
    """Created context with certain exit function.

    Parameters
    ----------
    exit_f : Callable[[], None]
        The function to call when exiting the context.

    Returns
    -------
    res : Any
        The created context.
    """

    @contextmanager
    def context():
        try:
            yield
        finally:
            exit_f()

    return context()


class ContextFrame:
    """The context frame.
    A frame of context stores the context created in one block or scope.
    """

    vars: set[str]

    def __init__(self):
        self.vars = set()

    def add(self, var_name: str, node: Optional[AST] = None) -> None:
        """Add a new context into context frame.

        Parameters
        ----------
        var_name : str
            The name of new context.
        node : Optional[AST]
            The AST node of variable, used for error reporting
        """
        if var_name in self.vars:
            if node is None:
                raise NameError(
                    f"Variable '{var_name}' already exists in the current scope"
                )
            else:
                raise ParserError(
                    node,
                    NameError(
                        f"Variable '{var_name}' already exists in the current scope"
                    ),
                )
        self.vars.add(var_name)

    def pop_all(self, fn_pop: Callable[[str], None]):
        """Pop out all variable in context frame.

        Parameters
        ----------
        fn_pop : Callable[[str], None]
            The methods to call when popping each variable.
        """
        for var_name in self.vars:
            fn_pop(var_name)
        self.vars.clear()


class Context:
    """The context.
    A context stores the all contexts when parsing PTO script.

    Parameters
    ----------
    frames : list[ContextFrame]
        The list or stack of context frame.
    """

    frames: list[ContextFrame]
    name2value: dict[str, list[Any]]
    marked_for_deletion: set[str]

    def __init__(self):
        self.frames = []
        self.name2value = defaultdict(list)
        self.marked_for_deletion = set()

    def with_frame(self) -> Iterator[None]:
        """Create a new variable table frame as with statement.

        Returns
        -------
        res : Iterator[None]
            The context manager for the new variable table frame.
        """

        def pop_frame() -> None:
            frame = self.frames.pop()
            frame.pop_all(lambda name: self.name2value[name].pop())

        self.frames.append(ContextFrame())
        return _deferred(pop_frame)

    def add(
        self,
        var: str,
        value: Any,
        node: Optional[AST] = None,
        allow_update: bool = True,
    ) -> None:
        """Add a new variable to variable table.

        Parameters
        ----------
        var : str
            The name of variable.
        value : Any
            The value of variable.
        node : Optional[AST]
            The AST node of variable, used for error reporting
        allow_update : bool
            The options of whether variable update allowed for this variable.
        """
        if allow_update and var in self.frames[-1].vars:
            # Update
            self.name2value[var][-1] = value
        else:
            self.frames[-1].add(var, node)
            self.name2value[var].append(value)

    def get(self) -> dict[str, Any]:
        """Get a context dictionary of latest contexts.

        Returns
        -------
        res : Any
            The variable dictionary copy of latest variables.
        """
        return {key: values[-1] for key, values in self.name2value.items() if values}

    def delete(self, var: str) -> None:
        """Delete a variable from the current context frame.

        Parameters
        ----------
        var : str
            The name of variable to delete.
        """
        if not self.frames:
            raise ValueError("Cannot delete variable outside of a frame")
        if var not in self.frames[-1].vars:
            raise NameError(f"Variable '{var}' is not defined in the current scope")

        # Remove from current frame's var set
        self.frames[-1].vars.discard(var)
        # Remove the last value from the stack
        if var in self.name2value and self.name2value[var]:
            self.name2value[var].pop()

    def mark_for_deletion(self, var_names: set[str]) -> None:
        """Mark variables for deletion.

        Parameters
        ----------
        var_names : set[str]
            The set of variable names to mark for deletion.
        """
        self.marked_for_deletion.update(var_names)

    def cleanup_marked(self) -> None:
        """Delete all variables marked for deletion."""
        for var_name in list(self.marked_for_deletion):
            # Check if variable exists in any frame
            if var_name in self.name2value and self.name2value[var_name]:
                # Find which frame contains this variable
                for frame in reversed(self.frames):
                    if var_name in frame.vars:
                        # Remove from frame
                        frame.vars.discard(var_name)
                        # Remove from value stack
                        self.name2value[var_name].pop()
                        break
        # Clear the marked set
        self.marked_for_deletion.clear()
