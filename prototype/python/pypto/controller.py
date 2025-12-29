
import inspect
import itertools
from contextlib import contextmanager

from . import pypto_impl
from typing import Union, Iterator, List, Optional
from .tensor import Scalar, SymInt
from .function import get_last_function
from . import utils

# Import ScalarValueKind from pypto_impl
ScalarValueKind = pypto_impl.ScalarValueKind

# Import IRBuilder for new API
from .builder import get_default_builder

__all__ = [
    "set_vec_tile_shapes",
    "loop",
    "cond"
]

class Controller:
    _loop_idx_generator = itertools.count(0)
    _if_stack: List[object] = []
    _block_stack: List[Optional[object]] = []

    @classmethod
    def next_loop_idx(cls) -> int:
        return next(cls._loop_idx_generator)

    @classmethod
    def reset(cls):
        cls._loop_idx_generator = itertools.count(0)
        cls._if_stack = []
        cls._block_stack = []

    @classmethod
    def set_active_block(cls, block):
        """Set the current active BlockStatement (deprecated).
        
        This method is kept for backward compatibility but does nothing.
        The active block is now managed by IRBuilder.
        
        Parameters
        ----------
        block : BlockStatement or None
            Ignored.
        """
        # Deprecated: active block is now managed by IRBuilder
        pass

    @classmethod
    def get_active_block(cls):
        """Get the current active BlockStatement from IRBuilder.
        
        Returns
        -------
        BlockStatement or None
            The current active BlockStatement from IRBuilder, or None if none is set.
        """
        # Get active block from IRBuilder
        builder = get_default_builder()
        return builder.get_active_block()

    @classmethod
    def push_if(cls, if_stmt, then_block) -> None:
        """Push a new IfStatement context (deprecated).
        
        The active block is now managed by IRBuilder, so this method
        only maintains the if_stmt stack for backward compatibility.
        
        Parameters
        ----------
        if_stmt : IfStatement
            The if statement to push
        then_block : BlockStatement or None
            Ignored (kept for backward compatibility)
        """
        cls._if_stack.append(if_stmt)
        cls._block_stack.append(None)  # Just maintain stack size

    @classmethod
    def pop_if(cls) -> None:
        """Pop the current IfStatement context (deprecated).
        
        The active block is now managed by IRBuilder.
        """
        if not cls._if_stack:
            return
        cls._if_stack.pop()
        if cls._block_stack:
            cls._block_stack.pop()

    @classmethod
    def current_if(cls):
        """Get the current active IfStatement, if any."""
        if not cls._if_stack:
            return None
        return cls._if_stack[-1]

def set_vec_tile_shapes(*shapes: int):
    """ set the tile shapes in vector computation

    This operation sets the value of the tile shapes
    in each dimension in vector computation.

    Parameters
    ----------
    shapes: *int
        the values of the tile shape in each dimension

    Returns
    -------
    None

    Examples
    --------
    >>> pypto.set_vec_tile_shapes(1, 1, 8, 8)
    >>> print(pypto.get_vec_tile_shapes())
    [1, 1, 8, 8]

    """
    # implementation
    pypto_impl.SetVecTile(*shapes)

def _get_loop_range(*args):
    nargs = len(args)
    if nargs == 1:
        start, stop, step = 0, args[0], 1
    elif nargs == 2:
        start, stop, step = args[0], args[1], 1
    elif nargs == 3:
        start, stop, step = args
    else:
        raise TypeError(
            f"loop() takes 1 to 3 positional arguments but {nargs} were given")
    return start, stop, step

def _loop_range(start: SymInt, stop: SymInt, step: SymInt) -> pypto_impl.LoopRange:
    """Convert loop range values to LoopRange object.
    
    Parameters
    ----------
    start : SymInt
        Start value (int or Scalar)
    stop : SymInt
        Stop value (int or Scalar)
    step : SymInt
        Step value (int or Scalar)
    
    Returns
    -------
    LoopRange
        LoopRange object containing start, end, and step as Scalar values
    """
    def _to_scalar(value: SymInt) -> pypto_impl.Scalar:
        if isinstance(value, int):
            # Create a constant Scalar from int
            return pypto_impl.Scalar(value)
        elif isinstance(value, Scalar):
            # Return the underlying C++ Scalar object
            return value._base
        else:
            raise TypeError(f"Cannot convert {type(value)} to Scalar")
    
    start_scalar = _to_scalar(start)
    stop_scalar = _to_scalar(stop)
    step_scalar = _to_scalar(step)
    
    return pypto_impl.LoopRange(start_scalar, stop_scalar, step_scalar)

@contextmanager
def _loop_function(
    name: str,
    idx_name: str,
    loop_range: pypto_impl.LoopRange
):
    """Context manager for creating a loop ForStatement.
    
    Parameters
    ----------
    name : str
        The name of the loop
    idx_name : str
        The name of the loop index variable
    loop_range : LoopRange
        LoopRange object containing start, end, and step as Scalar values
    
    Yields
    ------
    Iterator[SymInt]
        Generator that yields loop index values
    """
    # Get the default IRBuilder
    builder = get_default_builder()
    
    # Extract Scalar values from LoopRange
    start_scalar = loop_range.GetStart()
    end_scalar = loop_range.GetEnd()
    step_scalar = loop_range.GetStep()
    
    # Wrap C++ Scalar objects in Python Scalar wrappers
    from .scalar import Scalar
    start = Scalar.from_base(start_scalar)
    end = Scalar.from_base(end_scalar)
    step = Scalar.from_base(step_scalar)
    
    # Use IRBuilder's for_loop context manager
    # The active block is automatically managed by IRBuilder
    with builder.for_loop(idx_name, start, end, step) as loop_index_scalar:
        yield loop_index_scalar

def loop(
    *args,
    **kwargs,
) -> Iterator[SymInt]:
    """ set up a loop computation. Use as a for loop in python.

    Parameters
    ----------
    kwargs:
        name: str
            The name of the loop
        idx_name: str
            The name of the loop index

    Returns
    --------
    return a generator, which will be used for setting up the
    for loop in building computing graph
    """
    start, stop, step = _get_loop_range(*args)
    # implementation
    loop_idx = Controller.next_loop_idx()
    name = kwargs.get("name", f"loop_{loop_idx}")
    idx_name = kwargs.get("idx_name", f"loop_idx_{loop_idx}")
    
    with _loop_function(
        name, idx_name, _loop_range(start, stop, step)
    ) as rlf:
        # In symbolic execution, we only yield once so the loop body executes once
        # The actual loop structure is already created in the IR
        yield rlf

def cond(scalar: SymInt):
    """ set up a conditional computation. Use as a "if" condition in python.

    Parameters
    ----------
    scalar: Union[int, SymbolicScalar]
        expression to determine if condition is true or not

    Returns
    -------
    return a generator, which will be used for setting up the
    "if" in building computing graph

    Examples
    --------
    >>> if pto.cond(pto.is_loop_begin(bn)):
            pass
        elif pto.cond(pto.is_loop_end(bn)):
            pass
        elif pto.cond(1):
            pass
        else:
            pass
    
    Note
    ----
    This function is designed to work with Python's if statement, not as a context manager.
    The active block is automatically managed by IRBuilder. The if-then-else branches are
    handled implicitly: when this function returns True, the Python if body executes and
    operations are added to the then-branch. For else branches, users would typically call
    cond() again with a negated condition or use explicit Builder context managers.
    """
    # Get the default IRBuilder
    builder = get_default_builder()
    
    sym = utils.to_sym(scalar)
    
    # Prefer SSA name as condition string
    cond_str = sym.GetSSAName() if hasattr(sym, "GetSSAName") else str(sym)
    
    # Create IfStatement using IRBuilder's internal impl
    builder_impl = builder._impl
    if_stmt = builder_impl.CreateIfStmt(cond_str)
    
    # Enter then branch
    scope_guard = builder_impl.EnterIfThen(if_stmt)
    
    # Store if statement and scope guard for potential else branch or cleanup
    Controller.push_if(if_stmt, None)  # Block is managed by Builder
    
    # Store scope guard in builder for cleanup
    if not hasattr(builder, '_cond_guards'):
        builder._cond_guards = []
    builder._cond_guards.append(scope_guard)
    
    # Always return True so that the Python 'if' body is executed once and
    # the IR captures the control flow structurally.
    return True
