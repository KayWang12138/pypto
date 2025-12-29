from ..scalar import Scalar
from ..op_wrapper import op_wrapper
from ..tensor import Tensor
from .. import pypto_impl
from typing import Union

@op_wrapper
def amax(input: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    """Returns the maximum value of each slice of the input tensor in the given dimension dim.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    dim : int
        The dimension to reduce.
    keepdim : bool
        whether the output tensor has dim retained or not. Default: False.

    Returns
    -------
    Tensor
        If keepdim is True, the return tensor is of the same size as input except in the dimension dim where it
        is of size 1.
        Otherwise, dim is squeezed, resulting in the return tensor having 1 fewer dimension.

    Examples
    --------
    x = pto.tensor([2, 3], pto.DT_FP32)
    y = pto.amax(x, -1, True)

    Input x:[[1 2 3],
             [1 2 3]]
    Output y:[[3],
              [3]]

    """
    # Get current IRBuilder from default builder
    from ..builder import get_default_builder
    builder = get_default_builder()._impl
    
    return pypto_impl.Amax(input, dim, keepdim, builder)


@op_wrapper
def maximum(
    input: Union[Tensor, Scalar, int, float], other: Union[Tensor, Scalar, int, float]
) -> Tensor:
    """
    Computes the element-wise maximum of input and other.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor or Scalar
        The second input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise maximum.

    Examples
    --------
    a = pto.tensor([3], pto.DT_INT32)
    b = pto.tensor([3], pto.DT_INT32)
    out = pto.maximum(a, b)

    Input a:    [0 2 4]
    Input b:    [3 1 3]
    Output out: [3 2 4]
    """
    if not isinstance(input, pypto_impl.Tensor) and not isinstance(
        other, pypto_impl.Tensor
    ):
        raise TypeError("one of `input` and `other` should be `Tensor`")

    # Get current IRBuilder from default builder
    from ..builder import get_default_builder
    builder = get_default_builder()._impl
    
    if not isinstance(input, pypto_impl.Tensor) and isinstance(other, pypto_impl.Tensor):
        input, other = other, input
    if isinstance(other, (int, float)):
        other = pypto_impl.Scalar(other)
    return pypto_impl.Maximum(input, other, builder)


@op_wrapper
def minimum(
    input: Union[Tensor, Scalar, int, float], other: Union[Tensor, Scalar, int, float]
) -> Tensor:
    """
    Computes the element-wise minimum of input and other.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor or Scalar
        The second input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise minimum.

    Examples
    --------
    a = pto.tensor([3], pto.DT_INT32)
    b = pto.tensor([3], pto.DT_INT32)
    out = pto.minimum(a, b)

    Input a:    [0 2 4]
    Input b:    [3 1 3]
    Output out: [0 1 3]
    """
    if not isinstance(input, pypto_impl.Tensor) and not isinstance(
        other, pypto_impl.Tensor
    ):
        raise TypeError("one of `input` and `other` should be `Tensor`")

    # Get current IRBuilder from default builder
    from ..builder import get_default_builder
    builder = get_default_builder()._impl
    
    if not isinstance(input, pypto_impl.Tensor) and isinstance(other, pypto_impl.Tensor):
        input, other = other, input
    if isinstance(other, (int, float)):
        other = pypto_impl.Scalar(other)
    return pypto_impl.Minimum(input, other, builder)


@op_wrapper
def sum(input: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    """Returns the sum of each slice of the input tensor in the given dimension dim.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    dim : int
        The dimension to reduce.
    keepdim : bool
        whether the output tensor has dim retained or not. Default: False.

    Returns
    -------
    Tensor
        If keepdim is True, the return tensor is of the same size as input except in the dimension dim where it
        is of size 1.
        Otherwise, dim is squeezed, resulting in the return tensor having 1 fewer dimension.

    Examples
    --------
    x = pto.tensor([2, 3], pto.DT_FP32)
    y = pto.sum(x, -1, True)

    Input x:[[1 2 3],
             [4 5 6]]
    Output y:[[6],
              [15]]

    """
    # Get current IRBuilder from default builder
    from ..builder import get_default_builder
    builder = get_default_builder()._impl
    
    return pypto_impl.Sum(input, dim, keepdim, builder)
