from ..op_wrapper import op_wrapper
from ..tensor import Tensor, Scalar
from typing import Union, List, Tuple, Optional
from .. import pypto_impl
from ..controller import Controller

@op_wrapper
def add(
    input: Tensor, other: Union[Tensor, int, float], *, alpha: Union[int, float] = 1
) -> Tensor:
    """Computes the element-wise addition of `input` and `other`.

    This function calculates the formula: `out = input + alpha * other`.
    It supports broadcasting between the input tensors.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor or Number
        The second input tensor or a scalar to be added.
    alpha : float, optional, keyword-only
        A scaling factor for the `other` input. Default is 1.0.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise sum.

    Raises
    ------
    RuntimeError
        If the two tensors are not broadcastable to a common shape.

    See Also
    --------
    sub : The inverse operation, element-wise subtraction.
    mul : Element-wise multiplication.

    Examples
    --------
    a = pto.tensor([1, 3], pto.DT_FP32)
    b = pto.tensor([1, 3], pto.DT_FP32)
    out = pto.add(a, b)

    Input a:    [1 2 3]
    Input b:    [2 3 4]
    Output out: [3 5 7]
    """
    # Get current IRBuilder from default builder
    from ..builder import get_default_builder
    builder = get_default_builder()._impl
    
    if isinstance(other, pypto_impl.Tensor):
        if alpha == 1 or alpha == 1.0:
            return pypto_impl.Add(input, other, builder)
        else:
            return pypto_impl.Add(
                input, pypto_impl.Mul(other, pypto_impl.Scalar(alpha), builder), builder
            )
    else:
        if alpha == 1 or alpha == 1.0:
            return pypto_impl.Add(input, pypto_impl.Scalar(other), builder)
        else:
            if not isinstance(other, (int, float)):
                raise TypeError(f"alpha must be int or float, but got {type(other)}.")
            return pypto_impl.Add(input, pypto_impl.Scalar(other * alpha), builder)


@op_wrapper
def sub(
    input: Tensor, other: Union[Tensor, int, float], *, alpha: Union[int, float] = 1
) -> Tensor:
    """Computes the element-wise subtraction of `input` and `other`.

    This function calculates the formula: `out = input - alpha * other`.
    It supports broadcasting between the input tensors.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor or Number
        The second input tensor or a scalar to be subtracted.
    alpha : float, optional, keyword-only
        A scaling factor for the `other` input. Default is 1.0.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise subtraction.

    Raises
    ------
    RuntimeError
        If the two tensors are not broadcastable to a common shape.

    Examples
    --------
    x = pto.tensor([2, 3], pto.DT_FP32)
    y = pto.tensor([2, 3], pto.DT_FP32)
    out1 = pto.sub(a, b)

    Input x:      [[9 9 9],
                   [9 9 9]]
    Input y:      [[1 2 3],
                   [1 2 3]]
    Output out1 : [[8 7 6],
                   [8 7 6]]

    # Using a scalar and alpha
    c = pto.sub(x, 2, alpha=3) # Computes x - 2 * 3

    Output c:[[3 3 3],
              [3 3 3]]
    """    
    # Get current IRBuilder from default builder
    from ..builder import get_default_builder
    builder = get_default_builder()._impl
    
    if isinstance(other, pypto_impl.Tensor):
        if alpha == 1 or alpha == 1.0:
            return pypto_impl.Sub(input, other, builder)
        else:
            return pypto_impl.Sub(
                input, pypto_impl.Mul(other, pypto_impl.Scalar(alpha), builder), builder
            )
    else:
        if alpha == 1 or alpha == 1.0:
            return pypto_impl.Sub(input, pypto_impl.Scalar(other), builder)
        else:
            if not isinstance(other, (int, float)):
                raise TypeError(f"alpha must be int or float, but got {type(other)}.")
            return pypto_impl.Sub(input, pypto_impl.Scalar(other * alpha), builder)


@op_wrapper
def mul(input: Tensor, other: Union[Tensor, int, float]) -> Tensor:
    """Computes the element-wise multiplication of `input` and `other`.

    This function calculates the formula: `out = input * other`.
    It supports broadcasting between the input tensors.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor or Number
        The second input tensor or a scalar to be multiplied.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise multiplication.

    Raises
    ------
    RuntimeError
        If the two tensors are not broadcastable to a common shape.

    Examples
    --------
    x = pto.tensor([2, 3], pto.DT_FP32)
    y = pto.tensor([2, 3], pto.DT_FP32)
    z = pto.mul(a, b)

    Input x:[[1 2 3],
             [1 2 3]]
    Input y:[[1 2 3],
             [1 2 3]]
    Output z:[[1 4 9],
              [1 4 9]]
    """    
    # Get current IRBuilder from default builder
    from ..builder import get_default_builder
    builder = get_default_builder()._impl
    
    if isinstance(other, pypto_impl.Tensor):
        return pypto_impl.Mul(input, other, builder)
    else:
        return pypto_impl.Mul(input, pypto_impl.Scalar(other), builder)


@op_wrapper
def div(input: Tensor, other: Union[Tensor, int, float]) -> Tensor:
    """Computes the element-wise division of `input` and `other`.

    This function calculates the formula: `out = input / other`.
    It supports broadcasting between the input tensors.

    Parameters
    ----------
    input : Tensor
        The first input tensor (numerator).
    other : Tensor or Number
        The second input tensor or a scalar to divide by (denominator).

    Returns
    -------
    Tensor
        A new tensor containing the element-wise division.

    Raises
    ------
    RuntimeError
        If the two tensors are not broadcastable to a common shape.

    Examples
    --------
    x = pto.tensor([2, 3], pto.DT_FP32)
    y = pto.tensor([2, 3], pto.DT_FP32)
    z = pto.div(x, y)

    Input x:[[6 8 9],
             [3 4 6]]
    Input y:[[2 2 3],
             [1 2 2]]
    Output z:[[3 4 3],
              [3 2 3]]
    """    
    # Get current IRBuilder from default builder
    from ..builder import get_default_builder
    builder = get_default_builder()._impl
    
    if isinstance(other, pypto_impl.Tensor):
        return pypto_impl.Div(input, other, builder)
    else:
        return pypto_impl.Div(input, pypto_impl.Scalar(other), builder)


@op_wrapper
def cast(input: Tensor, dtype, mode=None) -> Tensor:
    """Cast a tensor to a different data type.

    This function converts the elements of the input tensor to the specified
    data type using the specified cast mode.

    Parameters
    ----------
    input : Tensor
        The input tensor to cast.
    dtype : DataType
        The target data type to cast to.
    mode : CastMode, optional
        The cast mode to use. Options are:
        - CAST_NONE: No rounding mode specified
        - CAST_RINT: Round to nearest integer (ties to even)
        - CAST_ROUND: Round to nearest integer (default)
        - CAST_FLOOR: Round down to nearest integer
        - CAST_CEIL: Round up to nearest integer
        - CAST_TRUNC: Truncate towards zero
        - CAST_ODD: Round to nearest odd integer

    Returns
    -------
    Tensor
        A new tensor with the same shape as input but with the target data type.

    Examples
    --------
    x = pto.tensor([4], pto.DT_FP32)
    y = pto.cast(x, pto.DT_INT32, pto.CastMode.CAST_ROUND)
    """    
    # Get current IRBuilder from default builder
    from ..builder import get_default_builder
    builder = get_default_builder()._impl
    
    return pypto_impl.Cast(input, dtype, builder, mode)

@op_wrapper
def exp(input: Tensor) -> Tensor:
    """Computes the element-wise exponential of `input`.

    This function calculates the formula: `out = e ** input`.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise exponential.

    See Also
    --------
    log : Element-wise natural logarithm
    sqrt : Element-wise square-root

    Examples
    --------
    x = pto.tensor([3], pto.DT_FP32)
    y = pto.exp(x)

    Input x: [0 1 2]
    Output y:[1.0000 2.7183 7.3891]
    """
    # Get current IRBuilder from default builder
    from ..builder import get_default_builder
    builder = get_default_builder()._impl
    
    return pypto_impl.Exp(input, builder)


@op_wrapper
def log(input: Tensor) -> Tensor:
    """Computes the element-wise natural logarithm of `input`.

    This function calculates the formula: `out = ln(input)`.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise natural logarithm.

    See Also
    --------
    exp : Element-wise exponential

    Examples
    --------
    x = pto.tensor([3], pto.DT_FP32)
    y = pto.log(x)

    Input x: [1.0000 2.7183 7.3891]
    Output y:[0 1 2]
    """
    # Get current IRBuilder from default builder
    from ..builder import get_default_builder
    builder = get_default_builder()._impl
    
    return pypto_impl.Log(input, builder)


@op_wrapper
def sqrt(input: Tensor) -> Tensor:
    """Computes the element-wise square root of `input`.

    This function calculates the formula: `out = sqrt(input)`.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise square root.

    See Also
    --------
    rsqrt : Element-wise reciprocal square root

    Examples
    --------
    x = pto.tensor([3], pto.DT_FP32)
    y = pto.sqrt(x)

    Input x: [1 4 9]
    Output y:[1 2 3]
    """
    # Get current IRBuilder from default builder
    from ..builder import get_default_builder
    builder = get_default_builder()._impl
    
    return pypto_impl.Sqrt(input, builder)


@op_wrapper
def rsqrt(input: Tensor) -> Tensor:
    """Computes the element-wise reciprocal square root of `input`.

    This function calculates the formula: `out = 1 / sqrt(input)`.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise reciprocal square root.

    See Also
    --------
    sqrt : Element-wise square root

    Examples
    --------
    x = pto.tensor([3], pto.DT_FP32)
    y = pto.rsqrt(x)

    Input x: [1 4 9]
    Output y:[1.0000 0.5000 0.3333]
    """
    # Get current IRBuilder from default builder
    from ..builder import get_default_builder
    builder = get_default_builder()._impl
    
    return pypto_impl.Rsqrt(input, builder)


