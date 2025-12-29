from ..op_wrapper import op_wrapper
from ..tensor import Tensor, Scalar
from typing import Union, List, Tuple, Optional
from .. import pypto_impl
from ..controller import Controller


@op_wrapper
def view(
    input: Tensor,
    shape: Union[List[int], Tuple[int, ...]],
    offsets: Union[List[Union[int, Scalar]], Tuple[Union[int, Scalar], ...]],
    *,
    valid_shape: Optional[Union[List[Union[int, Scalar]], Tuple[Union[int, Scalar], ...]]] = None
) -> Tensor:
    """Extract a sub-tensor (view) from a tensor with specified shape and offsets.

    This function creates a view of the input tensor starting at the specified
    offsets with the specified shape. In Python frontend mode, this returns a Tensor.
    The compiler may later optimize this to use Tile representation internally.

    Parameters
    ----------
    input : Tensor
        The input tensor to create a view from.
    shape : List[int] or Tuple[int, ...]
        The shape of the view sub-tensor.
    offsets : List[Union[int, Scalar]] or Tuple[Union[int, Scalar], ...]
        The offsets for each dimension to start the view.
    valid_shape : Optional[List[Union[int, Scalar]]], optional
        The valid shape of the view (for dynamic dimensions). Default is None.

    Returns
    -------
    Tensor
        A tensor representing the view (sub-region) of the input tensor.

    Examples
    --------
    >>> a = pypto.tensor((1024, 128), pypto.DT_FP32)
    >>> sub = pypto.view(a, (256, 128), [0, 0])  # Extract first 256 rows
    
    Notes
    -----
    At IR level, this operation works as:
    - Tensor -> Tensor (Python frontend, SSA semantics)
    - Tile -> Tile (compiler optimization passes)
    """
    from .. import utils
    from ..controller import Controller
    
    # Get current IRBuilder from default builder
    from ..builder import get_default_builder
    builder = get_default_builder()._impl
    
    # Convert shape to list of int
    shape_list = list(shape)
    
    # Convert offsets to list of Scalar
    offsets_list = utils.to_syms(list(offsets))
    
    # View now returns Tensor when input is Tensor (updated trait rules)
    result = pypto_impl.View(input, shape_list, offsets_list, builder)
    return result


@op_wrapper
def assemble(
    patch: Tensor,
    offsets: Union[List[Union[int, Scalar]], Tuple[Union[int, Scalar], ...]],
    base: Tensor,
) -> Tensor:
    """Assemble a patch tensor into a base tensor at specified offsets.

    This function places a patch tensor into a base tensor at the specified offsets,
    returning a new tensor (SSA semantics). In Python frontend mode, both inputs and
    output are Tensors. The compiler may later optimize this using Tile representation.

    Parameters
    ----------
    patch : Tensor
        The patch tensor to be assembled into the base.
    offsets : List[Union[int, Scalar]] or Tuple[Union[int, Scalar], ...]
        The offsets for each dimension to place the patch.
    base : Tensor
        The base tensor into which the patch will be assembled.

    Returns
    -------
    Tensor
        A new tensor with the patch assembled at the specified location.

    Examples
    --------
    >>> a = pypto.tensor((1024, 128), pypto.DT_FP32)
    >>> patch = pypto.tensor((256, 128), pypto.DT_FP32)
    >>> result = pypto.assemble(patch, [0, 0], a)  # Update first 256 rows
    
    Notes
    -----
    At IR level, this operation works as:
    - Tensor + Tensor -> Tensor (Python frontend, SSA semantics)
    - Tensor + Tile -> Tensor (mixed mode from compiler)
    - Tile + Tile -> Tile (compiler optimization passes)
    """
    from ..controller import Controller
    from ..utils import to_syms
    
    # Get current IRBuilder from default builder
    from ..builder import get_default_builder
    builder = get_default_builder()._impl
    
    # Convert offsets to list of Scalar
    offsets_list = to_syms(list(offsets))
    
    # Assemble now supports Tensor + Tensor -> Tensor (updated trait rules)
    output_wrapper = pypto_impl.Assemble(patch, offsets_list, base, builder)
    return output_wrapper

