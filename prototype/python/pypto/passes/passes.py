from .. import pypto_impl
from .block_to_func import BlockToFunc


def pass_manage():
    """
    Get the PassManager singleton instance.
    
    The PassManager is pre-configured with default passes including DumpIRPass.
    
    Returns:
        PassManager instance
    """
    pm = pypto_impl.PassManager.Instance()
    pm.AddPass(BlockToFunc())
    return pm