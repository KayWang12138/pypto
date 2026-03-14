from dataclasses import dataclass


@dataclass(slots=True)
class CompileConfig:
    optimize: bool = True
    debug_info: bool = False


@dataclass(slots=True)
class RuntimeConfig:
    enable_validation: bool = False
    enable_cache: bool = True
    fallback_to_torch: bool = True
