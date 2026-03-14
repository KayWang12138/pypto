from dataclasses import dataclass


@dataclass(slots=True)
class CompileConfig:
    optimize: bool = True
    debug_info: bool = False
    glslang_validator_path: str | None = None
    dump_artifacts: bool = False
    dump_dir: str | None = None


@dataclass(slots=True)
class RuntimeConfig:
    enable_validation: bool = False
    enable_cache: bool = True
    fallback_to_torch: bool = True
    prefer_real_vulkan: bool = True
    allow_cpu_fallback_for_dev: bool = False
    glslang_validator_path: str | None = None
    dump_artifacts: bool = False
    dump_dir: str | None = None
