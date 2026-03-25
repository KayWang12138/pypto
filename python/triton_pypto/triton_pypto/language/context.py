from typing import Optional
from . import dtypes


class Context:
    program_id = (0, 0, 0)
    num_programs = (1, 1, 1)
    dynamic = False

    class OverrideScope:
        _vec_stack = []
        _cube_stack = []

        @classmethod
        def empty_vec(cls) -> bool:
            return len(cls._vec_stack) == 0

        @classmethod
        def empty_cube(cls) -> bool:
            return len(cls._cube_stack) == 0

        @classmethod
        def empty(cls) -> bool:
            return cls.vec_empty() and cls.cube_empty()

        @classmethod
        def push_vec(cls, vec: dtypes.VecTile) -> None:
            if len(vec) == 0:
                raise ValueError("vec tile shape is empty")
            cls._vec_stack.append(vec)

        @classmethod
        def push_cube(cls, cube: dtypes.CubeTile) -> None:
            if len(cube) != 3:
                raise ValueError(f"cube must have 3 dimensions, got {len(cube)}")
            for i, dim in enumerate(cube):
                if len(dim) != 2:
                    raise ValueError(f"cube[{i}] must have 2 elements, got {len(dim)}")
            cls._cube_stack.append(cube)

        @classmethod
        def pop_vec(cls) -> dtypes.VecTile:
            if cls.empty_vec():
                raise RuntimeError("Cannot pop from empty vec stack")
            return cls._vec_stack.pop()

        @classmethod
        def pop_cube(cls) -> dtypes.CubeTile:
            if cls.empty_cube():
                raise RuntimeError("Cannot pop from empty cube stack")
            return cls._cube_stack.pop()

        @classmethod
        def top_vec(cls) -> Optional[dtypes.VecTile]:
            return cls._vec_stack[-1] if not cls.empty_vec() else None

        @classmethod
        def top_cube(cls) -> Optional[dtypes.CubeTile]:
            return cls._cube_stack[-1] if not cls.empty_cube() else None

        @classmethod
        def get_vec_depth(cls) -> int:
            return len(cls._vec_stack)

        @classmethod
        def get_cube_depth(cls) -> int:
            return len(cls._cube_stack)
