from typing import Optional

import numpy as np


def _format_values(values: np.ndarray, max_items: int = 16) -> str:
    shown = values[:max_items].tolist()
    suffix = "" if values.size <= max_items else f" ... total={values.size}"
    return f"{shown}{suffix}"


def validate_class_index_mask(mask, num_classes: int, source: Optional[str] = None) -> np.ndarray:
    """Validate and return a 2D class-index mask encoded as 0, 1, ..., C-1."""
    array = np.asarray(mask)
    src = source or "<unknown>"
    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError(f"num_classes must be an integer >= 2 for source={src}. Received {num_classes!r}.")
    if array.ndim != 2:
        raise ValueError(
            "Class-index masks must be 2D. RGB/color masks are not auto-mapped; "
            "convert them to class-index masks encoded as 0,1,...,C-1 first. "
            f"source={src}, num_classes={num_classes}, dtype={array.dtype}, shape={array.shape}."
        )
    if not np.isfinite(array).all():
        raise ValueError(
            f"Mask contains non-finite values. source={src}, num_classes={num_classes}, "
            f"dtype={array.dtype}, shape={array.shape}."
        )
    if not np.allclose(array, np.rint(array), atol=1e-6):
        unique_values = np.unique(array)
        raise ValueError(
            "Mask contains non-integer class values. Labels must be encoded as 0,1,...,C-1. "
            f"source={src}, num_classes={num_classes}, dtype={array.dtype}, shape={array.shape}, "
            f"unique_values={_format_values(unique_values)}."
        )

    rounded = np.rint(array).astype(np.int64)
    unique_values = np.unique(rounded)
    invalid_values = unique_values[(unique_values < 0) | (unique_values >= num_classes)]
    if invalid_values.size:
        raise ValueError(
            "Mask contains labels outside the valid class-index range [0, num_classes). "
            "Labels must be encoded as 0,1,...,C-1. "
            f"source={src}, num_classes={num_classes}, dtype={array.dtype}, shape={array.shape}, "
            f"unique_values={_format_values(unique_values)}, invalid_values={_format_values(invalid_values)}."
        )
    return rounded
