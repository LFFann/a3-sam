from typing import Tuple


def validate_ugda_batch_config(batch_size: int, labeled_bs: int) -> None:
    """Validate the one-to-one labeled/unlabeled pairing required by UGDA mixup."""
    if not isinstance(batch_size, int) or not isinstance(labeled_bs, int):
        raise ValueError(
            f"UGDA batch configuration requires integer batch_size and labeled_bs. "
            f"Received batch_size={batch_size!r}, labeled_bs={labeled_bs!r}."
        )
    if batch_size <= 0:
        raise ValueError(f"UGDA requires batch_size > 0. Received batch_size={batch_size}.")
    if labeled_bs <= 0:
        raise ValueError(f"UGDA requires labeled_bs > 0. Received labeled_bs={labeled_bs}.")
    if labeled_bs >= batch_size:
        raise ValueError(
            f"UGDA requires labeled_bs < batch_size. Received batch_size={batch_size}, labeled_bs={labeled_bs}."
        )
    if batch_size != 2 * labeled_bs:
        unlabeled_bs = batch_size - labeled_bs
        raise ValueError(
            "UGDA requires equal labeled and unlabeled batch sizes: "
            "batch_size must equal 2 * labeled_bs. "
            f"Received batch_size={batch_size}, labeled_bs={labeled_bs}, "
            f"which gives labeled_bs={labeled_bs} and unlabeled_bs={unlabeled_bs}."
        )


def validate_ugda_runtime_batch(actual_batch_size: int, labeled_bs: int) -> Tuple[int, int]:
    """Validate an actual tensor batch before UGDA indexing or reshaping."""
    if not isinstance(actual_batch_size, int) or not isinstance(labeled_bs, int):
        raise ValueError(
            f"UGDA runtime validation requires integer sizes. "
            f"Received actual_batch_size={actual_batch_size!r}, labeled_bs={labeled_bs!r}."
        )
    unlabeled_bs = actual_batch_size - labeled_bs
    if actual_batch_size <= 0 or labeled_bs <= 0 or unlabeled_bs <= 0 or unlabeled_bs != labeled_bs:
        raise ValueError(
            "UGDA runtime batch mismatch before mix_up reshape/indexing: "
            f"actual_batch_size={actual_batch_size}, labeled_bs={labeled_bs}, unlabeled_bs={unlabeled_bs}. "
            "Expected actual_batch_size == 2 * labeled_bs."
        )
    return labeled_bs, unlabeled_bs
